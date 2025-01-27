package orb

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	// Import the grep-ast library
	grepast "github.com/cyber-nic/grep-ast"
	"github.com/rs/zerolog/log"
	sitter "github.com/tree-sitter/go-tree-sitter"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/multi"
	"gonum.org/v1/gonum/graph/network"
)

// chat files: files that are part of the chat
// other files: files that are not (yet) part of the chat
// warned_files: files that have been warned about and are excluded

const (
	TagKindDef = "def"
	TagKindRef = "ref"
)

// Tag represents a “tag” extracted from a source file.
type Tag struct {
	FileName string
	FilePath string
	Line     int
	Name     string
	Kind     string
}

var (
	ErrOperational = errors.New("operational error")
	ErrDatabase    = errors.New("database error")
)

// RepoMap is the Go equivalent of the Python class `RepoMap`.
type RepoMap struct {
	Refresh           string
	Verbose           bool
	Root              string
	MainModel         *ModelStub
	RepoContentPx     string
	MaxMapTokens      int
	MaxCtxWindow      int
	MapMulNoFiles     int
	MapProcessingTime float64
	LastMap           string
	querySourceCache  map[string]string
}

// ModelStub simulates the main_model used in Python code (for token_count, etc.).
type ModelStub struct{}

// TokenCount is a naive token estimator. Real code might call tiktoken or other logic.
func (m *ModelStub) TokenCount(text string) int {
	// Very naive: 1 token ~ 4 chars
	return len(text) / 4
}

// ------------------------------------------------------------------------------------
// RepoMap Constructor
// ------------------------------------------------------------------------------------
func NewRepoMap(
	maxMapTokens int,
	root string,
	mainModel *ModelStub,
	repoContentPrefix string,
	verbose bool,
	maxContextWindow int,
	mapMulNoFiles int,
	refresh string,
) *RepoMap {
	if root == "" {
		cwd, err := os.Getwd()
		if err == nil {
			root = cwd
		}
	}

	r := &RepoMap{
		Refresh:          refresh,
		Verbose:          verbose,
		Root:             root,
		MainModel:        mainModel,
		RepoContentPx:    repoContentPrefix,
		MaxMapTokens:     maxMapTokens,
		MapMulNoFiles:    mapMulNoFiles,
		MaxCtxWindow:     maxContextWindow,
		querySourceCache: make(map[string]string),
	}

	if verbose {
		fmt.Printf("RepoMap initialized with map_mul_no_files: %d\n", mapMulNoFiles)
	}
	return r
}

// GetRelFname returns fname relative to r.Root. If that fails, returns fname as-is.
func (r *RepoMap) GetRelFname(fname string) string {
	rel, err := filepath.Rel(r.Root, fname)
	if err != nil {
		return fname
	}

	return rel
}

// TokenCount tries to mimic how the Python code estimates tokens (split into short vs. large).
func (r *RepoMap) TokenCount(text string) float64 {
	if len(text) < 200 {
		return float64(r.MainModel.TokenCount(text))
	}

	lines := strings.SplitAfter(text, "\n")
	numLines := len(lines)
	step := numLines / 100
	if step < 1 {
		step = 1
	}
	var sb strings.Builder
	for i := 0; i < numLines; i += step {
		sb.WriteString(lines[i])
	}
	sampleText := sb.String()
	sampleTokens := float64(r.MainModel.TokenCount(sampleText))
	ratio := sampleTokens / float64(len(sampleText))
	return ratio * float64(len(text))
}

// GetFileTags calls GetTagsRaw and filters out short names and common words.
func (r *RepoMap) GetFileTags(fname, relFname string, filter TagFilter) ([]Tag, error) {

	// Not cached or changed; re-parse
	data, err := r.GetTagsRaw(fname, relFname, filter)
	if err != nil {
		return nil, err
	}

	if data == nil {
		data = nil
	}

	return data, nil
}

// getSourceCodeMapQuery reads the query file for the given language.
func (r *RepoMap) getSourceCodeMapQuery(lang string) (string, error) {
	tpl := "queries/tree-sitter-%s-tags.scm"

	if _, ok := r.querySourceCache[lang]; ok {
		return r.querySourceCache[lang], nil
	}

	queryFilename := fmt.Sprintf(tpl, lang)

	// check if file exists
	if _, err := os.Stat(queryFilename); err != nil {
		return "", fmt.Errorf("query file not found: %s", queryFilename)
	}

	// read file
	querySource, err := os.ReadFile(queryFilename)
	if err != nil {
		return "", fmt.Errorf("failed to read query file (%s): %v", queryFilename, err)
	}

	// cache the query source
	r.querySourceCache[lang] = string(querySource)

	return r.querySourceCache[lang], nil
}

// LoadQuery loads the Tree-sitter query text and compiles a sitter.Query.
func (r *RepoMap) LoadQuery(lang *sitter.Language, langID string) (*sitter.Query, error) {
	querySource, err := r.getSourceCodeMapQuery(langID)
	if err != nil {
		return nil, fmt.Errorf("failed to read query file (%s): %w", langID, err)
	}
	if len(querySource) == 0 {
		return nil, fmt.Errorf("empty query file: %s", langID)
	}

	q, qErr := sitter.NewQuery(lang, querySource)
	if qErr != nil {
		var queryErr *sitter.QueryError
		if errors.As(qErr, &queryErr) {
			if queryErr != nil {
				return nil, fmt.Errorf(
					"query error: %s at row: %d, column: %d, offset: %d, kind: %v",
					queryErr.Message, queryErr.Row, queryErr.Column, queryErr.Offset, queryErr.Kind,
				)
			}
			return nil, fmt.Errorf("unexpected nil *sitter.QueryError")
		}
		return nil, fmt.Errorf("failed to create query: %w", qErr)
	}
	return q, nil
}

func ReadSourceCode(fname string) ([]byte, error) {
	sourceCode, err := os.ReadFile(fname)
	if err != nil {
		return nil, fmt.Errorf("failed to read file (%s): %w", fname, err)
	}
	if len(sourceCode) == 0 {
		return nil, fmt.Errorf("empty file: %s", fname)
	}
	return sourceCode, nil
}

type TagFilter func(name string) bool

// GetTagsFromQueryCapture extracts tags from the result
// of a Tree-sitter query on a given file. It iterates through
// the captures returned by the Tree-sitter query cursor and collects
// definitions (def) and references (ref). All other captures are ignored.
// filter is a function that accepts the name of a capture and returns bool false if it should be skipped.
func GetTagsFromQueryCapture(relFname, fname string, q *sitter.Query, tree *sitter.Tree, sourceCode []byte, filter TagFilter) []Tag {

	// Create a new query cursor that will be used to iterate through
	// the captures of our query on the provided parse tree. The query
	// cursor manages iteration state for match captures.
	qc := sitter.NewQueryCursor()
	defer qc.Close()

	// Execute the query against the provided parse tree, specifying the
	// source code as well. The captures method returns a Captures object
	// which allows iteration over matched captures in the parse tree.
	captures := qc.Captures(q, tree.RootNode(), sourceCode)

	tags := []Tag{}

	// Iterate over all of the query results (i.e., the captures). The Next
	// method returns a matched result (match) and the index of the capture
	// (index) within that match. Continue iterating until match is nil.
	for match, index := captures.Next(); match != nil; match, index = captures.Next() {

		// Retrieve the capture at the current index from the match's list
		// of captures. This capture includes the node in the AST and the
		// capture index used to look up the capture name.
		c := match.Captures[index]

		// Retrieve the name of the capture using the capture index stored in
		// c.Index. This references the actual capture label (e.g.,
		// "name.definition.function") in the query's capture names.
		tag := q.CaptureNames()[c.Index]

		// Convert the node's starting row position to an integer (ie. line number)
		row := int(c.Node.StartPosition().Row)

		// Extract the raw text from the matched node in the source code. We
		// convert it from a slice of bytes to a string.
		name := string(c.Node.Utf8Text(sourceCode))

		// Allows a user-provided list of terms to skip: eg. bool, string, etc.
		if filter != nil && !filter(name) {
			continue
		}

		// Determine if the capture corresponds to a definition or a reference
		// by checking prefixes in its name. If neither condition matches, we
		// skip it.
		switch {
		case strings.HasPrefix(tag, "name.definition."):
			// eg. function, method, type, etc.
			tags = append(tags, Tag{
				Name:     name,
				FileName: relFname,
				FilePath: fname,
				Line:     row,
				Kind:     TagKindDef,
			})

		case strings.HasPrefix(tag, "name.reference."):
			//eg. function call, type usage, etc.
			tags = append(tags, Tag{
				Name:     name,
				FileName: relFname,
				FilePath: fname,
				Line:     row,
				Kind:     TagKindRef,
			})

		default:
			// continue
		}
	}

	return tags
}

// GetTagsRaw parses the file with Tree-sitter and extracts "function definitions"
func (r *RepoMap) GetTagsRaw(fname, relFname string, filter TagFilter) ([]Tag, error) {
	// 1) Identify the file's language
	lang, langID, err := grepast.GetLanguageFromFileName(fname)
	if err != nil || lang == nil {
		return nil, grepast.ErrorUnsupportedLanguage
	}

	// 2) Read source code
	sourceCode, err := ReadSourceCode(fname)
	if err != nil {
		return nil, fmt.Errorf("failed to read file (%s): %v", fname, err)
	}

	// 3) Create parser
	parser := sitter.NewParser()
	parser.SetLanguage(lang)

	// 4) Parse
	tree := parser.Parse(sourceCode, nil)
	if tree == nil || tree.RootNode() == nil {
		return nil, fmt.Errorf("failed to parse file: %s", fname)
	}

	// 5) Load your query
	q, err := r.LoadQuery(lang, langID)
	if err != nil {
		return nil, fmt.Errorf("failed to read query file (%s): %v", langID, err)
	}
	defer q.Close()

	// 6) Execute the query
	qc := sitter.NewQueryCursor()
	defer qc.Close()

	// Get the tags from the query capture and source code
	tags := GetTagsFromQueryCapture(relFname, fname, q, tree, sourceCode, filter)

	// 7) Return the list of Tag objects
	return tags, nil
}

// getTagsFromFiles collect all tags from those files
func (r *RepoMap) getTagsFromFiles(
	allFnames []string,
	ignoreWords map[string]struct{},
) []Tag {

	var allTags []Tag

	for _, fname := range allFnames {
		// Get the relative file name
		rel := r.GetRelFname(fname)

		// Filter out short names and common words
		// tr@ck - where is the right place to put this filter?
		filter := func(name string) bool {
			if len(name) <= 3 {
				return false
			}
			if _, ok := ignoreWords[strings.ToLower(name)]; ok {
				return false
			}
			return true
		}

		// Get the tags for this file
		tg, err := r.GetFileTags(fname, rel, filter)
		if err != nil {
			if err == grepast.ErrorUnsupportedLanguage {
				log.Trace().Msgf("skip %s", fname)
			} else {
				log.Warn().Err(err).Msgf("Failed to get tags for %s", fname)
			}
			continue
		}
		if tg != nil {
			allTags = append(allTags, tg...)
		}
	}

	return allTags
}

type tagKey struct {
	fname  string // the file name (relative)
	symbol string // the actual identifier
}

func (r *RepoMap) getRankedTagsByPageRank(allTags []Tag, mentionedFnames, mentionedIdents map[string]bool) []Tag {
	// 1) Collect references, definitions
	defines := make(map[string]map[string]struct{}) // symbol -> set of filenames that define it
	references := make(map[string]map[string]int)   // symbol -> map of (referencerFile -> countOfRefs)
	definitions := make(map[tagKey][]Tag)           // (fname, symbol) -> slice of definition Tags

	for _, t := range allTags {
		rel := r.GetRelFname(t.FilePath)

		switch t.Kind {
		case TagKindDef:
			if defines[t.Name] == nil {
				defines[t.Name] = make(map[string]struct{})
			}
			defines[t.Name][rel] = struct{}{}

			k := tagKey{fname: rel, symbol: t.Name}
			definitions[k] = append(definitions[k], t)

		case TagKindRef:
			if references[t.Name] == nil {
				references[t.Name] = make(map[string]int)
			}
			references[t.Name][rel]++
		}
	}

	// If references is empty, fall back to references=defines
	if len(references) == 0 {
		references = make(map[string]map[string]int)
		for sym, defFiles := range defines {
			references[sym] = make(map[string]int)
			for df := range defFiles {
				references[sym][df]++
			}
		}
	}

	// 2) Build a multi directed graph
	g := multi.NewWeightedDirectedGraph()

	// Keep track of the node ID for each rel_fname
	nodeByFile := make(map[string]graph.Node)

	// Gather all relevant filenames
	fileSet := make(map[string]struct{})
	for _, defFiles := range defines {
		for f := range defFiles {
			fileSet[f] = struct{}{}
		}
	}
	for _, refMap := range references {
		for f := range refMap {
			fileSet[f] = struct{}{}
		}
	}

	// Create node for each file
	for f := range fileSet {
		n := g.NewNode()
		g.AddNode(n)
		nodeByFile[f] = n
	}

	fmt.Printf("Number of nodes (files): %d\n", g.Nodes().Len())

	// 3) For each ident, link referencing file -> defining file with weight
	for symbol, refMap := range references {
		defFiles := defines[symbol]
		if len(defFiles) == 0 {
			continue
		}

		var mul float64
		switch {
		case mentionedIdents[symbol]:
			mul = 10.0
		case strings.HasPrefix(symbol, "_"):
			mul = 0.1
		default:
			mul = 1.0
		}

		for refFile, numRefs := range refMap {
			// log.Trace().Msg(color.YellowString("refFile: %s, numRefs: %d"), refFile, numRefs))
			w := mul * math.Sqrt(float64(numRefs))
			for defFile := range defFiles {
				// If refFile == defFile, decide if you skip or not
				if refFile == defFile {
					continue
				}
				refNode := nodeByFile[refFile]
				defNode := nodeByFile[defFile]

				// Create a weighted edge
				edge := g.NewWeightedLine(refNode, defNode, w)
				g.SetWeightedLine(edge)
			}
		}
	}

	// 4) Personalization
	personal := make(map[int64]float64)
	totalFiles := float64(len(fileSet))
	defaultPersonal := 1.0 / totalFiles

	chatSet := make(map[string]struct{})
	for cf := range mentionedFnames {
		chatSet[cf] = struct{}{}
	}

	for f, node := range nodeByFile {
		if _, inChat := chatSet[f]; inChat {
			personal[node.ID()] = 100.0 / totalFiles
		} else {
			personal[node.ID()] = defaultPersonal
		}
	}

	// 5) Run PageRank (NOTE: gonum.network.PageRank might not natively handle personalization
	// the same way. If you need full personalized PageRank, you might have to modify or implement
	// your own. For now, we do unpersonalized for demonstration.)
	pr := network.PageRank(g, 0.85, 1e-6) // no direct personalization used

	// fmt.Println("PageRank results:")
	// PrintStructOut(pr)

	// 6) Distribute rank from each src node across its out edges
	edgeRanks := make(map[struct {
		dst    string
		symbol string
	}]float64)

	for symbol, refMap := range references {
		defFiles := defines[symbol]
		if defFiles == nil {
			continue
		}

		var mul float64
		switch {
		case mentionedIdents[symbol]:
			mul = 10.0
		case strings.HasPrefix(symbol, "_"):
			mul = 0.1
		default:
			mul = 1.0
		}

		for refFile, numRefs := range refMap {
			w := mul * math.Sqrt(float64(numRefs))
			sumW := float64(len(defFiles)) * w // If each defFile gets w from refFile

			srcRank := pr[nodeByFile[refFile].ID()]
			if sumW == 0 {
				continue
			}
			for defFile := range defFiles {
				portion := srcRank * (w / sumW)
				edgeRanks[struct {
					dst    string
					symbol string
				}{dst: defFile, symbol: symbol}] += portion
			}
		}
	}

	// 7) Convert to sorted list
	type defRank struct {
		fname  string
		symbol string
		rank   float64
	}

	var defRankSlice []defRank
	for k, v := range edgeRanks {
		defRankSlice = append(defRankSlice, defRank{
			fname:  k.dst,
			symbol: k.symbol,
			rank:   v,
		})
	}

	sort.Slice(defRankSlice, func(i, j int) bool {
		if defRankSlice[i].rank != defRankSlice[j].rank {
			return defRankSlice[i].rank > defRankSlice[j].rank
		}
		if defRankSlice[i].fname != defRankSlice[j].fname {
			return defRankSlice[i].fname < defRankSlice[j].fname
		}
		return defRankSlice[i].symbol < defRankSlice[j].symbol
	})

	chatRelFnames := make(map[string]bool)
	// If you had a slice of chatFnames, for example:
	/*
		for _, cf := range chatFnamesSlice {
			rel := r.GetRelFname(cf)
			chatRelFnames[rel] = true
		}
	*/

	var rankedTags []Tag
	for _, dr := range defRankSlice {
		if chatRelFnames[dr.fname] {
			continue
		}
		k := tagKey{fname: dr.fname, symbol: dr.symbol}
		defs := definitions[k]
		rankedTags = append(rankedTags, defs...)
	}

	// Possibly append files that have no tags, etc.
	return rankedTags
}

// GetRankedTagsMap orchestrates calls to getRankedTags and toTree to produce the final “map” string.
func (r *RepoMap) GetRankedTagsMap(
	chatFnames, otherFnames []string,
	maxMapTokens int,
	mentionedFnames, mentionedIdents map[string]bool,
	forceRefresh bool,
) string {

	startTime := time.Now()

	// Combine chatFnames and otherFnames into a map of unique elements
	allFnames := uniqueElements(chatFnames, otherFnames)

	// Collect all tags from those files
	allTags := r.getTagsFromFiles(allFnames, commonWords)

	// Get ranked tags by PageRank
	rankedTags := r.getRankedTagsByPageRank(allTags, mentionedFnames, mentionedIdents)

	// tr@ck
	topFive := 5
	for i, t := range rankedTags {
		if topFive == 0 {
			break
		}
		if t.Name == "doc" {
			continue
		}
		// first n chars
		log.Info().Int("index", i).Str("file", t.FileName).Int("line", t.Line).Str("tag", t.Name).Msg("tags")
		topFive--
	}

	// special := filterImportantFiles(otherFnames)

	// // Prepend special files as “important”.
	// var specialTags []Tag
	// for _, sf := range special {
	// 	specialTags = append(specialTags, Tag{Name: r.GetRelFname(sf)})
	// }
	// finalTags := append(specialTags, rankedTags...)

	finalTags := rankedTags

	bestTree := ""
	bestTreeTokens := 0.0

	lb := 0
	ub := len(finalTags)
	middle := ub
	if middle > 30 {
		middle = 30
	}

	for lb <= ub {
		tree := r.toTree(finalTags[:middle], chatFnames)
		numTokens := r.TokenCount(tree)

		diff := math.Abs(numTokens - float64(maxMapTokens))
		pctErr := diff / float64(maxMapTokens)
		if (numTokens <= float64(maxMapTokens) && numTokens > bestTreeTokens) || pctErr < 0.15 {
			bestTree = tree
			bestTreeTokens = numTokens
			if pctErr < 0.15 {
				break
			}
		}
		if numTokens < float64(maxMapTokens) {
			lb = middle + 1
		} else {
			ub = middle - 1
		}
		middle = (lb + ub) / 2
	}

	endTime := time.Now()
	r.MapProcessingTime = endTime.Sub(startTime).Seconds()

	r.LastMap = bestTree
	return bestTree
}

// tr@ck -- improve this chat vs other files. We should have repoFiles and chatFiles

// GetRepoMap is the top-level function (mirroring the Python method) that produces the “repo content”.
func (r *RepoMap) GetRepoMap(
	chatFiles, otherFiles []string,
	mentionedFnames, mentionedIdents map[string]bool,
	forceRefresh bool,
) string {

	if r.MaxMapTokens <= 0 {
		log.Warn().Msgf("Repo-map disabled by max_map_tokens: %d", r.MaxMapTokens)
		return ""
	}
	// if len(otherFiles) == 0 {
	// 	log.Warn().Msg("No other files found; disabling repo map")
	// 	return ""
	// }
	if mentionedFnames == nil {
		mentionedFnames = make(map[string]bool)
	}
	if mentionedIdents == nil {
		mentionedIdents = make(map[string]bool)
	}

	maxMapTokens := r.MaxMapTokens
	padding := 4096
	var target int
	if maxMapTokens > 0 && r.MaxCtxWindow > 0 {
		t := maxMapTokens * r.MapMulNoFiles
		t2 := r.MaxCtxWindow - padding
		if t2 < 0 {
			t2 = 0
		}
		if t < t2 {
			target = t
		} else {
			target = t2
		}
	}
	if len(chatFiles) == 0 && r.MaxCtxWindow > 0 && target > 0 {
		maxMapTokens = target
	}

	var filesListing string
	// defer func() {
	// 	if rec := recover(); rec != nil {
	// 		fmt.Printf("ERR: Disabling repo map, repository may be too large?")
	// 		r.MaxMapTokens = 0
	// 		filesListing = ""
	// 	}
	// }()

	filesListing = r.GetRankedTagsMap(chatFiles, otherFiles, maxMapTokens, mentionedFnames, mentionedIdents, forceRefresh)
	if filesListing == "" {
		return ""
	}

	// fmt.Println(filesListing)

	if r.Verbose {
		numTokens := r.TokenCount(filesListing)
		fmt.Printf("Repo-map: %.1f k-tokens\n", numTokens/1024.0)
	}

	other := ""
	if len(chatFiles) > 0 {
		other = "other "
	}

	var repoContent string
	if r.RepoContentPx != "" {
		repoContent = strings.ReplaceAll(r.RepoContentPx, "{other}", other)
	}

	repoContent += filesListing
	return repoContent
}

// ------------------------------------------------------------------------------------
// Rendering code blocks with TreeContext from grep-ast
// ------------------------------------------------------------------------------------
func (r *RepoMap) toTree(tags []Tag, chatFnames []string) string {
	if len(tags) == 0 {
		return ""
	}
	chatRelSet := make(map[string]bool)
	for _, c := range chatFnames {
		chatRelSet[r.GetRelFname(c)] = true
	}

	curFname := ""
	curAbsFname := ""
	var linesOfInterest []int
	var output strings.Builder

	dummyTag := Tag{Name: "____dummy____"}
	tagsWithDummy := append(tags, dummyTag)

	for _, tag := range tagsWithDummy {
		if chatRelSet[tag.Name] {
			continue
		}
		if tag.Name != curFname {
			if curFname != "" && linesOfInterest != nil {
				output.WriteString("\n" + curFname + ":\n")
				rendered, err := r.renderTree(curAbsFname, curFname, linesOfInterest)
				if err != nil {
					log.Warn().Err(err).Msgf("Failed to render tree for %s", curFname)
				}
				output.WriteString(rendered)
			}
			if tag.Name == "____dummy____" {
				break
			}
			linesOfInterest = []int{}
			curFname = tag.Name
			curAbsFname = tag.FilePath
		}
		if linesOfInterest != nil {
			linesOfInterest = append(linesOfInterest, tag.Line)
		}
	}

	lines := strings.Split(output.String(), "\n")
	for i, ln := range lines {
		if len(ln) > 100 {
			lines[i] = ln[:100]
		}
	}
	return strings.Join(lines, "\n") + "\n"
}

// renderTree uses a grep-ast TreeContext to produce a nice snippet with lines of interest expanded.
func (r *RepoMap) renderTree(absFname, relFname string, linesOfInterest []int) (string, error) {

	code, err := os.ReadFile(absFname)
	if err != nil {
		return "", fmt.Errorf("failed to read file (%s): %w", absFname, err)
	}

	// Build a grep-ast TreeContext.
	// (Below is an example usage; adapt to whatever the actual library API provides.)
	tc, err := grepast.NewTreeContext(relFname, code, grepast.TreeContextOptions{
		Color:                    false,
		Verbose:                  false,
		ShowLineNumber:           false,
		ShowParentContext:        false,
		ShowChildContext:         false,
		ShowLastLine:             false,
		MarginPadding:            0,
		MarkLinesOfInterest:      false,
		HeaderMax:                0,
		ShowTopOfFileParentScope: false,
		LinesOfInterestPadding:   0,
	})
	if err != nil {
		if err == grepast.ErrorUnsupportedLanguage || err == grepast.ErrorUnrecognizedFiletype {
			return "", nil
		}
		return "", fmt.Errorf("failed to create tree context: %w", err)
	}

	// Convert []int to map[int]struct{}
	// tr@ck -- could this be avoided?
	loiMap := make(map[int]struct{}, len(linesOfInterest))
	for _, ln := range linesOfInterest {
		loiMap[ln] = struct{}{}
	}

	// Add the lines of interest
	tc.AddLinesOfInterest(loiMap)
	// Expand context around those lines
	tc.AddContext()

	res := tc.Format()

	return res, nil
}

// getRandomColor replicates the Python get_random_color using HSV → RGB.
func getRandomColor() string {
	hue := rand.Float64()
	r, g, b := hsvToRGB(hue, 1.0, 0.75)
	return fmt.Sprintf("#%02x%02x%02x", r, g, b)
}

// hsvToRGB is standard. h, s, v in [0,1], output in [0,255].
func hsvToRGB(h, s, v float64) (int, int, int) {
	var r, g, b float64
	i := math.Floor(h * 6)
	f := h*6 - i
	p := v * (1 - s)
	q := v * (1 - f*s)
	t := v * (1 - (1-f)*s)

	switch int(i) % 6 {
	case 0:
		r, g, b = v, t, p
	case 1:
		r, g, b = q, v, p
	case 2:
		r, g, b = p, v, t
	case 3:
		r, g, b = p, q, v
	case 4:
		r, g, b = t, p, v
	case 5:
		r, g, b = v, p, q
	}
	return int(r * 255), int(g * 255), int(b * 255)
}

// FindSrcFiles gathers all files in a directory (or the file itself).
func FindSrcFiles(path string) []string {
	info, err := os.Stat(path)
	if err != nil {
		return []string{}
	}

	if !info.IsDir() {
		return []string{path}
	}

	var srcFiles []string
	filepath.Walk(path, func(p string, info os.FileInfo, err error) error {
		// Skip files that match the ignore patterns
		if err != nil {
			return nil
		}
		if grepast.MatchIgnorePattern(p, grepast.DefaultIgnorePatterns) {
			// log.Trace().Str("op", "source files").Str("path", p).Msg("skip")
			return nil
		}
		if info.IsDir() {
			return nil
		}
		// log.Debug().Str("op", "source files").Str("path", p).Msg("add")
		srcFiles = append(srcFiles, p)
		return nil
	})
	return srcFiles
}
