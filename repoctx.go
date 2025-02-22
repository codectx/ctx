// Package codectx contains the core logic for the codectx tool
package codectx

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	_ "embed"

	// Import the grep-ast library
	queries "github.com/codectx/ctx/internals/queries"
	"github.com/codectx/ctx/internals/services/chunker"
	embed "github.com/codectx/ctx/internals/services/embed"
	store "github.com/codectx/ctx/internals/services/store"
	"github.com/coder/hnsw"
	goignore "github.com/cyber-nic/go-gitignore"
	grepast "github.com/cyber-nic/grep-ast"
	ollama "github.com/ollama/ollama/api"
	sitter "github.com/tree-sitter/go-tree-sitter"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/multi"
	"gonum.org/v1/gonum/graph/network"
)

//go:embed .astignore
var defaultGlobIgnore string

// chat files: files that are part of the chat
// other files: files that are not (yet) part of the chat
// warned_files: files that have been warned about and are excluded

const (
	// TagKindDef is the kind of tag that represents a definition
	TagKindDef = "def"
	// TagKindRef is the kind of tag that represents a reference
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

type FileCTX struct {
	// Filename
	Filename string
	// Relative filename
	RelFilename string
	// Tree-sitter language
	Lang *sitter.Language
	// Language ID
	LangID string
	// Tree-sitter parse tree
	Tree *sitter.Tree
	// Source Code
	SourceCode []byte
	// Hash
	// MD5Hash string
}

// RepoCTX default options
const (
	defaultGraphQueryResults    = 25
	defaultGlobIgnoreEnabled    = true
	globIgnoreFileDefault       = true
	defaultMaxCtxFileMultiplier = 8
	defaultMaxCtxWindow         = 16000
	defaultMaxMapTokens         = 1024
	defaultRepoContentPrefix    = ""
	defaultVerbose              = false
)

// ModelStub simulates the main_model used in Python code (for token_count, etc.).
type ModelStub struct{}

// EmbeddingService defines an interface for obtaining embeddings from text.
type EmbeddingService interface {
	// Get generates an embedding for the given text.
	Get(ctx context.Context, text string) ([]float32, embed.Meta, error)
	// GetBatch generates embeddings for the given texts.
	GetBatch(ctx context.Context, text []string) ([][]float32, embed.Meta, error)
}

type ChunkerService interface {
	Chunk(tree *sitter.Tree, content []byte) ([]chunker.Chunk, error)
}

// StorageService defines the interface for CRUD operations on DuckDB.
type StorageService interface {
	// Upsert inserts or updates a row
	Upsert(context.Context, string, string, []float32) error
	// GetAll fetches all rows.
	GetAll(ctx context.Context) (map[string]store.Embedding, error)
	// Get fetches multiple rows by ids.
	Get(ctx context.Context, id []string) ([]store.Embedding, error)
	// Delete removes a row by id.
	Delete(ctx context.Context, id string) error
}

type RepoCTXServices struct {
	Chunk ChunkerService
	Embed EmbeddingService
	Store StorageService
}

// RepoCTX is the Go equivalent of the Python class `RepoCTX`.
type RepoCTX struct {
	// common words are ignored when traversing tags
	commonIgnoreWords    map[string]struct{}
	globIgnoreEnabled    bool
	globIgnoreFilePath   string
	globIgnorePatterns   *goignore.GitIgnore
	graph                *hnsw.Graph[string]
	graphQueryResults    int
	lastMap              string
	mainModel            *ModelStub
	maxMapTokens         int
	maxCtxWindow         int
	maxCtxFileMultiplier int // map_mul_no_files
	totalProcessingTime  float64
	contentPrefix        string
	root                 string
	ready                chan bool
	svc                  RepoCTXServices
	verbose              bool
	// per-file map options
	mapShowLineNumber         bool
	mapShowParentContext      bool
	mapShowLastLine           bool
	mapMarkLinesOfInterest    bool
	mapLinesOfInterestPadding int
}

// NewRepoCTX is the repo map constructor.
func NewRepoCTX(root string, mainModel *ModelStub, database *sql.DB, options ...func(*RepoCTX),
) *RepoCTX {
	if root == "" {
		cwd, err := os.Getwd()
		if err == nil {
			root = cwd
		}
	}

	zerolog.SetGlobalLevel(zerolog.ErrorLevel)

	// Setup storage service
	db := store.New(database)

	// tr@ck - pass ollama client as option WithEmbeddingClient()
	os.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	oClient, err := ollama.ClientFromEnvironment()
	if err != nil {
		log.Err(err).Msg("Failed to create Ollama client")
		os.Exit(1)
	}

	// Create embedding service
	emb := embed.New(oClient)

	chk := chunker.New(
		chunker.WithMaxChunkSize(1024),
		chunker.WithCoalesceThreshold(25),
	)

	ready := make(chan bool, 1)

	rm := &RepoCTX{
		commonIgnoreWords:    commonWords,
		contentPrefix:        defaultRepoContentPrefix,
		globIgnoreEnabled:    defaultGlobIgnoreEnabled,
		globIgnorePatterns:   &goignore.GitIgnore{},
		graph:                hnsw.NewGraph[string](),
		graphQueryResults:    defaultGraphQueryResults,
		mainModel:            mainModel,
		maxMapTokens:         defaultMaxMapTokens,
		maxCtxFileMultiplier: defaultMaxCtxFileMultiplier,
		maxCtxWindow:         defaultMaxCtxWindow,
		root:                 root,
		verbose:              defaultVerbose,
		ready:                ready,
		svc: RepoCTXServices{
			Chunk: chk,
			Embed: emb,
			Store: db,
		},
	}

	// Apply any additional options to the RepoCTX object
	for _, o := range options {
		o(rm)
	}

	// Max CTX File Multiplier has been explicitly set
	if rm.maxCtxFileMultiplier != defaultMaxCtxFileMultiplier {
		log.Debug().Int("multiplier", rm.maxCtxFileMultiplier).Msg("RepoCTX initialized with Max Context File Multiplier")
	}

	// Glob ignore has been explicitly disabled
	if !rm.globIgnoreEnabled {
		log.Debug().Str("glob_ignore", "disabled").Msg("RepoCTX initialized")
		return rm
	}
	log.Debug().Str("glob_ignore", "enabled").Msg("RepoCTX initialized")

	// Glob file path provided
	if rm.globIgnoreFilePath != "" {
		// handle path
		if _, err := os.Stat(rm.globIgnoreFilePath); err == nil {
			// Load the ignore file if it exists
			rm.globIgnorePatterns, err = goignore.CompileIgnoreFile(rm.globIgnoreFilePath)
			if err == nil {
				log.Info().Str("path", rm.globIgnoreFilePath).Msg("ignore file loaded")
				return rm
			}
		}

		// handle relative path / filename
		// 2. Find the root of the git repo
		root, err := FindGitRoot(rm.root)
		if err != nil {
			log.Fatal().Err(err).Msg("Error finding .git")
		}

		// handle full path
		p := filepath.Join(root, rm.globIgnoreFilePath)
		if _, err := os.Stat(p); err == nil {
			// Load the ignore file if it exists
			rm.globIgnorePatterns, err = goignore.CompileIgnoreFile(p)
			if err != nil {
				fmt.Fprintf(os.Stderr, "error loading ignore file: %v\n", err)
			}
			log.Info().Str("path", p).Msg("ignore file loaded")
			return rm
		}

		// Panic as the user-provided glob ignore file does not exist
		log.Fatal().Msgf("ignore file not found: %s", rm.globIgnoreFilePath)
	}

	// Use default glob ignore file
	// Load the ignore file if it exists
	defaultLines := strings.Split(defaultGlobIgnore, "\n")
	rm.globIgnorePatterns = goignore.CompileIgnoreLines(defaultLines...)

	return rm
}

// WithLogLevel sets the log level for the RepoCTX
func WithLogLevel(value int) func(*RepoCTX) {
	return func(_ *RepoCTX) {
		zerolog.SetGlobalLevel(zerolog.Level(value))
		log.Debug().Int("level", value).Msg("RepoCTX Log Level Set")
	}
}

// WithGlobIgnoreFilePath sets the glob ignore file path. Ignored if DisableGlobIgnore is set
func WithGlobIgnoreFilePath(value string) func(*RepoCTX) {
	return func(o *RepoCTX) {
		o.globIgnoreFilePath = value
	}
}

// DisableGlobIgnore disables the global ignore file
func DisableGlobIgnore() func(*RepoCTX) {
	return func(o *RepoCTX) {
		o.globIgnoreEnabled = false
	}
}

// WithMaxContextWindow set the maximum context window.
func WithMaxContextWindow(value int) func(*RepoCTX) {
	return func(o *RepoCTX) {
		o.maxMapTokens = value
	}
}

// WithMapMulNoFiles sets the number of files to multiply the map by.
func WithMapMulNoFiles(value int) func(*RepoCTX) {
	return func(o *RepoCTX) {
		o.maxMapTokens = value
	}
}

// WithMaxTokens sets the map's maximum number of tokens.
func WithMaxTokens(value int) func(*RepoCTX) {
	return func(o *RepoCTX) {
		o.maxMapTokens = value
	}
}

// WithContentPrefix sets the repository content prefix.
func WithContentPrefix(value string) func(*RepoCTX) {
	return func(o *RepoCTX) {
		o.contentPrefix = value
	}
}

// WithLineNumber enables or disables line numbers in the output.
func WithLineNumber(value bool) func(*RepoCTX) {
	return func(o *RepoCTX) {
		o.mapShowLineNumber = value
	}
}

// WithParentContext enables or disables the inclusion of parent context in the output.
func WithParentContext(value bool) func(*RepoCTX) {
	return func(o *RepoCTX) {
		o.mapShowParentContext = value
	}
}

// WithLinesOfInterestMarked enables or disables the marking of lines of interest in the output.
func WithLinesOfInterestMarked(value bool) func(*RepoCTX) {
	return func(o *RepoCTX) {
		o.mapMarkLinesOfInterest = value
	}
}

// WithLastLineContext enables or disables the inclusion of the context delimited by the last line in the output.
func WithLastLineContext(value bool) func(*RepoCTX) {
	return func(o *RepoCTX) {
		o.mapShowLastLine = value
	}
}

// WithLinesOfInterestPadding sets the number of lines of padding around lines of interest.
func WithLinesOfInterestPadding(value int) func(*RepoCTX) {
	return func(o *RepoCTX) {
		o.mapLinesOfInterestPadding = value
	}
}

// Verbose enables verbose output for debugging.
func Verbose(value bool) func(*RepoCTX) {
	return func(o *RepoCTX) {
		o.verbose = value
	}
}

// TokenCount is a naive token estimator. Real code might call tiktoken or other logic.
// tr@ck - this is a stub
func (m *ModelStub) TokenCount(text string) int {
	// Very naive: 1 token ~ 4 chars
	return len(text) / 4
}

// GetRelFilename returns fname relative to r.Root. If that fails, returns fname as-is.
func (r *RepoCTX) GetRelFilename(fname string) string {
	rel, err := filepath.Rel(r.root, fname)
	if err != nil {
		return fname
	}

	return rel
}

func (r *RepoCTX) Query(ctx context.Context, query string) (string, error) {
	// Get query embedding
	q, _, err := r.svc.Embed.Get(ctx, query)
	if err != nil {
		return "", fmt.Errorf("failed to embed query: %w", err)
	}

	// Ensure repo is ready before querying
	select {
	case <-r.ready:
	case <-ctx.Done():
		return "", ctx.Err()
	}

	// Compute distances and aggregate by file
	type neighbor struct {
		Filename string
		Weight   float32
	}
	neighborsByWeight := make(map[string]float32)

	neighbors := r.graph.Search(q, r.graphQueryResults)
	for _, n := range neighbors {
		distance := computeDistance(q, n.Value)
		filename := strings.SplitN(n.Key, ":", 2)[0] // Get filename only

		if existing, found := neighborsByWeight[filename]; !found || distance < existing {
			neighborsByWeight[filename] = distance
		}
		// neighborsByWeight[filename] += distance
		log.Debug().Float32("distance", distance).Str("file", filename).Msg("search result")
	}

	// Sort neighbors by weight (ascending)
	results := make([]neighbor, 0, len(neighborsByWeight))
	for filename, weight := range neighborsByWeight {
		results = append(results, neighbor{filename, weight})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Weight < results[j].Weight
	})

	// Log top results
	for _, res := range results {
		log.Info().Float32("weight", res.Weight).Str("file", res.Filename).Msg("search ranking")
	}

	return "", nil

	// // Compute distances and aggregate by file
	// type neighbor struct {
	// 	Filename string
	// 	Weight   float32
	// 	Count    int
	// }
	// neighborsByWeight := make(map[string]neighbor)

	// // perform query
	// neighbors := r.graph.Search(q, r.graphQueryResults)

	// neighborsByDistance := make(map[float32]string)

	// // sort neighbors by distance
	// for _, n := range neighbors {
	// 	distance := computeDistance(q, n.Value)
	// 	neighborsByDistance[distance] = n.Key
	// }

	// // sort distances
	// distances := make([]float32, 0, len(neighborsByDistance))
	// for d := range neighborsByDistance {
	// 	distances = append(distances, d)
	// }
	// sort.Slice(distances, func(i, j int) bool {
	// 	return distances[i] < distances[j]
	// })

	// for _, distance := range distances {
	// 	n := neighborsByDistance[distance]

	// 	// distance := computeDistance(q, n.Value)
	// 	filename := strings.SplitN(n, ":", 2)[0] // Get filename only

	// 	// Apply moving average formula
	// 	if existing, found := neighborsByWeight[filename]; found {
	// 		newWeight := (existing.Weight*float32(existing.Count) + distance) / float32(existing.Count+1)
	// 		neighborsByWeight[filename] = neighbor{filename, newWeight, existing.Count + 1}
	// 	} else {
	// 		neighborsByWeight[filename] = neighbor{filename, distance, 1}
	// 	}

	// 	log.Debug().Float32("distance", distance).Str("file", filename).Msg("search result")
	// }

	// // Sort neighbors by weight (ascending)
	// results := make([]neighbor, 0, len(neighborsByWeight))
	// for _, entry := range neighborsByWeight {
	// 	results = append(results, entry)
	// }
	// sort.Slice(results, func(i, j int) bool {
	// 	return results[i].Weight < results[j].Weight
	// })

	// // Log top results
	// for _, res := range results {
	// 	if res.Weight <= 0.8 {
	// 		log.Info().Float32("weight", res.Weight).Str("file", res.Filename).Msg("search ranking")
	// 	}
	// }

	// return "", nil
}

func computeDistance(q, v []float32) float32 {
	// var sum float32
	// for i := range q {
	// 	sum += (q[i] - v[i]) * (q[i] - v[i])
	// }
	// return sum

	return hnsw.CosineDistance(q, v)
	// return hnsw.EuclideanDistance(q, v)
}

// TokenCount tries to mimic how the Python code estimates tokens (split into short vs. large).
// tr@ck - this is a stub
func (r *RepoCTX) TokenCount(text string) float64 {
	if len(text) < 200 {
		return float64(r.mainModel.TokenCount(text))
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
	sampleTokens := float64(r.mainModel.TokenCount(sampleText))
	ratio := sampleTokens / float64(len(sampleText))
	return ratio * float64(len(text))
}

// LoadQuery loads the Tree-sitter query text and compiles a sitter.Query.
func (r *RepoCTX) LoadQuery(lang *sitter.Language, langID string) (*sitter.Query, error) {
	querySource, err := queries.GetSitterQuery(queries.SitterLanguage(langID))
	if err != nil {
		return nil, fmt.Errorf("failed to obtain query (%s): %w", langID, err)
	}
	if len(querySource) == 0 {
		return nil, fmt.Errorf("empty query file: %s", langID)
	}

	q, qErr := sitter.NewQuery(lang, string(querySource))
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

// readSourceCode reads the source code from a file.
func readSourceCode(fname string) ([]byte, error) {
	sourceCode, err := os.ReadFile(fname)
	if err != nil {
		return nil, fmt.Errorf("failed to read file (%s): %w", fname, err)
	}
	if len(sourceCode) == 0 {
		return nil, fmt.Errorf("empty file: %s", fname)
	}
	return sourceCode, nil
}

// TagFilter is a function that accepts the name of a capture and returns false if it should be skipped.
type TagFilter func(name string) bool

// GetTagsFromQueryCapture extracts tags from the result
// of a Tree-sitter query on a given file. It iterates through
// the captures returned by the Tree-sitter query cursor and collects
// definitions (def) and references (ref). All other captures are ignored.
// filter is a function that accepts the name of a capture and returns bool false if it should be skipped.
func GetTagsFromQueryCapture(f *FileCTX, q *sitter.Query, filter TagFilter) []Tag {

	// Create a new query cursor that will be used to iterate through
	// the captures of our query on the provided parse tree. The query
	// cursor manages iteration state for match captures.
	qc := sitter.NewQueryCursor()
	defer qc.Close()

	// Execute the query against the provided parse tree, specifying the
	// source code as well. The captures method returns a Captures object
	// which allows iteration over matched captures in the parse tree.
	captures := qc.Captures(q, f.Tree.RootNode(), f.SourceCode)

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
		name := string(c.Node.Utf8Text(f.SourceCode))

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
				FileName: f.RelFilename,
				FilePath: f.Filename,
				Line:     row,
				Kind:     TagKindDef,
			})

		case strings.HasPrefix(tag, "name.reference."):
			//eg. function call, type usage, etc.
			tags = append(tags, Tag{
				Name:     name,
				FileName: f.RelFilename,
				FilePath: f.Filename,
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
func (r *RepoCTX) GetFileTagsRaw(f *FileCTX, filter TagFilter) ([]Tag, error) {

	// Load langage query
	q, err := r.LoadQuery(f.Lang, f.LangID)
	if err != nil {
		return nil, fmt.Errorf("failed to read query file (%s): %v", f.LangID, err)
	}
	defer q.Close()

	// Execute the query
	qc := sitter.NewQueryCursor()
	defer qc.Close()

	// Get the tags from the query capture and source code
	tags := GetTagsFromQueryCapture(f, q, filter)

	// Return the list of Tag objects
	return tags, nil
}

var errStoreRequestFailed = errors.New("store request failed")

// GetBatchEmbedding embeds the source code of content of multiple files or retrieves them from the database.
func (r *RepoCTX) GetBatchEmbedding(ctx context.Context, filename string, data map[string]string) (map[string][]float32, error) {
	// safety check: handle no embedding service
	if r.svc.Embed == nil {
		return nil, fmt.Errorf("no embedding service")
	}

	// handle no storage service: return fresh embedding
	if r.svc.Store == nil {
		return nil, fmt.Errorf("no storage service")
	}

	have := make(map[string][]float32, len(data))
	wantKeys := []string{}
	wantContents := []string{}
	hashCache := make(map[string]string)

	// begin by fetching embeddings from the database
	// tr@ck -- optimize fetch all keys from db at once
	for key, content := range data {
		// compute hash
		hash := computeHash([]byte(content))
		hashCache[key] = hash

		// get from db
		b, _ := r.svc.Store.Get(ctx, []string{key})
		if b != nil && len(b) > 0 {
			if b[0].Hash == hash {
				have[key] = b[0].Vector
				continue
			}
		}

		wantKeys = append(wantKeys, key)
		wantContents = append(wantContents, content)

		// unexpected error
		if b != nil && len(b) == 0 {
			return nil, fmt.Errorf("failed to get embedding: %w", errStoreRequestFailed)
		}
	}

	log.Debug().Int("total", len(data)).Int("want", len(wantKeys)).Int("have", len(have)).Str("z", filename).Msg("chunks")

	// Batch in 10s
	batchSize := 10

	for i := 0; i < len(wantKeys); i += batchSize {
		end := i + batchSize
		if end > len(wantKeys) {
			end = len(wantKeys)
		}

		batchKeys := wantKeys[i:end]
		batchContents := wantContents[i:end]

		// Get all missing or stale embeddings
		vec, meta, err := r.svc.Embed.GetBatch(ctx, batchContents)
		log.Debug().Str("z", filename).Int("batch", len(batchKeys)).Int("tokens", meta.Tokens).Msg("batch")

		if err != nil {
			return nil, fmt.Errorf("failed to embed text: %w", err)
		}

		// anxiety check
		if len(vec) != len(batchContents) {
			return nil, fmt.Errorf("failed to embed text: %w", err)
		}

		for idx, key := range batchKeys {
			// add to have
			have[key] = vec[idx]

			// Upsert
			if err := r.svc.Store.Upsert(ctx, key, hashCache[key], vec[idx]); err != nil {
				return nil, fmt.Errorf("failed to upsert embedding: %w", err)
			}
		}
	}

	return have, nil
}

// GetFileTags calls GetTagsRaw and filters out short names and common words.
// tr@ck - add caching
func (r *RepoCTX) GetFileTags(f *FileCTX, filter TagFilter) ([]Tag, error) {

	// Not cached or changed; re-parse
	data, err := r.GetFileTagsRaw(f, filter)
	if err != nil {
		return nil, err
	}

	if data == nil {
		data = nil
	}

	return data, nil
}

// GetFileCTX collect tags and embeddings from a file
func (r *RepoCTX) GetFileCTX(ctx context.Context, fname string, filter TagFilter) (map[string][]float32, []Tag, error) {
	log.Trace().Str("z", fname).Msg("tags")

	// 1) Identify the file's language
	lang, langID, err := grepast.GetLanguageFromFileName(fname)
	if err != nil || lang == nil {
		return nil, nil, grepast.ErrorUnsupportedLanguage
	}

	// 2) Read source code
	sourceCode, err := readSourceCode(fname)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read file (%s): %v", fname, err)
	}

	// 3) Create parser
	parser := sitter.NewParser()
	parser.SetLanguage(lang)

	// 4) Parse to CST
	tree := parser.Parse(sourceCode, nil)
	if tree == nil || tree.RootNode() == nil {
		return nil, nil, fmt.Errorf("failed to parse file: %s", fname)
	}

	// 5) Create FileCTX
	f := &FileCTX{
		Filename:    fname,
		RelFilename: r.GetRelFilename(fname),
		Lang:        lang,
		LangID:      langID,
		SourceCode:  sourceCode,
		Tree:        tree,
	}

	errChan := make(chan error, 2)
	wg := sync.WaitGroup{}
	tags := []Tag{}
	vecs := map[string][]float32{}

	// 6) Get tags
	wg.Add(1)
	go func() {
		defer wg.Done()

		tags, err = r.GetFileTags(f, filter)
		if err != nil {
			// log.Warn().Err(err).Msgf("Failed to get tags for %s", fname)
			errChan <- fmt.Errorf("failed to get tags for %s: %w", fname, err)
		}
	}()

	// 8) Get file chunk embeddings
	wg.Add(1)
	go func() {
		defer wg.Done()

		// Chunk file
		chunks, err := r.svc.Chunk.Chunk(f.Tree, f.SourceCode)
		if err != nil {
			errChan <- fmt.Errorf("failed to chunk file: %w", err)
			return
		}

		// Prepare data for embeddings
		data := make(map[string]string, len(chunks))
		for _, chunk := range chunks {
			data[fmt.Sprintf("%s:%d", f.Filename, chunk.Start)] = chunk.Text
		}
		log.Trace().Str("z", f.Filename).Int("chunks", len(chunks)).Msgf("Chunk")

		// Get embeddings for each chunk
		vecs, err = r.GetBatchEmbedding(ctx, f.Filename, data)
		if err != nil {
			errChan <- fmt.Errorf("failed to get chunk embeddings for %s: %w", fname, err)
		}
	}()

	// Wait for all goroutines to finish
	wg.Wait()

	// Check for errors
	select {
	case err := <-errChan:
		return nil, nil, err
	default:
	}

	return vecs, tags, nil
}

// GetCTXFromFiles collect all tags from those files
func (r *RepoCTX) GetCTXFromFiles(ctx context.Context, allFilenames []string) []Tag {
	var allTags []Tag
	var mu sync.Mutex

	// Filter out short names and common words
	tagFilter := func(name string) bool {
		if len(name) <= 2 {
			return false
		}
		if _, ok := r.commonIgnoreWords[strings.ToLower(name)]; ok {
			return false
		}
		return true
	}

	type fileCTXResult struct {
		tags []Tag
		vecs map[string][]float32
	}

	// Create a worker pool with a maximum of 4 workers
	const maxWorkers = 4
	fileChan := make(chan string, len(allFilenames))
	resChan := make(chan fileCTXResult, len(allFilenames))
	errChan := make(chan error, len(allFilenames))

	var wg sync.WaitGroup

	// Worker function
	worker := func() {
		defer wg.Done()
		for fname := range fileChan {
			vecs, tags, err := r.GetFileCTX(ctx, fname, tagFilter)
			if err != nil {
				if err == grepast.ErrorUnsupportedLanguage {
					log.Trace().Msgf("skip %s", fname)
					continue
				}
				log.Error().Err(err).Msgf("Error getting tags for file: %s", fname)
				errChan <- err
				continue
			}

			resChan <- fileCTXResult{tags: tags, vecs: vecs}
		}
	}

	// Start workers
	for i := 0; i < maxWorkers; i++ {
		wg.Add(1)
		go worker()
	}

	// Send filenames to the file channel
	go func() {
		for _, fname := range allFilenames {
			fileChan <- fname
		}
		close(fileChan)
	}()

	// Wait for all workers to finish
	go func() {
		wg.Wait()
		close(resChan)
		close(errChan)
	}()

	// Collect results
	for res := range resChan {
		mu.Lock()
		allTags = append(allTags, res.tags...)
		for k, v := range res.vecs {
			r.graph.Add(hnsw.MakeNode(k, v))
		}
		mu.Unlock()
	}

	// Handle errors if needed
	for err := range errChan {
		log.Error().Err(err).Msg("Error processing files")
	}

	return allTags
}

type tagKey struct {
	fname  string // the file name (relative)
	symbol string // the actual identifier
}

func (r *RepoCTX) getRankedTagsByPageRank(allTags []Tag, mentionedFilenames, mentionedIdents map[string]bool) []Tag {

	//--------------------------------------------------------
	// 1) Build up references/defines data structures
	//--------------------------------------------------------
	defines, references, definitions, identifiers := r.buildReferenceMaps(allTags)

	if r.verbose {
		// tr@ck
		fmt.Printf("\n\n## defines:")
		for k, v := range defines {
			fmt.Printf("\n- %s: %v", k, v)
		}
		fmt.Printf("\n\n## definitions:")
		for k, v := range definitions {
			fmt.Printf("\n- %s: %v", k, v)
		}
		fmt.Printf("\n\n## references:")
		for k, v := range references {
			fmt.Printf("\n- %s: %v", k, v)
		}
		fmt.Printf("\n\n## idents:")
		for k := range identifiers {
			fmt.Printf("\n- %s", k)
		}
	}

	//--------------------------------------------------------
	// 2) Construct a multi-directed graph
	//--------------------------------------------------------
	g, nodeByFile, fileSet := r.buildFileGraph(defines, references, identifiers, mentionedIdents)

	// 4) Personalization
	personal := make(map[int64]float64)
	totalFiles := float64(len(fileSet))
	defaultPersonal := 1.0 / totalFiles

	chatSet := make(map[string]struct{})
	for cf := range mentionedFilenames {
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

	//--------------------------------------------------------
	// 3) Distribute each file’s rank across its out-edges
	//--------------------------------------------------------
	edgeRanks := distributeRank(pr, defines, references, nodeByFile, mentionedIdents)

	if r.verbose {
		fmt.Printf("\n\n## Ranked defs:")
		for edge, rank := range edgeRanks {
			fmt.Printf("\n- %v / %s / %s", rank, edge.dst, edge.symbol)
		}
	}

	//--------------------------------------------------------
	// 4) Convert edge-based rank to a sorted list
	//--------------------------------------------------------
	defRankSlice := toDefRankSlice(edgeRanks)

	// 8) Sort by rank, then by fname, then by symbol
	sort.Slice(defRankSlice, func(i, j int) bool {
		if defRankSlice[i].rank != defRankSlice[j].rank {
			return defRankSlice[i].rank > defRankSlice[j].rank
		}
		if defRankSlice[i].fname != defRankSlice[j].fname {
			return defRankSlice[i].fname < defRankSlice[j].fname
		}
		return defRankSlice[i].symbol < defRankSlice[j].symbol
	})

	chatRelFilenames := make(map[string]bool)
	// If you had a slice of chatFilenames, for example:
	/*
		for _, cf := range chatFilenamesSlice {
			rel := r.GetRelFilename(cf)
			chatRelFilenames[rel] = true
		}
	*/
	if r.verbose {
		fmt.Printf("\n\n## Ranked defs (SORTED):")
		for _, v := range defRankSlice {
			fmt.Printf("\n- %v / %s / %s", v.rank, v.fname, v.symbol)
		}

		fmt.Printf("\n\n")
	}

	//--------------------------------------------------------
	// 5) Gather final tags, skipping chat files if desired
	//--------------------------------------------------------
	var rankedTags []Tag
	for _, dr := range defRankSlice {
		if chatRelFilenames[dr.fname] {
			continue
		}
		k := tagKey{fname: dr.fname, symbol: dr.symbol}
		defs := definitions[k]
		rankedTags = append(rankedTags, defs...)
	}

	// Possibly append files that have no tags, etc.
	return rankedTags
}

// edgeData is a small struct to hold adjacency info for distributing rank
type edgeData struct {
	dstFile string
	symbol  string
	weight  float64
}

// EdgeRank is a struct to hold the edge data for distributing rank
type EdgeRank struct {
	dst    string
	symbol string
}

// DefRank is a struct to hold the rank for a definition
type DefRank struct {
	fname  string
	symbol string
	rank   float64
}

// toDefRankSlice converts the map[EdgeRank]rank into a slice for sorting
func toDefRankSlice(edgeRanks map[EdgeRank]float64) []DefRank {
	defRankSlice := make([]DefRank, 0, len(edgeRanks))
	for k, v := range edgeRanks {
		defRankSlice = append(defRankSlice, DefRank{
			fname:  k.dst,
			symbol: k.symbol,
			rank:   v,
		})
	}
	return defRankSlice
}

// distributeRank inspects each node's PageRank, sums the weights of all its out-edges,
// and then distributes that node's rank proportionally along those edges.
// The result is a mapping (defFile, symbol) -> rank. This parallels:
//
//	for src in G.nodes:
//	    srcRank = ranked[src]
//	    totalWeight = sum of out-edge weights
//	    for edge in out-edges:
//	        portion = srcRank * (edgeWeight / totalWeight)
//	        ranked_definitions[(edge.target, edge.symbol)] += portion
func distributeRank(
	pr map[int64]float64,
	defines map[string]map[string]struct{},
	references map[string][]string,
	nodeByFile map[string]graph.Node,
	mentionedIdents map[string]bool,
) map[EdgeRank]float64 {

	// 6) Distribute rank from each src node across its out edges
	edgeRanks := make(map[EdgeRank]float64)

	for symbol, refMap := range references {
		defFiles := defines[symbol]
		if defFiles == nil {
			continue
		}

		// // tr@ck - ranked tags
		// fmt.Println("\n\nRanked tags:")
		// for _, t := range rankedTags {
		// 	fmt.Printf("- %s / %d / %s\n", t.Kind, t.Line, t.Name)
		// }

		var mul float64
		switch {
		case mentionedIdents[symbol]:
			mul = 10.0
		case strings.HasPrefix(symbol, "_"):
			mul = 0.1
		default:
			mul = 1.0
		}

		for _, refFile := range refMap {
			w := mul * math.Sqrt(float64(len(refMap)))
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

	return edgeRanks
}

// buildFileGraph scans the union of (defines, references) to find all unique filenames
// and create a node for each. The return is a MultiDirectedGraph plus a lookup map to
// find that node by filename.
func (r *RepoCTX) buildFileGraph(
	defines map[string]map[string]struct{},
	references map[string][]string,
	identifiers map[string]bool,
	mentionedIdents map[string]bool,
) (
	g *multi.WeightedDirectedGraph,
	nodeByFile map[string]graph.Node,
	fileSet map[string]struct{},
) {
	// 2) Build a multi directed graph
	g = multi.NewWeightedDirectedGraph()

	// Keep track of the node ID for each rel_fname
	nodeByFile = make(map[string]graph.Node)

	// Gather all relevant filenames
	fileSet = make(map[string]struct{})
	for _, defFiles := range defines {
		for f := range defFiles {
			fileSet[f] = struct{}{}
		}
	}
	for _, refMap := range references {
		for _, f := range refMap {
			fileSet[f] = struct{}{}
		}
	}

	// Create node for each file
	for f := range fileSet {
		n := g.NewNode()
		g.AddNode(n)
		nodeByFile[f] = n
	}

	if r.verbose {
		fmt.Printf("\n\nNumber of nodes (files): %d\n", g.Nodes().Len())
	}

	// 3) For each ident, link referencing file -> defining file with weight
	for ident := range identifiers {
		defFiles := defines[ident]
		if len(defFiles) == 0 {
			continue
		}

		var mul float64
		switch {
		case mentionedIdents[ident]:
			mul = 10.0
		case strings.HasPrefix(ident, "_"):
			mul = 0.1
		default:
			mul = 1.0
		}

		for _, refFile := range references[ident] {
			// log.Trace().Msg(color.YellowString("refFile: %s, numRefs: %d"), refFile, numRefs))
			w := mul * math.Sqrt(float64(len(references[ident])))
			for defFile := range defFiles {
				refNode := nodeByFile[refFile]
				defNode := nodeByFile[defFile]

				// Create a weighted edge
				edge := g.NewWeightedLine(refNode, defNode, w)
				g.SetWeightedLine(edge)
			}
		}
	}

	return g, nodeByFile, fileSet
}

// buildReferenceMaps reads a slice of Tag objects and partitions them into
// (symbol -> set of files that define it) and (symbol -> map[file] countOfRefs).
// It also tracks the actual definition Tag objects for (file,symbol).
func (r *RepoCTX) buildReferenceMaps(allTags []Tag) (
	defines map[string]map[string]struct{}, // symbol -> set{relFilename}
	references map[string][]string, // symbol -> map[relFilename] -> # of references
	definitions map[tagKey][]Tag, // (relFilename, symbol) -> slices of definition tags
	identifiers map[string]bool, // set of symbols that have both defines and references
) {
	// 1) Collect references, definitions
	// defines is a set of filenames that define a symbol
	defines = make(map[string]map[string]struct{}) // symbol -> set of filenames that define it
	// references is a list of files per symbol
	references = make(map[string][]string) // symbol -> map of (referencerFile -> countOfRefs)
	// definitions is a set of symbols (tags) including file where they are defined
	definitions = make(map[tagKey][]Tag) // (fname, symbol) -> slice of definition Tags

	for _, t := range allTags {
		rel := r.GetRelFilename(t.FilePath)

		switch t.Kind {
		case TagKindDef:
			if defines[t.Name] == nil {
				defines[t.Name] = make(map[string]struct{})
			}
			defines[t.Name][rel] = struct{}{}

			k := tagKey{fname: rel, symbol: t.Name}
			definitions[k] = append(definitions[k], t)

		case TagKindRef:
			// if references[t.Name] == nil {
			// 	references[t.Name] = map[string][]string{t.FileName: {rel}}
			// }
			references[t.Name] = append(references[t.Name], rel)
		}
	}

	// If references is empty, fall back to references=defines
	// this code is needed as page rank will not work if references is empty
	if len(references) == 0 {
		for sym, defFiles := range defines {
			for df := range defFiles {
				references[sym] = append(references[sym], df)
			}
		}
	}

	// idents = set(defines.keys()).intersection(set(references.keys()))
	//
	identifiers = make(map[string]bool)
	for sym := range defines {
		if _, ok := references[sym]; ok {
			identifiers[sym] = true
		}
	}

	return defines, references, definitions, identifiers
}

// fallbackReferences is used when no references are found. Python code sets references = defines,
// effectively giving each symbol a trivial reference from its own definer.
func (r *RepoCTX) fallbackReferences(defines map[string]map[string]struct{}) map[string]map[string]int {
	refs := make(map[string]map[string]int)
	for sym, defFiles := range defines {
		refs[sym] = make(map[string]int)
		for df := range defFiles {
			// Just increment by 1 to indicate a trivial reference from the def file to itself
			refs[sym][df]++
		}
	}
	return refs
}

// GetRankedTagsMap orchestrates calls to getRankedTags and toTree to produce the final “map” string.
func (r *RepoCTX) GetRankedTags(
	tags []Tag,
	maxMapTokens int,
	mentionedFilenames, mentionedIdents map[string]bool,
) []Tag {
	// Handle empty tag list
	if len(tags) == 0 {
		return []Tag{}
	}

	startTime := time.Now()

	// Get ranked tags by PageRank
	rankedTags := r.getRankedTagsByPageRank(tags, mentionedFilenames, mentionedIdents)

	// special := filterImportantFiles(otherFilenames)

	// // Prepend special files as “important”.
	// var specialTags []Tag
	// for _, sf := range special {
	// 	specialTags = append(specialTags, Tag{Name: r.GetRelFilename(sf)})
	// }
	// finalTags := append(specialTags, rankedTags...)

	finalTags := rankedTags

	// Compute duration
	endTime := time.Now()
	r.totalProcessingTime = endTime.Sub(startTime).Seconds()

	return finalTags
}

// Generate is the top-level function (mirroring the Python method) that produces the “repo content”.
func (r *RepoCTX) Generate(
	ctx context.Context,
	chatFiles, otherFiles []string,
	mentionedFilenames, mentionedIdents map[string]bool,
) string {

	if r.maxMapTokens <= 0 {
		log.Warn().Msgf("Repo-map disabled by max_map_tokens: %d", r.maxMapTokens)
		return ""
	}
	// if len(otherFiles) == 0 {
	// 	log.Warn().Msg("No other files found; disabling repo map")
	// 	return ""
	// }
	if mentionedFilenames == nil {
		mentionedFilenames = make(map[string]bool)
	}
	if mentionedIdents == nil {
		mentionedIdents = make(map[string]bool)
	}

	maxMapTokens := r.maxMapTokens
	padding := 4096
	var target int
	if maxMapTokens > 0 && r.maxCtxWindow > 0 {
		t := maxMapTokens * r.maxCtxFileMultiplier
		t2 := r.maxCtxWindow - padding
		if t2 < 0 {
			t2 = 0
		}
		if t < t2 {
			target = t
		} else {
			target = t2
		}
	}
	if len(chatFiles) == 0 && r.maxCtxWindow > 0 && target > 0 {
		maxMapTokens = target
	}

	var treeMap string
	// defer func() {
	// 	if rec := recover(); rec != nil {
	// 		fmt.Printf("ERR: Disabling repo map, repository may be too large?")
	// 		r.MaxMapTokens = 0
	// 		filesListing = ""
	// 	}
	// }()

	// Combine chatFilenames and otherFilenames into a map of unique elements
	allFilenames := uniqueElements(chatFiles, otherFiles)

	// Collect all tags from those files
	allTags := r.GetCTXFromFiles(ctx, allFilenames)

	// Rank tags
	rankedTags := r.GetRankedTags(allTags, maxMapTokens, mentionedFilenames, mentionedIdents)

	// Build the tree map
	treeMap = r.buildTreeMapFromTags(rankedTags, chatFiles)
	if treeMap == "" {
		return ""
	}

	if r.verbose {
		numTokens := r.TokenCount(treeMap)
		fmt.Printf("Repo-map: %.1f k-tokens\n", numTokens/1024.0)
	}

	other := ""
	if len(chatFiles) > 0 {
		other = "other "
	}

	var repoContent string
	if r.contentPrefix != "" {
		repoContent = strings.ReplaceAll(r.contentPrefix, "{other}", other)
	}

	// ready for querying
	r.ready <- true

	repoContent += treeMap
	return repoContent
}

// BuildTreeMapFromTags builds a tree map from the tags.
// todo: handle max tokens
func (r *RepoCTX) buildTreeMapFromTags(tags []Tag, chatFilenames []string) string {

	bestTree := ""
	// bestTreeTokens := 0.0

	// lb := 0
	ub := len(tags)
	middle := ub
	if middle > 30 {
		middle = 30
	}

	bestTree = r.toTree(tags, chatFilenames)

	// for lb <= ub {
	// 	tree := r.toTree(finalTags[:middle], chatFilenames)
	// 	numTokens := r.TokenCount(tree)

	// 	diff := math.Abs(numTokens - float64(maxMapTokens))
	// 	pctErr := diff / float64(maxMapTokens)
	// 	if (numTokens <= float64(maxMapTokens) && numTokens > bestTreeTokens) || pctErr < 0.15 {
	// 		bestTree = tree
	// 		bestTreeTokens = numTokens
	// 		if pctErr < 0.15 {
	// 			break
	// 		}
	// 	}
	// 	if numTokens < float64(maxMapTokens) {
	// 		lb = middle + 1
	// 	} else {
	// 		ub = middle - 1
	// 	}
	// 	middle = (lb + ub) / 2
	// }

	// Set the last map
	r.lastMap = bestTree

	return bestTree
}

// toTree converts a list of Tag objects into a tree-like string representation.
func (r *RepoCTX) toTree(tags []Tag, chatFilenames []string) string {
	// Return immediately if no tags
	if len(tags) == 0 {
		return ""
	}

	// 1) Build a set of relative filenames that should be skipped
	chatRelSet := make(map[string]bool)
	for _, c := range chatFilenames {
		rel := r.GetRelFilename(c)
		chatRelSet[rel] = true
	}

	// tr@ck - verbose
	for i, c := range chatFilenames {
		log.Trace().Int("index", i).Str("z", c).Msg("chat files")
	}

	//  2) Sort the tags first by FileName in ascending order, and then by Line in ascending order
	// if two tags have the same FileName. This ensures a stable order where entries
	// are grouped by file and appear sequentially by their line numbers within each file.
	sort.Slice(tags, func(i, j int) bool {
		if tags[i].FileName != tags[j].FileName {
			return tags[i].FileName < tags[j].FileName
		}
		return tags[i].Line < tags[j].Line
	})

	// A sentinel value used to trigger a final flush of the current file's data in a streaming process.
	sentinel := "__sentinel_tag__"

	// 3) Append a sentinel tag, which triggers the final flush when we hit it in the loop.
	tags = append(tags, Tag{FileName: sentinel, Name: sentinel})

	// 4) Prepare to walk through each tag, grouping them by file.
	var output strings.Builder

	var curFilename string    // Tracks the *relative* file name of the current group
	var curAbsFilename string // Tracks the absolute path for rendering
	var linesOfInterest []int

	// sort tags by line number

	// 5) Process tags in a streaming fashion, flushing out each file's lines-of-interest
	//    when we detect a "new file name" or the dummy tag.
	for i, t := range tags {
		log.Trace().Int("index", i).Str("z", t.FileName).Int("line", t.Line).Str("tag", t.Name).Msg("tags")

		relFilename := t.FileName
		// // Skip tags that belong to a “chat” file. (Python: if this_rel_fname in chat_rel_fnames: continue)
		// if chatRelSet[relFilename] {
		// 	continue
		// }

		// If we've encountered a new file (i.e., the file name changed),
		// flush out the old file's lines-of-interest (if any).
		if relFilename != curFilename {
			if curFilename != "" && linesOfInterest != nil {
				// Write a blank line, then the file name plus colon
				output.WriteString("\n" + curFilename + ":\n")

				code, err := os.ReadFile(curAbsFilename)
				if err != nil {
					log.Warn().Err(err).Msgf("Failed to read file (%s)", curAbsFilename)
					continue
				}

				// Render the code snippet for the previous file.
				rendered, err := r.renderTree(curFilename, code, linesOfInterest)
				if err != nil {
					// If there's an error reading or parsing the file, just log and move on.
					log.Warn().Err(err).Msgf("Failed to render tree for %s", curFilename)
				}
				output.WriteString(rendered)
			}

			// If the new file name is the dummy sentinel, we've reached the end; stop.
			if relFilename == sentinel {
				break
			}

			// Otherwise, reset our state for the *new* file.
			curFilename = relFilename
			curAbsFilename = t.FilePath
			linesOfInterest = []int{}
		}

		// Accumulate the line number from this tag for the current file.
		if linesOfInterest != nil {
			linesOfInterest = append(linesOfInterest, t.Line)
		}
	}

	// 6) Truncate lines in the final output, in case of minified or extremely long content.
	//    This matches the Python code that does:  line[:100] for line in output.splitlines()
	lines := strings.Split(output.String(), "\n")
	for i, ln := range lines {
		if len(ln) > 100 {
			lines[i] = ln[:100]
		}
	}

	// 7) Return the final output (plus a newline).
	return strings.Join(lines, "\n") + "\n"
}

// renderTree uses a grep-ast TreeContext to produce a nice snippet with lines of interest expanded.
func (r *RepoCTX) renderTree(relFilename string, code []byte, linesOfInterest []int) (string, error) {
	if r.verbose {
		fmt.Printf("\nrender_tree:  %s, %v\n", relFilename, linesOfInterest)
	}

	// Build a grep-ast TreeContext.
	tc, err := grepast.NewTreeContext(
		relFilename, code,
		grepast.WithColor(false),
		grepast.WithChildContext(false),
		grepast.WithLastLineContext(false),
		grepast.WithTopMargin(0),
		grepast.WithLinesOfInterestMarked(false),
		grepast.WithLinesOfInterestPadding(2),
		grepast.WithTopOfFileParentScope(false),
	)
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

	// fmt.Println(loiMap)
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

// GetRepoFiles gathers all files in a directory (or a single file) and returns
// two values: the slice of file paths and a tree-like string representing
// the folder structure.
func (r *RepoCTX) GetRepoFiles(path string) ([]string, string) {
	info, err := os.Stat(path)
	if err != nil {
		// On error, return empty slices (or handle error as desired).
		return nil, ""
	}

	// If the path is a single file, we can simply return it. The "tree map" is trivial.
	if !info.IsDir() {
		fileName := filepath.Base(path)
		treeMap := fmt.Sprintf("└── %s\n", fileName)
		return []string{path}, treeMap
	}

	// Otherwise, build the tree and collect the file paths from the directory.
	tree, files := r.buildTree(path, "")
	return files, tree
}

// buildTree is a helper function that constructs a tree-like structure for the
// directory at 'path' and collects all non-ignored file paths recursively.
// 'prefix' is updated as we go deeper, to produce correct tree branches.
func (r *RepoCTX) buildTree(path, prefix string) (string, []string) {
	var (
		treeBuilder strings.Builder
		filePaths   []string
	)

	entries, err := os.ReadDir(path)
	if err != nil {
		// If there's an error reading the directory, simply return what we have.
		// You might prefer to log the error or handle it differently.
		log.Error().Err(err).Str("path", path).Msg("unable to read directory")
		return "", nil
	}

	// Filter out ignored entries first so we can accurately set the "last entry" connector.
	filtered := make([]os.DirEntry, 0, len(entries))
	for _, entry := range entries {
		fullPath := filepath.Join(path, entry.Name())
		// Use RepoMap’s ignore logic to skip undesired paths:
		if r.globIgnorePatterns.MatchesPath(fullPath) {
			continue
		}
		filtered = append(filtered, entry)
	}

	// Traverse each of the filtered entries in this directory.
	for i, entry := range filtered {
		connector := "├──"
		subPrefix := prefix + "│   "

		// Check if we are on the last entry; adjust prefix for child entries accordingly.
		isLast := i == len(filtered)-1
		if isLast {
			connector = "└──"
			subPrefix = prefix + "    "
		}

		// Print current node
		treeBuilder.WriteString(fmt.Sprintf("%s%s %s\n", prefix, connector, entry.Name()))
		fullPath := filepath.Join(path, entry.Name())

		// If directory, recurse and append the results
		if entry.IsDir() {
			subtree, subFiles := r.buildTree(fullPath, subPrefix)
			treeBuilder.WriteString(subtree)
			filePaths = append(filePaths, subFiles...)
		} else {
			// If a file, add to file paths
			filePaths = append(filePaths, fullPath)
		}
	}

	return treeBuilder.String(), filePaths
}

// FindGitRoot walks upward from the given path until
// it finds a directory containing a ".git" folder.
func FindGitRoot(start string) (string, error) {
	current, err := filepath.Abs(start)
	if err != nil {
		return "", fmt.Errorf("could not get absolute path of %q: %w", start, err)
	}

	for {
		// Does ".git" exist here?
		gitPath := filepath.Join(current, ".git")
		info, err := os.Stat(gitPath)
		if err == nil && info.IsDir() {
			// Found .git
			return current, nil
		}

		// can't go higher, stop
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
		current = parent
	}
	return "", fmt.Errorf("no .git folder found starting from %q and up", start)
}
