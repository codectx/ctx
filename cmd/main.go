// Package main provides the entry point for the codectx CLI.
package main

import (
	"bufio"
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"

	codectx "github.com/codectx/ctx"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	if len(os.Args) > 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s [path-to-file-or-dir]\n", filepath.Base(os.Args[0]))
		os.Exit(1)
	}

	trace := false
	debug := false
	lvl := ConfigLogging(&trace, &debug)

	inputPath := "."
	if len(os.Args) == 2 {
		inputPath = os.Args[1]
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 1. Get the path argument
	absPath, err := filepath.Abs(inputPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Error getting absolute path")
	}

	// 2. Find the root of the git repo
	root, err := codectx.FindGitRoot(absPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Error finding .git")
	}

	// tr@ck - pass database / or client as option WithDatabase()
	database, err := sql.Open("duckdb", "local.db")
	if err != nil {
		log.Err(err).Msg("Failed to connect to DuckDB")
		os.Exit(1)
	}
	defer database.Close()

	// 3. Build the RepoMap
	rm := codectx.NewRepoCTX(
		root,                 // pass the discovered root
		&codectx.ModelStub{}, // or your real model
		database,
		codectx.WithLogLevel(int(lvl)),
	)

	// 4. Decide which files are "chat files" vs. "other files"
	//    This part depends on your usage pattern. For a simple example:
	//    - If the input is a single file, treat that as the 'chat file'
	//    - If the input is a directory, gather all files from that directory as 'chat files'
	//    - Then, optionally, gather other files from the entire repo if you want a full map.

	// var chatFiles []string
	var otherFiles []string

	allFiles, treeMap := rm.GetRepoFiles(absPath)

	// chatSet := make(map[string]bool)
	// for _, cf := range chatFiles {
	// 	chatSet[filepath.Clean(cf)] = true
	// }

	// for _, f := range allFiles {
	// 	cleanF := filepath.Clean(f)
	// 	if !chatSet[cleanF] {
	// 		otherFiles = append(otherFiles, cleanF)
	// 	}
	// }

	// for f, _ := range otherFiles {
	// 	fmt.Printf("- %s\n", f)
	// }

	// for f, _ := range chatSet {
	// 	fmt.Printf("- %s\n", f)
	// }

	// 5. Generate Repo Map
	mentionedFnames := map[string]bool{}
	mentionedIdents := map[string]bool{}

	var codeMap string
	done := make(chan struct{})

	go func() {
		defer close(done)
		codeMap = rm.Generate(
			ctx,
			allFiles,
			otherFiles,
			mentionedFnames,
			mentionedIdents,
		)
	}()

	query, err := getUserQuery(os.Args)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	<-done

	// Search
	q, err := rm.Query(ctx, query)
	// q, _, err := emb.Voyage(vKey, query)
	if err != nil {
		log.Err(err).Msg("Failed to embed query")
		return
	}

	// tr@ck - flag to print the tree map
	fmt.Println(treeMap)

	if codeMap == "" {
		fmt.Println("Empty Code Map")
		return
	}

	// tr@ck - flag to print the code map
	fmt.Println(codeMap)

	fmt.Printf("Query Results: %s\n", q)
}

// getUserQuery returns the query
func getUserQuery(args []string) (string, error) {

	const usage = "Usage: <optional:query>"

	if len(args) < 1 || len(args) > 2 {
		return "", fmt.Errorf(usage)
	}

	query := ""

	if len(args) == 2 {
		query = args[2]
	} else {
		var err error
		query, err = promptForUserQuery()
		if err != nil {
			return "", err
		}
	}

	if query == "" {
		return "", fmt.Errorf(usage)
	}

	return query, nil
}

// promptForUserQuery prompts the user to input a search query
func promptForUserQuery() (string, error) {
	fmt.Printf("Query: ")
	q, err := bufio.NewReader(os.Stdin).ReadString('\n')
	if err != nil {
		return "", err
	}

	return q, nil
}

// ConfigLogging configures the logging level and format
func ConfigLogging(trace, debug *bool) zerolog.Level {
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
	log.Logger = log.With().Caller().Logger()

	if *trace {
		zerolog.SetGlobalLevel(zerolog.TraceLevel)
		log.Debug().Msg("Trace logging enabled")
		return zerolog.TraceLevel
	}

	if *debug {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
		log.Debug().Msg("Debug logging enabled")
		return zerolog.DebugLevel
	}

	// add CTX_LOG env variable to set log level
	if logLevel, ok := os.LookupEnv("CTX_LOG"); ok {
		switch logLevel {
		case "debug":
			zerolog.SetGlobalLevel(zerolog.DebugLevel)
			*debug = true
			log.Debug().Msg("debug logging enabled")
			return zerolog.DebugLevel
		case "trace":
			zerolog.SetGlobalLevel(zerolog.TraceLevel)
			*trace = true
			*debug = true
			log.Trace().Msg("trace logging enabled")
			return zerolog.TraceLevel
		case "error":
			zerolog.SetGlobalLevel(zerolog.ErrorLevel)
			return zerolog.ErrorLevel
		default:
			zerolog.SetGlobalLevel(zerolog.InfoLevel)
			log.Warn().Msgf("Invalid log level: %s", logLevel)
		}
		return zerolog.InfoLevel
	}

	// default log level
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	return zerolog.InfoLevel
}
