package main

import (
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
	ConfigLogging(&trace, &debug)

	inputPath := "."
	if len(os.Args) == 2 {
		inputPath = os.Args[1]
	}

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

	// 3. Build the RepoMap
	rm := codectx.NewRepoMap(
		root,                 // pass the discovered root
		&codectx.ModelStub{}, // or your real model
		codectx.WithLogLevel(int(zerolog.DebugLevel)),
	)

	// 4. Decide which files are "chat files" vs. "other files"
	//    This part depends on your usage pattern. For a simple example:
	//    - If the input is a single file, treat that as the 'chat file'
	//    - If the input is a directory, gather all files from that directory as 'chat files'
	//    - Then, optionally, gather other files from the entire repo if you want a full map.

	// var chatFiles []string
	var otherFiles []string

	allFiles, treeMap := rm.GetRepoFiles(absPath)

	fmt.Println(treeMap)

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

	codeMap := rm.Generate(
		allFiles,
		otherFiles,
		mentionedFnames,
		mentionedIdents,
	)

	if codeMap == "" {
		fmt.Println("Empty Code Map")
		return
	}

	fmt.Println(codeMap)
}

// ConfigLogging configures the logging level and format
func ConfigLogging(trace, debug *bool) {
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
	log.Logger = log.With().Caller().Logger()

	if *trace {
		zerolog.SetGlobalLevel(zerolog.TraceLevel)
		log.Debug().Msg("Trace logging enabled")
		return
	}

	if *debug {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
		log.Debug().Msg("Debug logging enabled")
		return
	}

	// add CTX_LOG env variable to set log level
	if logLevel, ok := os.LookupEnv("CTX_LOG"); ok {
		switch logLevel {
		case "debug":
			zerolog.SetGlobalLevel(zerolog.DebugLevel)
			*debug = true
			log.Debug().Msg("debug logging enabled")
		case "trace":
			zerolog.SetGlobalLevel(zerolog.TraceLevel)
			*trace = true
			*debug = true
			log.Trace().Msg("trace logging enabled")
		case "error":
			zerolog.SetGlobalLevel(zerolog.ErrorLevel)
		default:
			zerolog.SetGlobalLevel(zerolog.InfoLevel)
			log.Warn().Msgf("Invalid log level: %s", logLevel)
		}
		return
	}

	// default log level
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
}
