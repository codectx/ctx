# CodeCTX (ctx)

_A Go library for building repository code maps using AST/CST. Particularly usefuel with AI-powered coding assistants._

<p align="center">
  <img height="256" src="./ctx.svg">
</p>

> [!CAUTION]
> This project is still very much in development

## Overview

**CTX** (CTX) is a Go library designed to analyze and map code repositories by leveraging **Concrete Syntax Trees (CST)** (often referred to as **Abstract Syntax Trees (AST)**). It enables AI-powered tools to understand code structures, dependencies, and relationships efficiently.

CTX provides two distinct mechanisms for obtaining relevant repo code:

1. A compelete "snapshot" of the current repo which includes the file tree structure and the important code definitions. Much credit goes to by [Aider-AI/aider](https://github.com/Aider-AI/aider).
2. A chunk embedding search capabtility. Much credit goes to [sweepai/sweep](https://github.com/sweepai/sweep).

## Features

- **Graph-based repository mapping**: Extracts entities and relationships from source code.
- **AST/CST parsing**: Supports deep syntax analysis for code intelligence.
- **AI-friendly representation**: Generates structured data suitable for AI coding assistants.
- **Language support**: Primarily focused on Go, with extensibility for other languages.
- **Efficient and scalable**: Designed for large repositories with minimal performance overhead.

## Installation

```sh
go get github.com/codectx/ctx
```

## Usage

### Exlucding Files and Folers

Use a .gitignore or create a git-compatible .astignore. Alternatily copy the .astignore from this repo into yours.

### Example

See `cmd/main.go` for a working example.

# Outstanding Items

- cst caching
- "chat files" to increase weight

## License

[MIT License](LICENSE.md)
