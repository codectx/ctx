// Package embed provides a service for obtaining embeddings from text and code.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	ollama "github.com/ollama/ollama/api"
	"github.com/sugarme/tokenizer"
)

// EmbeddingsRequest represents the payload sent to the VoyageAI embeddings endpoint.
type EmbeddingsRequest struct {
	Input     interface{}                `json:"input"`                // Can be a string or []string
	Model     string                     `json:"model"`                // e.g. "voyage-code-3", "voyage-3-large", etc.
	InputType embeddingsRequestInputType `json:"input_type,omitempty"` // "query" or "document" (optional)
}

type embeddingsRequestInputType string

const (
	ollamaModelName = "unclemusclez/jina-embeddings-v2-base-code"
	voyageURL       = "https://api.voyageai.com/v1/embeddings"
	voyageModelName = "voyage-code-3"

	embeddingsRequestInputTypeQuery    embeddingsRequestInputType = "query"
	embeddingsRequestInputTypeDocument embeddingsRequestInputType = "document"
)

// EmbeddingService defines an interface for obtaining embeddings from text.
type EmbeddingService interface {
	// Get generates an embedding for the given text.
	Get(ctx context.Context, text []byte) ([]float32, Meta, error)
	// Get generates an embedding for the given text.
	Voyage(key, value string) ([]float32, Meta, error)
}

// embeddingService implements EmbeddingService.
type embeddingService struct {
	tk     *tokenizer.Tokenizer
	client *ollama.Client
}

// NewEmbedService returns an EmbeddingService instance.
// You might inject additional dependencies (e.g., Voyage clients) as needed.
func NewEmbedService(oClient *ollama.Client) EmbeddingService {
	if oClient == nil {
		panic("ollama client is not initialized")
	}

	return &embeddingService{
		client: oClient,
	}
}

// Meta holds metadata about an embedding
type Meta struct {
	// Tokens is the number of tokens in the input
	Tokens int
	// Duration is the duration in milliseconds
	Duration int
	// ProviderName is the name of the embedding provider
	ProviderName string
	// ProviderModel is the name of the embedding model
	ProviderModel string
}

// Get obtains an embedding using the Ollama client
// In production, handle tokens, model name, error checking, etc.
func (s *embeddingService) Get(ctx context.Context, value []byte) ([]float32, Meta, error) {
	start := time.Now()
	emb, err := s.client.Embed(ctx, &ollama.EmbedRequest{
		Model: ollamaModelName,
		Input: value,
	})

	if err != nil {
		return nil,
			Meta{
				Tokens:        0,
				Duration:      int(time.Since(start).Milliseconds()),
				ProviderName:  "ollama",
				ProviderModel: ollamaModelName,
			},
			fmt.Errorf("failed to embed text: %w", err)
	}

	return emb.Embeddings[0],
		Meta{
			Tokens:        emb.PromptEvalCount,
			Duration:      int(time.Since(start).Milliseconds()),
			ProviderName:  "ollama",
			ProviderModel: ollamaModelName,
		}, nil
}

// embedVoyage embeds the given value using the VoyageAI API.
func (s *embeddingService) Voyage(key, value string) ([]float32, Meta, error) {

	// Prepare request body
	requestBody := EmbeddingsRequest{
		Input:     value,
		Model:     voyageModelName,
		InputType: embeddingsRequestInputTypeDocument,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, Meta{}, fmt.Errorf("failed to marshal JSON: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequest(http.MethodPost, voyageURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, Meta{}, fmt.Errorf("failed to create HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)

	start := time.Now()
	// Execute HTTP request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, Meta{}, fmt.Errorf("failed to execute HTTP request: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, Meta{}, fmt.Errorf("failed to read response body: %w", err)
	}

	// Unmarshal response
	var res voyageAIResponse
	if err := json.Unmarshal(body, &res); err != nil {
		return nil, Meta{}, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	return res.Data[0].Embedding, Meta{
		Tokens:        res.Usage.TotalTokens,
		ProviderName:  "voyageai",
		ProviderModel: voyageModelName,
		Duration:      int(time.Since(start).Milliseconds()),
	}, nil
}

type voyageAIResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Embedding []float32 `json:"embedding"`
	}
	Model string `json:"model"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}
