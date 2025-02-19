// Package store provides a storage service for embeddings.
package store

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"

	// Import the DuckDB driver
	_ "github.com/marcboeker/go-duckdb"

	// import google uuid
	"github.com/google/uuid"
)

// Embedding holds a single row from the embeddings table.
type Embedding struct {
	ID     string
	Hash   string
	Vector []float32
	RunID  uuid.UUID
}

// StorageService defines the interface for CRUD operations on DuckDB.
type StorageService interface {
	// Upsert inserts or updates a row
	Upsert(context.Context, string, string, []float32) error
	// // BatchUpsert inserts or updates one or more rows
	// BatchUpsert(context.Context, string, string, map[string]Embedding) error
	// GetAll fetches all rows.
	GetAll(ctx context.Context) (map[string]Embedding, error)
	// Get fetches multiple rows by ids.
	Get(ctx context.Context, id []string) ([]Embedding, error)
	// MatchHash checks if the given hash matches the stored hash for the given id.
	MatchHash(ctx context.Context, id, hash string) (bool, error)
	// Delete removes a row by id.
	Delete(ctx context.Context, id string) error
}

// storageService implements StorageService.
type storageService struct {
	db    *sql.DB
	runID uuid.UUID
	// mu sync.Mutex
}

// NewStorageService opens or creates local.db and prepares the embeddings table.
// Panics on failure.
func NewStorageService(db *sql.DB) StorageService {

	// Create table if it doesn't exist.
	createTableSQL := `
    CREATE TABLE IF NOT EXISTS embeddings (
        id TEXT PRIMARY KEY,
				run_id TEXT,
        hash TEXT,
        embedding BLOB
    )
    `

	if _, err := db.Exec(createTableSQL); err != nil {
		panic(fmt.Sprintf("Failed to create embeddings table: %v", err))
	}

	// Generate a unique run ID.
	runID, _ := uuid.NewV7()

	return &storageService{db, runID}
}

// Upsert inserts or updates a row.
func (s *storageService) Upsert(ctx context.Context, id, hash string, vector []float32) error {
	// Insert or update the row.
	upsertSQL := `INSERT INTO embeddings (id, run_id, hash, embedding) VALUES (?, ?, ?, ?) 
		ON CONFLICT(id) DO UPDATE SET hash = excluded.hash, run_id = excluded.run_id, embedding = excluded.embedding;`

	// s.mu.Lock()
	// defer s.mu.Unlock()

	_, err := s.db.ExecContext(ctx, upsertSQL, id, s.runID.String(), hash, float32SliceToBytes(vector))
	if err != nil {
		return fmt.Errorf("Upsert failed: %w", err)
	}
	return nil
}

// Get fetches multiple rows by ids.
func (s *storageService) Get(ctx context.Context, id []string) ([]Embedding, error) {
	// build query with IN clause or do repeated SELECT.
	if len(id) == 0 {
		return nil, nil
	}
	// naive approach: SELECT * FROM embeddings WHERE id IN (?,?,?)
	query := "SELECT id, run_id, hash, embedding FROM embeddings WHERE id IN ("
	params := make([]interface{}, 0, len(id))
	for i, v := range id {
		if i > 0 {
			query += ","
		}
		query += "?"
		params = append(params, v)
	}
	query += ");"

	// s.mu.Lock()
	// defer s.mu.Unlock()

	rows, err := s.db.QueryContext(ctx, query, params...)
	if err != nil {
		return nil, fmt.Errorf("Get failed: %w", err)
	}
	defer rows.Close()
	// s.mu.Unlock()

	var results []Embedding
	for rows.Next() {
		var (
			e Embedding
			b []byte
		)
		err := rows.Scan(&e.ID, &e.RunID, &e.Hash, &b)
		if err != nil {
			return nil, fmt.Errorf("Get scan failed: %w", err)
		}
		e.Vector = bytesToFloat32Slice(b)
		results = append(results, e)
	}
	if rows.Err() != nil {
		return nil, rows.Err()
	}
	return results, nil
}

// Delete removes a row by key.
func (s *storageService) Delete(ctx context.Context, id string) error {
	// s.mu.Lock()
	// defer s.mu.Unlock()

	_, err := s.db.ExecContext(ctx, "DELETE FROM embeddings WHERE id = ?;", id)
	if err != nil {
		return fmt.Errorf("Delete failed: %w", err)
	}
	return nil
}

// MatchHash checks if the given hash matches the stored hash for the given id.
// Returns true if the hashes match, false if they don't, or an error.
func (s *storageService) MatchHash(ctx context.Context, id, hash string) (bool, error) {
	query := "SELECT COALESCE((SELECT CASE WHEN hash = ? THEN 1 ELSE 0 END FROM embeddings WHERE id = ?), 0);"

	var match int

	// s.mu.Lock()
	// defer s.mu.Unlock()

	err := s.db.QueryRowContext(ctx, query, hash, id).Scan(&match)
	if err != nil {
		return false, fmt.Errorf("MatchHash query failed: %w", err)
	}
	return match == 1, nil
}

// GetAll fetches all rows from the embeddings table.
func (s *storageService) GetAll(ctx context.Context) (map[string]Embedding, error) {
	// s.mu.Lock()
	// defer s.mu.Unlock()

	rows, err := s.db.QueryContext(ctx, "SELECT id, run_id, hash, embedding FROM embeddings;")
	if err != nil {
		return nil, fmt.Errorf("GetAll failed: %w", err)
	}
	defer rows.Close()
	// s.mu.Unlock()

	// map of results
	results := map[string]Embedding{}

	// iterate over rows
	for rows.Next() {
		var (
			e Embedding
			b []byte
		)
		err := rows.Scan(&e.ID, &e.RunID, &e.Hash, &b)
		if err != nil {
			return nil, fmt.Errorf("GetAll scan failed: %w", err)
		}

		e.Vector = bytesToFloat32Slice(b)
		results[e.ID] = e
	}

	// check for errors
	if rows.Err() != nil {
		return nil, rows.Err()
	}

	return results, nil
}

// float32SliceToBytes converts []float32 to a binary representation.
// This is a naive approach. For production, consider carefully handling endianness.
func float32SliceToBytes(vec []float32) []byte {
	buf := new(bytes.Buffer)
	for _, f := range vec {
		binary.Write(buf, binary.LittleEndian, f)
	}
	return buf.Bytes()
}

// bytesToFloat32Slice converts raw bytes into a []float32.
func bytesToFloat32Slice(b []byte) []float32 {
	r := bytes.NewReader(b)
	var out []float32
	for {
		var val float32
		err := binary.Read(r, binary.LittleEndian, &val)
		if err != nil {
			break
		}
		out = append(out, val)
	}
	return out
}
