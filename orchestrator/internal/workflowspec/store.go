package workflowspec

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Store persists workflow definitions for later orchestration.
type Store interface {
	Save(wf *Workflow) (*Workflow, error)
	Get(id string) (*Workflow, error)
	List() ([]TrimmedWorkflow, error)
	History(id string, limit int) ([]HistoryEntry, error)
}

// FileStore writes workflow definitions to disk as JSON documents.
type FileStore struct {
	mu  sync.RWMutex
	dir string
}

// NewFileStore creates a store rooted at dir.
func NewFileStore(dir string) (*FileStore, error) {
	if dir == "" {
		return nil, errors.New("dir is required")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("ensure workflow dir: %w", err)
	}
	return &FileStore{dir: dir}, nil
}

// Save upserts a workflow definition, returning a deep copy with timestamps populated.
func (s *FileStore) Save(in *Workflow) (*Workflow, error) {
	if in == nil {
		return nil, errors.New("workflow is nil")
	}

	wf := in.Copy()
	if wf.ID == "" {
		wf.ID = generateWorkflowID()
	}
	now := time.Now().UTC()
	if wf.CreatedAt.IsZero() {
		wf.CreatedAt = now
	}
	wf.UpdatedAt = now

	if err := ValidateWorkflow(&wf); err != nil {
		return nil, err
	}

	data, err := json.MarshalIndent(wf, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("serialise workflow: %w", err)
	}

	tmp := filepath.Join(s.dir, wf.ID+".tmp")
	dest := filepath.Join(s.dir, wf.ID+".json")

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, err := os.Stat(dest); err == nil {
		current, readErr := os.ReadFile(dest)
		if readErr == nil {
			historyDir := filepath.Join(s.dir, wf.ID, "history")
			if err := os.MkdirAll(historyDir, 0o755); err != nil {
				return nil, fmt.Errorf("create history dir: %w", err)
			}
			stamp := wf.UpdatedAt.Format(time.RFC3339Nano)
			historyFile := filepath.Join(historyDir, stamp+".json")
			if writeErr := os.WriteFile(historyFile, current, 0o644); writeErr != nil {
				return nil, fmt.Errorf("write history: %w", writeErr)
			}
		}
	}

	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return nil, fmt.Errorf("write tmp workflow: %w", err)
	}
	if err := os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return nil, fmt.Errorf("commit workflow: %w", err)
	}

	stored := wf.Copy()
	return &stored, nil
}

func generateWorkflowID() string {
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err == nil {
		return "wf-" + hex.EncodeToString(buf)
	}
	return fmt.Sprintf("wf-%d", time.Now().UnixNano())
}

// HistoryEntry describes a stored workflow revision.
type HistoryEntry struct {
	ID        string    `json:"id"`
	UpdatedAt time.Time `json:"updated_at"`
	Path      string    `json:"path"`
}

// Get loads a workflow by identifier.
func (s *FileStore) Get(id string) (*Workflow, error) {
	if strings.TrimSpace(id) == "" {
		return nil, errors.New("id is required")
	}
	path := filepath.Join(s.dir, id+".json")

	s.mu.RLock()
	data, err := os.ReadFile(path)
	s.mu.RUnlock()
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("workflow %s not found", id)
		}
		return nil, fmt.Errorf("read workflow: %w", err)
	}

	var wf Workflow
	if err := json.Unmarshal(data, &wf); err != nil {
		return nil, fmt.Errorf("decode workflow: %w", err)
	}
	return &wf, nil
}

// List returns workflow summaries sorted by most recent update.
func (s *FileStore) List() ([]TrimmedWorkflow, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, fmt.Errorf("list workflows: %w", err)
	}

	summaries := make([]TrimmedWorkflow, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		id := strings.TrimSuffix(entry.Name(), ".json")
		wf, err := s.Get(id)
		if err != nil {
			return nil, err
		}
		summaries = append(summaries, wf.ToSummary())
	}

	sort.SliceStable(summaries, func(i, j int) bool {
		return summaries[i].UpdatedAt.After(summaries[j].UpdatedAt)
	})

	return summaries, nil
}

// History returns workflow revisions sorted newest-first. If limit <= 0 all entries are returned.
func (s *FileStore) History(id string, limit int) ([]HistoryEntry, error) {
	if strings.TrimSpace(id) == "" {
		return nil, errors.New("id is required")
	}
	historyDir := filepath.Join(s.dir, id, "history")
	entries, err := os.ReadDir(historyDir)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return []HistoryEntry{}, nil
		}
		return nil, fmt.Errorf("read history dir: %w", err)
	}
	revisions := make([]HistoryEntry, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		stamp := strings.TrimSuffix(entry.Name(), ".json")
		ts, parseErr := time.Parse(time.RFC3339Nano, stamp)
		if parseErr != nil {
			continue
		}
		revisions = append(revisions, HistoryEntry{
			ID:        id,
			UpdatedAt: ts,
			Path:      filepath.Join(historyDir, entry.Name()),
		})
	}
	sort.SliceStable(revisions, func(i, j int) bool {
		return revisions[i].UpdatedAt.After(revisions[j].UpdatedAt)
	})
	if limit > 0 && len(revisions) > limit {
		revisions = revisions[:limit]
	}
	return revisions, nil
}
