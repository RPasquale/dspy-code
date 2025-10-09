package workflowspec

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFileStore_SaveAndGet(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	wf := &Workflow{
		Name: "Go Build Loop",
		Nodes: []Node{
			{
				ID:        "sig",
				Name:      "Signature",
				Type:      NodeSignature,
				Signature: &SignatureConfig{Prompt: "compile"},
			},
			{
				ID:         "deploy",
				Name:       "Deploy",
				Type:       NodeDeployment,
				Deployment: &DeploymentConfig{Tenant: "prod", Domain: "go", Channel: "build"},
			},
		},
		Edges: []Edge{
			{ID: "flow", Source: "sig", Target: "deploy", Kind: EdgeKindControl},
		},
	}

	saved, err := store.Save(wf)
	if err != nil {
		t.Fatalf("save workflow: %v", err)
	}
	if saved.ID == "" {
		t.Fatal("expected ID to be assigned")
	}

	loaded, err := store.Get(saved.ID)
	if err != nil {
		t.Fatalf("get workflow: %v", err)
	}
	if loaded.Name != wf.Name {
		t.Fatalf("unexpected name: %s", loaded.Name)
	}

	summaries, err := store.List()
	if err != nil {
		t.Fatalf("list workflows: %v", err)
	}
	if len(summaries) != 1 {
		t.Fatalf("expected 1 summary, got %d", len(summaries))
	}
	if summaries[0].ID != saved.ID {
		t.Fatalf("summary id mismatch: %s", summaries[0].ID)
	}

	revisions, err := store.History(saved.ID, 0)
	if err != nil {
		t.Fatalf("history: %v", err)
	}
	if len(revisions) != 0 {
		t.Fatalf("expected no revisions yet, got %d", len(revisions))
	}
}

func TestFileStore_InvalidWorkflow(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	_, err = store.Save(&Workflow{Name: "bad"})
	if err == nil {
		t.Fatal("expected validation error for invalid workflow")
	}
}

func TestFileStore_History(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	wf := &Workflow{
		Name:  "Workflow",
		Nodes: []Node{{ID: "n", Name: "N", Type: NodeCustom, Custom: CustomConfig{"k": true}}},
		Edges: []Edge{{ID: "e", Source: "n", Target: "n", Kind: EdgeKindControl}},
	}
	saved, err := store.Save(wf)
	if err != nil {
		t.Fatalf("save: %v", err)
	}
	// second save should archive first revision
	saved.Description = "updated"
	updated, err := store.Save(saved)
	if err != nil {
		t.Fatalf("resave: %v", err)
	}
	revisions, err := store.History(saved.ID, 0)
	if err != nil {
		t.Fatalf("history: %v", err)
	}
	if len(revisions) != 1 {
		t.Fatalf("expected 1 revision, got %d", len(revisions))
	}
	if !revisions[0].UpdatedAt.Before(updated.UpdatedAt) && !revisions[0].UpdatedAt.Equal(updated.UpdatedAt) {
		t.Fatalf("history timestamp unexpected: %v vs %v", revisions[0].UpdatedAt, updated.UpdatedAt)
	}
}

func TestFileStore_RejectsMissingDir(t *testing.T) {
	_, err := NewFileStore("")
	if err == nil {
		t.Fatal("expected error for empty dir")
	}
}

func TestFileStore_ListSkipsNonJSON(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("ignore"), 0o644); err != nil {
		t.Fatalf("write stray file: %v", err)
	}

	wf := &Workflow{
		Name: "Minimal",
		Nodes: []Node{{
			ID:     "custom",
			Name:   "C",
			Type:   NodeCustom,
			Custom: CustomConfig{"k": "v"},
		}},
		Edges: []Edge{{ID: "edge", Source: "custom", Target: "custom", Kind: EdgeKindControl}},
	}
	if _, err := store.Save(wf); err != nil {
		t.Fatalf("save workflow: %v", err)
	}

	summaries, err := store.List()
	if err != nil {
		t.Fatalf("list workflows: %v", err)
	}
	if len(summaries) != 1 {
		t.Fatalf("expected 1 summary, got %d", len(summaries))
	}
}
