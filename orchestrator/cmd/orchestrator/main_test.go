package main

import (
	"errors"
	"testing"

	"github.com/dspy/orchestrator/internal/workflowspec"
)

type stubWorkflowStore struct {
	wf     *workflowspec.Workflow
	getErr error
}

func (s stubWorkflowStore) Save(wf *workflowspec.Workflow) (*workflowspec.Workflow, error) {
	return nil, errors.New("not implemented")
}

func (s stubWorkflowStore) Get(id string) (*workflowspec.Workflow, error) {
	if s.getErr != nil {
		return nil, s.getErr
	}
	return s.wf, nil
}

func (s stubWorkflowStore) List() ([]workflowspec.TrimmedWorkflow, error) {
	return nil, nil
}

func (s stubWorkflowStore) History(string, int) ([]workflowspec.HistoryEntry, error) {
	return nil, nil
}

func TestAttachWorkflowContext(t *testing.T) {
	wf := &workflowspec.Workflow{
		ID:   "wf-1",
		Name: "Go Build Curriculum",
		Nodes: []workflowspec.Node{
			{
				ID:        "sig",
				Name:      "Signature",
				Type:      workflowspec.NodeSignature,
				Signature: &workflowspec.SignatureConfig{Prompt: "fn main() {}"},
			},
		},
		Edges: []workflowspec.Edge{{ID: "edge", Source: "sig", Target: "sig", Kind: workflowspec.EdgeKindData}},
	}
	payload := map[string]interface{}{"workflow_id": "wf-1"}

	attachWorkflowContext(payload, stubWorkflowStore{wf: wf})

	ctxVal, ok := payload["workflow_context"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected workflow_context to be injected, payload=%v", payload)
	}
	if ctxVal["id"] != "wf-1" {
		t.Fatalf("unexpected workflow id: %v", ctxVal["id"])
	}
	if _, hasIndex := ctxVal["node_index"].(map[string][]string); !hasIndex {
		t.Fatalf("expected node_index to be present: %v", ctxVal)
	}
	if payload["tenant"] != nil {
		t.Fatalf("tenant should be unset when workflow has no tenant")
	}
}

func TestAttachWorkflowContextMissing(t *testing.T) {
	payload := map[string]interface{}{}
	attachWorkflowContext(payload, stubWorkflowStore{})
	if _, ok := payload["workflow_context"]; ok {
		t.Fatalf("expected no workflow context when id missing")
	}
}
