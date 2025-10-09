package workflowspec

import (
	"testing"
	"time"
)

func TestValidateWorkflow_Success(t *testing.T) {
	wf := &Workflow{
		Name: "Go Agent Curriculum",
		Nodes: []Node{
			{
				ID:   "sig1",
				Name: "Signature",
				Type: NodeSignature,
				Signature: &SignatureConfig{
					Prompt: "fn main() {}",
				},
			},
			{
				ID:   "ver1",
				Name: "Verifier",
				Type: NodeVerifier,
				Verifier: &VerifierConfig{
					Command: "go test ./...",
					Weight:  1.0,
				},
			},
		},
		Edges: []Edge{
			{ID: "e1", Source: "sig1", Target: "ver1", Kind: EdgeKindData},
		},
	}

	if err := ValidateWorkflow(wf); err != nil {
		t.Fatalf("expected valid workflow, got %v", err)
	}
}

func TestValidateWorkflow_FailsWithoutNodes(t *testing.T) {
	wf := &Workflow{Name: "Empty"}
	if err := ValidateWorkflow(wf); err == nil {
		t.Fatal("expected error when workflow has no nodes")
	}
}

func TestValidateWorkflow_DuplicateNode(t *testing.T) {
	wf := &Workflow{
		Name: "Invalid",
		Nodes: []Node{
			{ID: "dup", Name: "n1", Type: NodeCustom, Custom: CustomConfig{"foo": true}},
			{ID: "dup", Name: "n2", Type: NodeCustom, Custom: CustomConfig{"foo": false}},
		},
		Edges: []Edge{{ID: "e", Source: "dup", Target: "dup", Kind: EdgeKindControl}},
	}
	if err := ValidateWorkflow(wf); err == nil {
		t.Fatal("expected duplicate node validation error")
	}
}

func TestWorkflowSummary(t *testing.T) {
	now := time.Now().UTC()
	wf := &Workflow{
		ID:        "wf-1",
		Name:      "Workflow",
		UpdatedAt: now,
		Tags:      []string{"beta", "alpha"},
		Nodes:     []Node{{ID: "a", Name: "A", Type: NodeCustom, Custom: CustomConfig{"k": 1}}},
		Edges:     []Edge{{ID: "e", Source: "a", Target: "a", Kind: EdgeKindControl}},
	}
	if err := ValidateWorkflow(wf); err != nil {
		t.Fatalf("expected workflow to validate: %v", err)
	}
	summary := wf.ToSummary()
	if summary.NodeCount != 1 || summary.EdgeCount != 1 {
		t.Fatalf("unexpected summary counts: %+v", summary)
	}
	if summary.Tags[0] != "alpha" {
		t.Fatalf("expected tags to be sorted, got %v", summary.Tags)
	}
}
