package workflow

import (
	"github.com/dspy/orchestrator/internal/workflowspec"
)

// BuildWorkflowContext produces the runtime context payload injected into downstream tasks.
func BuildWorkflowContext(wf *workflowspec.Workflow) map[string]any {
	if wf == nil {
		return nil
	}
	ctx := map[string]any{
		"id":      wf.ID,
		"name":    wf.Name,
		"version": wf.Version,
		"tenant":  wf.Tenant,
		"tags":    append([]string(nil), wf.Tags...),
		"nodes":   wf.Nodes,
		"edges":   wf.Edges,
	}
	index := make(map[string][]string)
	for _, node := range wf.Nodes {
		index[string(node.Type)] = append(index[string(node.Type)], node.ID)
	}
	ctx["node_index"] = index
	return ctx
}
