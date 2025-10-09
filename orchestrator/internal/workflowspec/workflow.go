package workflowspec

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// NodeType represents the type of a workflow node exposed to the UI composer.
type NodeType string

const (
	NodeSignature  NodeType = "signature"
	NodeVerifier   NodeType = "verifier"
	NodeReward     NodeType = "reward"
	NodeTraining   NodeType = "training"
	NodeDeployment NodeType = "deployment"
	NodeCustom     NodeType = "custom"
)

// EdgeKind describes how data flows between two nodes.
type EdgeKind string

const (
	EdgeKindData    EdgeKind = "data"
	EdgeKindControl EdgeKind = "control"
	EdgeKindReward  EdgeKind = "reward"
)

// Position contains layout coordinates from the frontend graph builder.
type Position struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// SignatureConfig captures settings for a signature/program node.
type SignatureConfig struct {
	Prompt          string   `json:"prompt"`
	Tools           []string `json:"tools,omitempty"`
	Runtime         string   `json:"runtime,omitempty"`
	MaxTokens       int      `json:"max_tokens,omitempty"`
	Temperature     float64  `json:"temperature,omitempty"`
	ExecutionDomain string   `json:"execution_domain,omitempty"`
}

// VerifierConfig represents verifier execution metadata.
type VerifierConfig struct {
	Command       string  `json:"command"`
	Weight        float64 `json:"weight"`
	Penalty       float64 `json:"penalty,omitempty"`
	TimeoutSec    int     `json:"timeout_sec,omitempty"`
	SuccessSignal string  `json:"success_signal,omitempty"`
}

// RewardConfig stores reward calculation settings.
type RewardConfig struct {
	Metric     string  `json:"metric"`
	Target     float64 `json:"target,omitempty"`
	Aggregator string  `json:"aggregator,omitempty"`
	Window     int     `json:"window,omitempty"`
	Scale      float64 `json:"scale,omitempty"`
}

// TrainingConfig captures RL/GEPA scheduling metadata.
type TrainingConfig struct {
	Method        string `json:"method"`
	Skill         string `json:"skill"`
	Dataset       string `json:"dataset,omitempty"`
	Schedule      string `json:"schedule,omitempty"`
	MaxSteps      int    `json:"max_steps,omitempty"`
	SlurmProfile  string `json:"slurm_profile,omitempty"`
	TriggerMetric string `json:"trigger_metric,omitempty"`
}

// DeploymentConfig configures inference/serving behaviour.
type DeploymentConfig struct {
	Tenant      string `json:"tenant"`
	Domain      string `json:"domain"`
	Channel     string `json:"channel"`
	Strategy    string `json:"strategy,omitempty"`
	Autopromote bool   `json:"autopromote,omitempty"`
}

// CustomConfig allows arbitrary configuration when the UI expands beyond the core types.
type CustomConfig map[string]any

// Node defines a node within the workflow graph.
type Node struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Type        NodeType          `json:"type"`
	Description string            `json:"description,omitempty"`
	Position    *Position         `json:"position,omitempty"`
	Signature   *SignatureConfig  `json:"signature,omitempty"`
	Verifier    *VerifierConfig   `json:"verifier,omitempty"`
	Reward      *RewardConfig     `json:"reward,omitempty"`
	Training    *TrainingConfig   `json:"training,omitempty"`
	Deployment  *DeploymentConfig `json:"deployment,omitempty"`
	Custom      CustomConfig      `json:"custom,omitempty"`
}

// Edge links two nodes together.
type Edge struct {
	ID     string   `json:"id"`
	Source string   `json:"source"`
	Target string   `json:"target"`
	Kind   EdgeKind `json:"kind"`
	Label  string   `json:"label,omitempty"`
}

// Workflow represents an entire workflow graph that can be materialised by the orchestrator.
type Workflow struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Description string    `json:"description,omitempty"`
	Tags        []string  `json:"tags,omitempty"`
	Tenant      string    `json:"tenant,omitempty"`
	Version     string    `json:"version,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	Nodes       []Node    `json:"nodes"`
	Edges       []Edge    `json:"edges"`
}

// TrimmedWorkflow is a projection used for listing workflows in the UI.
type TrimmedWorkflow struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Description string    `json:"description,omitempty"`
	Tags        []string  `json:"tags,omitempty"`
	Tenant      string    `json:"tenant,omitempty"`
	Version     string    `json:"version,omitempty"`
	UpdatedAt   time.Time `json:"updated_at"`
	NodeCount   int       `json:"node_count"`
	EdgeCount   int       `json:"edge_count"`
}

// ValidateWorkflow checks structural correctness and returns a descriptive error if invalid.
func ValidateWorkflow(wf *Workflow) error {
	if wf == nil {
		return errors.New("workflow is nil")
	}
	if strings.TrimSpace(wf.Name) == "" {
		return errors.New("workflow name is required")
	}

	nodeIDs := make(map[string]Node)
	for idx, node := range wf.Nodes {
		if strings.TrimSpace(node.ID) == "" {
			return fmt.Errorf("node[%d] missing id", idx)
		}
		if _, exists := nodeIDs[node.ID]; exists {
			return fmt.Errorf("duplicate node id %q", node.ID)
		}
		if strings.TrimSpace(node.Name) == "" {
			return fmt.Errorf("node %q missing name", node.ID)
		}
		if err := validateNode(&node); err != nil {
			return fmt.Errorf("node %q invalid: %w", node.ID, err)
		}
		nodeIDs[node.ID] = node
	}
	if len(nodeIDs) == 0 {
		return errors.New("workflow requires at least one node")
	}

	if len(wf.Edges) == 0 {
		return errors.New("workflow requires at least one edge to express flow")
	}

	seenEdgeIDs := make(map[string]struct{})
	for idx, edge := range wf.Edges {
		if strings.TrimSpace(edge.ID) == "" {
			return fmt.Errorf("edge[%d] missing id", idx)
		}
		if _, exists := seenEdgeIDs[edge.ID]; exists {
			return fmt.Errorf("duplicate edge id %q", edge.ID)
		}
		if strings.TrimSpace(edge.Source) == "" || strings.TrimSpace(edge.Target) == "" {
			return fmt.Errorf("edge %q requires both source and target", edge.ID)
		}
		if _, ok := nodeIDs[edge.Source]; !ok {
			return fmt.Errorf("edge %q references unknown source %q", edge.ID, edge.Source)
		}
		if _, ok := nodeIDs[edge.Target]; !ok {
			return fmt.Errorf("edge %q references unknown target %q", edge.ID, edge.Target)
		}
		if edge.Kind == "" {
			return fmt.Errorf("edge %q missing kind", edge.ID)
		}
		seenEdgeIDs[edge.ID] = struct{}{}
	}

	// Normalise tags for deterministic rendering.
	sort.Strings(wf.Tags)

	return nil
}

func validateNode(node *Node) error {
	switch node.Type {
	case NodeSignature:
		if node.Signature == nil {
			return errors.New("signature node requires signature config")
		}
		if strings.TrimSpace(node.Signature.Prompt) == "" {
			return errors.New("signature prompt is required")
		}
	case NodeVerifier:
		if node.Verifier == nil {
			return errors.New("verifier node requires verifier config")
		}
		if strings.TrimSpace(node.Verifier.Command) == "" {
			return errors.New("verifier command is required")
		}
		if node.Verifier.Weight < 0 {
			return errors.New("verifier weight must be non-negative")
		}
	case NodeReward:
		if node.Reward == nil {
			return errors.New("reward node requires reward config")
		}
		if strings.TrimSpace(node.Reward.Metric) == "" {
			return errors.New("reward metric is required")
		}
	case NodeTraining:
		if node.Training == nil {
			return errors.New("training node requires training config")
		}
		if strings.TrimSpace(node.Training.Method) == "" {
			return errors.New("training method is required")
		}
		if strings.TrimSpace(node.Training.Skill) == "" {
			return errors.New("training skill is required")
		}
	case NodeDeployment:
		if node.Deployment == nil {
			return errors.New("deployment node requires deployment config")
		}
		if strings.TrimSpace(node.Deployment.Tenant) == "" {
			return errors.New("deployment tenant is required")
		}
		if strings.TrimSpace(node.Deployment.Domain) == "" {
			return errors.New("deployment domain is required")
		}
		if strings.TrimSpace(node.Deployment.Channel) == "" {
			return errors.New("deployment channel is required")
		}
	case NodeCustom:
		if len(node.Custom) == 0 {
			return errors.New("custom node requires configuration")
		}
	default:
		return fmt.Errorf("unknown node type %q", node.Type)
	}
	return nil
}

// ToSummary creates a lightweight summary representation for listings.
func (wf *Workflow) ToSummary() TrimmedWorkflow {
	return TrimmedWorkflow{
		ID:          wf.ID,
		Name:        wf.Name,
		Description: wf.Description,
		Tags:        append([]string(nil), wf.Tags...),
		Tenant:      wf.Tenant,
		Version:     wf.Version,
		UpdatedAt:   wf.UpdatedAt,
		NodeCount:   len(wf.Nodes),
		EdgeCount:   len(wf.Edges),
	}
}

// Copy produces a deep copy of the workflow for safe use by callers.
func (wf *Workflow) Copy() Workflow {
	clone := *wf
	if wf.Nodes != nil {
		clone.Nodes = make([]Node, len(wf.Nodes))
		copy(clone.Nodes, wf.Nodes)
	}
	if wf.Edges != nil {
		clone.Edges = make([]Edge, len(wf.Edges))
		copy(clone.Edges, wf.Edges)
	}
	if wf.Tags != nil {
		clone.Tags = append([]string(nil), wf.Tags...)
	}
	return clone
}
