package workflow

import (
	"fmt"
	"sort"

	"github.com/dspy/orchestrator/internal/workflowspec"
)

// ActionType enumerates the possible execution actions produced from a workflow node.
type ActionType string

const (
	ActionSignature  ActionType = "signature"
	ActionVerifier   ActionType = "verifier"
	ActionReward     ActionType = "reward"
	ActionTraining   ActionType = "training"
	ActionDeployment ActionType = "deployment"
	ActionCustom     ActionType = "custom"
)

// StepAction captures the concrete execution parameters for a plan step.
type StepAction struct {
	Type       ActionType                     `json:"type"`
	Signature  *workflowspec.SignatureConfig  `json:"signature,omitempty"`
	Verifier   *workflowspec.VerifierConfig   `json:"verifier,omitempty"`
	Reward     *workflowspec.RewardConfig     `json:"reward,omitempty"`
	Training   *workflowspec.TrainingConfig   `json:"training,omitempty"`
	Deployment *workflowspec.DeploymentConfig `json:"deployment,omitempty"`
	Custom     workflowspec.CustomConfig      `json:"custom,omitempty"`
}

// PlanStep represents a single actionable unit derived from a workflow node.
type PlanStep struct {
	NodeID string                `json:"node_id"`
	Name   string                `json:"name"`
	Type   workflowspec.NodeType `json:"type"`
	Action StepAction            `json:"action"`
	After  []string              `json:"after"`
}

// Plan defines the runnable plan compiled from a workflow graph.
type Plan struct {
	WorkflowID string     `json:"workflow_id"`
	Steps      []PlanStep `json:"steps"`
	nodeIndex  map[string]PlanStep
}

// BuildPlan compiles a workflow graph into a deterministic runnable plan.
func BuildPlan(wf *workflowspec.Workflow) (*Plan, error) {
	if wf == nil {
		return nil, fmt.Errorf("workflow is nil")
	}

	adjacency := make(map[string][]string, len(wf.Nodes))
	reverse := make(map[string][]string, len(wf.Nodes))
	indegree := make(map[string]int, len(wf.Nodes))
	nodes := make(map[string]workflowspec.Node, len(wf.Nodes))

	for _, node := range wf.Nodes {
		nodeCopy := node
		adjacency[node.ID] = nil
		reverse[node.ID] = nil
		indegree[node.ID] = 0
		nodes[node.ID] = nodeCopy
	}

	for _, edge := range wf.Edges {
		adjacency[edge.Source] = append(adjacency[edge.Source], edge.Target)
		reverse[edge.Target] = append(reverse[edge.Target], edge.Source)
		indegree[edge.Target]++
	}

	queue := make([]string, 0, len(nodes))
	for id, deg := range indegree {
		if deg == 0 {
			queue = append(queue, id)
		}
	}
	sort.Strings(queue)

	order := make([]string, 0, len(nodes))
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		order = append(order, id)

		successors := adjacency[id]
		sort.Strings(successors)
		for _, succ := range successors {
			indegree[succ]--
			if indegree[succ] == 0 {
				insertSorted(&queue, succ)
			}
		}
	}

	if len(order) != len(nodes) {
		return nil, fmt.Errorf("workflow contains cycles; unable to build plan")
	}

	steps := make([]PlanStep, 0, len(order))
	index := make(map[string]PlanStep, len(order))
	for _, nodeID := range order {
		node := nodes[nodeID]
		action, err := buildActionForNode(node)
		if err != nil {
			return nil, err
		}
		parents := append([]string(nil), reverse[nodeID]...)
		sort.Strings(parents)
		step := PlanStep{
			NodeID: node.ID,
			Name:   node.Name,
			Type:   node.Type,
			Action: action,
			After:  parents,
		}
		steps = append(steps, step)
		index[step.NodeID] = step
	}

	return &Plan{WorkflowID: wf.ID, Steps: steps, nodeIndex: index}, nil
}

func insertSorted(queue *[]string, value string) {
	q := *queue
	q = append(q, value)
	sort.Strings(q)
	*queue = q
}

func buildActionForNode(node workflowspec.Node) (StepAction, error) {
	switch node.Type {
	case workflowspec.NodeSignature:
		if node.Signature == nil {
			return StepAction{}, fmt.Errorf("node %s missing signature config", node.ID)
		}
		sigCopy := *node.Signature
		sigCopy.Tools = append([]string(nil), node.Signature.Tools...)
		return StepAction{Type: ActionSignature, Signature: &sigCopy}, nil
	case workflowspec.NodeVerifier:
		if node.Verifier == nil {
			return StepAction{}, fmt.Errorf("node %s missing verifier config", node.ID)
		}
		verCopy := *node.Verifier
		return StepAction{Type: ActionVerifier, Verifier: &verCopy}, nil
	case workflowspec.NodeReward:
		if node.Reward == nil {
			return StepAction{}, fmt.Errorf("node %s missing reward config", node.ID)
		}
		rewCopy := *node.Reward
		return StepAction{Type: ActionReward, Reward: &rewCopy}, nil
	case workflowspec.NodeTraining:
		if node.Training == nil {
			return StepAction{}, fmt.Errorf("node %s missing training config", node.ID)
		}
		trainCopy := *node.Training
		return StepAction{Type: ActionTraining, Training: &trainCopy}, nil
	case workflowspec.NodeDeployment:
		if node.Deployment == nil {
			return StepAction{}, fmt.Errorf("node %s missing deployment config", node.ID)
		}
		depCopy := *node.Deployment
		return StepAction{Type: ActionDeployment, Deployment: &depCopy}, nil
	case workflowspec.NodeCustom:
		cfg := workflowspec.CustomConfig{}
		for k, v := range node.Custom {
			cfg[k] = v
		}
		return StepAction{Type: ActionCustom, Custom: cfg}, nil
	default:
		return StepAction{}, fmt.Errorf("unsupported node type %s", node.Type)
	}
}

// StepByID looks up a step by node identifier.
func (p *Plan) StepByID(nodeID string) (PlanStep, bool) {
	if p == nil {
		return PlanStep{}, false
	}
	step, ok := p.nodeIndex[nodeID]
	return step, ok
}
