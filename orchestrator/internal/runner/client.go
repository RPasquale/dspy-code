package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// TaskRequest models the payload sent to the Rust environment runner.
type TaskRequest struct {
	ID      string                 `json:"id"`
	Class   string                 `json:"class"`
	Payload map[string]interface{} `json:"payload"`
}

// TaskResponse captures the execution result returned by the Rust runner.
type TaskResponse struct {
	ID         string         `json:"id"`
	Embeddings [][]float64    `json:"embeddings"`
	LatencyMs  float64        `json:"latency_ms"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

// Metrics mirrors the Rust runner metrics endpoint.
type Metrics struct {
	TasksProcessed uint64            `json:"tasks_processed"`
	QueueDepth     uint64            `json:"queue_depth"`
	GPUUtilization float64           `json:"gpu_utilization"`
	LatencyP95Ms   uint64            `json:"latency_p95_ms"`
	AvgDurationMs  uint64            `json:"avg_duration_ms"`
	TotalErrors    uint64            `json:"total_errors"`
	ErrorsByClass  map[string]uint64 `json:"errors_by_class"`
	UptimeSeconds  uint64            `json:"uptime_seconds"`
}

// HTTPClient wraps communication with the Rust environment runner.
type HTTPClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewHTTPClient creates an HTTP-based runner client. If baseURL is empty, it defaults to http://localhost:8083.
func NewHTTPClient(baseURL string) *HTTPClient {
	clean := strings.TrimSpace(baseURL)
	if clean == "" {
		clean = "http://localhost:8083"
	}
	return &HTTPClient{
		baseURL:    clean,
		httpClient: &http.Client{Timeout: 45 * time.Second},
	}
}

// ExecuteTask forwards a task to the Rust runner and returns the execution response.
func (c *HTTPClient) ExecuteTask(ctx context.Context, req TaskRequest) (*TaskResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("runner: marshal task: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/tasks/execute", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("runner: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("runner: execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var errPayload map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&errPayload)
		return nil, fmt.Errorf("runner: unexpected status %d: %v", resp.StatusCode, errPayload)
	}

	var taskResp TaskResponse
	if err := json.NewDecoder(resp.Body).Decode(&taskResp); err != nil {
		return nil, fmt.Errorf("runner: decode response: %w", err)
	}

	return &taskResp, nil
}

// FetchMetrics retrieves the latest runner metrics and returns them for dashboard ingestion.
func (c *HTTPClient) FetchMetrics(ctx context.Context) (*Metrics, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/metrics", nil)
	if err != nil {
		return nil, fmt.Errorf("runner: build metrics request: %w", err)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("runner: fetch metrics: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("runner: metrics status %d", resp.StatusCode)
	}

	var metrics Metrics
	if err := json.NewDecoder(resp.Body).Decode(&metrics); err != nil {
		return nil, fmt.Errorf("runner: decode metrics: %w", err)
	}

	return &metrics, nil
}

// BaseURL exposes the configured base URL.
func (c *HTTPClient) BaseURL() string {
	return c.baseURL
}
