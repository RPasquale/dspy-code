package infermesh

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// InferMeshClient provides high-performance, concurrent access to InferMesh embedding service
type InferMeshClient struct {
	baseURL        string
	apiKey         string
	model          string
	httpClient     *http.Client
	connectionPool *ConnectionPool
	batchProcessor *BatchProcessor
	metrics        *ClientMetrics
	mu             sync.RWMutex
}

// ConnectionPool manages HTTP connections for optimal performance
type ConnectionPool struct {
	client      *http.Client
	maxConns    int
	maxIdle     int
	idleTimeout time.Duration
}

// BatchProcessor handles intelligent batching of embedding requests
type BatchProcessor struct {
	batchSize      int
	maxWaitTime    time.Duration
	pendingBatches map[string]*PendingBatch
	mu             sync.RWMutex
}

// PendingBatch represents a batch of texts waiting to be processed
type PendingBatch struct {
	Texts     []string
	Results   chan BatchResult
	CreatedAt time.Time
}

// BatchResult contains the results of a batch processing operation
type BatchResult struct {
	Embeddings [][]float32
	Error      error
}

// ClientMetrics tracks performance metrics for the InferMesh client
type ClientMetrics struct {
	TotalRequests      int64
	SuccessfulRequests int64
	FailedRequests     int64
	AverageLatency     time.Duration
	LastRequestTime    time.Time
	ConcurrentRequests int64
}

// EmbedRequest represents a request to embed texts
type EmbedRequest struct {
	Model  string   `json:"model"`
	Inputs []string `json:"inputs"`
}

// EmbedResponse represents the response from InferMesh
type EmbedResponse struct {
	Vectors    [][]float32 `json:"vectors,omitempty"`
	Embeddings [][]float32 `json:"embeddings,omitempty"`
}

// NewInferMeshClient creates a new high-performance InferMesh client
func NewInferMeshClient(baseURL, apiKey, model string) *InferMeshClient {
	// Configure HTTP client with connection pooling
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 50,
		IdleConnTimeout:     90 * time.Second,
		DisableKeepAlives:   false,
	}

	httpClient := &http.Client{
		Transport: transport,
		Timeout:   30 * time.Second,
	}

	connectionPool := &ConnectionPool{
		client:      httpClient,
		maxConns:    100,
		maxIdle:     50,
		idleTimeout: 90 * time.Second,
	}

	batchProcessor := &BatchProcessor{
		batchSize:      512, // Larger default batch size
		maxWaitTime:    100 * time.Millisecond,
		pendingBatches: make(map[string]*PendingBatch),
	}

	return &InferMeshClient{
		baseURL:        baseURL,
		apiKey:         apiKey,
		model:          model,
		httpClient:     httpClient,
		connectionPool: connectionPool,
		batchProcessor: batchProcessor,
		metrics:        &ClientMetrics{},
	}
}

// Embed processes a single embedding request with intelligent batching
func (c *InferMeshClient) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	// Use batch processing for optimal performance
	return c.batchProcessor.ProcessBatch(ctx, texts, c)
}

// EmbedBatch processes a large batch of texts with optimized HTTP requests
func (c *InferMeshClient) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	// Update metrics
	c.mu.Lock()
	c.metrics.TotalRequests++
	c.metrics.ConcurrentRequests++
	c.metrics.LastRequestTime = time.Now()
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		c.metrics.ConcurrentRequests--
		c.mu.Unlock()
	}()

	// Prepare request
	request := EmbedRequest{
		Model:  c.model,
		Inputs: texts,
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		c.mu.Lock()
		c.metrics.FailedRequests++
		c.mu.Unlock()
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Execute request with retries
	startTime := time.Now()
	var response *http.Response

	for attempt := 0; attempt < 3; attempt++ {
		req, newReqErr := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/embed", bytes.NewReader(requestBody))
		if newReqErr != nil {
			err = newReqErr
			break
		}

		req.Header.Set("Content-Type", "application/json")
		if c.apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+c.apiKey)
		}

		response, err = c.httpClient.Do(req)
		if err == nil {
			break
		}

		if attempt < 2 {
			time.Sleep(time.Duration(attempt+1) * 100 * time.Millisecond)
		}
	}

	if err != nil {
		c.mu.Lock()
		c.metrics.FailedRequests++
		c.mu.Unlock()
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer response.Body.Close()

	// Update latency metrics
	latency := time.Since(startTime)
	c.mu.Lock()
	c.metrics.AverageLatency = (c.metrics.AverageLatency + latency) / 2
	c.mu.Unlock()

	// Check response status
	if response.StatusCode != http.StatusOK {
		c.mu.Lock()
		c.metrics.FailedRequests++
		c.mu.Unlock()
		return nil, fmt.Errorf("infermesh returned status %d", response.StatusCode)
	}

	// Parse response
	var embedResponse EmbedResponse
	if err := json.NewDecoder(response.Body).Decode(&embedResponse); err != nil {
		c.mu.Lock()
		c.metrics.FailedRequests++
		c.mu.Unlock()
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Extract embeddings (support both 'vectors' and 'embeddings' fields)
	embeddings := embedResponse.Vectors
	if len(embeddings) == 0 {
		embeddings = embedResponse.Embeddings
	}

	// Validate response length
	if len(embeddings) != len(texts) {
		c.mu.Lock()
		c.metrics.FailedRequests++
		c.mu.Unlock()
		return nil, fmt.Errorf("mismatch between input texts (%d) and embeddings (%d)",
			len(texts), len(embeddings))
	}

	// Update success metrics
	c.mu.Lock()
	c.metrics.SuccessfulRequests++
	c.mu.Unlock()

	return embeddings, nil
}

// ProcessBatch handles intelligent batching of requests
func (bp *BatchProcessor) ProcessBatch(ctx context.Context, texts []string, client *InferMeshClient) ([][]float32, error) {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	// Create batch key for grouping similar requests
	batchKey := fmt.Sprintf("batch_%d", time.Now().UnixNano()/int64(bp.maxWaitTime))

	// Create pending batch
	pendingBatch := &PendingBatch{
		Texts:     texts,
		Results:   make(chan BatchResult, 1),
		CreatedAt: time.Now(),
	}

	bp.pendingBatches[batchKey] = pendingBatch

	// Start batch processing goroutine
	go bp.processBatchAsync(ctx, batchKey, client)

	// Wait for results
	select {
	case result := <-pendingBatch.Results:
		return result.Embeddings, result.Error
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// processBatchAsync processes a batch asynchronously
func (bp *BatchProcessor) processBatchAsync(ctx context.Context, batchKey string, client *InferMeshClient) {
	bp.mu.RLock()
	pendingBatch, exists := bp.pendingBatches[batchKey]
	bp.mu.RUnlock()

	if !exists {
		return
	}

	// Wait for batch to fill up or timeout
	time.Sleep(bp.maxWaitTime)

	// Process the batch
	embeddings, err := client.EmbedBatch(ctx, pendingBatch.Texts)

	// Send results
	select {
	case pendingBatch.Results <- BatchResult{Embeddings: embeddings, Error: err}:
	default:
	}

	// Clean up
	bp.mu.Lock()
	delete(bp.pendingBatches, batchKey)
	bp.mu.Unlock()
}

// GetMetrics returns current client metrics
func (c *InferMeshClient) GetMetrics() ClientMetrics {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return *c.metrics
}

// SetBatchSize configures the batch size for optimal performance
func (c *InferMeshClient) SetBatchSize(size int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.batchProcessor.batchSize = size
}

// SetMaxWaitTime configures the maximum wait time for batching
func (c *InferMeshClient) SetMaxWaitTime(duration time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.batchProcessor.maxWaitTime = duration
}

// Close cleans up resources
func (c *InferMeshClient) Close() error {
	// Close HTTP client
	if transport, ok := c.httpClient.Transport.(*http.Transport); ok {
		transport.CloseIdleConnections()
	}
	return nil
}
