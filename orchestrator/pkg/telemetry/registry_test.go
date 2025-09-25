package telemetry

import (
	"io/ioutil"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRegistryGauge(t *testing.T) {
	reg := NewRegistry()
	g := NewGauge(reg, "test_gauge", "help text")
	g.Set(5)
	if v, ok := reg.Value("test_gauge"); !ok || v != 5 {
		t.Fatalf("expected gauge value 5, got %v ok=%v", v, ok)
	}
}

func TestCounterVec(t *testing.T) {
	reg := NewRegistry()
	c := NewCounterVec(reg, "task_errors_total", "help", []string{"task"})
	c.WithLabelValues("build").Inc()
	c.WithLabelValues("build").Add(2)
	if total, ok := reg.Value("task_errors_total"); !ok || total != 3 {
		t.Fatalf("expected total 3, got %v ok=%v", total, ok)
	}
}

func TestHandlerOutput(t *testing.T) {
	reg := NewRegistry()
	g := NewGauge(reg, "foo", "demo gauge")
	g.Set(7)
	c := NewCounterVec(reg, "bar_total", "demo counter", []string{"label"})
	c.WithLabelValues("x").Inc()

	req := httptest.NewRequest("GET", "/metrics", nil)
	res := httptest.NewRecorder()
	reg.Handler().ServeHTTP(res, req)

	body, _ := ioutil.ReadAll(res.Body)
	content := string(body)
	if !strings.Contains(content, "foo 7") {
		t.Fatalf("metrics body missing gauge value: %s", content)
	}
	if !strings.Contains(content, "bar_total{label=\"x\"} 1") {
		t.Fatalf("metrics body missing counter value: %s", content)
	}
}
