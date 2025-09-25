package telemetry

import (
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
)

// Registry is a lightweight Prometheus-compatible collector registry tailored for unit tests
// and air-gapped environments. It supports gauges and counter vectors and exposes an HTTP
// handler that renders metrics using the standard text exposition format.
type Registry struct {
	mu       sync.RWMutex
	gauges   map[string]*Gauge
	counters map[string]*CounterVec
}

// NewRegistry creates a fresh registry instance.
func NewRegistry() *Registry {
	return &Registry{gauges: map[string]*Gauge{}, counters: map[string]*CounterVec{}}
}

// RegisterGauge adds a gauge to the registry.
func (r *Registry) RegisterGauge(g *Gauge) { r.mu.Lock(); r.gauges[g.name] = g; r.mu.Unlock() }

// RegisterCounterVec adds a counter vector to the registry.
func (r *Registry) RegisterCounterVec(c *CounterVec) {
	r.mu.Lock()
	r.counters[c.name] = c
	r.mu.Unlock()
}

// Value returns the latest recorded value for the provided metric name if it exists.
func (r *Registry) Value(name string) (float64, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if g, ok := r.gauges[name]; ok {
		return g.Value(), true
	}
	if c, ok := r.counters[name]; ok {
		return c.Total(), true
	}
	return 0, false
}

// Handler returns an http.Handler emitting metrics in Prometheus exposition format.
func (r *Registry) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		r.mu.RLock()
		defer r.mu.RUnlock()
		for _, g := range r.gauges {
			fmt.Fprintf(w, "# HELP %s %s\n", g.name, escapeHelp(g.help))
			fmt.Fprintf(w, "# TYPE %s gauge\n", g.name)
			fmt.Fprintf(w, "%s %g\n", g.name, g.Value())
		}
		for _, c := range r.counters {
			fmt.Fprintf(w, "# HELP %s %s\n", c.name, escapeHelp(c.help))
			fmt.Fprintf(w, "# TYPE %s counter\n", c.name)
			series := c.seriesSnapshot()
			sort.Strings(series)
			for _, line := range series {
				fmt.Fprintln(w, line)
			}
		}
	})
}

func escapeHelp(help string) string {
	help = strings.ReplaceAll(help, "\\", "\\\\")
	help = strings.ReplaceAll(help, "\n", "\\n")
	return help
}

// Gauge represents a single numeric value that can go up and down.
type Gauge struct {
	mu         sync.RWMutex
	name, help string
	val        float64
}

// NewGauge creates a gauge and automatically registers it with the provided registry.
func NewGauge(reg *Registry, name, help string) *Gauge {
	g := &Gauge{name: name, help: help}
	if reg != nil {
		reg.RegisterGauge(g)
	}
	return g
}
func (g *Gauge) Set(v float64)  { g.mu.Lock(); g.val = v; g.mu.Unlock() }
func (g *Gauge) Inc()           { g.Add(1) }
func (g *Gauge) Dec()           { g.Add(-1) }
func (g *Gauge) Add(d float64)  { g.mu.Lock(); g.val += d; g.mu.Unlock() }
func (g *Gauge) Value() float64 { g.mu.RLock(); defer g.mu.RUnlock(); return g.val }

// CounterVec represents a set of counters partitioned by label values.
type CounterVec struct {
	mu         sync.RWMutex
	name, help string
	labelNames []string
	values     map[string]float64
}

func NewCounterVec(reg *Registry, name, help string, labels []string) *CounterVec {
	c := &CounterVec{name: name, help: help, labelNames: append([]string(nil), labels...), values: map[string]float64{}}
	if reg != nil {
		reg.RegisterCounterVec(c)
	}
	return c
}
func (c *CounterVec) WithLabelValues(vals ...string) *Counter {
	if len(vals) != len(c.labelNames) {
		panic("telemetry: label value count mismatch")
	}
	key := strings.Join(vals, "\xff")
	c.mu.Lock()
	if _, ok := c.values[key]; !ok {
		c.values[key] = 0
	}
	c.mu.Unlock()
	return &Counter{parent: c, key: key, labels: append([]string(nil), vals...)}
}
func (c *CounterVec) add(key string, d float64) { c.mu.Lock(); c.values[key] += d; c.mu.Unlock() }
func (c *CounterVec) Total() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	sum := 0.0
	for _, v := range c.values {
		sum += v
	}
	return sum
}
func (c *CounterVec) seriesSnapshot() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	series := make([]string, 0, len(c.values))
	for key, v := range c.values {
		labels := strings.Split(key, "\xff")
		pairs := make([]string, len(labels))
		for i, val := range labels {
			pairs[i] = fmt.Sprintf("%s=\"%s\"", c.labelNames[i], escapeValue(val))
		}
		series = append(series, fmt.Sprintf("%s{%s} %g", c.name, strings.Join(pairs, ","), v))
	}
	return series
}
func escapeValue(v string) string {
	v = strings.ReplaceAll(v, "\\", "\\\\")
	v = strings.ReplaceAll(v, "\"", "\\\"")
	v = strings.ReplaceAll(v, "\n", "\\n")
	return v
}

// Counter is a handle for updating an underlying counter vector entry.
type Counter struct {
	parent *CounterVec
	key    string
	labels []string
}

func (c *Counter) Inc()          { c.Add(1) }
func (c *Counter) Add(d float64) { c.parent.add(c.key, d) }
