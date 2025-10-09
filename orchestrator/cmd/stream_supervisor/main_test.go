package main

import "testing"

func TestParseTenantDomains(t *testing.T) {
	cases := []struct {
		name   string
		input  string
		expect map[string]string
	}{
		{"empty", "", nil},
		{"single", "tenant-alpha=edge", map[string]string{"tenant-alpha": "edge"}},
		{"spaces", "alpha =default , beta = edge", map[string]string{"alpha": "default", "beta": "edge"}},
		{"invalid", "foo", nil},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := parseTenantDomains(tc.input)
			if len(tc.expect) == 0 {
				if result != nil && len(result) > 0 {
					t.Fatalf("expected nil map, got %v", result)
				}
				return
			}
			if len(result) != len(tc.expect) {
				t.Fatalf("expected %d entries, got %d", len(tc.expect), len(result))
			}
			for k, v := range tc.expect {
				if got := result[k]; got != v {
					t.Fatalf("expected %s => %s, got %s", k, v, got)
				}
			}
		})
	}
}

func TestLinkPenaltyForService(t *testing.T) {
	svc := meshServiceConf{RttMs: 20, Weight: 2}
	penalty := linkPenaltyForService(svc)
	if penalty >= 1.0 {
		t.Fatalf("expected penalty < 1.0 when weight > 1, got %f", penalty)
	}

	svc = meshServiceConf{RttMs: 0, Weight: 0}
	penalty = linkPenaltyForService(svc)
	if penalty != 1.0 {
		t.Fatalf("expected default penalty of 1.0, got %f", penalty)
	}
}
