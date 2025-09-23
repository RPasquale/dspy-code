from dspy_agent.analytics.utils import compute_correlations, compute_direction, kmeans_clusters


def test_compute_correlations_simple():
    # X: 2 dims; y correlates strongly with dim0
    X = [[1.0, 0.0], [0.8, 0.2], [0.6, 0.4], [0.2, 0.8], [0.0, 1.0]]
    y = [1.0, 0.8, 0.6, 0.2, 0.0]
    corr = compute_correlations(X, y)
    assert len(corr) == 2
    assert corr[0] > corr[1]


def test_compute_direction_points_to_dim0():
    # y approx equals dim0; direction weight for dim0 should be larger than dim1
    X = [[1.0, 0.0], [0.8, 0.2], [0.6, 0.4], [0.2, 0.8], [0.0, 1.0]]
    y = [1.0, 0.8, 0.6, 0.2, 0.0]
    w = compute_direction(X, y, lr=0.05, iters=300)
    assert len(w) == 2
    assert abs(sum(v*v for v in w) - 1.0) < 1e-3
    assert w[0] > w[1]


def test_kmeans_clusters_cosine():
    # Two clusters: near [1,0] and [0,1]
    A = [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]]
    B = [[0.0, 1.0], [0.1, 0.9], [0.05, 0.95]]
    vecs = A + B
    groups = kmeans_clusters(vecs, k=2, iters=5)
    # We should have 2 clusters with total count 6
    assert len(groups) == 2
    assert sum(len(g['indices']) for g in groups) == len(vecs)

