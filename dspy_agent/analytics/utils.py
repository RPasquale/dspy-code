from __future__ import annotations

from typing import List, Tuple, Dict, Any


def compute_correlations(X: List[List[float]], y: List[float]) -> List[float]:
    """Pearson correlation per feature dimension between X (unit vectors) and y (rewards).

    Returns a list of correlation coefficients for each dimension in X.
    """
    if not X or not y or len(X) != len(y):
        return []
    n = len(X)
    d = len(X[0])
    # means
    y_mean = sum(y) / n
    y_var = sum((v - y_mean) ** 2 for v in y) / max(1, n - 1)
    y_std = (y_var ** 0.5) if y_var > 1e-12 else 1.0
    x_mean = [0.0] * d
    for row in X:
        for j in range(d):
            x_mean[j] += float(row[j])
    x_mean = [m / n for m in x_mean]
    x_var = [0.0] * d
    for row in X:
        for j in range(d):
            x_var[j] += (float(row[j]) - x_mean[j]) ** 2
    x_std = [((v / max(1, n - 1)) ** 0.5) if v > 1e-12 else 1.0 for v in x_var]
    cov = [0.0] * d
    for row, yy in zip(X, y):
        dy = yy - y_mean
        for j in range(d):
            cov[j] += (float(row[j]) - x_mean[j]) * dy
    cov = [c / max(1, n - 1) for c in cov]
    corr = [cov[j] / (x_std[j] * y_std) if (x_std[j] * y_std) > 1e-12 else 0.0 for j in range(d)]
    return corr


def compute_direction(X: List[List[float]], y: List[float], *, lr: float = 0.01, iters: int = 200) -> List[float]:
    """Compute a regression 'direction' vector via simple gradient descent on standardized features.

    Returns a unit-norm coefficient vector of length equal to X[0].
    """
    if not X or not y or len(X) != len(y):
        return []
    n = len(X); d = len(X[0])
    # Standardize features
    x_mean = [0.0] * d
    for row in X:
        for j in range(d):
            x_mean[j] += float(row[j])
    x_mean = [m / n for m in x_mean]
    x_var = [0.0] * d
    for row in X:
        for j in range(d):
            x_var[j] += (float(row[j]) - x_mean[j]) ** 2
    x_std = [((v / max(1, n - 1)) ** 0.5) if v > 1e-12 else 1.0 for v in x_var]
    Xs = [[(float(row[j]) - x_mean[j]) / (x_std[j] or 1.0) for j in range(d)] for row in X]
    # Gradient descent linear regression
    w = [0.0] * d
    for _ in range(max(1, iters)):
        grad = [0.0] * d
        for i in range(n):
            yhat = 0.0
            xi = Xs[i]
            for j in range(d):
                yhat += w[j] * xi[j]
            err = (yhat - float(y[i]))
            for j in range(d):
                grad[j] += err * xi[j]
        for j in range(d):
            w[j] -= float(lr) * (grad[j] / n)
    # Normalize
    norm = sum(v * v for v in w) ** 0.5 or 1.0
    return [float(v / norm) for v in w]


def kmeans_clusters(vecs: List[List[float]], *, k: int = 3, iters: int = 5) -> List[Dict[str, Any]]:
    """Naive k-means over cosine similarity (dot product for unit vectors).

    vecs: list of unit vectors; returns list of cluster dicts with 'center' and 'indices'.
    """
    if not vecs or k <= 1:
        return [{'id': 0, 'center': vecs[0] if vecs else [], 'indices': list(range(len(vecs)))}]
    k = min(k, len(vecs))
    centers = [list(vecs[i]) for i in range(k)]
    def dot(a: List[float], b: List[float]) -> float:
        n = min(len(a), len(b))
        return sum(float(a[i]) * float(b[i]) for i in range(n))
    assign = [[] for _ in range(k)]
    for _ in range(max(1, iters)):
        assign = [[] for _ in range(k)]
        for idx, v in enumerate(vecs):
            sims = [dot(v, c) for c in centers]
            ci = max(range(k), key=lambda i: sims[i])
            assign[ci].append(idx)
        new_centers: List[List[float]] = []
        for group in assign:
            if not group:
                new_centers.append(centers[0])
                continue
            dim = len(vecs[group[0]]); c = [0.0] * dim
            for idx in group:
                v = vecs[idx]
                for j in range(dim):
                    c[j] += float(v[j])
            z = max(1.0, float(len(group)))
            c = [x / z for x in c]
            new_centers.append(c)
        centers = new_centers
    return [{'id': i, 'center': centers[i], 'indices': assign[i]} for i in range(k)]

