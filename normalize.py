import numpy as np

def row_standardize(W: np.ndarray) -> np.ndarray:
    """Row-standardize so each row sums to 1 (if row sum is 0, keep row unchanged)."""
    W = np.asarray(W, dtype=float)
    rs = W.sum(axis=1, keepdims=True)
    rs = np.where(rs == 0, 1.0, rs)
    return W / rs