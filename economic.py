from __future__ import annotations
import numpy as np
import pandas as pd

def per_capita_series_mean(
    df: pd.DataFrame,
    provinces: list[str],
    prov_col: str,
    value_col: str,
    exp_if_log: bool = True,
) -> np.ndarray:
    """
    Paper-style time-invariant perGDP_i: sample-period mean by province.
    """
    s = (
        df[[prov_col, value_col]]
        .dropna(subset=[value_col])
        .groupby(prov_col)[value_col]
        .mean()
        .reindex(provinces)
    )
    if s.isna().any():
        missing = s[s.isna()].index.tolist()
        raise ValueError(f"Missing {value_col} for provinces: {missing}")

    vals = s.to_numpy(dtype=float)
    if exp_if_log and str(value_col).lower().startswith("ln"):
        vals = np.exp(vals)
    return vals

def economic_diff_matrix(perGDP: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Build |perGDP_i - perGDP_j| matrix (n,n).
    """
    perGDP = np.asarray(perGDP, dtype=float)
    diff = np.abs(perGDP[:, None] - perGDP[None, :])
    # keep diagonal 0; eps handled later
    return diff + eps