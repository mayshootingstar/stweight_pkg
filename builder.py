from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from distance import distance_matrix_km
from economic import per_capita_series_mean
from normalize import row_standardize


@dataclass(frozen=True)
class TWResult:
    provinces: list[str]
    years: list[int]
    D_km: np.ndarray
    W_raw: np.ndarray
    W_used: np.ndarray
    moran_I: pd.DataFrame
    T: np.ndarray
    TW: sparse.csr_matrix


def global_morans_I(y: np.ndarray, W: np.ndarray) -> float:
    """
    Global Moran's I:
      I = (n/S0) * (z' W z) / (z'z), z = y - mean(y), S0 = sum(W_ij)
    """
    y = np.asarray(y, dtype=float)
    z = y - y.mean()
    denom = float(z @ z)
    S0 = float(W.sum())
    if denom <= 0 or S0 <= 0:
        return np.nan
    num = float(z @ (W @ z))
    n = len(y)
    return float((n / S0) * (num / denom))


def build_T_from_moran(I_vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Paper Eq.(12): lower triangular with ratios I_r/I_d; diag=1; upper=0
    """
    I_vec = np.asarray(I_vec, dtype=float)
    T_len = len(I_vec)
    T = np.zeros((T_len, T_len), dtype=float)
    for r in range(T_len):
        for d in range(T_len):
            if r == d:
                T[r, d] = 1.0
            elif r > d:
                T[r, d] = (I_vec[r] + eps) / (I_vec[d] + eps)
            else:
                T[r, d] = 0.0
    return T


def build_W_paper(
    df: pd.DataFrame,
    provinces: list[str],
    *,
    prov_col: str,
    lon_col: str,
    lat_col: str,
    pergdp_col: str,
    row_standardize_w: bool = True,
    exp_if_log_pergdp: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build paper-style spatial weight matrix W (Eq.10) and return (D_km, W_raw, W_used).

    Eq.(10): W_ij = 1 / ( |perGDP_i - perGDP_j| * d_ij ), i!=j; W_ii=0
    """
    # Coordinates (one row per province)
    coord_df = (
        df[[prov_col, lon_col, lat_col]]
        .dropna(subset=[prov_col, lon_col, lat_col])
        .drop_duplicates(subset=[prov_col])
        .set_index(prov_col)
        .reindex(provinces)
    )
    if coord_df.isna().any().any():
        missing = coord_df[coord_df.isna().any(axis=1)].index.tolist()
        raise ValueError(f"Missing lon/lat for provinces: {missing}")

    lon = coord_df[lon_col].to_numpy(dtype=float)
    lat = coord_df[lat_col].to_numpy(dtype=float)

    # Distance matrix
    D = distance_matrix_km(lon, lat)

    # perGDP (time-invariant mean by province)
    perGDP = per_capita_series_mean(
        df=df, provinces=provinces, prov_col=prov_col, value_col=pergdp_col, exp_if_log=exp_if_log_pergdp
    )

    # Build W
    n = len(provinces)
    W_raw = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            econ_diff = abs(perGDP[i] - perGDP[j]) + eps
            dij = D[i, j] + eps
            W_raw[i, j] = 1.0 / (econ_diff * dij)

    W_used = row_standardize(W_raw) if row_standardize_w else W_raw
    return D, W_raw, W_used


def build_TW_paper(
    df: pd.DataFrame,
    *,
    prov_col: str = "Province",
    year_col: str = "year",
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    co2_col: str = "CO2",
    pergdp_col: str = "pinc",
    provinces_order: Optional[Sequence[str]] = None,
    years_order: Optional[Sequence[int]] = None,
    row_standardize_w: bool = True,
    exp_if_log_pergdp: bool = True,
    eps: float = 1e-12,
) -> TWResult:
    """
    Full reproduction of paper-style spatio-temporal weight matrix TW:
      - W: Eq.(10)
      - Moran's I by year (on CO2): used to construct T
      - T: Eq.(12)
      - TW: Eq.(13 block structure) => TW = kron(T, W_used) = T ⊗ W_used
    """
    df = df.copy()
    df[prov_col] = df[prov_col].astype(str).str.strip()

    # Orders
    if provinces_order is None:
        provinces = sorted(df[prov_col].dropna().unique().tolist())
    else:
        provinces = list(provinces_order)

    if years_order is None:
        years = sorted(pd.Series(df[year_col].dropna().unique()).astype(int).tolist())
    else:
        years = [int(y) for y in years_order]

    if len(provinces) < 2:
        raise ValueError("Not enough provinces.")
    if len(years) < 2:
        raise ValueError("Not enough years.")

    # Build W
    D, W_raw, W_used = build_W_paper(
        df=df,
        provinces=provinces,
        prov_col=prov_col,
        lon_col=lon_col,
        lat_col=lat_col,
        pergdp_col=pergdp_col,
        row_standardize_w=row_standardize_w,
        exp_if_log_pergdp=exp_if_log_pergdp,
        eps=eps,
    )

    # Moran's I per year
    I_vals = []
    for y in years:
        sub = df[df[year_col].astype(int) == int(y)][[prov_col, co2_col]].dropna()
        sub[prov_col] = sub[prov_col].astype(str).str.strip()
        y_map = dict(zip(sub[prov_col], sub[co2_col].astype(float)))

        y_vec = np.array([y_map.get(p, np.nan) for p in provinces], dtype=float)
        if np.isnan(y_vec).any():
            miss = [provinces[i] for i in np.where(np.isnan(y_vec))[0]]
            raise ValueError(f"Year {y}: missing {co2_col} for provinces: {miss}")

        I = global_morans_I(y_vec, W_used)
        if not np.isfinite(I):
            raise ValueError(f"Year {y}: Moran's I is NaN/Inf. Check CO2 variance and W.")
        I_vals.append(I)

    I_vec = np.array(I_vals, dtype=float)
    moran_df = pd.DataFrame({"Year": years, "Moran_I": I_vec})

    # Build T
    T = build_T_from_moran(I_vec, eps=eps)

    # Build TW (block structure): TW = T ⊗ W_used
    TW = sparse.kron(sparse.csr_matrix(T), sparse.csr_matrix(W_used), format="csr")

    return TWResult(
        provinces=provinces,
        years=years,
        D_km=D,
        W_raw=W_raw,
        W_used=W_used,
        moran_I=moran_df,
        T=T,
        TW=TW,
    )


def save_result(res: TWResult, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.Series(res.provinces, name="Province").to_csv(out_dir / "provinces_order.csv", index=False, encoding="utf-8-sig")
    pd.Series(res.years, name="Year").to_csv(out_dir / "years_order.csv", index=False, encoding="utf-8-sig")

    np.save(out_dir / "D_km.npy", res.D_km)
    np.save(out_dir / "W_raw.npy", res.W_raw)
    np.save(out_dir / "W_used.npy", res.W_used)
    np.save(out_dir / "T.npy", res.T)

    res.moran_I.to_csv(out_dir / "moran_I_by_year.csv", index=False, encoding="utf-8-sig")

    sparse.save_npz(out_dir / "TW_csr.npz", res.TW)

    meta = {
        "n_provinces": len(res.provinces),
        "n_years": len(res.years),
        "TW_shape": list(res.TW.shape),
        "TW_nnz": int(res.TW.nnz),
    }
    pd.Series(meta).to_csv(out_dir / "meta.csv", header=["value"], encoding="utf-8-sig")

    return out_dir