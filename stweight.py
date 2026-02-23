from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


# -----------------------------
# Result container
# -----------------------------
@dataclass(frozen=True)
class TWResult:
    provinces: list[str]
    years: list[int]
    D_km: np.ndarray              # (n,n)
    W_raw: np.ndarray             # (n,n) Eq.(10) before normalization
    W_used: np.ndarray            # (n,n) after optional row-standardization
    moran_I: pd.DataFrame         # columns: Year, Moran_I
    T: np.ndarray                 # (T,T) Eq.(12)
    TW: sparse.csr_matrix         # (n*T, n*T) Eq.(13) block-structure (T ⊗ W)


# -----------------------------
# Helpers
# -----------------------------
def _pick_first_existing(columns: Iterable[str], candidates: Sequence[str], name: str) -> str:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Missing {name} column. Candidates={list(candidates)}; Available={list(columns)}")


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Great-circle distance in kilometers using Haversine formula.
    """
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float(R * c)


def compute_distance_matrix_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    n = len(lon)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i, j] = 0.0
            else:
                D[i, j] = haversine_km(float(lon[i]), float(lat[i]), float(lon[j]), float(lat[j]))
    return D


def row_standardize(W: np.ndarray) -> np.ndarray:
    rs = W.sum(axis=1, keepdims=True)
    rs = np.where(rs == 0, 1.0, rs)
    return W / rs


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
    Paper Eq.(12): lower-triangular temporal weights
    diag = 1, upper = 0, lower = I_r / I_d
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


# -----------------------------
# Core builder
# -----------------------------
def build_tw_from_df(
    df: pd.DataFrame,
    *,
    prov_col: str = "Province",
    year_col: str = "year",
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    co2_col: str = "CO2",
    pergdp_col: str = "pinc",
    row_standardize_w: bool = True,
    exp_if_log_pergdp: bool = True,
    eps: float = 1e-12,
    provinces_order: Optional[Sequence[str]] = None,
    years_order: Optional[Sequence[int]] = None,
) -> TWResult:
    """
    Reproduce paper-style spatio-temporal weight matrix:

    1) Spatial W (Eq.10):
       W_ij = 1 / (|perGDP_i - perGDP_j| * d_ij), i != j; W_ii = 0
       perGDP_i: time-invariant; default uses sample-period mean by province.
       d_ij: great-circle distance based on lon/lat.

    2) Temporal T (Eq.12):
       Uses yearly Global Moran's I (computed on CO2 with W).
       Lower-triangular ratios: T[r,d] = I_r / I_d for r>d; diag=1; upper=0.

    3) TW (Eq.13 block structure):
       TW = kron(T, W)  (i.e., T ⊗ W) to match paper's block-matrix illustration.

    Returns TWResult with all intermediate artifacts.
    """
    # ---- basic checks
    for c in [prov_col, year_col, lon_col, lat_col, co2_col, pergdp_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}. Available={list(df.columns)}")

    df = df.copy()
    df[prov_col] = df[prov_col].astype(str).str.strip()

    # ---- provinces order
    if provinces_order is None:
        provinces = sorted(df[prov_col].dropna().unique().tolist())
    else:
        provinces = list(provinces_order)

    n = len(provinces)
    if n < 2:
        raise ValueError(f"Province count abnormal: {n}")

    # ---- years order
    if years_order is None:
        years = sorted(pd.Series(df[year_col].dropna().unique()).astype(int).tolist())
    else:
        years = [int(y) for y in years_order]

    T_len = len(years)
    if T_len < 2:
        raise ValueError(f"Year count abnormal: {T_len}")

    # ---- coordinates per province (must exist)
    coord_df = (
        df[[prov_col, lon_col, lat_col]]
        .dropna(subset=[prov_col, lon_col, lat_col])
        .drop_duplicates(subset=[prov_col])
        .set_index(prov_col)
        .reindex(provinces)
    )
    if coord_df.isna().any().any():
        miss = coord_df[coord_df.isna().any(axis=1)].index.tolist()
        raise ValueError(f"Missing lon/lat for provinces: {miss}")

    lon = coord_df[lon_col].to_numpy(dtype=float)
    lat = coord_df[lat_col].to_numpy(dtype=float)

    # ---- distance matrix d_ij
    D = compute_distance_matrix_km(lon, lat)

    # ---- perGDP_i (time-invariant): sample mean by province
    pgdp_series = (
        df[[prov_col, pergdp_col]]
        .dropna(subset=[pergdp_col])
        .groupby(prov_col)[pergdp_col]
        .mean()
        .reindex(provinces)
    )
    if pgdp_series.isna().any():
        miss = pgdp_series[pgdp_series.isna()].index.tolist()
        raise ValueError(f"Missing {pergdp_col} for provinces: {miss}")

    perGDP = pgdp_series.to_numpy(dtype=float)

    # if pergdp is log (ln...), optionally exp() back to level
    if exp_if_log_pergdp and str(pergdp_col).lower().startswith("ln"):
        perGDP = np.exp(perGDP)

    # ---- build W Eq.(10)
    W_raw = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            econ_diff = abs(perGDP[i] - perGDP[j]) + eps
            dij = D[i, j] + eps
            W_raw[i, j] = 1.0 / (econ_diff * dij)

    W_used = row_standardize(W_raw) if row_standardize_w else W_raw

    # ---- yearly Moran's I (on co2_col)
    I_vals = []
    for y in years:
        sub = df[df[year_col].astype(int) == int(y)][[prov_col, co2_col]].dropna()
        sub[prov_col] = sub[prov_col].astype(str).str.strip()
        y_map = dict(zip(sub[prov_col], sub[co2_col].astype(float)))

        y_vec = np.array([y_map.get(p, np.nan) for p in provinces], dtype=float)
        if np.isnan(y_vec).any():
            miss = [provinces[i] for i in np.where(np.isnan(y_vec))[0]]
            raise ValueError(f"Year={y}: missing {co2_col} for provinces: {miss}")

        I = global_morans_I(y_vec, W_used)
        if not np.isfinite(I):
            raise ValueError(
                f"Year={y}: Moran's I is NaN/Inf. "
                f"Check {co2_col} variability and W (S0>0, z'z>0)."
            )
        I_vals.append(I)

    I_vec = np.array(I_vals, dtype=float)
    moran_df = pd.DataFrame({"Year": years, "Moran_I": I_vec})

    # ---- build T Eq.(12)
    T = build_T_from_moran(I_vec, eps=eps)

    # ---- build TW Eq.(13 block structure): TW = T ⊗ W
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


def build_tw_from_csv(
    csv_path: str | Path,
    *,
    prov_col: str = "Province",
    year_col: str = "year",
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    co2_col: Optional[str] = None,
    pergdp_col: Optional[str] = None,
    row_standardize_w: bool = True,
    exp_if_log_pergdp: bool = True,
    eps: float = 1e-12,
    provinces_order: Optional[Sequence[str]] = None,
    years_order: Optional[Sequence[int]] = None,
) -> TWResult:
    """
    Convenience wrapper: read CSV then build TW.

    If co2_col/pergdp_col are None, tries to auto-detect common names.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if co2_col is None:
        co2_col = _pick_first_existing(df.columns, ["CO2", "co2", "lnCO2", "lnco2"], "CO2")
    if pergdp_col is None:
        pergdp_col = _pick_first_existing(
            df.columns,
            ["perGDP", "pergdp", "pinc", "Pinc", "pgdp", "PGDP", "lnpgdp", "lnPGDP", "lnpinc", "lnPinc"],
            "perGDP/pinc",
        )

    return build_tw_from_df(
        df,
        prov_col=prov_col,
        year_col=year_col,
        lon_col=lon_col,
        lat_col=lat_col,
        co2_col=co2_col,
        pergdp_col=pergdp_col,
        row_standardize_w=row_standardize_w,
        exp_if_log_pergdp=exp_if_log_pergdp,
        eps=eps,
        provinces_order=provinces_order,
        years_order=years_order,
    )


def save_tw_result(result: TWResult, out_dir: str | Path) -> Path:
    """
    Save W/T/TW and index orders in a directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.Series(result.provinces, name="Province").to_csv(out_dir / "provinces_order.csv", index=False, encoding="utf-8-sig")
    pd.Series(result.years, name="Year").to_csv(out_dir / "years_order.csv", index=False, encoding="utf-8-sig")

    np.save(out_dir / "D_km.npy", result.D_km)
    np.save(out_dir / "W_raw.npy", result.W_raw)
    np.save(out_dir / "W_used.npy", result.W_used)
    np.save(out_dir / "T.npy", result.T)

    result.moran_I.to_csv(out_dir / "moran_I_by_year.csv", index=False, encoding="utf-8-sig")

    sparse.save_npz(out_dir / "TW_csr.npz", result.TW)

    # Optional: quick metadata
    meta = {
        "n_provinces": len(result.provinces),
        "n_years": len(result.years),
        "TW_shape": list(result.TW.shape),
        "TW_nnz": int(result.TW.nnz),
    }
    pd.Series(meta).to_csv(out_dir / "meta.csv", header=["value"], encoding="utf-8-sig")

    return out_dir


# -----------------------------
# Optional: run as script
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build paper-style spatio-temporal weight matrix TW (Eq.10–13).")
    parser.add_argument("--input", required=True, help="Path to data_with_lon_lat.csv")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--year-col", default="year")
    parser.add_argument("--prov-col", default="Province")
    parser.add_argument("--lon-col", default="Longitude")
    parser.add_argument("--lat-col", default="Latitude")
    parser.add_argument("--co2-col", default=None, help="CO2 column name (default: auto-detect)")
    parser.add_argument("--pergdp-col", default=None, help="perGDP/pinc column name (default: auto-detect)")
    parser.add_argument("--no-row-standardize", action="store_true", help="Disable row-standardization for W")
    args = parser.parse_args()

    res = build_tw_from_csv(
        args.input,
        prov_col=args.prov_col,
        year_col=args.year_col,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        co2_col=args.co2_col,
        pergdp_col=args.pergdp_col,
        row_standardize_w=not args.no_row_standardize,
    )
    save_tw_result(res, args.out)
    print("✅ Done. TW shape:", res.TW.shape, "nnz:", res.TW.nnz)