import argparse
import pandas as pd

from builder import build_TW_paper, save_result

def main():
    parser = argparse.ArgumentParser(description="Build paper-style spatio-temporal weight matrix TW.")
    parser.add_argument("--input", default=r"C:\Users\mhmy\Desktop\套磁\26年1月\Pengyu ZHU\stweight_pkg\data_with_lon_lat.csv", help="Path to data_with_lon_lat.csv")
    parser.add_argument("--out", default=r"C:\Users\mhmy\Desktop\套磁\26年1月\Pengyu ZHU\stweight_pkg\paper_TW_outputs", help="Output directory")

    parser.add_argument("--prov-col", default="Province")
    parser.add_argument("--year-col", default="year")
    parser.add_argument("--lon-col", default="Longitude")
    parser.add_argument("--lat-col", default="Latitude")
    parser.add_argument("--co2-col", default="CO2")
    parser.add_argument("--pergdp-col", default="pinc")

    parser.add_argument("--no-row-standardize", action="store_true", help="Disable row-standardization for W")
    parser.add_argument("--no-exp-log", action="store_true", help="Do NOT exp() if pergdp_col starts with 'ln'")
    parser.add_argument("--eps", type=float, default=1e-12)

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    res = build_TW_paper(
        df,
        prov_col=args.prov_col,
        year_col=args.year_col,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        co2_col=args.co2_col,
        pergdp_col=args.pergdp_col,
        row_standardize_w=not args.no_row_standardize,
        exp_if_log_pergdp=not args.no_exp_log,
        eps=args.eps,
    )

    out_dir = save_result(res, args.out)
    print("✅ Done.")
    print("Saved to:", out_dir)
    print("TW shape:", res.TW.shape, "nnz:", res.TW.nnz)

if __name__ == "__main__":
    main()