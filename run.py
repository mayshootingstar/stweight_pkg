import stweight

INP = r"C:\Users\mhmy\Desktop\套磁\26年1月\Pengyu ZHU\data_with_lon_lat.csv"
OUT = r"C:\Users\mhmy\Desktop\套磁\26年1月\Pengyu ZHU\paper_TW_outputs"

print("Start building TW...")
print("Input:", INP)
print("Output dir:", OUT)

res = stweight.build_tw_from_csv(
    INP,
    year_col="year",
    prov_col="Province",
    lon_col="Longitude",
    lat_col="Latitude",
    co2_col="CO2",       # 如果你文件里叫 lnCO2，就改成 "lnCO2"
    pergdp_col="pinc",   # 如果你文件里叫 lnpgdp，就改成 "lnpgdp"
    row_standardize_w=True
)

print("Build done.")
print("n_provinces =", len(res.provinces))
print("n_years     =", len(res.years))
print("TW shape    =", res.TW.shape)
print("TW nnz      =", res.TW.nnz)

out_dir = stweight.save_tw_result(res, OUT)
print("Saved to:", out_dir)