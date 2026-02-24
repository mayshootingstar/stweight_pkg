# Spatio-Temporal Weight Matrix Construction (TW)

This repository provides a **reproducible Python implementation** for constructing the  
**spatio-temporal weight matrix (TW)** used in spatial econometric models, following
Eq. (10)–(13) in the paper:

> *Trade of Eco-Friendly Products and CO₂ Emissions in China*

The implementation is designed for **research use**, with an emphasis on:
- correctness
- transparency
- ease of execution (no complex package structure)

---

## 1. What This Code Does

Given provincial panel data with geographic coordinates and economic indicators, the code constructs:

1. **Spatial weight matrix \(W\)**  

2. **Yearly Global Moran’s I** (based on CO₂ emissions)

3. **Temporal weight matrix \(T\)**  

4. **Spatio-temporal weight matrix \(TW\)**  
---

## 2. Directory Structure

All files are placed in **one single directory** for simplicity:

```text
.
├─ distance.py     # geographic distance (Haversine)
├─ economic.py     # economic weights (perGDP)
├─ normalize.py    # row standardization
├─ builder.py      # W, T, TW construction
├─ run.py          # executable script
└─ README.md
