# stweight_pkg

A lightweight Python package to reproduce the spatio-temporal weight matrix (TW)
used in spatial econometric models, following Eq. (10)–(13) in:

> *Trade of Eco-Friendly Products and CO₂ Emissions in China*

The package constructs:
- Spatial weight matrix **W** (economic–geographic nested distance)
- Yearly **Global Moran’s I**
- Temporal weight matrix **T**
- Spatio-temporal weight matrix **TW = T ⊗ W**

## Installation

```bash
git clone https://github.com/yourname/stweight.git
cd stweight
pip install -e .
