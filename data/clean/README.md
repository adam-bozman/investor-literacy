# data/clean/ — final analysis-ready files (NOT tracked in git)

This directory holds the cleaned, merged panel(s) the analysis scripts read.
**These files are not committed** (see `.gitignore`) because they are derived
from license-restricted CRSP / Compustat / Thomson Reuters data and inherit
those redistribution restrictions.

Primary analysis-ready file produced by the cleaning step:

- `panel_corrected_standardized.parquet` — the firm-month panel: monthly returns,
  momentum (12-2), idiosyncratic volatility, institutional-ownership share
  (persistent and time-varying), static and point-in-time headquarters state,
  state NFCS Big-Five literacy (z-scored within month), and the standardized
  three-/four-way interaction terms.

Regenerate it from `data/raw/` by running the data-collection and data-cleaning
steps in `code/00_master.py`.
