# data/raw/ — original downloaded/pulled data (NOT tracked in git)

This directory holds the original, untransformed source extracts. **None of these
files are committed to the repository** (see `.gitignore`); they are either
license-restricted or too large to redistribute. To reproduce the analysis you
must obtain them yourself from the sources below and place them here.

| Expected file(s) | Source | Access |
|---|---|---|
| CRSP monthly/daily stock files (`msf`, `dsf`, `stocknames`) | WRDS → CRSP | Subscription (institutional) |
| Compustat `funda` / `company` (state, CIK) | WRDS → Compustat | Subscription (institutional) |
| Thomson Reuters s34 13F holdings, 2009Q1–2023Q4 | WRDS → TR Institutional (s34) | Subscription (institutional) |
| CRSP–Compustat link table | WRDS → CCM | Subscription (institutional) |
| SEC EDGAR 10-K SGML headers / Form 13F-HR | SEC EDGAR (public) | Free — see `code/utils/edgar_utils.py` |
| NFCS State-by-State waves 2009, 2012, 2015, 2018, 2021 | FINRA Investor Education Foundation | Free public-use — https://www.finrafoundation.org/ |
| State unemployment, Philadelphia Fed coincident index | FRED | Free — `code/utils/fred_utils.py` |

The NFCS waves are the only fully public, redistributable inputs; they are still
left untracked here for a uniform data policy. Everything else is governed by the
WRDS / CRSP / Compustat / Thomson Reuters license and cannot be shared.
