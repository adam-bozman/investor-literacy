# Locally Biased, Financially Blind: Where State Investor Literacy Moderates the Momentum–Idiosyncratic-Volatility Anomaly

Replication package.

## Abstract

Retail investors trade locally, and they trade more like the people around them
than the textbook investor: chasing recent winners, anchoring on familiar names,
paying limited attention to volatility-adjusted risk. State-level financial
literacy varies substantially across the United States, and a large
household-finance literature documents that less literate investors make
systematically worse portfolio choices. If those choices show up in prices, they
should show up as a state-literacy moderation of return anomalies that depend on
retail mispricing. Yet in aggregate, the empirical signature is faint. This paper
asks a sharper question: not whether literacy moderates the
momentum–idiosyncratic-volatility anomaly, but *where in the cross-section* it
does, and what microfoundation that location identifies. On the
stable-headquarters subsample of CRSP-listed U.S. common stock from 2009 to 2023,
the moderation lives in firms that occupy an interaction zone — enough retail
demand to move prices, enough institutional activity to translate that demand into
return differences across literate and illiterate states. In the full panel these
firms fall in the middle of the institutional-ownership distribution; under
restrictions to larger firms they migrate to the low-IO tercile, exactly as the
interaction-zone reading predicts. Firms dominated by institutions show no literacy
gradient at all, consistent with arbitrage neutralizing the local-state channel.
The contribution is a precisely-bounded characterization of where a literacy
mechanism operates, and an empirical signature — migration with size composition —
that identifies the mechanism without requiring an instrumental-variables strategy.

## Authors

[AUTHOR NAMES]

## Paper status

[JOURNAL SUBMISSION STATUS]

## Repository structure

```
investor-attention-submission/
├── README.md               This file.
├── requirements.txt        Python dependencies (see "Software").
├── .gitignore
│
├── manuscript/             The main paper.
│   ├── main.tex            Canonical manuscript source (\input{}s sections/).
│   ├── main.pdf            Compiled manuscript (58 pp; pre-trim — see note below).
│   ├── references.bib      Bibliography.
│   ├── papersetup.sty      Style file required to compile.
│   ├── sections/           introduction, data, identification, results,
│   │                       mechanism, robustness, discussion, conclusion, appendix.
│   ├── figures/            Manuscript figures (PNG).
│   └── _alt/               Archived earlier, alternate version of the paper
│                           ("Where Literacy Moderates Momentum", 44 pp). Provenance
│                           only; not part of the submission.
│
├── appendix/               Internet Appendix (compiles standalone).
│   ├── internet_appendix.tex
│   ├── internet_appendix.pdf   (15 pp)
│   ├── references.bib
│   └── papersetup.sty
│
├── slides/                 Presentation decks (none yet — see slides/README.md).
│
├── code/
│   ├── 00_master.py        Run-order manifest + table/figure provenance for the
│   │                       analysis (the authoritative map of which script makes
│   │                       which exhibit). READ ITS HEADER FIRST.
│   ├── empirical/          53 analysis scripts (each carries a header block with
│   │                       its inputs, outputs, and paper location). Includes a
│   │                       version ladder; superseded iterations are kept for
│   │                       provenance and flagged in their headers + 00_master.py.
│   └── utils/              Shared helper modules (WRDS client/server, EDGAR, FRED,
│                           CRSP processing, etc.).
│
├── data/
│   ├── raw/                Original source extracts — NOT committed (see below).
│   └── clean/              Analysis-ready panel — NOT committed (see below).
│
└── output/
    ├── tables/             Generated table fragments (.tex) and result CSVs.
    └── figures/            Generated figures (.png / .pdf).
```

## Data

The analysis combines the following sources. **No data files are committed to
this repository** (see `.gitignore`); the bulk are subscription/license-restricted
and cannot be redistributed. `data/raw/README.md` and `data/clean/README.md` list
exactly what is expected and where to obtain it.

| Source | Used for | Access |
|---|---|---|
| CRSP monthly/daily stock files | returns, momentum, idiosyncratic volatility, shares outstanding | WRDS (subscription) |
| Compustat (funda / company) | firm state, CIK linkage | WRDS (subscription) |
| Thomson Reuters s34 13F holdings (2009Q1–2023Q4) | institutional-ownership share (persistent + time-varying) | WRDS (subscription) |
| CRSP–Compustat link table | permno↔gvkey↔CIK | WRDS (subscription) |
| SEC EDGAR (10-K SGML headers, Form 13F-HR) | point-in-time HQ state; EDGAR-proxy IO cross-check | Public (free) |
| NFCS State-by-State (2009, 2012, 2015, 2018, 2021) | state Big-Five financial-literacy share | Public — FINRA Investor Education Foundation |
| FRED | state unemployment, Phila. Fed coincident index (KK business-cycle controls) | Public (free) |

Only the NFCS waves and EDGAR/FRED pulls are public; everything from WRDS is
license-restricted. The cleaned panel (`panel_corrected_standardized.parquet`)
inherits those restrictions because it is derived from CRSP/Compustat/Thomson.

## Replication instructions

1. **Environment.** Python 3.9+. Create a virtual environment and
   `pip install -r requirements.txt`.

2. **Credentials.** Put WRDS credentials (and any API keys) in a local `.env`
   file at the repo root — it is git-ignored and must never be committed. The
   WRDS access layer runs as a small persistent server: see
   `code/utils/wrds_server.py` and `code/utils/wrds_client.py`.

3. **⚠ Repoint paths.** The analysis scripts in `code/empirical/` hard-code their
   `ROOT` to the *original* project tree
   (`.../Research/investor-attention-empirical`), not this `-submission` repo.
   Before running, edit `ROOT` at the top of each script (or run from a checkout
   matching the original layout). Outputs are written under `output/stage3a/`
   (and `output/stage3a/tables/`) relative to whatever `ROOT` names.

4. **Order to run.** Follow `code/00_master.py`, which lists every script in
   dependency order grouped into stages (data collection → cleaning → headline →
   descriptives → robustness → secondary DiD → figures) and records which
   manuscript/appendix exhibit each one produces. Set `RUN = True` inside it to
   execute end-to-end, or run the stages by hand. The shared library modules
   (`code/empirical/deepen_estimators.py`, `code/empirical/v9_helpers.py`) are
   imported by the analysis scripts and must be on `sys.path` — run scripts from
   inside `code/empirical/`.

5. **Manual steps / caveats.** Data-collection steps require live WRDS access and
   are long-running (the 60-quarter Thomson s34 pull and the EDGAR 10-K-header
   pull were run in stages with reconnects). Superseded scripts (an earlier
   EDGAR-13F IO proxy, and an empirical-Bayes shrinkage line later replaced by
   classical errors-in-variables) are kept for provenance and are **not** needed
   to reproduce the paper; they are flagged in `00_master.py` and in each file's
   header.

6. **Expected outputs.** Table fragments (`tab_*.tex`) and result CSVs land in
   `output/tables/`; figures in `output/figures/`. The manuscript's headline is
   Table `tab:headline` (mid-IO concentration, stable-HQ); see `00_master.py` for
   the full exhibit map.

7. **Compile the paper.** From `manuscript/`:
   `pdflatex main → bibtex main → pdflatex main → pdflatex main`.
   The Internet Appendix compiles independently from `appendix/` the same way; its
   cross-references to the main text resolve once `manuscript/main.aux` exists.

## Software and package dependencies

- **Python** 3.9 or newer.
- Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `pyarrow`.
- Data access: `wrds`, `python-dotenv`, `requests`.
- Optional helper modules only: `statsmodels`.
- See `requirements.txt`. (No `environment.yml` is provided; a plain
  `pip install -r requirements.txt` into a venv is sufficient.)
- **LaTeX**: a standard TeX distribution (TeX Live / MiKTeX) with `natbib`,
  `hyperref`, `booktabs`, `amsmath`, `graphicx`, `setspace`, and `xr`/`xr-hyper`.

## Notes

- `manuscript/main.pdf` is the **pre-trim** 58-page compile. If a length-trimmed
  version is produced, this PDF and the page count above should be updated.
- The cross-document reference between `manuscript/main.tex` and
  `appendix/internet_appendix.tex` uses `xr`/`xr-hyper` with relative paths and is
  guarded by `\IfFileExists`, so each document still compiles on its own.
