# The Interaction Zone: State Financial Literacy and the Cross-Section of the Momentum Anomaly

Replication package. This README maps out the question and the result before the replication steps: why the study matters, what it tests, what data it uses, and what it finds. The replication workflow follows at the end.

## Motivation: why we are doing this

Two well-established facts motivate the paper. First, household-finance research shows that financially literate people make better investment decisions: they diversify more, trade less on noise, and pay attention to risk and fees. Second, retail investors hold portfolios tilted toward locally headquartered firms, and that local demand moves prices. Put together, the quality of a firm's local investors should leave a footprint in its stock price. In states where retail investors are less financially literate, the return anomalies that feed on retail mistakes should be stronger.

The anomaly I use is the interaction of momentum with idiosyncratic volatility: high-volatility momentum stocks are exactly where retail chasing of recent winners is least disciplined by arbitrage. State financial literacy, applied to firms by headquarters state, is the moderator.

The economic stake is whether individual-level literacy aggregates up into market prices at all. If it does, investor education has consequences beyond personal portfolios: it shapes how efficiently prices work in parts of the market, and it tells us where retail mistakes survive arbitrage long enough to matter. The complication is that the obvious test fails. Run the regression across all firms and the result is weak and fragile. So the paper asks a sharper question: not whether literacy moves prices, but **where in the cross-section** it does, and what that location reveals about the mechanism.

## Hypotheses (informal)

1. State financial literacy moderates the momentum and idiosyncratic-volatility anomaly: the anomaly is weaker in more literate states.
2. The moderation is not uniform across firms. It should appear only where two things hold at once: enough retail ownership for local mistakes to move the price, and enough institutional ownership for those mistakes to show up as return differences across states. This is the **interaction zone**.
3. Because the interaction zone is defined by economics (a mix of retail and institutional ownership at meaningful size), not by a fixed ownership label, it should move predictably when the sample changes. Drop the smallest firms and the moderation should shift from mid-ownership firms to low-ownership firms, with the cross-state literacy gradient following it.
4. Where institutions dominate, arbitrage absorbs retail demand, so the most institution-held firms should show no literacy moderation under any sample.

## Data

The analysis combines the following sources. **No data files are committed to this repository** (see `.gitignore`); most are subscription or license-restricted and cannot be redistributed. `data/raw/README.md` and `data/clean/README.md` list exactly what is expected and where to obtain it.

| Source | Used for | Access |
|---|---|---|
| CRSP monthly/daily stock files | returns, momentum, idiosyncratic volatility, shares outstanding | WRDS (subscription) |
| Compustat (funda / company) | firm state, CIK linkage | WRDS (subscription) |
| Thomson Reuters s34 13F holdings (2009Q1 to 2023Q4) | institutional-ownership share (persistent and time-varying) | WRDS (subscription) |
| CRSP–Compustat link table | permno to gvkey to CIK | WRDS (subscription) |
| SEC EDGAR (10-K SGML headers, Form 13F-HR) | point-in-time headquarters state | Public (free) |
| NFCS State-by-State (2009, 2012, 2015, 2018, 2021) | state Big-Five financial-literacy share | Public, FINRA Investor Education Foundation |
| FRED | state unemployment, Philadelphia Fed coincident index (business-cycle controls) | Public (free) |

The sample is CRSP common stock from January 2009 through December 2023. The headline results use the **stable-headquarters subsample**: firms whose headquarters state does not change during the sample, so that the local investor base is well-defined. Only the NFCS, EDGAR, and FRED inputs are public; everything from WRDS is license-restricted, and the cleaned panel inherits those restrictions.

## What the paper finds (the punchline)

On the stable-headquarters subsample, 2009 to 2023:

- **The literacy effect is concentrated, not uniform.** It shows up in firms with mid-level institutional ownership and is absent in both the most retail-held and the most institution-held firms. The all-firms test misses it because it averages this concentration away. That is why the aggregate signal looked weak.

- **The effect belongs to an economic population, not an ownership label.** This is the core of the paper. When the smallest half of firms is dropped from the sample, the moderation moves out of the mid-ownership tercile and into the low-ownership tercile, and the cross-state literacy gradient moves with it. The same kind of firm (mixed ownership at meaningful size) carries the effect throughout; only its ownership label changes with the sample.

- **That migration is the identification.** A genuine mechanism should relocate on cue when you change which firms are in the sample; a spurious correlation should not. The migration confirms the interaction-zone reading without an instrument, which matters because the natural instrument (state K-12 financial-education mandates) does not work: states adopted mandates because their adults were less literate, so the instrument points the wrong way.

- **Bottom line.** Financial literacy does leave a footprint in stock prices, but only in the interaction zone, where retail demand and institutional activity coexist. Investor sophistication matters for asset prices conditionally, in a segment of the market that can be located precisely and tracked as it moves. Methodologically, "watch the effect migrate as the sample changes" is a portable way to identify a cross-sectional mechanism when no clean instrument exists.

**Scope.** The result holds for firms with stable headquarters; a point-in-time correction to headquarters location shrinks the effect on the full panel, which bounds the population it applies to. The U-shape across the ownership distribution is robust as a discrete three-tercile contrast rather than as a precise parametric curve.

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
├── manuscript/             The main paper (37 pp).
│   ├── main.tex            Canonical manuscript source (\input{}s sections/).
│   ├── main.pdf            Compiled manuscript.
│   ├── references.bib      Bibliography.
│   ├── papersetup.sty      Style file required to compile.
│   ├── sections/           introduction, data, identification, results,
│   │                       mechanism, robustness, discussion, conclusion, appendix.
│   ├── figures/            Manuscript figures (PNG).
│   └── _alt/               Archived earlier, alternate version of the paper
│                           ("Where Literacy Moderates Momentum", 44 pp). Provenance
│                           only; not part of the submission.
│
├── appendix/               Internet Appendix (20 pp; compiles standalone).
│   ├── internet_appendix.tex
│   ├── internet_appendix.pdf
│   ├── references.bib
│   └── papersetup.sty
│
├── slides/                 Presentation decks (none yet; see slides/README.md).
│
├── code/
│   ├── 00_master.py        Run-order manifest and table/figure provenance for the
│   │                       analysis (the authoritative map of which script makes
│   │                       which exhibit). READ ITS HEADER FIRST.
│   ├── empirical/          53 analysis scripts (each carries a header block with
│   │                       its inputs, outputs, and paper location). Includes a
│   │                       version ladder; superseded iterations are kept for
│   │                       provenance and flagged in their headers and 00_master.py.
│   └── utils/              Shared helper modules (WRDS client/server, EDGAR, FRED,
│                           CRSP processing, etc.).
│
├── data/
│   ├── raw/                Original source extracts, NOT committed (see Data).
│   └── clean/              Analysis-ready panel, NOT committed (see Data).
│
└── output/
    ├── tables/             Generated table fragments (.tex) and result CSVs.
    └── figures/            Generated figures (.png / .pdf).
```

## Replication

1. **Environment.** Python 3.9 or newer. Create a virtual environment and `pip install -r requirements.txt`.

2. **Credentials.** Put WRDS credentials (and any API keys) in a local `.env` file at the repo root. It is git-ignored and must never be committed. The WRDS access layer runs as a small persistent server: see `code/utils/wrds_server.py` and `code/utils/wrds_client.py`.

3. **Repoint paths.** The analysis scripts in `code/empirical/` hard-code their `ROOT` to the original project tree (`.../Research/investor-attention-empirical`), not this `-submission` repo. Before running, edit `ROOT` at the top of each script, or run from a checkout matching the original layout. Outputs are written under `output/stage3a/` (and `output/stage3a/tables/`) relative to whatever `ROOT` names.

4. **Order to run.** Follow `code/00_master.py`, which lists every script in dependency order grouped into stages (data collection, cleaning, headline, descriptives, robustness, secondary DiD, figures) and records which manuscript or appendix exhibit each one produces. Set `RUN = True` inside it to execute end to end, or run the stages by hand. The shared library modules (`code/empirical/deepen_estimators.py`, `code/empirical/v9_helpers.py`) are imported by the analysis scripts and must be on `sys.path`, so run scripts from inside `code/empirical/`.

5. **Manual steps and caveats.** Data-collection steps require live WRDS access and are long-running (the 60-quarter Thomson s34 pull and the EDGAR 10-K-header pull were run in stages with reconnects). Superseded scripts (an earlier EDGAR-13F ownership proxy, and an empirical-Bayes shrinkage line later replaced by classical errors-in-variables) are kept for provenance and are not needed to reproduce the paper; they are flagged in `00_master.py` and in each file's header.

6. **Expected outputs.** Table fragments (`tab_*.tex`) and result CSVs land in `output/tables/`; figures in `output/figures/`. The headline result is the mid-ownership concentration table (`tab:headline`); see `00_master.py` for the full exhibit map.

7. **Compile the paper.** From `manuscript/`: `pdflatex main`, `bibtex main`, then `pdflatex main` twice. The Internet Appendix compiles independently from `appendix/` the same way; its cross-references to the main text resolve once `manuscript/main.aux` exists.

## Software and package dependencies

- **Python** 3.9 or newer.
- Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `pyarrow`.
- Data access: `wrds`, `python-dotenv`, `requests`.
- Optional helper modules only: `statsmodels`.
- See `requirements.txt`. A plain `pip install -r requirements.txt` into a virtual environment is sufficient; no `environment.yml` is provided.
- **LaTeX**: a standard TeX distribution (TeX Live or MiKTeX) with `natbib`, `hyperref`, `booktabs`, `amsmath`, `graphicx`, `setspace`, and `xr`/`xr-hyper`.

## Notes

- The cross-document reference between `manuscript/main.tex` and `appendix/internet_appendix.tex` uses `xr`/`xr-hyper` with relative paths and is guarded by `\IfFileExists`, so each document still compiles on its own.
