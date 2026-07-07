# Data

Real data files are gitignored. This directory only holds this README — the raw
dataset is never committed (see "Why real data is not committed" below).

## Dataset

**Lending Club loan dataset for granting models**
- Source: https://zenodo.org/records/11295916
- Authors: Miller Janny Ariza-Garzón, Mario Sanz-Guerrero, and Javier Arroyo
  Gallardo (data originally collected by Lending Club).
- Published: 25 May 2024 (Zenodo).
- License: CC BY 4.0 (Creative Commons Attribution 4.0 International) — you must
  attribute the original authors if you reproduce or redistribute results.

Peer-to-peer Lending Club loans spanning **2007–2018** (~1.35M records, about
60% of the original dataset, cleaned for completeness). Features are borrower
attributes available at application time — income, debt-to-income ratio, credit
scores, employment length, loan purpose, and state/zip — with a binary target
indicating whether each loan was fully paid or defaulted.

> Note: the Zenodo record was *published* in 2024, but the *loans* it contains
> are from 2007–2018. The modeling code splits temporally on those loan years
> (see `src/data_loader.py`).

## Download

1. Open the Zenodo record: https://zenodo.org/records/11295916
2. Download `LC_loans_granting_model_dataset.csv` (~167 MB).
3. Place it in this `data/` directory **without renaming**. The file must be
   named exactly `LC_loans_granting_model_dataset.csv` and live under `data/`,
   so the final path is:

   ```
   data/LC_loans_granting_model_dataset.csv
   ```

   This is the path `src/data_loader.py` expects (its `DEFAULT_DATA_PATH`); the
   code will not find the data under any other name or location.
4. It is automatically ignored by git (see the `data/*.csv` rule in `.gitignore`).

## Why real data is not committed

- The raw CSV is large (~167 MB).
- CC BY 4.0 permits redistribution with attribution, but committing the raw data
  to a public repo creates an uncontrolled mirror — link to the Zenodo record of
  record instead.
- Keeping data out of git history prevents accidental exposure in future dataset
  updates and keeps the repository lightweight.

## Expected files after download

```
data/
├── README.md                            ← this file (committed)
└── LC_loans_granting_model_dataset.csv  ← raw download from Zenodo (gitignored)
```
