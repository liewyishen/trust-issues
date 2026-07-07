# Data

Real data files are gitignored. This directory only holds this README.

## Dataset

**LendingClub 2024 granting-model dataset**
- Source: Zenodo (https://zenodo.org/)
- License: CC BY 4.0 — you must attribute the original authors if you reproduce results.

## Download

1. Visit the Zenodo record linked above and download the CSV file(s).
2. Place them in this `data/` directory.
3. They will be automatically ignored by git (see `.gitignore`).

## Why real data is not committed

- Files are large (hundreds of MB).
- The CC BY 4.0 license permits redistribution with attribution, but committing raw data
  to a public repo creates an uncontrolled mirror — link to Zenodo instead.
- Keeping data out of git history prevents accidental PII exposure in future dataset updates.

## Expected files after download

```
data/
├── README.md          ← this file (committed)
├── lending_club.csv   ← raw download (gitignored)
└── lending_club.parquet  ← converted for faster I/O (gitignored)
```
