---
type: algorithm
status: implemented
tags: [easyrobot, colorscience, spectral]
workstream: W1
module: "[[colorscience]]"
summary: "an in-memory spectral library with two ranked searches."
source: [x/math/colorscience/database.go]
tests: []
test_result: none
complete: yes
correct: unknown
evidence: "no tests, not probed; code read only (2026-11-09)"
---
# Spectrum library search (correlation and peak matching)

**What it does:** an in-memory spectral library with two ranked searches.
1. **Shape similarity:** Pearson correlation between the query and each entry, after resampling the entry to the query grid. Negative correlation counts as 0.
2. **Peak pattern:** for each query peak, find the nearest entry peak within 50 nm. Score = matched ratio × 1/(1 + mean distance/10 nm).

**Where:** `database.go:33-53` `SpectrumDatabase` interface; `:56-65` `InMemoryDatabase`, `NewSpectrumDatabase`; `:88-159` `SearchBySimilarity`; `:162-225` `SearchByPeaks`; `:68-85,228-253` Add/Get/List/Remove.

**Inputs → outputs:** `SearchBySimilarity(query SPD, maxResults) ([]MatchResult{Entry, Score, Confidence}, error)`; `SearchByPeaks([]Peak, maxResults)`.

## Completeness review

- Complete: yes for a toy library.
  - `Add` stores the caller's matrix without copying (`:77-82`, comment admits it), so later mutation of the SPD changes the library.
  - Entries are resampled with flat end extrapolation (`spd.go:115-120`), which inflates correlation when ranges differ.
  - No range-overlap check and no baseline or continuum removal (needed for Raman/absorption matching).
  - Peak matching is many-to-one, and the 50 nm and 10 nm constants are hard-coded.
  - Not concurrency-safe (plain map).
  - Entry peaks are recomputed on every search (`:171`).
- Tested: none.
- Correct: unknown. The formulas are standard Pearson; the single-pass variance (`:128-130`) loses float32 precision for large offsets.

## TODOs

- [ ] Deep-copy SPD on Add — `database.go:79` #p2
- [ ] Restrict similarity to overlapping wavelength range; optional baseline removal — `database.go:102-110` #p2
- [ ] Tests (self-match = 1, ranking) — (none) #p2

## Links

- Module: [[colorscience]]
