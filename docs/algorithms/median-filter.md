---
type: algorithm
status: partial
tags: [easyrobot, filter, dsp]
workstream: W7
module: "[[filters]]"
summary: "Periodic passthrough"
source: [x/math/filter/median/median.go]
tests: []
test_result: none
complete: yes
correct: no
evidence: "no tests; probe dspprobe: spike 100 passes through at sample 20, warm-up outputs 0"
---
# Median filter (streaming)

**What it does:** Sliding-window median over the last `windowSize` (forced odd) samples using a ring buffer; `ProcessBuffer`/`Median` helpers.

**Where:** `x/math/filter/median/median.go`: `New` :23, `Process` :51-77, `ProcessBuffer` :81-94, `Median` :98-101.

**Inputs → outputs:** float32 sample → float32; window ≥1.

## Completeness review

- Complete: yes in API; warm-up handling wrong.
- Tested: no test file. Package compiles in the run: go test -count=1 -json ./x/math/filter/... (2026-11-09, raw tmp/easyrobot/filters/test-2026-11-09.json) → `[no test files]`.
- Correct: **no**.
  - `median.go:58` gates output with `mf.index < halfWindow`, where `index` is the ring write position, not a sample count. After warm-up the filter returns the raw sample every time the index wraps into `[0, halfWindow)` — i.e. periodically bypasses filtering. During warm-up it takes the median of a zero-filled buffer.
  - Probe (`dspprobe`, window 5, constant 5 with a 100 spike every 7th sample): output `5 0 5 … 5 100` — sample 1 is 0 (zero-padded median), sample 20 passes the spike through.
  - Allocates two slices per sample (:63, :73) and sorts with `sort.Slice` (reflection-based swapper) — O(w log w) and GC pressure on embedded targets.

## TODOs

- [ ] Track sample count; compute median over min(count, w) samples; never bypass after warm-up — median.go:58 #p1 #ER-019
- [ ] Preallocate scratch; use insertion into a sorted window (O(w)) instead of sort.Slice — median.go:63-75 #p2
- [ ] Unit tests (spike rejection, step edge preservation) #p2

## Links

- Module: [[filters]]
