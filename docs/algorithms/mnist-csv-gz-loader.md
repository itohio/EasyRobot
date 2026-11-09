---
type: algorithm
status: implemented
tags: [easyrobot, learn, nn]
workstream: W9
module: "[[learn]]"
summary: "only tagged tests (not run)"
source: [x/math/learn/datasets/mnist/loader.go]
tests: [x/math/learn/mnist_test.go, x/math/learn/mnist_convnet_test.go, x/math/learn/mnist_large_test.go]
test_result: not-run
complete: yes
correct: unknown
evidence: "only used by build-tagged tests (mnist_test, mnist_large_test) — not run per contract"
---
# MNIST CSV.gz loader

**What it does:** reads gzip-compressed MNIST **CSV** (label + 784 pixel columns) into `[]Sample{Label int, Image tensor [1,28,28] float32 in [0,1]}`. Not an IDX (ubyte) reader.

**Where:** `x/math/learn/datasets/mnist/loader.go:13-94` (`Sample`, `Load`).

**Inputs → outputs:** `Load(filename string, maxSamples int) ([]Sample, error)`; `maxSamples<=0` = all. Rows with <785 columns error (:42-44); labels outside 0–9 skipped (:53-56); pixels `/255` (:66). Allocates one 784-float slice + tensor per sample (~3.1 KB each; 60k samples ≈ 188 MB as float32 plus tensor headers).

## Completeness review

- Complete: yes for CSV.gz. No IDX format, no uncompressed CSV, no `io.Reader` input, no batching/one-hot helpers.
- Tested: no direct test. Used by tagged tests `mnist_test.go:24-33`, `mnist_large_test.go:32`, `mnist_convnet_test.go` expecting `datasets/mnist/mnist_{train,test}.csv.gz`; those assert only test accuracy >15% (mnist_test.go:247-249) and are **not run** here (build tags; data presence not checked).
- Correct: unknown — not exercised.

## TODOs

- [ ] Unit test with a tiny in-memory gz CSV — x/math/learn/datasets/mnist/loader.go:20 #p2
- [ ] Accept `io.Reader`; optional IDX reader — x/math/learn/datasets/mnist/loader.go:20 #p3

## Links

- Module: [[learn]]
