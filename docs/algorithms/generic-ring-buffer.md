---
type: algorithm
status: tested
tags: [easyrobot, buffer, streaming]
workstream: [W2, W7]
module: "[[bit-buffer]]"
summary: "9 pass (+ probe for untested views/Grow)"
source: [x/buffer/ring_buffer.go]
tests: [x/buffer/ring_buffer_test.go, x/buffer/ring_bench_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/buffer → ok, 89/89 PASS (2026-11-09); probe tmp/easyrobot/coding/bufprobe/ring → views, commits and Grow correct across wraparound"
---
# Generic ring buffer `buffer.Ring[T]`

**What it does:** fixed-capacity circular FIFO over a caller-supplied backing slice (capacity = `cap(backing)`), mutex-protected, with copy-based `Read/Write`, single-item `ReadItem/WriteItem`, and zero-copy `ReadView/WriteView` + `CommitRead/CommitWrite` (each view returns ≤ 2 slices for wraparound). Implements `buffer.Reader/Writer`, `Initializer`, `ViewReader/ViewWriter`, and `io.Reader/io.Writer` for `T=byte`.

**Where:** `x/buffer/ring_buffer.go` — `NewRing` :28, `Write` :41-88 (partial write when nearly full, `io.ErrShortWrite` when full), `Read` :95-145 (`io.EOF` when empty or short), `Grow` :271-321 (linearises into a larger slice), `Init` :326, views :349-403, commits :408-467.

**Inputs → outputs:** `[]T` in/out; `n` = elements.

## Completeness review

- Complete: yes for a bounded FIFO.
- Tested: 9 tests (write/read, wraparound, item I/O, full/empty, reset, io interfaces, generic types, zero capacity) + 14 benchmarks pass. Views, commits and `Grow` have **no unit tests**; probe exercised them across wraparound (`ReadView` → `[[4 5] [6 7]]`, commit accounting, `Grow(4)` preserves order) — all correct.
- Correct: yes for single-goroutine use. Caveats:
  - `Write` returns a short count with `nil` error when space runs out (`:60-63`); every `x/math/dsp/coding` processor ignores `n` (`_, err := dst.Write(...)`), so encoding into a too-small `Ring` silently drops output (probe: Manchester 8 B → 16 B into cap-10 ring: returns 16, ring holds 10).
  - `Read` returns `io.EOF` when merely empty — a live producer/consumer loop cannot distinguish "drained for now" from "stream ended".
  - `Grow` and `cap(rb.buf)` reads in `Cap`, `WriteItem`, `ReadItem`, `CommitRead/Write` happen outside the lock (`:151,178,211,413,445`) → racy if `Grow` runs concurrently (documented as not thread-safe, `:270`).
  - Views allocate a `[][]T` per call (`:363-402`).

## TODOs

- [ ] Add unit tests for `ReadView/WriteView/CommitRead/CommitWrite/Grow` — x/buffer/ring_buffer_test.go #p2
- [ ] Return `io.ErrShortWrite` (or document) on partial `Write`, and make coding processors honour `n` — x/buffer/ring_buffer.go:60-88 #p2
- [ ] Non-allocating view API (fill caller-provided `[2][]T`) — x/buffer/ring_buffer.go:349-403 #p3
- [ ] Distinguish empty from closed (`ErrEmpty` vs `io.EOF`) — x/buffer/ring_buffer.go:109-111 #p3

## Links

- Module: [[bit-buffer]]
- Related: [[pipeline-and-chain]], [[line-coding-and-fec]]
