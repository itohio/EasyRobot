---
type: algorithm
status: tested
tags: [easyrobot, buffer, memory]
workstream: [W2]
module: "[[bit-buffer]]"
summary: "7 pass"
source: [x/buffer/pool.go]
tests: [x/buffer/pool_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/buffer → ok (7 Pool tests incl. concurrent Get/Put pass, 2026-11-09); probe bufprobe: Put(cap 20) then Get(30) → fresh cap-32 slice (cap-20 buffer unusable for its tier)"
---
# Tiered slice pool `buffer.Pool[T]`

**What it does:** size-class allocator over `sync.Pool`: tiers with increasing max lengths (default 16, 32, …, 2048), plus an unbounded tier. `Get(n)` takes from the smallest tier whose bound ≥ n and allocates `make([]T, 0, bound)` on miss; `Put(buf)` files the slice by `cap(buf)`.

**Where:** `x/buffer/pool.go` — `Reconfigure(lengths...)` :24, `Get` :43-66, `Put` :69-90, lazy default config `ensureDefault` :92, `configureLocked` :107-136, `tierIndex` :148, `capacityFor` :157.

**Inputs → outputs:** `Get(n) []T` (len n); `Put([]T)`.

## Completeness review

- Complete: yes.
- Tested: default config, zeroth-tier reuse, zero-length put, unbounded-tier reuse, reconfigure (valid/invalid), concurrent Get/Put — pass.
- Correct: yes, with inefficiencies:
  - `Put` files by the tier whose **bound ≥ cap** (`:85`), so a slice with cap between two bounds lands in the higher tier and is rejected by `Get` there (`cap < n` → new allocation, `:61-63`) — such slices are effectively thrown away (probe). Filing by the largest bound ≤ cap avoids this.
  - `pool.Put(buf[:0])` boxes the slice header into an interface → one allocation per `Put` (staticcheck SA6002) [INFERENCE: not measured].
  - Exported embedded `sync.RWMutex` and exported `Tiers` field leak locking into the API.

## TODOs

- [ ] File returned slices by floor tier (largest bound ≤ cap) — x/buffer/pool.go:69-90 #p3
- [ ] Store `*[]T` in `sync.Pool` to avoid per-Put allocation — x/buffer/pool.go:168-173 #p3
- [ ] Unexport the mutex and `Tiers` — x/buffer/pool.go:16-21 #p3

## Links

- Module: [[bit-buffer]]
