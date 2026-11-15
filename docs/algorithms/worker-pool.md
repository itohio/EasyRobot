---
type: algorithm
status: partial
tags: [easyrobot, workers, concurrency]
workstream: W9
module: "[[iter-workers]]"
summary: "nested `Execute` from a callback deadlocks (`worker_pool.go:411`); default-build…"
source: [x/workers/worker_pool.go, x/workers/execute_st.go, x/workers/execute_mt.go, x/workers/mt/pool.go]
tests: [x/workers/worker_pool_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09): x/workers 19/19 pass; go build -tags use_mt ./x/workers fails (undefined: mt); probe InfraNotes/poolprobe: nested Execute deadlocks"
---
# Worker pool (x/workers)

**What it does:** Fixed-size goroutine pool. `Execute(total, fn(start,end) error)` splits `[0,total)` into chunks (default `ceil(total/workers)`) and blocks until all finish; `Do(ctx, callbacks…)` runs independent callbacks with context cancellation; `Iterator(total)` yields chunk ranges sequentially (no parallelism). Package-level helpers `ParallelChunks/ParallelRows/Execute/ShouldParallelize` are the hook used by `x/iter/mt`.

**Where:**
- `x/workers/worker_pool.go:43-56` `WorkerPool`; `:166-202` `Init` (spawns `Size` or `GOMAXPROCS` goroutines, task chan buffer = workers); `:206-242` `Execute`; `:246-283` `Do`; `:286-311` `Iterator`; `:315-328` `Close`; `:330-379` worker loop; `:411-418` submit (blocks = backpressure); chunk sizers `:450-498`.
- `x/workers/execute_st.go` (`!use_mt`): `ShouldParallelize` → false; `ParallelChunks/Rows/IteratorChunks/Execute` → `fn(0,n)`; `ParallelTensorChunks` (`:17`) is an **empty body** (never calls fn).
- `x/workers/execute_mt.go` (`use_mt`): calls `mt.*` without importing it (`:6-29`), `return` of a void call (`:33`) → does not compile.
- `x/workers/mt/pool.go:11-83` package-level pool (`init()` creates `NumCPU` workers; `minParallelSize = 128·NumCPU`). Imported by nobody; importing it from `x/workers` would be a cycle (`x/workers/mt` imports `x/workers`).

**Inputs → outputs:** `total int`, callback over half-open ranges; returns first callback error, `ErrPoolClosed`, `ErrPoolNotInitialized`, `ErrWorkerCallbackNil`.

## Completeness review

- Complete: **no** — the `use_mt` wiring that would make tensor ops parallel does not build; default build is strictly sequential.
- Tested: 19 tests pass (`TestWorkerPoolExecute`, `…Do`, `…DoParallelExecution`, `…DoContextCancellation/Timeout`, `…Close`, `…DoAfterClose`, `…BlocksWhenNoWorkers`, `…IteratorRange{Equal,Unequal}Chunks`) — check coverage of ranges, error propagation, cancellation. No test for nested `Execute` or `Close` racing an in-flight `Execute`.
- Correct: **no** — probe: `Execute` called from inside an `Execute` callback on a 2-worker pool → **deadlock** (no return in 2 s): workers block in `submit` (`worker_pool.go:411-418`) waiting for themselves. `Close` during `Execute` returned `ErrPoolClosed` (no hang in this run), but queued jobs are dropped when a worker picks `stopCh` first (`:333-336`) — the doc "Outstanding tasks complete before shutdown" (`:314`) is not guaranteed [INFERENCE from select semantics; `state.done()` is never called for dropped jobs → potential hang].

**Defects**
- `x/workers/execute_st.go:17` — `ParallelTensorChunks` no-op: callers silently skip work.
- `x/workers/execute_mt.go:6-33` — undefined `mt`, `return mt.Execute(...)` in void func.
- `x/workers/worker_pool.go:411-418` — re-entrant `Execute` deadlocks (no caller-runs fallback).
- `x/workers/worker_pool.go:133-145` — `Do` wait spawns an extra goroutine per call; `:220` `newExecutionState` heap-allocates per `Execute`.
- `x/workers/worker_pool.go:431-436` — `defaultPoolConfig` (NumCPU) unused; `normalisePoolConfig` uses GOMAXPROCS.

**Embedded suitability:** goroutines + channels + `sync.Pool` + atomics. On ESP32/TinyGo (single core scheduler, `sync.Pool` no-op [INFERENCE]) the pool only adds overhead; keep the default `!use_mt` path there. Useful only for native/K230 multi-core.

## TODOs

- [ ] Make `ParallelTensorChunks` call `fn(0, shape[0])` — x/workers/execute_st.go:17 #p1 #ER-020
- [ ] Fix `use_mt` build (import, cycle with x/workers/mt, void return) or delete it — x/workers/execute_mt.go:6 #p1 #ER-026
- [ ] Caller-runs fallback (run chunk inline when submit would block / when called from a worker) — x/workers/worker_pool.go:411 #p1 #ER-020
- [ ] Drain queued jobs on Close (or mark them done) — x/workers/worker_pool.go:333 #p2
- [ ] Tests for nested Execute and Close-during-Execute — x/workers/worker_pool_test.go #p2

## Links

- Module: [[iter-workers]]
- Related: [[tensor-iterators]]
