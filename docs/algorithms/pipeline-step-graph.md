---
type: algorithm
status: partial
tags: [easyrobot, pipeline, deprecated, streaming]
workstream: [W1]
module: "[[pipeline-and-chain]]"
summary: "Timestamp sync broken"
source: [x/pipeline/pipeline.go, x/pipeline/step.go, x/pipeline/registry.go, x/pipeline/steps/source.go, x/pipeline/steps/processor.go, x/pipeline/steps/sink.go, x/pipeline/steps/fan.in.go, x/pipeline/steps/fan.out.go, x/pipeline/steps/join.go, x/pipeline/steps/sync.go, x/pipeline/steps/fps/fps.go]
tests: [x/pipeline/pipeline_test.go, x/pipeline/step_test.go, x/pipeline/registry_test.go, x/pipeline/steps/sync_test.go, x/pipeline/steps/sink_test.go]
test_result: build-fail
complete: no
correct: no
evidence: "go test -count=1 ./x/pipeline → ok 17 PASS; ./x/pipeline/steps → vet build fail; with -vet=off → 28 PASS, 1 FAIL (TestOptions) (2026-11-09)"
---
# Channel-based step pipeline (deprecated)

**What it does:** goroutine-per-step dataflow graph: steps (`In(<-chan Data)`, `Out()`, `Run(ctx)`, `Reset()`) exchange `store.Store` maps over channels; `Pipeline` assigns ids, connects chains and runs every step via `concurrency.Submit`. Step library: source/reader, processor (func), sink, fan-in, fan-out, join, timestamp sync, fps counter; steps self-register in `plugin.Global`. `DEPRECATED.md`: "Pipeline, Plugin, Store will be replaced by dndm package and associated patterns." Nothing outside `x/pipeline` imports it.

**Where:** `Pipeline` `pipeline.go:20-141` (`FromJSON` stub returns empty pipeline :33-35; ids start at 123 :27); `StepSend` (blocking vs drop) / `StepReceive` / `StepMakeChan` `step.go`; timestamp sync `steps/sync.go:19-196` (`FindClosest` :54-78, `doSyncronize` :159-196).

## Completeness review

- Complete: no — `FromJSON` is a stub (`pipeline.go:33-35`); `MarshallableStep` has no implementations.
- **Timestamp sync never fires**: `s.inArr[idx].Add(data)` discards the returned slice (`steps/sync.go:169`); `Add` has a value receiver and appends when len < cap (:21-31), and buffers start with len 0 (:117), so the input buffers stay empty and `FindClosest` returns −1 forever. The test that would catch it (`Testsyncronize_doSyncronize`, `sync_test.go:291`) never runs because of its malformed name.
- Tested: `x/pipeline` 17 PASS. `x/pipeline/steps` fails to build under `go test` because the vet `tests` analyzer rejects 30+ lower-case test names (`Testfanin_In`, `Testsyncronize_Run`, … e.g. `fan.in_test.go:48`, `sync_test.go:13`) and unkeyed `plugin.Options` literals (`sink_test.go:33-38`). With `-vet=off`: 28 PASS, 1 FAIL — `TestOptions` (`sink_test.go:55` expects `BufferSize 0` but `plugin.DefaultOptions` now sets 1; :61 compares func values with `reflect.DeepEqual`, which is never true for non-nil funcs). Malformed-name functions are not executed at all.
- Correct: no (sync defect).

## TODOs

- [ ] Decide: delete `x/pipeline` (+ `x/plugin`, `x/store`) after dndm migration, or keep and fix — x/pipeline/DEPRECATED.md #p3
- [ ] If kept: `s.inArr[idx] = s.inArr[idx].Add(data)` — x/pipeline/steps/sync.go:169 #p1 #backlog
- [ ] If kept: rename `TestX_y` tests to `TestX_Y`, key `plugin.Options` literals — x/pipeline/steps/*_test.go #p2

## Links

- Module: [[pipeline-and-chain]]; replacement [[dndm-and-protobuf]]; data container [[store-and-plugin]]
