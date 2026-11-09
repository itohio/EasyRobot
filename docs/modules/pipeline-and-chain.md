---
type: module
group: "IO, camera, vision"
status: partial
tags: [easyrobot, chain, pipeline, streaming, deprecated]
workstream: [W1, W5, W7]
packages: [x/chain, x/pipeline, x/pipeline/steps, x/pipeline/steps/fps, internal/concurrency]
test_result: build-fail
evidence: "x/chain: no tests, probe tmp/easyrobot/io/chainprobe (2026-11-09); go test ./x/pipeline → 17 PASS; ./x/pipeline/steps → build fail (vet), -vet=off → 28 PASS 1 FAIL (2026-11-09)"
---
# pipeline-and-chain — `x/chain` (live) and `x/pipeline` (deprecated)

**Purpose:** two generations of stream composition. `x/chain` is the current synchronous, allocation-aware, generic processor composition used by DSP line codes/FEC, the audio modem CLI and all vision processors. `x/pipeline` is the older goroutine-and-channel step graph over `store.Store` maps, marked deprecated ("Pipeline, Plugin, Store will be replaced by dndm", `x/pipeline/DEPRECATED.md`) and imported by nothing outside itself.

**Packages:**

| Package | LOC (non-test/test) | Notes |
|---|---|---|
| `x/chain` | 390 / 0 | `interfaces.go` (Processor, Streamer), `chain.go`, `adaper.go` (sic), `helpers.go`; `README.md`. 74 files import it (`x/math/dsp/coding/*`, `cmd/audio`, `x/vision/*`) |
| `x/pipeline` | 252 / 627 | `Pipeline`, `Step`, registry; `SPEC.md`; examples `example.step`, `example.processor`, `example.processor.embed` |
| `x/pipeline/steps` | 998 / 1930 | source, processor, sink, fan-in, fan-out, join, sync |
| `x/pipeline/steps/fps` | 122 / 0 | FPS counter step |
| `internal/concurrency` | 15 / 0 | `Submit(f)` = `go f()`; `ants` tag variant is a FIXME with the same body (`concurrency_ants.go:5-7`) |

## Public API

- `chain.Processor[DST,SRC]{ Process(dst buffer.Writer[DST], src buffer.Reader[SRC]) int; Reset() }` — `dst == nil` means "return the output size for this input". `chain.New[T, BUF](factory, procs...) Chain[T,BUF]`, `chain.NewAdapter[DST,MID,SRC,BUF](factory, dst, src)`. `BUF` must be a `buffer.Writer+Reader+Initializer` (e.g. `*buffer.Values[T]`, `*buffer.Ring[T]`, `*buffer.BitsRing`).
- `pipeline.New()`, `AddStep`, `ConnectSteps(steps...) (<-chan Data, error)`, `Run(ctx)`, `Reset()`; steps built from `plugin.Global` by name with `options.Option` funcs.

## Algorithms

<!-- vault:algorithms h=bd05cdc0619c -->

Counts: 0 tested · 0 implemented · 2 partial · 0 stub · 0 unverified (2 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[pipeline-step-graph]] | partial | build-fail | Timestamp sync broken |
| [[processor-chain-composition]] | partial | none | Adapter nil-deref, wrong size query |

<!-- /vault:algorithms -->

## Build targets and dependencies

- `x/chain`: pure Go generics, no goroutines, no reflection; per-call allocations only in stages. TinyGo-friendly in principle (generics supported by TinyGo ≥0.27 [INFERENCE]); intermediate buffers are allocated once by the factory — good for MCUs.
- `x/pipeline`: goroutine per step, channels, `store.Store` maps of `interface{}`, zerolog — not suitable for MCUs.

## Tests and evidence

- `x/chain`: no tests. Probe `~/src/tmp/easyrobot/io/chainprobe/main.go` (`GOFLAGS=-mod=mod go run .`, 2026-11-09): 3× "doubler" chain on 4 bytes → processing 32 (correct), `Process(nil, src)` → 16 (wrong); `Adapter` with `*buffer.Values[byte]` → nil-pointer panic on `Process` and on `Reset`.
- `go test -count=1 ./x/pipeline` → ok, 17 PASS.
- `go test -count=1 ./x/pipeline/steps` → **build failed**: vet `tests` analyzer: 30+ malformed names `Testfanin_In` … `Testsyncronize_doSyncronize` (`fan.in_test.go:48`, `fan.out_test.go:48`, `join_test.go:49`, `processor_test.go:324`, `sink_test.go:333`, `source_test.go:121`, `sync_test.go:13`), plus vet `composites`: unkeyed `plugin.Options` literals `sink_test.go:33-38`. `go test` runs this vet subset automatically, so the package fails before any test runs.
- `go test -count=1 -vet=off ./x/pipeline/steps` → 28 PASS, 1 FAIL `TestOptions` (`sink_test.go:55`: `BufferSize` 1 vs expected 0 after the `plugin.DefaultOptions` change; `:61` compares function values with `reflect.DeepEqual`). Malformed-name functions are silently not executed.

## API notes (ease of use, composability)

- **`x/chain` is the composable core.** One interface, generic over sample type, explicit buffers, `Process(nil)` sizing so callers can pre-allocate — good fit for audio and MCU code. Weak spots: the int return conflates "samples written" with errors (no error channel — vision processors return 0 for "not a GoCV tensor", "no keypoints" and "OpenCV failed" alike); the nil-dst sizing convention is non-obvious and currently wrong for chains; `Adapter` handles only two stages (longer heterogeneous chains nest adapters); argument order `NewAdapter(factory, dst, src)` is reverse of data flow and easy to swap (`cmd/audio/common.go:502` passes `mod, encoder`).
- Vision processors use `chain.Processor` with structs as samples (`KeypointsWithImage`, `StereoPair`), so the "sample" can be a whole frame — workable, but the size-query semantics ("number of frames") differ from DSP ("number of samples").
- **No bridge** to `types.FrameStream` (channels) — see [[camera-io]].
- `x/pipeline` (deprecated): steps configured via untyped `options.Option func(interface{})` with silent type-assertion no-ops; `store.Store` keys are `uint16` codes with collisions (see [[store-and-plugin]]); data passes as `interface{}`. dndm is the stated replacement ([[dndm-and-protobuf]]).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Unit tests for `x/chain` — x/chain #p2
- [ ] Error reporting in `Processor` (e.g. `Process(...) (int, error)` or `Err()`) — x/chain/interfaces.go:17-25 #p3

## Inferred objectives (proposals)

- **Proposal:** treat `x/chain` as the one composition primitive for both DSP and vision; add `chain.Multi`/`Tee` for fan-out and a `FrameStream` source adapter, and move remaining `x/pipeline` users (none today) to dndm.
- **Proposal ([[W7-filters-codecs-voice]]):** fix `Adapter` before any further voice-codec work — the CLI encoder/decoder path depends on it.
