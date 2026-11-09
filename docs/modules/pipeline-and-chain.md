---
type: module
group: "IO, camera, vision"
status: partial
tags: [easyrobot, chain, streaming]
workstream: [W1, W5, W7]
packages: [x/chain]
test_result: none
evidence: "x/chain: no tests, probe tmp/easyrobot/io/chainprobe (2026-11-09); x/pipeline, x/pipeline/steps and internal/concurrency deleted in ER-cleanup"
---
# pipeline-and-chain — `x/chain`

**Purpose:** synchronous, allocation-aware, generic processor composition, used by the DSP line codes/FEC, the audio modem CLI and all vision processors. The older goroutine-and-channel step graph `x/pipeline` (with `x/pipeline/steps` and `internal/concurrency`) was deprecated ("Pipeline, Plugin, Store will be replaced by dndm") and imported by nothing outside itself. It was deleted in ER-cleanup (evidence: `~/src/docs/evidence/ER-cleanup.md`, outside the repo) (owner-approved), along with its broken timestamp sync and its stale or malformed tests.

**Packages:**

| Package | LOC (non-test/test) | Notes |
|---|---|---|
| `x/chain` | 390 / 0 | `interfaces.go` (Processor, Streamer), `chain.go`, `adaper.go` (sic), `helpers.go`; `README.md`. 74 files import it (`x/math/dsp/coding/*`, `cmd/audio`, `x/vision/*`) |

## Public API

- `chain.Processor[DST,SRC]{ Process(dst buffer.Writer[DST], src buffer.Reader[SRC]) int; Reset() }` — `dst == nil` means "return the output size for this input". `chain.New[T, BUF](factory, procs...) Chain[T,BUF]`, `chain.NewAdapter[DST,MID,SRC,BUF](factory, dst, src)`. `BUF` must be a `buffer.Writer+Reader+Initializer` (e.g. `*buffer.Values[T]`, `*buffer.Ring[T]`, `*buffer.BitsRing`).

## Algorithms

<!-- vault:algorithms h=10ad21162501 -->

Counts: 0 tested · 0 implemented · 1 partial · 0 stub · 0 unverified (1 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[processor-chain-composition]] | partial | none | Adapter nil-deref, wrong size query |

<!-- /vault:algorithms -->

## Build targets and dependencies

- `x/chain`: pure Go generics, no goroutines, no reflection; per-call allocations only in stages. TinyGo-friendly in principle (generics supported by TinyGo ≥0.27 [INFERENCE]); intermediate buffers are allocated once by the factory — good for MCUs.

## Tests and evidence

- `x/chain`: no tests. Probe `~/src/tmp/easyrobot/io/chainprobe/main.go` (`GOFLAGS=-mod=mod go run .`, 2026-11-09): 3× "doubler" chain on 4 bytes → processing 32 (correct), `Process(nil, src)` → 16 (wrong); `Adapter` with `*buffer.Values[byte]` → nil-pointer panic on `Process` and on `Reset`.

## API notes (ease of use, composability)

- **`x/chain` is the composable core.** One interface, generic over sample type, explicit buffers, `Process(nil)` sizing so callers can pre-allocate — good fit for audio and MCU code. Weak spots: the int return conflates "samples written" with errors (no error channel — vision processors return 0 for "not a GoCV tensor", "no keypoints" and "OpenCV failed" alike); the nil-dst sizing convention is non-obvious and currently wrong for chains; `Adapter` handles only two stages (longer heterogeneous chains nest adapters); argument order `NewAdapter(factory, dst, src)` is reverse of data flow and easy to swap (`cmd/audio/common.go:502` passes `mod, encoder`).
- Vision processors use `chain.Processor` with structs as samples (`KeypointsWithImage`, `StereoPair`), so the "sample" can be a whole frame — workable, but the size-query semantics ("number of frames") differ from DSP ("number of samples").
- **No bridge** to `types.FrameStream` (channels) — see [[camera-io]].
- `x/pipeline` (deleted): dndm is the transport that replaces it ([[dndm-and-protobuf]]).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Unit tests for `x/chain` — x/chain #p2
- [ ] Error reporting in `Processor` (e.g. `Process(...) (int, error)` or `Err()`) — x/chain/interfaces.go:17-25 #p3

## Inferred objectives (proposals)

- **Proposal:** treat `x/chain` as the one composition primitive for both DSP and vision; add `chain.Multi`/`Tee` for fan-out and a `FrameStream` source adapter.
- **Proposal ([[W7-filters-codecs-voice]]):** fix `Adapter` before any further voice-codec work — the CLI encoder/decoder path depends on it.
