---
type: algorithm
status: partial
tags: [easyrobot, chain, streaming, composition]
workstream: [W7, W5]
module: "[[pipeline-and-chain]]"
summary: "Adapter nil-deref, wrong size query"
source: [x/chain/chain.go, x/chain/adaper.go, x/chain/interfaces.go, x/chain/helpers.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe tmp/easyrobot/io/chainprobe (go run ., 2026-11-09): Chain size estimate 16 for 3×doubler on 4 samples (expected 32); Adapter with *buffer.Values panics nil-deref on Process and on Reset"
---
# Processor chain and adapter (`x/chain`)

**What it does:** generic synchronous composition of `Processor[DST,SRC]` stages (`Process(dst Writer[DST], src Reader[SRC]) int`, `Reset()`) over `x/buffer` readers/writers. `Chain[T,BUF]` links N same-typed stages through intermediate buffers made by a user factory; `Adapter[DST,MID,SRC,BUF]` links two stages with different types (e.g. bytes → FEC encoder → modulator → float32 samples). `Process(nil, src)` is the size-query convention (theoretical output length). Used by every line code / FEC in `x/math/dsp/coding` (74 importing files), `cmd/audio`, and all `x/vision` processors.

**Where:** `Processor`/`Streamer` `interfaces.go:9-25`; `Chain` `chain.go:30-34`, `New` :60-71, `Process` :88-150 (size query :91-115, buffer init :121-123, stage loop :129-146), `initializeBuffers` :153-185, `Reset` :188-196; `Adapter` `adaper.go:35-40`, `NewAdapter` :66-79, `Process` :97-133, `Reset` :135-141; `sizeReader` `helpers.go:7-28`.

## Completeness review

- Complete: no.
  - **Adapter panics on first use with pointer buffers**: `NewAdapter` never allocates `buf`; `Process` calls `c.buf.Cap()` on the zero value (`adaper.go:111`) and `Reset` does the same (:138). For `BUF = *buffer.Values[T]` (what `cmd/audio/common.go:502, :542` uses) that is a nil-pointer dereference (`x/buffer/buffer.go:171-172`). Any `cmd/audio` run with a non-`none` encoding/decoding takes this path [INFERENCE from code + probe; CLI not run]. Independently reproduced by the DSP coding slice (`~/src/tmp/easyrobot/coding/bufprobe/adapter/`).
  - **Size query skips stage 0 when `src` has `Len()`**: `chain.go:101-110` starts from `src.Len()` and only propagates through `processors[1:]`, so the first stage's expansion/compression is ignored (probe: 3 doublers on 4 samples → 16, expected 32). `initializeBuffers` (:153-185) does it right, so real processing works, but callers sizing `dst` from `Process(nil, …)` under-allocate.
  - Buffers are sized once (:121-123); a later larger input relies on the buffer's own growth. The doc says "call Process with nil dst to recalculate", but the nil path never resets `c.buffers`.
  - A stage that writes 0 (e.g. decoder waiting for more input) aborts the whole chain for that call (:138-141) — fine for stateless stages; stateful stages must buffer internally.
  - Not concurrency-safe; one call per chain at a time.
- Tested: **no tests in `x/chain`**. Exercised indirectly by DSP coding tests that build chains [see [[line-coding-and-fec]]].
- Correct: no (two probe-confirmed defects above). Probe source: `~/src/tmp/easyrobot/io/chainprobe/main.go` — real processing through the chain produced 32 samples (correct), size query 16 (wrong), adapter `runtime error: invalid memory address or nil pointer dereference` on `Process` and on `Reset`.

## TODOs

- [ ] Allocate `Adapter.buf` lazily with a nil/zero check that works for pointer BUF (track `initialized bool`) — x/chain/adaper.go:111,138 #p1 #ER-022
- [ ] Propagate size through stage 0 in the nil-dst path — x/chain/chain.go:101-110 #p1 #ER-022
- [ ] Unit tests for Chain/Adapter (size query, multi-stage, Reset before Process) — x/chain #p2
- [ ] Rename `adaper.go` → `adapter.go` — x/chain/adaper.go #p3

## Links

- Module: [[pipeline-and-chain]]; users [[line-coding-and-fec]], [[modulation]], [[audio-modem-cli]], [[vision]]; buffers [[bit-buffer]]
