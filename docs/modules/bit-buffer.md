---
type: module
group: "Signal, codecs, audio"
status: partial
tags: [easyrobot, buffer, streaming, bits]
workstream: [W2, W7]
packages: [x/buffer]
test_result: pass
evidence: "go test -count=1 ./x/buffer → ok, 89/89 PASS (2026-11-09); go vet ./x/buffer → FAIL stdmethods (bit_buffer.go:144 WriteByte, :210 ReadByte); probed bit buffer, adapter and ring"
---
# bit-buffer — `x/buffer`

**Purpose:** the shared stream containers and reader/writer interfaces of the whole DSP/coding/vision pipeline: `buffer.Reader[T]`/`Writer[T]` are the argument types of every `chain.Processor` ([[pipeline-and-chain]]); `Bits`/`BitRing` carry k-bit symbols between coders ([[line-coding-and-fec]]) and modulators ([[modulation]]); `Pool[T]` recycles slices in [[audio-io]], v4l capture ([[camera-io]]) and primitive kernels ([[primitive-kernels]]).

**Package:** 1 665 LOC non-test (`buffer.go` 265, `ring_buffer.go` 475, `bit_buffer.go` 402, `bit_ring.go` 350, `pool.go` 173), 2 466 LOC tests (89 tests, 16 benchmarks). No build tags. Imported by `cmd/audio`, `x/chain`, all `x/math/dsp/coding/*`, `x/math/dsp/modulation/*`, `x/marshaller/{audio,v4l}`, `x/math/primitive/*`, several `x/vision/*` packages (`buffer.New` ≈ 413 call sites, mostly tests).

## Public API

- Interfaces (`buffer.go:17-84`): `Writer[T]{Write, Len, Cap}`, `Reader[T]{Read, Len, Cap}`, `Initializer{Init(numBits,len,cap), Grow, Cap, Reset}`, `BitsReader/BitsWriter` (+`Bits()`, `BitsLength()`), `Viewer[T]{Items}`, zero-copy `ViewReader/ViewWriter`.
- `Values[T]` (`New(backing)`): growable `bytes.Buffer` analogue — `Write`, `Read` (EOF on short), `WriteItem/ReadItem`, `Items`, `Next`, `Truncate`, `UnreadByte`, `Grow`, `Reset`. Trivial; covered by 12 tests; kept in this note.
- `Ring[T]` → [[generic-ring-buffer]]; `Bits` → [[bit-packing-buffer]]; `BitRing` → [[bit-ring-buffer]]; `Pool[T]` → [[tiered-buffer-pool]].

## Algorithms

<!-- vault:algorithms h=daba9093cfd8 -->

Counts: 2 tested · 0 implemented · 2 partial · 0 stub · 0 unverified (4 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[generic-ring-buffer]] | tested | pass | 9 pass (+ probe for untested views/Grow) |
| [[tiered-buffer-pool]] | tested | pass | 7 pass |
| [[bit-packing-buffer]] | partial | pass | 37 pass; vet stdmethods; Write/Read return bits |
| [[bit-ring-buffer]] | partial | pass | 24 pass; WriteBit counts bits as values |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Std only (`io`, `errors`, `fmt`, `sync`). Generics (Go ≥1.18; TinyGo supports generics) [INFERENCE: not built with TinyGo].
- `Ring` takes a `sync.RWMutex` on every call, including single-item ops — measurable on MCUs; `Values`, `Bits`, `BitRing` are unsynchronised.
- No allocation in steady-state `Ring`/`Values` read/write; `Ring` views allocate `[][]T`; `Pool.Put` boxes slices.

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 ./x/buffer` → ok, 89/89 (2026-11-09; baseline-2026-11-08 also ok).
- `go vet ./x/buffer` → `bit_buffer.go:144:17: method WriteByte(value byte) should have signature WriteByte(byte) error`; `bit_buffer.go:210:17: method ReadByte() (byte, bool) should have signature ReadByte() (byte, error)` (stdmethods; the only vet failure in this slice, matches baseline).
- Probes: `bufprobe` — `Bits.Write` returns 24 for 3 bytes; `io.Copy` into `Bits` → `invalid write result`; shared read/write cursor overwrites unread data; `BitRing` 4×`WriteBit` → `Len()=4`; Manchester encoder into a cap-10 `Ring` returns 16 but ring holds 10; `Pool` floor/ceil tier mismatch; `bufprobe/adapter` — `chain.NewAdapter[..., *buffer.Values[byte]]` panics with nil pointer on first `Process`; `bufprobe/ring` — views, commits and `Grow` correct.

## API notes (ease of use, composability)

- The generic `Reader/Writer` pair mirrors `io.Reader/io.Writer` and lets any element type (bytes, `float32` samples, `complex64`) flow through `x/chain` — good, simple contract.
- **Contract drift across implementations:** `n` means elements for `Values`/`Ring` but bits for `Bits`/`BitRing`; `Len()` means unread elements for `Values`/`Ring`/`BitRing` but backing bytes for `Bits`. Generic code (`chain`, coders) cannot rely on either. Current modulators only test `n == 0`, which is why nothing crashes today (`x/math/dsp/modulation/mask/modulator.go:110-113`).
- Short writes: `Ring.Write` returns `n < len(p)` with `nil` error; callers across the repo ignore `n`. Either make short writes an error or audit callers.
- `io.EOF` doubles as "empty right now" for `Ring`/`BitRing`/`Values`, so a live pipeline cannot tell an underrun from end-of-stream.
- `Bits` single cursor + `Reset()` to read back what was written is a footgun; a FIFO (separate read/write positions, like `BitRing`) is what the coder → modulator hand-off actually needs.
- Naming: `x/chain` docs reference `buffer.NewBitsRing` (`x/chain/adaper.go:32`, `chain.go:27`); the constructor is `NewBitRing`.
- `chain.Streamer` requires `Initializer`, which forces `Init(numBits, len, cap)` on byte-only buffers (`Values.Init` ignores `numBits`) — a leaky abstraction.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Unit tests for `Ring` views/commits/`Grow` — x/buffer/ring_buffer_test.go #p2
- [ ] Fix `NewBitsRing` references in chain docs — x/chain/adaper.go:32, x/chain/chain.go:27 #p3

## Inferred objectives (proposals)

- **Proposal:** freeze a written contract for `Reader/Writer` (n = elements, short write = error, `ErrEmpty` vs `io.EOF`) and add a shared conformance test run against `Values`, `Ring`, `Bits`, `BitRing` — the whole DSP chain depends on it.
- **Proposal:** a lock-free SPSC variant of `Ring` for ISR/DMA → task hand-off on ESP32 (audio, SPI feature stream for [[W4-k230-feature-stream]]).
