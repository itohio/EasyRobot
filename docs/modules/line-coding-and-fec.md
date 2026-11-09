---
type: module
group: "Signal, codecs, audio"
status: partial
tags: [easyrobot, dsp, coding, fec, line-code]
workstream: W7
packages: [x/math/dsp/coding/biphase_i, x/math/dsp/coding/biphase_m, x/math/dsp/coding/crc, x/math/dsp/coding/eight10b, x/math/dsp/coding/four5b, x/math/dsp/coding/gray, x/math/dsp/coding/hamming, x/math/dsp/coding/lfsr, x/math/dsp/coding/manchester, x/math/dsp/coding/miller, x/math/dsp/coding/nrzi, x/math/dsp/coding/rs]
test_result: fail
evidence: "go test -count=1 ./x/math/dsp/coding/... → crc FAIL (5), four5b FAIL (1), gray/hamming/manchester/nrzi/rs ok, 6 packages without tests; go vet clean (2026-11-09); probe tmp/easyrobot/coding/probe → probe_out.txt; AMI/B8ZS/HDB3/RLL/PPM placeholders deleted in ER-cleanup"
---
# line-coding-and-fec — `x/math/dsp/coding/*`

**Purpose:** byte-stream line codes, scramblers, error detection and error correction meant to sit between data and a modulator ([[modulation]]) in the audio modem ([[audio-modem-cli]]) and other links. Every coder is a `chain.Processor[byte,byte]` so it can be chained with modulators via `x/chain` ([[pipeline-and-chain]]). Design docs: `x/math/dsp/coding/SPEC.md` (describes the old `ecc.NewXxxEncoder` API — stale), `PLAN.md` (rename to `Encoder/Decoder`, done; mentions a `diff_manchester` package that does not exist).

**Packages:** 12 packages; no build tags, no cgo, no reflection, no goroutines; only `rs/rs_utils.go` uses floats. The wireline and storage placeholders AMI, B8ZS, HDB3, RLL and PPM (1 231 LOC, no tests; probe: identity, lossy, panic, lossy, identity) were deleted in [[ER-cleanup]] (owner-approved, no importers).

| Package | LOC src/test | Algorithm note | Probe result (2026-11-09) |
|---|---|---|---|
| manchester | 611 / 569 | [[manchester-line-code]] | aligned RT ok; odd `SetBitOffset` wrong |
| biphase_m, biphase_i | 321 + 308 / 0 | [[biphase-fm0-differential-manchester]] | RT ok, polarity-proof (same algorithm twice, misnamed) |
| miller | 330 / 0 | [[miller-delay-code]] | RT ok, polarity-proof |
| nrzi | 242 / 94 | [[nrzi-line-code]] | RT ok |
| four5b | 324 / 63 | [[four-b-five-b-code]] | nibble swap |
| eight10b | 366 / 0 | [[eight-b-ten-b-code]] | drops 0xFF, no RD |
| lfsr | 311 / 0 | [[lfsr-additive-scrambler]] | RT ok; 7-bit period 21, 23-bit period 1 |
| gray | 290 / 223 | [[gray-code-mapping]] | widths 4, 8 ok; 1–3, 5–7 lose data |
| crc | 559 / 629 | [[crc-frame-check]] | non-standard CRCs; CRC-8 ≈ 3-bit |
| hamming | 349 / 732 | [[hamming-7-4-code]] | SEC exhaustive ok, no DED |
| rs | 761 / 1621 | [[reed-solomon-code]] | corrects nothing |

## Public API

- Uniform shape: `pkg.NewEncoder()`, `pkg.NewDecoder()` plus variants (`manchester.NewEncoderIEEE8023`, `gray.NewEncoderWithBits(n)`, `lfsr.NewEncoderWithPolynomial(poly, width)`, `crc.NewEncoderCRC8/16/32`, `crc.NewEncoderWith(poly, init, xorOut, width)`, `rs.NewEncoder(n,k)`). Constructors panic on invalid parameters.
- Each type implements `chain.Processor[byte,byte]` (`x/chain/interfaces.go:17-25`): `Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int` and `Reset()`. `Process(nil, src)` is a "size query" that **reads (drains) `src`** to count bytes.
- RS helpers: `rs.CalculateRSParams`, `CalculateCodeRate`, `CalculateOverhead`, `MaxCorrectableErrors`, `CalculateOptimalBlockSize`, `EstimateErrorCorrectionCapability` (heuristic, wrong — see [[reed-solomon-code]]), `CalculateBlockCount/EncodedLength/DecodedLength`.
- Consumers: `cmd/audio` uses `gray(8)`, `hamming`, `rs` (`cmd/audio/common.go:479-532`, `gen_mask.go`, `recv_mask.go`) and `manchester` (mask tools). No other package imports coding.

## Streaming behaviour (what exists)

- All coders pull 256-byte chunks (Manchester decoder: 16) until `io.EOF`, keep state across `Process` calls, and can be fed incrementally. Probe fed every encoded stream one byte per `Process` call: biphase_i/m, manchester(±IEEE), miller, nrzi, eight10b (except 0xFF), lfsr, hamming, gray(4,8) reproduce the input; four5b and gray(other widths) do not.
- Block/frame coders are not streaming-friendly: CRC appends only at EOF of the whole input and validates only at EOF (no frame boundaries); RS keeps a trailing partial block forever and silently skips undecodable blocks.

## Voice/audio-channel suitability (owner's "robust codecs over voice")

Audio paths (speaker→mic, phone/VoIP) are AC-coupled, band-limited, may invert polarity, drift in sample clock, and produce bursty errors; a voice codec (AMR/Opus) further distorts non-speech signals [general background, not measured here].

| Need | In repo, usable now | Needs work |
|---|---|---|
| DC-free + self-clocking chips (baseband over audio) | FM0 biphase (polarity-proof), Miller (polarity-proof, narrower), Manchester (polarity-sensitive) | chip-timing recovery and erasure output are missing in all three |
| Whitening / avoid tones from long runs | lfsr 15-bit preset only | fix 7/23-bit presets; self-synchronising option |
| Neighbour-symbol error mapping (MFSK/PSK/QAM) | Gray width 4 or 8 | width = bits/symbol (2 for 4-FSK/QPSK) is the broken path |
| Bit-error correction | Hamming(7,4) (single bit per 7) | interleaving, SECDED, soft decisions |
| Burst/symbol correction | — | real RS (or convolutional/Viterbi, LDPC) + interleaver; current RS is a stub |
| Frame integrity | — (CRC values non-standard, CRC-8 weak) | fixed CRC-16/CCITT or CRC-32 with framing |
| Not applicable to audio | 4B5B/8B10B (wired SerDes) | — |

[INFERENCE] Part of the "noise figures" disappointment in `cmd/audio` is explained by the coding layer: `--encoding rs` corrects nothing, Hamming without interleaving does not survive bursts, and 8-bit Gray on bytes does not match a 2-bit/symbol modulator. Measured audio BER/SNR curves are owned by [[audio-modem-cli]] / [[W7-filters-codecs-voice]].

## Algorithms

<!-- vault:algorithms h=1426f52bd9b3 -->

Counts: 2 tested · 2 implemented · 6 partial · 1 stub · 0 unverified (11 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[hamming-7-4-code]] | tested | pass | each nibble → 7-bit codeword (parity at positions 1, 2, 4; data at 3, 5, 6, 7), decoder computes a… |
| [[nrzi-line-code]] | tested | pass | differential coding — a `1` toggles the output level, a `0` holds it. |
| [[biphase-fm0-differential-manchester]] | implemented | none | differential biphase: the line level always toggles at the start of each bit cell; a `0` adds a… |
| [[miller-delay-code]] | implemented | none | `1` → transition in mid-cell; `0` → no mid-cell transition, plus a start-of-cell transition only… |
| [[crc-frame-check]] | partial | fail | appends a CRC (little-endian) to the whole message at EOF; the decoder recomputes and compares… |
| [[eight-b-ten-b-code]] | partial | none |  |
| [[four-b-five-b-code]] | partial | fail | each nibble → 5-bit code word from the FDDI/100BASE-TX table (max 3 consecutive zeros, ≥2 ones per… |
| [[gray-code-mapping]] | partial | pass | maps values to binary-reflected Gray code (`g = b ^ (b>>1)`) and back (prefix XOR), so adjacent… |
| [[lfsr-additive-scrambler]] | partial | none | XORs data bits (MSB first) with a Fibonacci LFSR keystream; descrambler runs the identical LFSR. |
| [[manchester-line-code]] | partial | pass | maps every data bit to a two-chip symbol with a guaranteed mid-bit transition: standard (G.E. |
| [[reed-solomon-code]] | stub | pass |  |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Pure Go, std + `x/buffer`, `x/chain`. No floats (except RS helpers), no reflection, no goroutines → TinyGo/ESP32-compatible in principle [INFERENCE: not built with TinyGo].
- Allocation: almost every `Process` does `make([]byte, read)` (or `read*2`) per 256-byte chunk (e.g. `fourb5b_encoder.go:98`, `rs_library.go:24,80`); Hamming allocates per output byte (`hamming_encoder.go:123,130`). On ESP32 this is GC churn in the hot path; fixed-size stack buffers (as Manchester already does, `encoder.go:65`) would remove it.
- Lookup tables: Manchester 64 B, 4B5B 48 B, 8B10B 512 B + 1 KiB, CRC 1 KiB per instance (`[]uint32` of 256, allocated per encoder *and* decoder).

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 ./x/math/dsp/coding/...` (2026-11-09): `crc` FAIL (16 pass / 5 fail — all 5 stale after `NewEncoder()` became CRC-32), `four5b` FAIL (`TestFourB5B_Roundtrip`: decoder nibble swap — implementation bug), `gray` ok (5), `hamming` ok (20), `manchester` ok (15), `nrzi` ok (2), `rs` ok (36); biphase_i, biphase_m, eight10b, lfsr, miller: no test files. Matches baseline-2026-11-08 §4–5. Log: `tmp/easyrobot/coding/go_test_out.txt`.
- `go vet ./x/math/dsp/coding/...` clean.
- Probe `~/src/tmp/easyrobot/coding/probe/main.go` (output `probe_out.txt`): roundtrip, byte-wise streaming, polarity inversion, run length/ones density, CRC catalog check values and random-error detection, Hamming exhaustive single/double errors, RS error trials, LFSR periods, Manchester bit offsets, Gray widths, 8b/10b disparity (the probe also covered the now-deleted B8ZS/HDB3).

## API notes (ease of use, composability)

- Good: one interface for every coder; stateful streaming with `Reset()`; composable with modulators through `x/chain`; consistent constructor naming after PLAN.md.
- `Process(nil, src)` size queries consume the reader, so "ask size, then process" needs two readers (tests construct the source twice). `chain.Adapter.Process` does exactly this on its first call and, with the pointer buffer type used in `cmd/audio`, **panics** before that (nil `*buffer.Values` receiver: probe `bufprobe/adapter` → nil pointer dereference at first `Process`) — owner: [[pipeline-and-chain]].
- The `int` return mixes meanings (bytes written; 0 for CRC failure after data was already written; Manchester encoder returns input×2 even when `dst` accepted less). There is no error return, so read errors other than EOF and decode failures are invisible.
- `dst.Write`'s `n` is ignored everywhere; with a bounded `buffer.Ring` output, overflow is silent data loss ([[generic-ring-buffer]]).
- No erasure/soft-information side channel: line decoders cannot tell FEC which symbols were invalid (Manchester `00/11`, 4B5B invalid codes, 8B10B misses are mapped to 0 or dropped). Erasures would double RS capacity and enable soft Viterbi/LDPC.
- Byte-only interface: codes with 5-, 7- or 10-bit words store one word per byte (Hamming, 4B5B, 8B10B ⇒ 2× expansion instead of 1.75×/1.25×). `buffer.Bits` could carry packed bits but its `Write/Read` return bit counts ([[bit-packing-buffer]]).
- ~100 lines of identical read/EOF boilerplate in each coder file; a shared `mapBytes(dst, src, fn)` helper would remove most of it and the per-chunk allocations.
- Duplicates: `biphase_i` ≡ `biphase_m`; CRC table/CRC code duplicated in encoder and decoder; LFSR encoder ≡ decoder.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Fix 8B10B 0xFF sentinel collision — x/math/dsp/coding/eight10b/eightb10b_tables.go:86 #p1 #ER-023
- [ ] Add a block interleaver processor (depth × width) for burst channels — x/math/dsp/coding/ #p2
- [ ] Add a framing layer (preamble/sync word + length + CRC) for streams — x/math/dsp/coding/ #p2
- [ ] Add erasure/soft output path from line decoders to FEC — x/math/dsp/coding/ #p2
- [ ] Tests for the 5 untested packages (roundtrip + streaming + polarity) — x/math/dsp/coding/ #p2
- [ ] Remove per-chunk `make` in `Process` loops — e.g. x/math/dsp/coding/four5b/fourb5b_encoder.go:98 #p3
- [ ] Merge biphase_i/biphase_m, rename to FM0 — x/math/dsp/coding/biphase_i/ #p3
- [ ] Rewrite SPEC.md for the current API and real statuses; drop `diff_manchester` from PLAN.md — x/math/dsp/coding/SPEC.md, PLAN.md #p3
- [x] Decide fate of AMI/B8ZS/HDB3/RLL/PPM stubs — deleted in [[ER-cleanup]] #p3

## Inferred objectives (proposals)

- **Proposal:** a robust audio data-link stack for W7: `framer(sync+len+CRC-16) → RS(255,223) or conv(7,1/2) → interleaver → Gray(bits/symbol) → MFSK/OFDM`, with erasure/soft hand-off from the demodulator; measure BER vs SNR per layer with the existing `cmd/audio` WAV loopback.
- **Proposal:** shrink the catalog to what serves audio/radio links (FM0, Manchester, NRZI+scrambler, Gray, Hamming/SECDED, CRC, RS, convolutional); the wireline stubs are already gone.
- **Proposal:** introduce a bit-level `Processor[bit]` or fix `buffer.Bits` so codes stop wasting bits on byte padding.
