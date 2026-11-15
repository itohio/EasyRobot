---
type: algorithm
status: implemented
tags: [easyrobot, dsp, coding, line-code]
workstream: W7
module: "[[line-coding-and-fec]]"
summary: "differential biphase: the line level always toggles at the start of each bit cell; a `0` adds a…"
source: [x/math/dsp/coding/biphase_m/encoder.go, x/math/dsp/coding/biphase_m/decoder.go, x/math/dsp/coding/biphase_i/biphase_l_encoder.go, x/math/dsp/coding/biphase_i/biphase_l_decoder.go]
tests: []
test_result: none
complete: yes
correct: yes
evidence: "no repo tests; probe (2026-11-09) → both packages round-trip random payloads, byte-by-byte streaming OK, and decode correctly after full polarity inversion"
---
# Biphase-space (FM0) — packages `biphase_m` and `biphase_i`

**What it does:** differential biphase: the line level always toggles at the start of each bit cell; a `0` adds a mid-cell toggle, a `1` does not. This is biphase-space / FM0 (ISO 18000-6 style). Rate 1/2, DC-free, max run 2 chips, polarity-independent.

**Naming defect:** `biphase_m` calls it "Differential Manchester (Biphase-M)" and `biphase_i` calls it "Biphase-L". Neither label matches: Biphase-L is plain Manchester ([[manchester-line-code]]), Biphase-M puts the mid-cell toggle on `1`, and IEEE 802.5 differential Manchester always toggles mid-cell and encodes in the start-of-cell transition. The two packages contain the **same algorithm** with copy-pasted code (`biphase_m/encoder.go:75-113` ≡ `biphase_i/biphase_l_encoder.go:74-110`).

**Where:** encoder loop `biphase_m/encoder.go:82-112` (toggle at :87, mid toggle for 0 at :91-93, chip packing :97-111); decoder `biphase_m/decoder.go:95-140` (mid-transition test). `biphase_i` mirrors it (`biphase_l_encoder.go:80-108`, `biphase_l_decoder.go`).

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1 byte → 2 bytes (16 chips, MSB first). Decoder buffers an odd trailing byte across calls (`buffer` field).

## Completeness review

- Complete: yes for encode/decode. The decoder ignores the start-of-cell transition, so it detects no code violations (lost chance for error detection / resync); decoder state `lastState` is written but never used for decisions.
- Tested: no tests in either package.
- Correct: yes — probe round-trips random 1/7/300/1000-byte payloads plus all-zero/all-one blocks, byte-by-byte streaming, and still decodes after inverting every chip (probe column `rt-after-polarity-inversion = yes`). Max encoded run = 2 chips, 50.0 % ones.
- Voice-channel fit: good candidate for audio (DC-free, polarity-proof, run ≤ 2), but needs chip-timing recovery upstream; the coder works on already-sliced chips.

## TODOs

- [ ] Delete one of the duplicate packages and give the survivor the correct name (FM0 / biphase-space) — biphase_i/, biphase_m/ #p3
- [ ] Add roundtrip, streaming and polarity-inversion tests — biphase_m/ #p2
- [ ] Flag missing start-of-cell transitions as violations (erasure output) — biphase_m/decoder.go:124-130 #p2

## Links

- Module: [[line-coding-and-fec]]
- Workstream: [[W7-filters-codecs-voice]]
