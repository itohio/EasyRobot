---
type: algorithm
status: tested
tags: [easyrobot, dsp, coding, line-code]
workstream: W7
module: "[[line-coding-and-fec]]"
summary: "differential coding — a `1` toggles the output level, a `0` holds it."
source: [x/math/dsp/coding/nrzi/nrzi_encoder.go, x/math/dsp/coding/nrzi/nrzi_decoder.go]
tests: [x/math/dsp/coding/nrzi/nrzi_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/math/dsp/coding/nrzi → ok, 2/2 PASS (2026-11-09); probe roundtrip + byte-by-byte streaming OK"
---
# NRZI (non-return-to-zero inverted, mark)

**What it does:** differential coding — a `1` toggles the output level, a `0` holds it. Rate 1, removes absolute polarity dependence (except the first bit, which is referenced to the initial state `lastState=false`).

**Where:** `nrzi/nrzi_encoder.go:81-92` (toggle on 1), `nrzi/nrzi_decoder.go:80-88` (transition ⇒ 1). State persists across `Process` calls; `Reset()` restores low.

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1:1 bytes, bits MSB first.

## Completeness review

- Complete: yes.
- Tested: `TestNRZI_Roundtrip` (roundtrip on several payloads), `TestNRZI_Transitions` (exact output for known inputs); both pass.
- Correct: yes. Probe: roundtrip OK for random and edge payloads and byte-wise streaming. After full inversion only the first bit flips (probe reports "no" because of that single bit). No run-length limit: long `0` runs give constant level (probe max run 512 bits for all-zero input) — pair with a scrambler ([[lfsr-additive-scrambler]]) or use FM0 for audio.

## TODOs

- [ ] Document/offer configurable initial level so the first bit is polarity-safe — nrzi/nrzi_encoder.go:20-25 #p3

## Links

- Module: [[line-coding-and-fec]]
