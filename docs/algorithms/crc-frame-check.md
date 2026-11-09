---
type: algorithm
status: partial
tags: [easyrobot, dsp, coding, error-detection]
workstream: W7
module: "[[line-coding-and-fec]]"
summary: "appends a CRC (little-endian) to the whole message at EOF; the decoder recomputes and compares…"
source: [x/math/dsp/coding/crc/crc_encoder.go, x/math/dsp/coding/crc/crc_decoder.go]
tests: [x/math/dsp/coding/crc/crc_test.go, x/math/dsp/coding/crc/crc_roundtrip_test.go, x/math/dsp/coding/crc/crc_lossy_test.go]
test_result: fail
complete: no
correct: no
evidence: "go test -count=1 ./x/math/dsp/coding/crc → FAIL, 16 PASS / 5 FAIL top-level (2026-11-09); probe: check values for \"123456789\" match no catalog CRC; CRC-8 misses 23.6 % of 2-bit errors"
---
# CRC frame check (CRC-8 / CRC-16 / CRC-32)

**What it does:** appends a CRC (little-endian) to the whole message at EOF; the decoder recomputes and compares, returning 0 on mismatch. Detection only.

**Where:** constants `crc/crc_encoder.go:23-44`; table generation `generateTable` :86-153 (bitwise **right-shift / reflected** algorithm, `crc = (crc>>1) ^ poly`, first at :100); `calculateCRC` :226-241; `Encoder.Process` :156-224 (buffers all input, appends CRC once). Decoder `crc_decoder.go:138-262` (holds back the last `width/8` bytes, writes data as it goes, validates at EOF), duplicate table/CRC code in the decoder.

**Inputs → outputs:** `chain.Processor[byte,byte]`; message → message‖CRC (1/2/4 bytes LE). `NewEncoder()` = CRC-32; `NewEncoderCRC8/16/32`; `NewEncoderWith(poly, init, xorOut, width ∈ {8,16,32})`.

## Completeness review

- Complete: no — no `refin/refout` parameters, no frame boundaries other than EOF (a continuous stream never gets a CRC), corrupted data is already written to `dst` before the decoder reports failure (`crc_decoder.go:215`, return 0 at :239-257), empty input: encoder reports `crcBytes` written but writes nothing (`crc_encoder.go:212-224`).
- Tested: 21 top-level tests, 5 fail. **All 5 failures are stale tests, not implementation faults:** `TestCRCEncoder_NewEncoderCRC8/16`, `TestCRCDecoder_NewCRCDecoder` call `NewEncoder()`/`NewDecoder()` (now CRC-32) and expect width 8/16 (`crc_test.go:11-21,94-105`); `TestCRCEncoder_Process_SizeCalculation`, `TestCRCDecoder_Process_SizeCalculation` expect CRC-8 sizes from the CRC-32 default (`crc_test.go:71-91,127-147`). Passing roundtrip tests check data preservation. `TestCRC_LossyRoundtrip_InvalidCRC_DataCorrupted` accepts an undetected error and logs "placeholder implementation" (`crc_lossy_test.go:64-70`) — it hides the defect below.
- Correct: **no.** The polynomial constants are in normal (MSB-first) form but are fed to the reflected (LSB-first) algorithm. That computes a CRC whose generator is x^w + bitreverse(poly), which here is divisible by x^k:
  - CRC-8 `0x07` → generator x^8+x^7+x^6+x^5 = x^5(x+1)^3: effectively a 3-bit check. Probe: 4 715/20 000 (23.6 %) random 2-bit errors in a 32-byte frame undetected (theory 25 %: (x+1)^3 divides x^i+x^j iff 4 | (j−i)); odd error counts are all caught.
  - CRC-16 `0x1021` → x^3·H(x), H of degree 13. Probe: 7/20 000 4-bit errors undetected; real CRC-16/CCITT at this length detects all ≤3-bit errors and all bursts ≤16 — burst capability here is reduced to 13 bits [derived].
  - CRC-32 `0x04C11DB7` → x^5·H(x). Probe found no undetected errors in 20 000 trials per class (expected; 2^-27 floor) but values are non-standard.
  - Check values for ASCII "123456789": CRC-8 → `0x02` (CRC-8/SMBUS = `0xF4`), CRC-16 → `0x1DBA` (CCITT-FALSE `0x29B1`, MCRF4XX `0x6F91`, KERMIT `0x2189`), CRC-32 → `0xFC4F2BE9` (ISO-HDLC `0xCBF43926`). Not interoperable with any standard.
- Fix is one line per preset: use reflected constants (`0xE0`, `0x8408`, `0xEDB88320`) with the current algorithm, or switch to MSB-first for CCITT-FALSE.

## TODOs

- [ ] Use reflected polynomial constants (0xEDB88320 / 0x8408 / 0xE0) or an MSB-first table; add catalog check-value tests ("123456789") — crc/crc_encoder.go:23-44,86-153 #p1 #ER-023
- [ ] Fix stale tests to call `NewEncoderCRC8/16` and expect matching sizes; remove "placeholder" acceptance in lossy test — crc/crc_test.go:11-21,71-91,93-105,127-147; crc_lossy_test.go:64-70 #p2
- [ ] Do not emit data before CRC validation (or provide a frame-level API returning `([]byte, ok)`) — crc/crc_decoder.go:205-262 #p2
- [ ] Add frame delimiting (length or delimiter) so CRC works on continuous streams — crc/crc_encoder.go:156-224 #p2
- [ ] Deduplicate table code between encoder and decoder — crc/crc_decoder.go #p3

## Links

- Module: [[line-coding-and-fec]]
- Workstream: [[W7-filters-codecs-voice]]
