---
type: algorithm
status: stub
tags: [easyrobot, dsp, coding, fec]
workstream: W7
module: "[[line-coding-and-fec]]"
source: [x/math/dsp/coding/rs/rs_library.go, x/math/dsp/coding/rs/rs_encoder.go, x/math/dsp/coding/rs/rs_decoder.go, x/math/dsp/coding/rs/rs_utils.go]
tests: [x/math/dsp/coding/rs/rs_test.go, x/math/dsp/coding/rs/rs_roundtrip_test.go, x/math/dsp/coding/rs/rs_lossy_test.go, x/math/dsp/coding/rs/rs_streaming_test.go, x/math/dsp/coding/rs/rs_utils_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/math/dsp/coding/rs → ok, 36/36 PASS (2026-11-09) but lossy tests assert only bytesDecoded >= 0; probe: RS(15,11) and RS(255,223) correct 0/200 single-symbol errors (all dropped), two equal-delta errors pass undetected"
---
# Reed–Solomon codec (placeholder)

**What it claims:** systematic RS(n,k) over GF(2^8), n ≤ 255, correcting t = (n−k)/2 symbol errors per block.

**What it actually does:** no Galois-field arithmetic. Parity byte i = XOR of all data bytes XOR (constants i+j+1) (`rs/rs_library.go:21-47`, comments say "placeholder", "not real RS"). Every parity byte therefore carries the same information (the XOR sum of the block). Decoding counts mismatching parity bytes; any data-byte error changes all n−k parity bytes → "too many errors" → block dropped (`:66-123`, :92). The "correction" loop (:96-121) only decrements a counter and never changes data.

**Where:** `Encoder.Process` `rs_encoder.go:77-180` (k-byte blocks, encodes each through `rsEncoderImpl`), `Decoder.Process` `rs_decoder.go:76-219` (block alignment search at :153: tries offsets 0..n−1 until a block "decodes"), helpers `rs_utils.go` (`CalculateRSParams`, `CalculateCodeRate`, `CalculateOverhead`, `MaxCorrectableErrors`, `CalculateOptimalBlockSize`, `EstimateErrorCorrectionCapability` :130, `CalculateBlockCount/EncodedLength/DecodedLength`).

**Inputs → outputs:** `chain.Processor[byte,byte]`; k data bytes → n bytes; decoder n → k (or nothing for a failed block). Constructor panics on n ≤ k, k ≤ 0, n > 255, t < 1.

## Completeness review

- Complete: no — stub. Also: the trailing partial block (< k bytes) is never flushed or padded (probe: 7 bytes into RS(15,11) → 0 bytes out; `rs_encoder.go:164-166` keeps it buffered forever); failed blocks are silently skipped (`rs_decoder.go:163-169`), so output length shrinks with no erasure signal; the offset search can lock onto a wrong alignment because the stub accepts blocks with ≤ t parity mismatches.
- Tested: 36 tests pass, but the lossy tests assert only `bytesDecoded >= 0` (`rs_lossy_test.go:82,113,144,173,254`) and comment that correction is not expected (:80, :252). Clean-channel roundtrips and helper arithmetic are checked.
- Correct: no. Probe (200 trials each): 0 data-symbol errors → 200 ok; 1 error → 0 corrected / 200 dropped; 2 errors → 199 dropped, 1 wrong data passed. Two errors with equal XOR delta → corrupted data accepted as valid. Theory for real RS(255,223): all patterns of ≤ 16 symbol errors corrected.
- `EstimateErrorCorrectionCapability` is a heuristic, not the block success probability Σ_{i≤t} C(n,i)p^i(1−p)^{n−i}: probe gives 0.92 for RS(255,223) at p = 0.01 (binomial ≈ 1.000) and 0.96 at p = 0.1 (binomial ≈ 0.000) — increasing error rate *raises* the estimate in that range.
- Impact: `cmd/audio --encoding rs` (`cmd/audio/common.go:492`, gen/recv_mask.go) uses this stub, so "RS" runs in the audio modem provide error detection-by-dropping only.

## TODOs

- [ ] Implement real RS over GF(256): generator polynomial, syndromes, Berlekamp–Massey, Chien search, Forney; support erasures — rs/rs_library.go:21-123 #p1 #ER-023
- [ ] Flush/pad the final partial block (with length header) — rs/rs_encoder.go:164-176 #p1 #ER-023
- [ ] Replace `EstimateErrorCorrectionCapability` with the binomial tail — rs/rs_utils.go:130-160 #p1 #ER-023
- [ ] Make lossy tests assert corrected output for ≤ t errors and failure for > t — rs/rs_lossy_test.go:53-257 #p2
- [ ] Emit erasure/failed-block info instead of silently skipping; frame sync via marker, not offset search — rs/rs_decoder.go:150-200 #p2
- [ ] Add block interleaving (depth D) for burst errors on audio — rs/ #p2

## Links

- Module: [[line-coding-and-fec]]
- Workstream: [[W7-filters-codecs-voice]]
- Related: [[audio-modem-cli]]
