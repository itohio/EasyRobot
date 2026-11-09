---
type: algorithm
status: tested
tags: [easyrobot, audio, wav]
workstream: W7
module: "[[audio-io]]"
summary: "21 pass (value checks, loose)"
source: [x/marshaller/audio/marshaller.go, x/marshaller/audio/unmarshaller.go]
tests: [x/marshaller/audio/marshaller_test.go, x/marshaller/audio/unmarshaller_test.go]
test_result: pass
complete: no
correct: unknown
evidence: "go test -count=1 -v ./x/marshaller/audio/ → ok, 21 PASS 0 SKIP (2026-11-09); offline cmd/audio gen→recv roundtrips decode payload (see [[audio-modem-cli]])"
---
# WAV codec (tensor ↔ WAV file)

**What it does:** writes a `[frames, channels]` tensor to a PCM WAV file and reads WAV (single file, or several files channel-concatenated) back into a tensor, using go-audio/wav.

**Where:**
- Write: `Marshaller.writeSingleWAV` marshaller.go:179-235; per-dtype `writeInt16PCM` :238, `writeInt32PCM` :267, `writeFloat32PCM` :309, `writeUint8PCM` :369.
- Read: `Unmarshaller.readSingleWAV` unmarshaller.go:186-320, `wavFormatToDtype` :323, `fillTensorFromPCM` :341-425, `readMultipleWAV` :428.

**Inputs → outputs:** tensor dtype INT16/INT32/FP32/UINT8 → WAV 16/32/32/8-bit **integer** PCM (bit depth from tensor dtype, marshaller.go:188-201; FP32 written as int ×2147483647, :327). WAV → tensor of the file's native dtype, or the dtype given by `WithDestinationType` (normalised to [-1,1] when FP32 is requested, unmarshaller.go:369-401). Rate/channel options on read are validated, not converted (:242-253).

## Completeness review

- Complete: no — no IEEE-float WAV output (FP32 → int32 PCM); `WithDestinationType` ignored on write (marshaller.go:83-84); float WAV input path divides ints by 2^31-1 with a comment saying it "shouldn't happen" (unmarshaller.go:375-380); no 24-bit write; only one output file; no streaming (whole file buffered into `[]int`, :257-287).
- Tested: 21 tests pass. Write→read roundtrips per dtype assert shape/rate/channels and that ≥80/100 first samples are within quantisation (INT16) or 1% (FP32 read back as INT32 accepted, marshaller_test.go:243). Read tests on non-empty testdata WAVs check dtype/shape, Goertzel 1 kHz amplitude and frequency within 1% (unmarshaller_test.go:410-420), and error cases.
- Correct: unknown/partial — round trips are loose (80% match). Scale asymmetry: write ×32767, read ÷32768 (marshaller.go:319 vs unmarshaller.go:394-401) → −0.00026 dB gain error per cycle, harmless but inconsistent. UINT8 write subtracts 128 then go-audio adds the offset back [INFERENCE, consistent with UINT8 test passing].

## TODOs

- [ ] Honour `WithDestinationType` when writing — x/marshaller/audio/marshaller.go:83-84 #p1 #ER-022
- [ ] Write FP32 as WAV format 3 (IEEE float) — marshaller.go:197-198 #p2
- [ ] Tighten roundtrip tests to all samples / exact for integer dtypes — marshaller_test.go:77 #p2
- [ ] Symmetric scale constants — marshaller.go:319, unmarshaller.go:394-401 #p3
- [ ] Block writes instead of one `encoder.Write` per frame — marshaller.go:243-257 #p3

## Links

- Module: [[audio-io]]; used by [[audio-modem-cli]]; [[marshallers]]
