---
type: algorithm
status: tested
tags: [easyrobot, devices, crsf, protocol]
workstream: [W6, W3]
module: "[[crsf-elrs]]"
summary: "pass (+ probe: CRC 0xD5/0xBA bitwise 256/256)"
source: [x/devices/crsf/frame.go, x/devices/crsf/decoder.go, x/devices/crsf/encoder.go]
tests: [x/devices/crsf/crsf_test.go, x/devices/crsf/advanced_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 -json ./x/devices/crsf/... → 38 top-level pass (2026-11-09); probe CRC8 table == bitwise poly 0xD5 256/256, CRC8(\"123456789\")=0xBC, CommandCRC8 == bitwise 0xBA 256/256, resync over 0x00 noise + bad-CRC false sync ok"
---
# CRSF frame codec, CRC8 and stream decoder

**What it does:** serialises/parses CRSF frames `[sync][len][type][payload…][crc8]`, computes the frame CRC-8 (poly 0xD5, MSB-first, init 0, no xorout = CRC-8/DVB-S2) and the direct-command CRC-8 (poly 0xBA), and recovers frames from a byte stream by sync scanning with 1-byte slip on any framing error.

**Where:**
- `x/devices/crsf/frame.go:5-11` constants (`MaxFrameSize`=64, `MaxPayload`=60, `SyncByte`=0xC8); `:62-64` `Type.IsExtended` (type ≥ 0x28); `:84-107` `Dest`/`Origin`/`ExtPayload`.
- `:111-127` `(*Frame).Marshal`, `:132-162` `(*Frame).Unmarshal` (zero-copy payload sub-slice), `:164-172` `isDeviceAddress`.
- `:175-217` `crc8tab` + `CRC8`; `:220-263` `commandCRC8tab` + `CommandCRC8`.
- `x/devices/crsf/decoder.go:20-57` `(*Decoder).ReadFrame`, `:59-71` `ensureBytes`, `:73-79` `advance`, `:81-83` `isSyncByte`.
- `x/devices/crsf/encoder.go:16-23` `WriteFrame`, `:26-40` `WriteChannels` (frame built in place, no allocation).

**Inputs → outputs:** `Marshal(buf []byte) (n int, err)`, needs `len(buf) ≥ 4+len(Payload)`, payload ≤ 60 else `ErrPayloadTooLarge`. `Unmarshal(buf) (consumed int, err)`: first byte must be 0xC8 **or any known device address** (incl. 0x00), `len` ∈ [2,62], CRC over `type+payload`. `Decoder.ReadFrame() (*Frame, error)` returns a frame owning a copied payload; returns the reader's error (e.g. `io.EOF`) only when it cannot fill the bytes it needs.

## Completeness review

- Complete: yes for framing. Extended-frame dest/origin are exposed as payload prefix accessors only; routing by address is left to callers (`LOW-LEVEL.md:162`). `ErrFrameTooLong` (frame.go:69) is declared but never returned.
- Tested: `TestCRC8Empty`, `TestCRC8SingleByte` (CRC8(0x01)=0xD5 — the only independent CRC value), `TestCRC8KnownVector` (tautological: compares `Marshal` against `CRC8` itself, crsf_test.go:9-26), `TestFrameMarshalUnmarshal`, `TestFrameCRCReject`, `TestFrameInvalidSync`, `TestFrameInvalidLength`, `TestDecoderStream` (3 garbage bytes then two frames), `TestEncoderDecoder`, `TestCommandCRC8` (only nil→0 and determinism, advanced_test.go:5-14), `TestCommandCRCValidation`. All pass (2026-11-09).
- Correct: yes. Probe: table CRC8 equals an independent bitwise poly-0xD5 CRC for all 256 single bytes and a 61-byte buffer; `CRC8("123456789")=0xBC`, the published CRC-8/DVB-S2 check value [INFERENCE: catalogue value recalled, not fetched]; CommandCRC8 equals bitwise poly 0xBA (256/256). All-992 channels frame encodes to `c8 18 16 e0 03 1f f8 … 7c ad`, CRC 0xAD [INFERENCE: matches the commonly cited CRSF example frame from memory]. Decoder recovers a frame after `00 00 00` noise and a bad-CRC false sync starting with 0xEC; accepts a frame whose first byte is 0xEE; truncated stream → `io.EOF`.

**Defects / risks**
- `Decoder.ensureBytes` loops forever when the reader returns `(0, nil)` (decoder.go:60-70); the Linux serial `Read` returns exactly that on `EAGAIN` (x/devices/serial_linux.go:178-180) and Windows sets immediate-return timeouts (serial_windows.go:82-88) → busy spin, and the caller never sees an error (breaks rx failsafe, see [[crsf-rx-failsafe-detection]]).
- Sync accepts 0x00 and every device address (frame.go:136, decoder.go:82): more false-sync candidates on noisy lines; each costs a wait for up to 64 bytes before CRC rejection (latency, not data loss).
- `ReadFrame` allocates `*Frame` + payload copy per frame (decoder.go:42,51) — GC pressure on TinyGo at 500 Hz.
- `Marshal` reports a short destination buffer as `ErrFrameTooShort` (frame.go:118) — misleading name.
- `Encoder` reuses one `buf` without locking (encoder.go:8); concurrent writers race (see module note).

## TODOs

- [ ] Handle `(0, nil)` reads in `Decoder.ensureBytes` (yield/sleep or return a sentinel) so callers regain control — x/devices/crsf/decoder.go:60 #p1 #backlog
- [ ] Replace tautological `TestCRC8KnownVector` with a fixed vector (e.g. all-992 frame CRC 0xAD, `CRC8("123456789")=0xBC`) and add a CommandCRC8 known value — x/devices/crsf/crsf_test.go:9 #p2
- [ ] Zero-alloc `ReadFrameInto(*Frame)` variant for TinyGo hot path — x/devices/crsf/decoder.go:42 #p2
- [ ] Remove unused `ErrFrameTooLong` or use it; return a buffer-size error from `Marshal` — x/devices/crsf/frame.go:69 #p3

## Links

- Module: [[crsf-elrs]] · Related: [[crsf-rc-channels-packing]], [[crsf-direct-commands]] · Interfaces: [[device-io]]
