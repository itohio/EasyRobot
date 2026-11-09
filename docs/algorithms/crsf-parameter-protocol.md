---
type: algorithm
status: partial
tags: [easyrobot, devices, crsf, protocol]
workstream: [W6]
module: "[[crsf-elrs]]"
summary: "pass (round trips)"
source: [x/devices/crsf/discovery.go, x/devices/crsf/param.go]
tests: [x/devices/crsf/advanced_test.go]
test_result: pass
complete: no
correct: unknown
evidence: "go test → TestPingRoundTrip, TestDeviceInfoRoundTrip, TestParamReadRoundTrip, TestParamWriteRoundTrip, TestParamEntryChunk0RoundTrip pass (2026-11-09); probe ParamEntry wire = EA EC 01 00 00 00 00 41 00 05 00 0A 01"
---
# CRSF device discovery and parameter protocol

**What it does:** frame codecs for device discovery (Ping 0x28 → DeviceInfo 0x29) and the Lua-style parameter menu (ParamEntry 0x2B, ParamRead 0x2C, ParamWrite 0x2D), plus value extractors for uint8/uint16/text-select entries. No state machine: callers must drive ping → read each index/chunk → reassemble → write.

**Where:**
- `x/devices/crsf/discovery.go:6-15` `DeviceInfo`; `:18-34` `EncodePing`/`DecodePing`; `:37-67` `DecodeDeviceInfo` (NUL name + serial/hw/sw uint32 BE + param count + protocol version); `:70-85` `EncodeDeviceInfo`.
- `x/devices/crsf/param.go:9-24` `ParamType` constants; `:27-33` `ParamAccess`; `:41-71` `ParamEntry`/`ParamRead`/`ParamWrite`; `:74-120` read/write codecs; `:124-158` `DecodeParamEntry`; `:161-187` `EncodeParamEntry`; `:190-221` `ParamValueUint8`/`ParamValueUint16`/`ParamValueTextSelect`.

**Inputs → outputs:** extended frames, payload `[dest][origin]…`. ParamEntry as implemented: `[dest][origin][index][chunks][chunk]` then on chunk 0 `[parent][type&0x7F][name\0][value…]`, later chunks raw value bytes. Value extractors return `ErrParamInvalid` on short data.

## Completeness review

- Complete: no.
  - No chunk splitter for oversized entries and no reassembly helper; `ErrParamChunkIncomplete` declared but unused (param.go:36). LOW-LEVEL.md:164 still lists the "chunked param read/write state machine" as needed.
  - Value extractors only for uint8, uint16, text-select; no int8/int16/int32/float/string/folder/command/info decoding.
  - Hidden flag (type bit 7) is masked away and not exposed (param.go:142); `ParamAccess` constants unused.
  - `DecodeParamEntry` chunk 0 without a NUL terminator treats all remaining bytes as `Name` and returns no error, `Value` nil (param.go:145-152); a name split across chunks is therefore truncated silently.
- Tested: `TestPingRoundTrip`, `TestDeviceInfoRoundTrip` (struct equality), `TestParamReadRoundTrip`, `TestParamWriteRoundTrip`, `TestParamEntryChunk0RoundTrip` (text-select "Rate"). Round trips only; non-zero chunk and extractor functions untested. Pass (2026-11-09).
- Correct: unknown — self-consistent, but interop with ELRS/EdgeTX is doubtful [INFERENCE: comparison with the TBS CRSF spec as recalled; spec not fetched, no hardware]:
  - TBS 0x2B header is `[index][chunks_remaining]` then data; this code adds a separate `Chunk` byte (param.go:134-136, :180-184), shifting every following field by one.
  - TBS type ids are UINT32=4, INT32=5, FLOAT=8, TEXT_SELECTION=9, STRING=10, FOLDER=11, INFO=12, COMMAND=13; code uses 8, 9, 10, 11, 12, 13, 14 and has no INFO (param.go:16-22).
  - Multi-byte numeric values are big-endian in CRSF (SPEC.md:75 itself says "Big-endian unless noted"); `ParamValueUint16` reads little-endian (param.go:202-205).
  - DeviceInfo layout (name, serial, hw, sw, count, version) matches the recalled spec.

## TODOs

- [ ] Verify ParamEntry header layout (extra `Chunk` byte) against TBS spec / ELRS capture and fix — x/devices/crsf/param.go:134 #p1 #backlog
- [ ] Verify `ParamType` numbering (UINT32=4 … COMMAND=13, INFO=12) — x/devices/crsf/param.go:16 #p1 #backlog
- [ ] `ParamValueUint16` endianness vs SPEC.md big-endian rule — x/devices/crsf/param.go:202 #p1 #backlog
- [ ] Chunk splitter + reassembler (use `ErrParamChunkIncomplete`) so rx/tx can serve/consume Lua menus — x/devices/crsf/param.go:36 #p2
- [ ] Tests for non-zero chunks and value extractors — x/devices/crsf/advanced_test.go:125 #p2
- [ ] Expose hidden flag; extractors for int/float/string/folder/command — x/devices/crsf/param.go:142 #p3

## Links

- Module: [[crsf-elrs]] · Framing: [[crsf-frame-codec-and-crc8]] · Commands: [[crsf-direct-commands]]
