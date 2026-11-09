---
type: algorithm
status: partial
tags: [easyrobot, devices, crsf, protocol, msp]
workstream: [W6]
module: "[[crsf-elrs]]"
summary: "pass; probe shows >255 B truncation"
source: [x/devices/crsf/msp.go]
tests: [x/devices/crsf/advanced_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test → TestMSPFrameRoundTrip, TestMSPStatus, TestMSPChunkAndReassemble (120 B, 3 chunks), TestMSPResponseChunking pass (2026-11-09); probe: 300-byte payload → size byte 44 (len truncated to uint8)"
---
# CRSF MSP tunnel (0x7A/0x7B) chunking and reassembly

**What it does:** wraps MSP messages in extended CRSF frames `[dest][origin][status][data…]`, splitting into ≤57-byte chunks with a status byte (start flag, error flag, 4-bit sequence), and reassembles in-order chunks into `MSPMessage{Function, Payload}`.

**Where:** `x/devices/crsf/msp.go:9-36` `MSPStatus` (bit0 start, bit1 error, bits 7:4 seq) and `NewMSPStatus`; `:39-50` `MSPMessage`/`MSPFrame`; `:53-68` `DecodeMSPFrame`; `:71-83` `EncodeMSPFrame`; `:87` `maxMSPChunkPayload`=57; `:92-109` `ChunkMSPRequest`/`ChunkMSPResponse` (first-chunk header `[fn_lo][fn_hi][size]`); `:111-148` `chunkMSP`/`chunkMSPResp` (duplicated loops); `:151-174` `ReassembleMSP`.

**Inputs → outputs:** `ChunkMSPRequest(dest, origin, *MSPMessage, seq) []*Frame`; `ReassembleMSP([]*MSPFrame) (*MSPMessage, error)` — first frame must have start flag, concatenates payloads, trims to header size.

## Completeness review

- Complete: no. No incremental reassembler (caller must buffer frames), no sequence-continuity or duplicate check (msp.go:156-159), all failures reported as `ErrFrameTooShort` (msp.go:153,162,167), `IsError` never set on non-start chunks (msp.go:140).
- Tested: `TestMSPFrameRoundTrip`, `TestMSPStatus` (seq 7 + flags), `TestMSPChunkAndReassemble` (120-byte payload, start flag only on chunk 0, byte-exact reassembly), `TestMSPResponseChunking` (1 frame, type 0x7B). Pass (2026-11-09).
- Correct: no.
  - Payload length stored as one byte (`byte(len(msg.Payload))`, msp.go:95,105): probe with 300 bytes emits 6 frames but size byte 44 → reassembly returns 44 bytes. Messages > 255 B silently truncated.
  - Interop doubtful [INFERENCE: Betaflight/ELRS MSP-over-CRSF recalled, not fetched]: Betaflight uses status bits seq 3:0, start 0x10, version 6:5, error 0x80 and an MSPv1 body `[size][cmd]…`; this code uses seq 7:4, start 0x01, error 0x02 and `[fn_lo][fn_hi][size]` (msp.go:11-15, :93-95). Self round-trips pass, so the tests cannot detect this.

## TODOs

- [ ] Reject or encode (MSPv2/jumbo) payloads > 255 bytes instead of truncating the size byte — x/devices/crsf/msp.go:95 #p1 #backlog
- [ ] Verify status-byte layout and first-chunk header against Betaflight `msp_over_crsf` / ELRS and fix — x/devices/crsf/msp.go:11 #p1 #backlog
- [ ] Sequence continuity check + streaming reassembler; distinct errors — x/devices/crsf/msp.go:151 #p2
- [ ] Merge duplicated `chunkMSP`/`chunkMSPResp` — x/devices/crsf/msp.go:111 #p3

## Links

- Module: [[crsf-elrs]] · Framing: [[crsf-frame-codec-and-crc8]]
