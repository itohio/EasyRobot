---
type: algorithm
status: partial
tags: [easyrobot, devices, crsf, protocol]
workstream: [W6]
module: "[[crsf-elrs]]"
summary: "pass (round trips)"
source: [x/devices/crsf/command.go, x/devices/crsf/baud.go, x/devices/crsf/timing.go, x/devices/crsf/frame.go]
tests: [x/devices/crsf/advanced_test.go]
test_result: pass
complete: no
correct: unknown
evidence: "go test → TestCommandCRC8, TestCommandRoundTrip, TestCommandCRCValidation, TestBaudProposalRoundTrip, TestTimingCorrectionRoundTrip pass (2026-11-09); probe: CommandCRC8 == bitwise poly 0xBA; baud payload C8 EC 0A 70 60 E3 16 00 42 (LE, no port id); timing payload 00 00 10 A0 0F 00 00 38 FF FF FF (LE)"
---
# CRSF direct commands (0x32), baud negotiation and timing correction (0x3A)

**What it does:** encodes/decodes direct-command frames `[dest][origin][cmdID][subCmd][data…][cmdCRC8]` with the inner CRC-8 (poly 0xBA) over `type+dest+origin+id+sub+data`; builds CRSFv3 baud proposal/confirm commands; encodes/decodes timing-correction ("CRSF Shot") frames.

**Where:** `x/devices/crsf/command.go:4-23` `CommandID`/`SubCommandID` constants; `:37-53` `EncodeCommand`; `:56-80` `DecodeCommand` (validates inner CRC → `ErrCRCMismatch`); CRC itself `x/devices/crsf/frame.go:220-263` `CommandCRC8`. `x/devices/crsf/baud.go:6-15` `BaudRate` constants, `:20-32` `EncodeBaudProposal`, `:36-46` `EncodeBaudConfirm`, `:49-54` `DecodeBaudRate`. `x/devices/crsf/timing.go:5` sub-command 0x10, `:19-36` `DecodeTimingCorrection`, `:39-47` `EncodeTimingCorrection`.

**Inputs → outputs:** `EncodeCommand(*Command) *Frame` (allocates payload + a temp CRC buffer, command.go:39,47); `DecodeCommand` returns `Data` as a sub-slice. Baud: uint32 little-endian in `Data`. Timing: `UpdateInterval`, `Offset` int32 little-endian.

## Completeness review

- Complete: no. No bind/model-select/VTX helpers, no ACK (0xFF) handling; baud proposal lacks the port-id byte that LOW-LEVEL.md:113 says the frame carries and there is no "accepted" response codec.
- Tested: `TestCommandCRC8` (nil → 0, determinism only), `TestCommandRoundTrip`, `TestCommandCRCValidation` (flipped inner CRC rejected), `TestBaudProposalRoundTrip` (1.5 Mbaud), `TestTimingCorrectionRoundTrip` (4000 / −200). Round trips only. Pass (2026-11-09).
- Correct: unknown.
  - CRC: correct — probe shows table == bitwise poly 0xBA for all 256 bytes.
  - Command IDs disagree with the package's own LOW-LEVEL.md: bind is documented as `0x32.0x10.0x01` (LOW-LEVEL.md:79) but `0x10` is named `CmdGeneralBluetooth` and RX commands are `CmdGeneralRX=0x20` (command.go:8,12). [INFERENCE: TBS spec as recalled: 0x10 = Crossfire (bind 0x01, model select 0x05), 0x20 = flow control, OSD = 0x05, 0x09 = LED.]
  - Endianness: baud and timing use little-endian (baud.go:24,38,53; timing.go:33-34,44-45) while SPEC.md:75 states big-endian unless noted; nothing notes the exception.
  - Units: timing comments say µs (timing.go:12-14) while LOW-LEVEL.md:123-124 says LSB = 100 ns.
  - `EncodeBaudProposal` builds a 5-byte buffer with the sub-command at `data[0]` and then discards it (baud.go:21-30) — dead code.

## TODOs

- [ ] Reconcile command IDs (bind = 0x10/0x01 per LOW-LEVEL.md) and fix constants — x/devices/crsf/command.go:8 #p1 #backlog
- [ ] Verify baud/timing endianness and add port-id + accept response to baud codec — x/devices/crsf/baud.go:24 #p1 #backlog
- [ ] Fix timing units (100 ns vs µs) in field docs or conversion — x/devices/crsf/timing.go:12 #p2
- [ ] Known-value test for `CommandCRC8` and a captured bind/baud frame — x/devices/crsf/advanced_test.go:5 #p2
- [ ] Drop dead `data[0]` in `EncodeBaudProposal`; avoid temp CRC buffer alloc in Encode/DecodeCommand — x/devices/crsf/baud.go:21 #p3

## Links

- Module: [[crsf-elrs]] · CRC/framing: [[crsf-frame-codec-and-crc8]] · Params: [[crsf-parameter-protocol]]
