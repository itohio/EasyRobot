---
type: algorithm
status: unverified
tags: [easyrobot, devices, sensors, colorimetry, protocol, serial]
workstream: W1
module: "[[sensors]]"
summary: "unverified (no device; self-consistent per probe; receive checksum unchecked)"
source: [x/devices/cr30/packet.go, x/devices/cr30/device.go, x/devices/cr30/DESIGN.md]
tests: []
test_result: none
complete: no
correct: unknown
evidence: "no test files; probe tmp/easyrobot/devices/sensors (go run ., 2026-11-09): checksums AA0A0000→0xB4, BB010000→0xBB; fake-device Measure → wl 400..700, SPD 31/31 values match; silent device Connect → nil after 11.1 s; bad-checksum packet accepted; VerifyChecksum false then true. Real CR30 not available."
---
# CR30 colorimeter serial packet protocol

**What it does:** frames 60-byte CR30 packets (start/cmd/subcmd/param/52-byte payload/marker/checksum), runs the handshake (device-info queries + init/check/param queries), triggers or waits for a measurement, pulls SPD chunks 0x10–0x13 and decodes 31 little-endian float32 values (400–700 nm, 10 nm) into a `devices.Spectrometer` matrix.

**Where:**
- `x/devices/cr30/packet.go:9-22` layout constants; `:32-190` `Packet` accessors, `CalculateChecksum` (`:147-162`), `VerifyChecksum` (`:165-169`), `IsValid` (`:172-184`), `Bytes` (`:187-190`).
- `:193-225` `PacketBuilder.BuildPacket/BuildHandshakePacket/BuildCommandPacket`.
- `:228-363` `PacketParser` (`ParsePacket`, `ExtractPayload`, `ParseSPDChunk` `:296-347`, `IsSPDComplete`); `:366-378` `parseFloats`.
- `x/devices/cr30/device.go:20` compile-time `devices.Spectrometer` assertion; `:119-134` `Connect`; `:153-202` `Handshake`; `:235-273` `Measure`/`WaitMeasurement`; `:276-342` `readMeasurementCtx`; `:345-356` `parseSPDFloats`; `:358-388` info parsers; `:391-499` send/recv/flush.

**Inputs → outputs:** `devices.Serial` (19200 8N1 per DESIGN.md; the driver does not open the port). `Measure(ctx, dst)`: `dst` must be a `mat.Matrix` with ≥31 cols; 1 row → SPD, ≥2 rows → row0 wavelengths nm, row1 SPD (device units, no scaling). Packet: byte0 0xAA (handshake) / 0xBB (command); bytes 4–55 payload; 56–57 unused; 58 marker (0xFF for AA, 0x00/0xFF for BB); 59 checksum = Σ bytes 0..57 mod 256, minus 1 for 0xBB.

**Protocol sequence** (as implemented; from DESIGN.md, apparently reverse-engineered — no vendor spec in repo [INFERENCE]):
1. Handshake: `AA 0A 00..03` (name/model, serial, firmware/build), `BB 17` (init), `BB 13` + "Check" payload, `BB 28 00 {00,01,02,03,FF}`.
2. Measure: `BB 01 00 00` → header (subcmd 0x09, not checked in `Measure`), then `BB 01 {10,11,12,13} 00`; chunks 0x10–0x12 contribute payload[2:50] (48 B each = 144 B), first 124 B → 31 floats; chunk 0x13 content discarded.

## Completeness review

- Complete: no. Receive path never checks the checksum (`ParsePacket` packet.go:262-274, `recv` device.go:429); responses are not matched to the request cmd/subcmd (device.go:474-487); chunk 0x13 is ignored (packet.go:335-337); handshake errors are swallowed and `Handshake` always returns nil (device.go:168-201); `verbose` logging is empty (device.go:171-173, 397-399); no calibration command (DESIGN.md open question 4; `cmd/cr30/main.go:219 performCalibration` exists in the app).
- Tested: no test files (`go test ./x/devices/cr30` → `[no test files]`, 2026-11-09). Probe `tmp/easyrobot/devices/sensors/main.go` with a fake serial device that answers with packets built by the same `PacketBuilder`:
  - `BuildHandshakePacket(0x0A,0,0,nil)` → marker 0xFF, checksum 0xB4 (= 0xAA+0x0A); `BuildCommandPacket(0x01,0,0,nil)` → marker 0xFF, checksum 0xBB (= 0xBB+0x01−1). Payload of 53 B → `ErrPayloadTooLarge`.
  - `Connect` on fake → info `{Name:CR30-FAKE Model:MODELX Serial:SN123 Firmware:FW1.0 Build:BUILD1}`; `Measure` into 2×31 → wavelengths 400…700, SPD values equal to the 31 encoded floats (31/31).
  - Silent device (writes swallowed, reads return 0): `Connect` → `err=nil`, `Connected()=true` after 11.1 s.
  - Packet with corrupted checksum: `ParsePacket` → nil error; `VerifyChecksum` → false, second call → true.
  - Marker 0x12 on a 0xBB packet → error text "start byte must be 0xAA or 0xBB".
- Correct: unknown vs a real CR30 (the fake only proves self-consistency of encoder/decoder and the chunk→float layout the code assumes).

**Defects / risks**
- `Handshake` cannot fail → `Connect` reports success with no device attached (device.go:153-202; probe 11.1 s, nil).
- No checksum verification on receive (packet.go:262-274, device.go:429-433); `VerifyChecksum` overwrites the stored checksum, so it is only valid once (packet.go:165-169).
- Wrong error for bad marker: `ParsePacket` returns `ErrInvalidStartByte` (packet.go:270-272); `ErrInvalidMarker` only used by `SetMarker`.
- `Measure` never checks the header subcmd 0x09 (device.go:241-246); a stale/unsolicited packet becomes the "header".
- Chunk errors are dropped silently (device.go:291-301); failure surfaces only as `ErrIncompleteData` (<124 B).
- `flush` loops until `Read` returns 0 (device.go:490-499): fine with the non-blocking Linux serial (`VMIN=0,VTIME=0`, serial_linux.go:86-87) but hangs with a blocking `Read` or a streaming device; ignores `Serial.Buffered()`.
- Type assertions panic for non-`vec.Vector`/`mat.Matrix` destinations (device.go:218, 318).
- Allocation per packet: `make(60)` in recv/recvCtx (device.go:407, 438), `make(1024)` per flush (491), `time.After` per poll (457), maps per parser reset (packet.go:244-246) and per `IsSPDComplete` (356).
- `Disconnect` only clears a flag (device.go:137-140) though DESIGN.md says it closes the connection.
- App build: `go build -o /dev/null ./cmd/cr30` fails on darwin (gocv/opencv4 pkg-config missing; also `NewSerialWithConfig` linux/windows-only, cmd/cr30/main.go:99, baseline F2).

## TODOs

- [ ] Verify checksum on receive (`recv`/`recvCtx`/`ParsePacket`) and make `VerifyChecksum` non-mutating — x/devices/cr30/packet.go:165 #p1 #backlog
- [ ] Make `Handshake` fail when no device answers the name query; propagate to `Connect` — x/devices/cr30/device.go:168 #p1 #backlog
- [ ] Check header subcmd 0x09 in `Measure` and match response cmd/subcmd to request — x/devices/cr30/device.go:241 #p2
- [ ] Unit tests: checksum vectors (AA0A0000→0xB4, BB010000→0xBB), fake-serial measure round trip, bad checksum/marker rejection — x/devices/cr30/packet.go:147 #p2
- [ ] Bound `flush` by `Buffered()` / deadline instead of reading until 0 — x/devices/cr30/device.go:490 #p2
- [ ] Return `ErrInvalidMarker` for marker failures in `ParsePacket` — x/devices/cr30/packet.go:270 #p3
- [ ] Reuse packet/flush buffers; drop per-poll `time.After` — x/devices/cr30/device.go:407 #p3
- [ ] Implement or remove `verbose` logging stubs — x/devices/cr30/device.go:171 #p3
- [ ] Validate SPD layout (payload[2:50] of 0x10–0x12, chunk 0x13 meaning) against a real capture — x/devices/cr30/packet.go:315 #p2

## Links

- Module: [[sensors]] · Consumers: [[colorscience]] (cmd/cr30, cmd/spectrometer) · Interfaces: [[device-io]]
