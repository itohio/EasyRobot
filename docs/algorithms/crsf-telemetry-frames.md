---
type: algorithm
status: partial
tags: [easyrobot, devices, crsf, protocol, telemetry]
workstream: [W6, W3]
module: "[[crsf-elrs]]"
summary: "pass (round trips); probe shows vario/RSSI overflow"
source: [x/devices/crsf/telemetry.go]
tests: [x/devices/crsf/crsf_test.go, x/devices/crsf/advanced_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test → Battery/GPS/Attitude/LinkStats(RX,TX)/FlightMode/Heartbeat/BaroAltitude(4 subtests) round trips pass (2026-11-09); probe: vario 30 m/s → raw −124 → −24.1 m/s (sign flip), RSSI raw 130 → +126 dBm"
---
# CRSF telemetry frame codecs

**What it does:** typed encode/decode of broadcast telemetry payloads, big-endian, fixed-point → float32/float64 engineering units.

**Where (`x/devices/crsf/telemetry.go`):**

| Frame | Type | Lines | Wire → units |
|---|---|---|---|
| `LinkStatistics` | 0x14 | 9-60 | 10 bytes; RSSI bytes negated to dBm; `RFPowerDBm` enum table :23 |
| `LinkStatsRX` | 0x1C | 64-98 | 5 bytes |
| `LinkStatsTX` | 0x1D | 101-137 | 7 bytes, FPS uint16 BE |
| `Battery` | 0x08 | 140-172 | V·10, A·10 (uint16), 24-bit mAh, % |
| `GPS` | 0x02 | 175-211 | lat/lon int32 ·1e7, speed/heading uint16 ·100, alt uint16 +1000 m, sats |
| `Attitude` | 0x1E | 214-243 | int16 rad·1e4 pitch/roll/yaw |
| `BaroAltitude` | 0x09 | 246-320 | dm+10000 below 0x8000, else 0x8000\|m; vario int8 log `(e^{|v|·0.026}−1)·100` cm/s |
| `FlightMode` | 0x21 | 323-342 | NUL-terminated string |
| `Heartbeat` | 0x0B | 345-363 | origin as uint16 BE |

**Inputs → outputs:** `DecodeX(*Frame) (*X, error)` checks type (`ErrTypeMismatch`) and min length (`ErrFrameTooShort`); `EncodeX(*X) *Frame` allocates the payload. `Encoder.WriteX` wrappers in encoder.go:42-68.

## Completeness review

- Complete: no. `TypeVario` 0x07 and `TypeGPSTime` 0x03 are declared (frame.go:36-37) with no codec although SPEC.md:39 lists a Variometer payload; no clamping on any encode path.
- Tested: `TestBatteryRoundTrip` (±0.15 V/A), `TestGPSRoundTrip` (lat/lon 1e-6, speed/heading 0.02, alt 1 m), `TestAttitudeRoundTrip` (1e-3 rad), `TestLinkStatsRoundTrip`, `TestLinkStatsRXRoundTrip`, `TestLinkStatsTXRoundTrip`, `TestFlightModeRoundTrip`, `TestHeartbeatRoundTrip`, `TestBaroAltitudeRoundTrip` (zero, 50 m/2 m/s, −100 m/−5 m/s, 2500 m/10 m/s), `TestTypeMismatch`. Round-trip only (encoder and decoder share the same scale assumptions); no fixed wire vectors. All pass (2026-11-09).
- Correct: no — confirmed out-of-range defects (probe, darwin/arm64):
  - Vario encode overflows int8 above ≈26 m/s: 30 m/s → raw −124 → decodes −24.1 m/s; 50 m/s → −14.3 m/s (telemetry.go:315; float→int8 out-of-range conversion is implementation-specific in Go).
  - RSSI below −127 dBm wraps: raw 130 → +126 dBm (telemetry.go:34-35,41; same in :81,:119). Raw 128 → −128 by luck.
  - Negative `Speed`/`Heading`/`Voltage`/`Current` or `Altitude < −1000` convert negative float to uint16 (telemetry.go:165-166,206-208) — implementation-specific value.
  - Encode truncates (`uint16(v*10)`), no rounding (12.6 V happened to survive in float32).
  - GPS speed scale ·100 (raw 1250 for 12.5 km/h, telemetry.go:194,206): TBS spec defines ground speed in km/h·10 [INFERENCE: spec recalled, not fetched]; would make interop speeds 10× off. SPEC.md does not state the scale.

## TODOs

- [ ] Clamp vario packing to int8 range before conversion — x/devices/crsf/telemetry.go:315 #p1 #backlog
- [ ] Decode RSSI as `-int(uint8)` (int16/int) to cover −128…−255 dBm — x/devices/crsf/telemetry.go:34 #p1 #backlog
- [ ] Verify GPS ground-speed scale (km/h·10 vs ·100) against TBS spec / ELRS captures — x/devices/crsf/telemetry.go:206 #p1 #backlog
- [ ] Clamp/round all float→unsigned encodes (battery, GPS) — x/devices/crsf/telemetry.go:165 #p2
- [ ] Fixed wire-vector tests from real FC/ELRS captures instead of round trips only — x/devices/crsf/crsf_test.go:203 #p2
- [ ] Add Vario (0x07) and GPS time (0x03) codecs or drop the constants/SPEC claim — x/devices/crsf/frame.go:36 #p3

## Links

- Module: [[crsf-elrs]] · Framing: [[crsf-frame-codec-and-crc8]] · GPS units elsewhere: [[nmea]], [[gps-geodesy]]
