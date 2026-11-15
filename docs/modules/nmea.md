---
type: module
group: "Devices and apps"
status: partial
tags: [easyrobot, devices, gps]
workstream: W1
packages: [x/devices/nmea]
test_result: fail
---
# NMEA GPS

**Purpose:** Read a GPS receiver's NMEA 0183 stream from a `devices.Serial` and keep a merged fix (time, position, altitude, quality, speed, course).

**Packages:** `x/devices/nmea` — `parser.go` 162 (framing/checksum, stream parser), `data.go` 264 (GGA/RMC decoding, merge), `device.go` 129 (read loop + callback), `parser_test.go` 159. No build tags, no CGO.

## Public API

`ParseSentence(line) (*Sentence, error)`; `NewParser().ParseBytes([]byte) ([]*Sentence, int, error)`; `ParseGGA/ParseRMC(*Sentence) (*Data, error)`; `MergeData(dst, src *Data)`; `New(ctx, devices.Serial) *Device` with `Start()`, `OnUpdate(func(*Data))`, `Data() *Data` (copy), `Close()`.

Composition: serial → `Parser` → `processSentence` (GGA/RMC only) → `MergeData` under RWMutex → callback with a copy. Positions are `float32` degrees (≈1 m resolution at 48° lat for float32 [INFERENCE: 24-bit mantissa ⇒ ~1e-5° steps]).

## Algorithms

<!-- vault:algorithms h=9af0cd317ff6 -->

Counts: 1 tested · 0 implemented · 1 partial · 0 stub · 0 unverified (2 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[nmea-sentence-framing-and-checksum]] | tested | pass | pass (TestParseSentence, TestParser_ParseBytes, _Incomplete) |
| [[nmea-gga-rmc-decoding]] | partial | fail | RMC 2.0 rejected, century bug, panic on bad coords, altitude wiped |

<!-- /vault:algorithms -->

## Build targets and dependencies

Pure Go, stdlib only (`strings`, `strconv`, `time`). Allocates per sentence and per update (strings, slices, `*Data`); one goroutine; `time.Now()` per sentence. Works on TinyGo in principle; float32 output, float64 parsing.

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 ./x/devices/nmea` (2026-11-09): FAIL — 5 pass, `TestParseRMC` fails "nmea: invalid sentence" (same as baseline §5).
- Probe (synthetic sentences): see algorithm notes.

## API notes (ease of use, composability)

- Simple and readable; copy-on-read avoids races. `Device` mirrors the lidar pattern (ctx, `OnUpdate`), good.
- Friction: `Start()` vs lidar's `Configure(bool)` — inconsistent lifecycle names across devices; `ParseBytes` returns a misleading `consumed`; validity is a few booleans for the whole struct, so merges clobber fields (altitude); no way to feed other talkers or sentence types without editing the switch (a `map[string]func` registry would make it extensible); no conversion to the local ENU frame — belongs in [[gps-geodesy]].

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Accept all talker IDs; add VTG/GSA — x/devices/nmea/device.go:103-111 #p2
- [ ] Unify lifecycle (`Start` vs `Configure`) across device drivers — x/devices/nmea/device.go:63 #p3

## Inferred objectives (proposals)

- **Proposal:** GPS as an outdoor pose prior for the wheeled rover: NMEA → [[gps-geodesy]] (lat/lon → ENU) → [[filters]] EKF fusion with wheel odometry/IMU.

Related: [[device-io]], [[gps-geodesy]], [[filters]]
