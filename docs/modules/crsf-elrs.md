---
type: module
group: "Devices and apps"
status: partial
tags: [easyrobot, devices, crsf, elrs, protocol]
workstream: [W6, W3]
packages: [x/devices/crsf, x/devices/crsf/rx, x/devices/crsf/tx, cmd/elrs-rx]
test_result: pass
---
# CRSF / ExpressLRS

**Purpose:** pure-Go TBS Crossfire (CRSF) protocol stack for ELRS/Crossfire radio links: frame codec + CRC, RC channels, telemetry, extended frames (discovery, parameters, commands, baud, timing, MSP), plus `rx` (receiver-side: read sticks, send telemetry) and `tx` (controller-side: send sticks, read telemetry) device wrappers over any `io.Reader`/`io.Writer`. Consumer app `cmd/elrs-rx` maps an ELRS receiver to PWM outputs (embedded) or console (desktop). Relevant as the RC/teleop link for mobile and legged robots ([[W3-legged-ik]], W6 wheeled ESP32) — workstream mapping is a **proposal**.

**Packages:**

| Package | Files (LOC) | Tags |
|---|---|---|
| `x/devices/crsf` | frame.go 263, decoder.go 84, encoder.go 68, channels.go 107, telemetry.go 363, param.go 221, msp.go 174, command.go 80, discovery.go 85, baud.go 54, timing.go 47 (1546 non-test); crsf_test.go 449, advanced_test.go 377; SPEC.md 371, LOW-LEVEL.md 228 | none |
| `x/devices/crsf/rx` | device.go 235, scheduler.go 40; scheduler_test.go 55 | none |
| `x/devices/crsf/tx` | device.go 220; **no tests** | none |
| `cmd/elrs-rx` | main.go 55; platform_desktop.go 53 (`linux \|\| windows`), platform_esp32.go 60 (`esp32`), platform_rp2040.go 60 (`rp2040`), platform_xiao.go 60 (`sam && xiao`) | per platform |

Imports only stdlib (`encoding/binary`, `math`, `io`, `context`, `sync`, `time`); `cmd/elrs-rx` desktop adds `x/devices` serial (see [[device-io]]), embedded adds TinyGo `machine`.

## Public API

- Framing: `Frame{Type, Payload}` with `Marshal`/`Unmarshal`, `Dest()/Origin()/ExtPayload()` for extended types (≥0x28); `CRC8`, `CommandCRC8`; `Address`, `Type` constants.
- Stream: `NewDecoder(io.Reader).ReadFrame() (*Frame, error)`; `NewEncoder(io.Writer)` with `WriteFrame`, zero-alloc `WriteChannels`, `WriteBattery/GPS/Attitude/BaroAltitude/LinkStats/FlightMode/Heartbeat`.
- Payloads: `EncodeX(*X) *Frame` / `DecodeX(*Frame) (*X, error)` for Channels, LinkStatistics(/RX/TX), Battery, GPS, Attitude, BaroAltitude, FlightMode, Heartbeat, Ping/DeviceInfo, ParamEntry/Read/Write, Command, Baud, TimingCorrection, MSP (+ `ChunkMSPRequest/Response`, `ReassembleMSP`).
- `rx.New(ctx, r, w) *Device`: `OnChannels/OnLinkStats/OnFailsafe/OnFrame` setters, `Start`, `Close`, snapshots `Channels()`, `LinkStats()`, `IsFailsafe()`, `SendBattery/GPS/Attitude/FlightMode/SendFrame`, field `Telemetry Scheduler`.
- `tx.New(ctx, r, w) *Device`: `OnBattery/OnGPS/OnAttitude/OnFlightMode/OnLinkStats/OnFrame`, `SendChannels`, `SendFrame`, snapshots `Battery()/GPS()/Attitude()/LinkStats()/FlightMode()`.

Composition: transport (`devices.Serial`, `machine.UART`, `io.Pipe`, file) → `Decoder` → `Frame` → `DecodeX` → typed struct → callback/snapshot; reverse via `EncodeX` → `Encoder`. Unhandled types go to `OnFrame`, so params/commands/MSP are app-level.

## Algorithms

<!-- vault:algorithms h=2674cba532f0 -->

Counts: 3 tested · 0 implemented · 5 partial · 0 stub · 0 unverified (8 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[crsf-frame-codec-and-crc8]] | tested | pass | pass (+ probe: CRC 0xD5/0xBA bitwise 256/256) |
| [[crsf-rc-channels-packing]] | tested | pass | pass (+ probe vs bitwise reference) |
| [[crsf-rx-telemetry-scheduler]] | tested | pass | round-robins registered telemetry providers and returns the first non-nil frame, at most one per… |
| [[crsf-direct-commands]] | partial | pass | pass (round trips) |
| [[crsf-msp-tunnel]] | partial | pass | pass; probe shows >255 B truncation |
| [[crsf-parameter-protocol]] | partial | pass | pass (round trips) |
| [[crsf-rx-failsafe-detection]] | partial | none | marks the link lost when no RC-channels frame has arrived for > 1 s and notifies via callback… |
| [[crsf-telemetry-frames]] | partial | pass | pass (round trips); probe shows vario/RSSI overflow |

<!-- /vault:algorithms -->

Trivial helpers kept here: `tx.Device` dispatch/snapshots (no algorithm), `estimatePercent` in cmd/elrs-rx (linear 9.0–12.6 V → 0–100 %, 3S LiPo assumption, main.go:46-55).

## Build targets and dependencies

- No CGO, no reflection, no build tags in library packages. Builds and vets on darwin/arm64.
- TinyGo suitability (not compiled — TinyGo not run): CRC/pack/`WriteChannels` are allocation-free; `Decoder.ReadFrame` allocates frame + payload per frame (decoder.go:42,51), `UnmarshalChannels` and every `EncodeX` allocate; telemetry uses float32 plus float64 GPS lat/lon and `math.Exp/Log` (soft-float on rp2040/sam). rx/tx use a goroutine, `context`, `sync.RWMutex`, `time` — supported by TinyGo but the decoder's busy loop on `(0,nil)` reads is a starvation risk on the cooperative scheduler [INFERENCE].
- `cmd/elrs-rx`: `GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o /dev/null ./cmd/elrs-rx` → ok; vet (linux) ok. Darwin `go build`/`go vet ./cmd/elrs-rx` → FAIL `undefined: initHardware, getStream, onChannels, onLinkStats, heartbeat, readBatteryVoltage` (gap F3: platform_desktop.go:1 is `linux || windows`; F2: no darwin `NewSerialWithConfig`). Embedded files need TinyGo targets `pico`, `esp32-coreboard-v2`, `xiao` (main.go:5-10) — unverified; the comment `go build -tags linux .` is misleading (GOOS, not a tag).
- Baud: app uses 420000 (platform_desktop.go:26, platform_*.go:13); SPEC.md:82 says 416666 for RX↔FC.

## Tests and evidence (2026-11-09)

All tests read first: pure in-memory (`bytes.Buffer`/`bytes.Reader`), no devices.

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -json ./x/devices/crsf/...` → `crsf` ok 0.22 s, 38 top-level pass (+4 BaroAltitude subtests); `crsf/rx` ok 0.41 s, 4 pass (TestSchedulerEmpty/RoundRobin/AllNil/Len); `crsf/tx` no test files. Matches baseline (38/4).
- `go vet ./x/devices/crsf/...` → ok.
- Probe `tmp/easyrobot/devices/crsf/main.go` (`GOFLAGS=-mod=mod go run .` in scratch module): CRC tables vs bitwise, `CRC8("123456789")=0xBC`, all-992 frame `c81816e0031ff8…7cad`, pack vs reference (0 mismatches), scaling table, decoder resync, telemetry/param/MSP/baud/timing wire dumps. Findings in the algorithm notes.
- Most tests are encode→decode round trips: they cannot catch wire-format errors shared by both directions (param layout, MSP status, endianness, GPS speed scale).

## API notes (ease of use, composability)

- Good: io.Reader/io.Writer everywhere; typed structs; `OnFrame`/`SendFrame` extension points; zero-copy `Unmarshal`; zero-alloc channel send.
- `SPEC.md` drifted from code: `rx.New(ctx, r, w, handler)` vs actual `New(ctx, r, w)` + setters (SPEC.md:269 vs rx/device.go:42); `Frame{Sync, Dest, Origin}`/`FrameType` (SPEC.md:176,248-254) vs `Frame{Type, Payload}`/`Type`; float64 fields vs float32; acceptance criteria all unchecked (SPEC.md:290-296); Non-Goals list params/MSP/baud (SPEC.md:300-302) that are now implemented.
- `rx` routes 0x1C to `DecodeLinkStats`, which rejects it → `LinkStatsRX` frames silently dropped (rx/device.go:168,208; probe: `unexpected frame type`). `tx` does the same for 0x1C/0x1D (tx/device.go:148,210).
- No mutex around the shared `Encoder` (encoder.go:8): rx scheduled telemetry vs user `SendX`, and in tx `SendChannels` vs `SendFrame` from different goroutines race.
- Callback setters unsynchronised; `Close` cannot interrupt a blocked `Read`.
- `Send*` silently return nil when `w == nil` (rx/device.go:105).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Same fix in tx dispatch — x/devices/crsf/tx/device.go:148 #p1 #backlog
- [ ] Add darwin to desktop platform (needs darwin serial, F2) — cmd/elrs-rx/platform_desktop.go:1 #p2
- [ ] Implement battery ADC read on esp32/rp2040/xiao (returns constant 11.1 V) — cmd/elrs-rx/platform_esp32.go:53 #p2
- [ ] Same ADC TODO — cmd/elrs-rx/platform_rp2040.go:53 #p2
- [ ] Same ADC TODO — cmd/elrs-rx/platform_xiao.go:53 #p2
- [ ] Implement `setPWMMicroseconds` (ESP32 LEDC) — PWM outputs are a no-op so the app drives nothing — cmd/elrs-rx/platform_esp32.go:57 #p2
- [ ] Implement `setPWMMicroseconds` (rp2040 machine.PWM) — cmd/elrs-rx/platform_rp2040.go:57 #p2
- [ ] Implement `setPWMMicroseconds` (XIAO TCC) — cmd/elrs-rx/platform_xiao.go:57 #p2
- [ ] Tests for `tx.Device` (pipe-based: send channels, receive telemetry snapshots) — x/devices/crsf/tx/device.go:40 #p2
- [ ] TinyGo build check of cmd/elrs-rx for pico/esp32/xiao — cmd/elrs-rx/main.go:5 #p2
- [ ] Update SPEC.md (constructor, Frame struct, float32, acceptance checklist, non-goals) — x/devices/crsf/SPEC.md:248 #p3
- [ ] Fix build comment `go build -tags linux` and baud 420000 vs SPEC 416666 note — cmd/elrs-rx/main.go:9 #p3

## Inferred objectives (proposals)

- **Proposal:** use `crsf/rx` as the standard RC/teleop input for robot apps (gait selection, speed set-points) via a `devices`-level "RC input" interface, with failsafe feeding a safe-stop in [[motion-and-gait]].
- **Proposal:** route `rx` channels into PWM/servo drivers from [[actuators]] instead of per-board `setPWMMicroseconds` stubs, making `cmd/elrs-rx` the reference app for [[apps]].
- **Proposal:** capture-based golden tests (recorded ELRS/Betaflight UART streams) to validate the extended-frame codecs offline.
