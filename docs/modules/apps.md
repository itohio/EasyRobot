---
type: module
group: "Devices and apps"
status: partial
tags: [easyrobot, apps, devices]
workstream: [W3, W8, W1]
packages: [cmd/elrs-rx, cmd/manipulator, cmd/monitor, cmd/cr30, cmd/spectrometer, cmd/calibrate, cmd/display, drivers/lidar, drivers/manipulator]
test_result: none
---
# Device apps and nodes (cmd/*, drivers/*)

**Purpose:** Executables that wire `x/devices` drivers to users or to the dndm bus: desktop tools (serial protocol monitor, colorimeter/spectrometer, camera calibration, display), host clients, and TinyGo firmware "nodes" (`drivers/README.md`: ROS-style nodes exposing hardware over [[dndm-and-protobuf]]).

## Inventory

| App | LOC (Go) | Tags / deps | What it does | Build here (darwin) |
|---|---|---|---|---|
| `drivers/lidar` | 883 | host `!rp2040 && cgo` + `linux\|\|windows` (GoCV); `rp2040` + `ld06`/`xwpftb` | lidar → `LIDARReading` over dndm; host GoCV polar plot | no: OpenCV missing; rp2040 tag sets have duplicate consts (see [[lidar]]) |
| `drivers/manipulator` | 552 | `sam && xiao` (+`planar`/`dh`), TinyGo; flash via the confirm-first `make flash-manipulator` (the `//go:generate tinygo flash` line was removed in [[ER-cleanup]]) | XIAO servo-arm node: consumes `ManipulatorConfig`/`ManipulatorTarget`, runs `vaj` jerk-limited profiles, drives `servo.ServoArray`, publishes `ManipulatorState` on `easyrobot.manipulator.{raw,planar,dh}` | excluded (TinyGo); planar/dh kinematics are TODO stubs (`kinematics.planar.go:26-55`, `kinematics.dh.go:26-55`) |
| `cmd/manipulator` | 473 | dndm serial (tarm/serial → CGO on darwin) | host client for the node above: sends config/targets, prints state | builds (baseline); no tests. The committed 9.3 MB ELF `cmd/manipulator/manipulator` was removed and gitignored in [[ER-cleanup]] |
| `cmd/elrs-rx` | 288 | `linux\|\|windows` desktop; `esp32`, `rp2040`, `sam&&xiao` | CRSF receiver app — see [[crsf-elrs]] | fails (F3); linux cross-build OK |
| `cmd/monitor` | 685 + SPEC | `x/devices` serial, [[peg-protocol-parser]] | pattern-driven serial sniffer: `-header "AA*L…"` framing, hex dump, CRC check (additive 8/16-bit sums only, `crc.go:84-118`), raw `-capture` mode — the tool for capturing lidar/CR30 protocols | fails (F2); linux/windows cross-build OK |
| `cmd/cr30` | 1035 | GoCV, cobra | CR30 colorimeter measurement UI (spectrum plot, colorscience) — driver in [[sensors]] | no: OpenCV |
| `cmd/spectrometer` | ~3.9 k Go + ~6 k lines of SPEC/PLAN/REVIEW/WORK md | GoCV, cobra, serial | camera spectrometer: calibrate, measure via CR30, freerun; 5 subcommands return "not implemented yet" (`main.go:184-233`) | no: OpenCV + F2 (`ports`, `internal/acquirer`) |
| `cmd/calibrate` | 1906 | GoCV, cobra | mono/stereo camera calibration (`cam-mono`, stereo) — vision side, see [[camera-io]]/[[vision]] | no: OpenCV |
| `cmd/display` | 2.0 k + DESIGN/README | GoCV, cobra | generic source→destination frame/tensor viewer; `destination.IntentDestination` is reused by `drivers/lidar` and `cmd/cr30` for dndm publishing | no: OpenCV |

## Public API

Apps, not libraries. Reused pieces: `cmd/display/destination` (`RegisterAllFlags`, `NewIntentFromFlags`, `IntentDestination.SetLIDARReading`) is imported by `drivers/lidar/main.go:17` and `cmd/cr30` — a `cmd/` package used as a library (should live under `x/`).

## Algorithms

<!-- vault:algorithms h=f56d515d3b10 -->

No algorithm notes belong to this module.

<!-- /vault:algorithms -->

App-level logic only; algorithms live in the driver notes: [[ld06-packet-decoder-and-scan-assembly]], [[xwpftb-frame-decoder-and-scan-assembly]], [[servo-angle-mapping-and-array]], [[xiao-samd21-tcc-pwm]], CR30 protocol in [[sensors]], CRSF in [[crsf-elrs]]. `cmd/monitor` CRC heuristic: treats the last u8/u16 pattern field as a checksum and validates it as a byte sum (`crc.go:30-118`) — would flag LD06 (CRC-8 poly 0x4D) and CRSF (CRC-8 poly 0xD5) packets as bad; only matches additive-sum protocols such as the xwpftb frames.

## Build targets and dependencies

- GoCV/OpenCV: `drivers/lidar` (host), `cmd/cr30`, `cmd/spectrometer`, `cmd/calibrate`, `cmd/display` — not buildable or runnable here; runtime `unverified`.
- Serial desktop apps are linux/windows only (F2/F3); darwin needs a serial implementation ([[device-io]]).
- TinyGo nodes: `drivers/manipulator` (`sam && xiao`), `drivers/lidar` (`rp2040`), `cmd/elrs-rx` (`esp32`/`rp2040`/`sam&&xiao`). None compiled (no TinyGo). `drivers/lidar` rp2040 fails type-check (duplicate consts). dndm `network/serial` imports `github.com/tarm/serial` (`../dndm/network/serial/serial.go:9`), which is unlikely to build on TinyGo bare-metal [INFERENCE] — the nodes' dndm-over-UART transport is unverified.
- `make gen` / `go generate ./...` no longer reaches `tinygo flash`; flashing needs `make flash-manipulator` and a typed `y` ([[ER-cleanup]]).

## Tests and evidence

- No app has tests. `GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o /dev/null` for `cmd/elrs-rx`, `cmd/monitor`, `cmd/manipulator`, `cmd/spectrometer/ports`, `cmd/spectrometer/internal/acquirer` → OK; same for `GOOS=windows` (2026-11-09). darwin results from baseline §4–5 (F2, F3, OpenCV).
- go/types tag check of `drivers/lidar` (`tmp/easyrobot/devices/tagcheck`): `rp2040,ld06` → `targetPoints` redeclared; `rp2040,xwpftb` → `targetPoints`, `maxPoints` redeclared.
- Not run: any binary (all open serial ports, cameras or GUI windows).

## API notes (ease of use, composability)

- Strong pattern: device node = driver + `bus.Producer`/`Consumer` on a named intent; host client mirrors it (`cmd/manipulator` ↔ `drivers/manipulator`). Topic naming is ad hoc (`LIDARReading@lidar.scan` vs `easyrobot.manipulator.raw`) — a naming convention doc would help ([[dndm-and-protobuf]]).
- Each app re-implements router setup (`setupRouter` in `drivers/lidar/main.go:297-312`, `main_rp2040.go:154-215`, `drivers/manipulator/main.go:192-...`, `cmd/manipulator`) — a shared `x/node` helper (serial peer, container endpoint, remote endpoint init) would remove ~60 lines per app.
- Shared code living in `cmd/display/destination` couples device apps to the GoCV display package; moving `IntentDestination` to `x/` would let `drivers/lidar` host mode build without OpenCV when `--display` is off.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [x] Remove committed binary `cmd/manipulator/manipulator` and gitignore it — [[ER-cleanup]] #p3
- [x] Remove `//go:generate tinygo flash` (or guard it) so `make gen` cannot flash hardware — [[ER-cleanup]] #p1 #ER-026
- [ ] Implement planar/DH kinematics hooks in the manipulator node (now stubs) — drivers/manipulator/kinematics.planar.go:26-55; kinematics.dh.go:26-55 #p2
- [ ] Move `cmd/display/destination.IntentDestination` to `x/` and make GoCV display optional in `drivers/lidar` — drivers/lidar/main.go:17,194-216 #p2
- [ ] Fix rp2040 duplicate consts in drivers/lidar — drivers/lidar/main_rp2040.go:41-42 #p1 #backlog
- [ ] Add real CRC options (CRC-8 polys 0x4D/0xD5/0x07, CRC-16 CCITT/Modbus) to `cmd/monitor` — cmd/monitor/crc.go:84-118 #p3
- [ ] Spectrometer subcommands return "not implemented yet" — cmd/spectrometer/main.go:184-233 #p3
- [ ] Shared dndm node bootstrap helper — drivers/lidar/main_rp2040.go:154-215; drivers/manipulator/main.go:192 #p3

## Inferred objectives (proposals)

- **Proposal:** use `cmd/monitor -capture` (linux) to record LD06/XWPFTB/CR30 golden captures into `testdata/` — unblocks decoder tests without hardware in CI.
- **Proposal:** a generic `drivers/<sensor>` node template (TinyGo + host) that the K230/ESP32 feature-stream node ([[W4-k230-feature-stream]]) and lidar node share.

Related: [[device-io]], [[lidar]], [[sensors]], [[crsf-elrs]], [[actuators]], [[dndm-and-protobuf]], [[build-targets]], [[W8-lidar]]
