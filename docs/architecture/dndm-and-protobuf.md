---
type: architecture
status: active
tags: [easyrobot, architecture, dndm, protobuf]
workstream: [W1, W4]
source: [go.mod, proto/buf.yaml, proto/buf.gen.yaml, proto/buf.lock, cmd/manipulator/main.go, cmd/display/destination/intent.go, cmd/display/source/interest.go, drivers/lidar/main.go, drivers/lidar/main_rp2040.go, drivers/manipulator/main.go, x/marshaller/proto/convert.go, x/marshaller/gocv/sink.go]
evidence: "grep -rn itohio/dndm --include=*.go (2026-11-09); probes ~/src/tmp/easyrobot/arch/{protoprobe,regprobe} run with GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod go run . (2026-11-09)"
---
# dndm and protobuf

How EasyRobot uses [dndm](../../../dndm) (Decentralised Named Data Mesh) and its protobuf schemas. Layering: [[overview]]. Where messages travel: [[data-flow]]. Build constraints: [[build-targets]].

## dndm dependency

- `go.mod`: `require github.com/itohio/dndm v0.0.0` + `replace github.com/itohio/dndm => ../dndm`. The build uses whatever is checked out in `../dndm`: HEAD `c7b5b61` at baseline time, and on 2026-11-09 that tree had uncommitted edits in 25 files (`git -C ../dndm status --short`, `git diff --stat`), so EasyRobot's dndm-dependent build follows in-flight dndm work. Pinning is an open owner decision (`~/src/docs/decisions.md`, "EasyRobot: pin dndm to a fixed revision").
- The CPU core does not import dndm (baseline §1). Importers (`grep -rn itohio/dndm --include=*.go`):

| Importer | Build | dndm packages | Endpoint wiring |
|---|---|---|---|
| `cmd/manipulator/main.go` | desktop (builds on darwin, baseline) | `dndm`, `endpoint/remote`, `network`, `network/serial`, `network/stream`, `x/bus` | `setupRouter` (`main.go:250–305`): peers `serial://<port>/easyrobot.host?baud=` and `…/easyrobot.manipulator`, `network.New(serial.New(host))`, `factory.Dial`, `stream.NewWithContext`, `remote.New(host, conn, 10, 10 s, 3 s)`, `dndm.New(WithQueueSize(10), WithEndpoint(remoteEP))` |
| `drivers/manipulator/main.go` | TinyGo `sam && xiao` (not built) | same set | `setupRouter` (`main.go:192–240`): peer `serial:///easyrobot.manipulator`, `dndm.NewContainer("manipulator", 10)`, `factory.Serve` → per connection `remote.New` added to the container |
| `drivers/lidar/main_rp2040.go` | TinyGo `rp2040` (not built) | same set | `setupRouter` (`main_rp2040.go:154–205`): as the manipulator firmware, peer `serial:///easyrobot.lidar` |
| `drivers/lidar/main.go` | desktop, `!rp2040 && cgo` (OpenCV missing → not built) | `dndm`, `endpoint/direct` | `setupRouter` (`main.go:297–306`): in-process `direct.New(queueSize)` only — no network endpoint |
| `cmd/display/destination/intent.go` | desktop (OpenCV missing → not built) | `dndm`, `x/bus` | `IntentDestination` holds one `bus.Producer[*pbdevices.LIDARReading]` per `--intent` route (`intent.go:65–75`) |

### Routes (intents and interests)

dndm routes are `<proto full name>@<path>`; `Router.Publish(path, msg)` builds them with `NewRoute`, which rejects paths containing `@` or `#` (`dndm/route.go:81–89` at HEAD `c7b5b61`). `bus.NewProducer` = `router.Publish` (intent), `bus.NewConsumer` = subscribe (interest) (`dndm/x/bus/producer.go:24–40`).

| Route | Intent (producer) | Interest (consumer) |
|---|---|---|
| `types.control.ManipulatorState@easyrobot.manipulator.{raw,planar,dh}` | `drivers/manipulator/main.go:111` | `cmd/manipulator/main.go:88` |
| `types.control.ManipulatorConfig@easyrobot.manipulator.<mode>.config` | `cmd/manipulator/main.go:97` | `drivers/manipulator/main.go:120` |
| `types.control.ManipulatorTarget@easyrobot.manipulator.<mode>.target` | `cmd/manipulator/main.go:106` | `drivers/manipulator/main.go:129` |
| `types.devices.LIDARReading@<--intent route>` | `drivers/lidar/main.go` via `IntentDestination.SetLIDARReading` (`intent.go:102–121`) | none in-tree |
| path `"LIDARReading@lidar.scan"` | `drivers/lidar/main_rp2040.go:79` | none — the path contains `@`, so `NewRoute` returns `ErrInvalidRoute` and the firmware exits at start-up (static reading; not run) |

The mode suffix is chosen at compile time on the firmware (`drivers/manipulator/{raw,kinematics.planar,kinematics.dh}.go` `getIntentPathImpl`) and by `--mode` on the host (`cmd/manipulator/main.go:237–248`).

### Defects and gaps in the dndm wiring (static reading)

1. `drivers/lidar/main_rp2040.go:79` — invalid route path (above).
2. `drivers/lidar/main.go:52,93` call `destination.RegisterIntentFlags()` and `destination.NewIntentFromFlags(router)`, which do not exist; the package exports `AddIntentFlags(cmd)` and `NewIntentFromCobra(cmd, router)` (`cmd/display/destination/intent.go:137,143`). The OpenCV absence masks this compile error in the baseline.
3. `cmd/display` registers `--intent` (`cmd/display/main.go:32`) but nothing calls `NewIntentFromCobra`, and `IntentDestination.AddFrame` is a no-op (`intent.go:85–98`): frames are never published.
4. `cmd/display/source/interest.go:28–45` — the dndm Interest source is a placeholder that always returns "DNDM interest source not yet implemented". No in-tree consumer receives camera frames or lidar readings over dndm.
5. Firmware transport: dndm's `network/serial` opens ports with `github.com/tarm/serial` (`dndm/network/serial/serial.go:9,76`), and `Serve` just dials its own peer (`serial.go:81–87`). The firmware mains never hand their `machine.UART` to dndm. [INFERENCE] tarm/serial targets OS serial devices and is unlikely to work on TinyGo bare metal; `stream.NewWithContext(ctx, local, remote, rw io.ReadWriter, …)` accepts any `io.ReadWriter`, so the UART could be passed directly. Not verified (no TinyGo build, no board).
6. `remoteEP.Init(ctx, nil, …)` passes a nil logger on firmware (`drivers/lidar/main_rp2040.go:192`); whether dndm tolerates nil was not checked.

## Protobuf schemas

Tooling: `proto/buf.yaml` (buf **v1** config, dep `buf.build/protocolbuffers/wellknowntypes`, lint `DEFAULT`+`COMMENTS`+`FILE_LOWER_SNAKE_CASE`, breaking `FILE`), `proto/buf.gen.yaml` (v1, managed mode, `go_package_prefix github.com/itohio/EasyRobot/`, plugin `go`, `paths=source_relative`, out `../`), `proto/buf.lock` (wellknowntypes commit `9220c3cb…`). `make proto` = `cd proto && buf generate`; `make install-tools` installs `protoc-gen-go@latest` and `buf@latest` (unpinned). Runtime `google.golang.org/protobuf v1.36.10` (`go.mod`).

Generated code: 27 `.proto` files → 28 `types/**/*.pb.go`. 27 headers say `protoc-gen-go v1.36.10`; **`types/core/types.pb.go` says `v1.36.5` and its source `types/core/types.proto` does not exist in `proto/`** — an orphan generated file declaring proto package `easyrobot.core.types` with its own `Tensor` (string dtype, f32/f64/i32/i64 only), `Parameter`, `Layer`, `Model`, `Matrix`, `Vector`, `Slice`, `Value` (`types/core/types.pb.go:25–660`). dndm's own 6 `.pb.go` files are all `v1.36.5` (`grep -rh "protoc-gen-go v" ../dndm/types`). Regeneration was not run (needs `buf`, a tool install).

| Schema (`proto/types/…`) | Proto package | Messages / enums | Purpose | Go users |
|---|---|---|---|---|
| `math/vectors.proto` | `types.math` | Vector2D/3D/4D, Quaternion, Vector | geometry values | manipulator host+firmware, spectrometer |
| `math/matrices.proto` | `types.math` | Matrix2x2…4x4, 3x4, 4x3, Matrix | geometry values | — |
| `math/tensor.proto` | `types.math` | `DataType`, `Tensor` | tensor wire format (`data_f32/f64/i32/i64/i16/i8/ui8`, `data_bytes`) | via `core.Frame` |
| `core/frame.proto` | `types.core` | `Frame{index, timestamp, metadata, repeated types.math.Tensor}` | multi-tensor stream frame | `x/marshaller/gocv/sink.go` (metadata only, see below) |
| `core/arbitrary.proto` | `types.core` | ArbitraryValue, ArbitraryMessage | in-process "reference" wrapper | — |
| `nn/neural_network.proto` | `types.nn` | Parameter, Layer, Model | NN model serialisation | — (the proto marshaller uses the orphan schema instead) |
| `control/kinematics/{joints,wheels,rigidbody,types,state}.proto` | `types.control.kinematics` | DH, PlanarJoint, JointConstraint, JointsConfig; Differential/Mecanum/Steer4/Steer4Dual/Steer6/WheelsConfig; RigidBodyConfig; Config, Dimensions, Capabilities, KinematicsConfig; IMUData, BodyPose, Target, SensorData, State | kinematics configuration/state | manipulator host+firmware |
| `control/{manipulator,motion,motor,path}.proto` | `types.control` | ServoMotorConfig, ManipulatorConfig/Target/State; State, Controls, Trajectory, Constraints, PIDGains, Parameters; MotorType/Config/State/ArrayConfig/ArrayState; Path, PathSegment, PathPlan | control commands/state | manipulator host+firmware |
| `devices/{encoder,motor,sensors,servo}.proto` | `types.devices` | EncoderReading/Config; Motor* (+SpeedCommand); IMU/Distance/Temperature/Pressure/GPS/LIDARReading, SensorReading; Servo* (+AngleCommand) | device readings/commands | lidar desktop+firmware, display intent |
| `filter/slam.proto` | `types.filter` | OccupancyGrid, SLAMPose, SLAMConfig, SLAMMeasurement, SLAMState | SLAM state | — |
| `interaction/{events,messages,requests}.proto` | `types.interaction` | mouse/keyboard/touch/controller events; UserMessage; UserRequest/Response (`Any`) | UI/user I/O | — |
| `marshaller/gocv.proto` | `types.marshaller` | FrameStreamManifest, SourceSpec, SourceKind, DeviceSpec, MatMetadata | capture manifests | `x/marshaller/gocv/manifest.go` |
| `marshaller/graph_metadata.proto` | `types.marshaller` | GraphKind, DecisionEdgeOp, DecisionMetadata, ExpressionMetadata, GraphMetadata | stored graph/decision-tree metadata | `x/marshaller/graph/*` |
| `spectrometer/{config,measurement}.proto` | `types.spectrometer` | 25 config messages + 2 enums (`SpectrometerConfiguration`, `CameraSettings`, `WavelengthCalibration`, …); SpectrumMeasurement, ColorimetryData, … | spectrometer app | `cmd/spectrometer/*` |

`types.control.MotorType/MotorConfig/…` and `types.devices.MotorType/MotorConfig/…` are two parallel motor schemas (`control/motor.proto`, `devices/motor.proto`).

### Findings on the schemas (with evidence)

- **Two tensor wire formats.** `x/marshaller/proto` (`convert.go:6`, `pb "…/types/core"`) encodes tensors with the orphan `easyrobot.core.types.Tensor`, not `types.math.Tensor`; `types.core.Frame` embeds `types.math.Tensor`. A consumer must know which one it is reading.
- **Quantised tensors are lost by `x/marshaller/proto`.** `tensorToProto` has cases only for FP32, FP64, INT32, INT64, INT, C64 (`convert.go:88–122`). Probe (`tmp/easyrobot/arch/protoprobe`, 4×4 tensors, values −3…3): FP32 round-trips (0/16 mismatches); INT8 and INT16 keep dtype but lose data (14/16 mismatches, only zeros survive); UINT8 **panics** on unmarshal (`unsupported dtype: 0`, `eager/tensor.go:36` via `convert.go:144`).
- **`types.math.Tensor` int8 encoding is 5.5× the raw size.** `data_i8` is `repeated int32` (varint; negative values take 10 bytes). Probe: a 7×7×1280 int8 map is 344,972 B via `data_i8` vs 62,732 B via `data_bytes` (raw 62,720 B). Use `data_bytes` (or `sint32`) for int8 feature tensors — relevant to [[W4-k230-feature-stream]].
- **Enum numbering differs from memory.** In-memory `DataType` is `DT_UNKNOWN, INT64, C128, C64, FP64, INT32, FP32, …, UINT8` (`x/math/tensor/types/dtype.go:18–32`); the proto enum is `UNSPECIFIED, INT64, FP64, INT32, FP32, INT, INT16, FP16, INT8, INT48, UINT8` (`tensor.proto`) — no C64/C128; conversion must map by name, not by value.
- **`core.Frame` sink drops pixel data by policy:** `frameToProto` writes index/timestamp/metadata and `Tensors: nil` (`x/marshaller/gocv/sink.go:185–210`, "GoCV-first policy").
- **Shared proto package name.** dndm's `header.proto`/`intents.proto`/`interests.proto` and EasyRobot's `frame.proto`/`arbitrary.proto` are all proto package `types.core` with file paths `types/core/*.proto`. Probe `tmp/easyrobot/arch/regprobe` links both into one binary: no registration conflict today (5 distinct files listed under `types.core`), but any future same-named message or file in either repo would panic at init.

## Owner decisions about protobuf as the wire contract (facts from `~/src/docs/decisions.md`)

- Owner (verbatim, dndm section): "dndm uses protobufs for wire layer. why the fuck do you need a separate spec for c++?! what if we wanted to add typescript client/server? invent new wire ? or did you misderstood the protobuf schemas? (btw, probably must update buf versions and so on)". Recorded consequence: protobuf schemas are the language-neutral wire contract, no hand-written binary spec; the link-context model is expressed in proto; C++/embedded use generated code (nanopb / protobuf-c / EmbeddedProto), TinyGo uses protobuf-go-lite, TypeScript protobuf-es; buf and protobuf tooling versions to be updated.
- Peer decision taken on the owner's behalf (reversible): plain protobuf data frames everywhere (4 B + payload), route id in the CAN identifier on CAN links; legacy v1 fallback only via `Insecure("legacy v1 peer")`.
- Tooling download (buf/protoc-gen-go pins): **not approved, pending**.
- Overhead requirement (verbatim excerpt): "on esp32 or rp2040 memory and i2c/spi/can/elrs bandwidth is at premium!" — zones must not add bytes to every name/message.
- C++ reference core wanted "when we must use nncase" (K230 toolchain) — ties dndm's wire to [[W4-k230-feature-stream]].

## TODOs

- [ ] Fix the firmware lidar route path (drop `LIDARReading@`) — `drivers/lidar/main_rp2040.go:79` #p1 #ER-027
- [ ] Fix the desktop lidar driver's calls to non-existent `RegisterIntentFlags`/`NewIntentFromFlags` — `drivers/lidar/main.go:52,93` #p1 #ER-027
- [ ] Implement the dndm Interest source (proto `Frame` → `types.Frame`) — `cmd/display/source/interest.go:28–45` #p2
- [ ] Wire `--intent` in `cmd/display` and publish frames in `IntentDestination.AddFrame` — `cmd/display/main.go:32`, `cmd/display/destination/intent.go:85–98` #p2
- [ ] Add INT8/INT16/UINT8/FP16 to `x/marshaller/proto` encode/decode, with round-trip tests — `x/marshaller/proto/convert.go:88–122`, probe above #p1 #ER-027
- [ ] Delete the orphan `types/core/types.pb.go` or restore its `.proto`; converge on `types.math.Tensor` — `types/core/types.pb.go:1–5` #p2
- [ ] Document/encode int8 tensors via `data_bytes` (or change `data_i8`/`data_i16` to `sint32`) — `proto/types/math/tensor.proto`, probe above #p2
- [ ] Rename EasyRobot's proto packages to an `easyrobot.*` namespace to avoid sharing `types.core` with dndm — `proto/types/core/frame.proto` #p3
- [ ] Merge the duplicate motor schemas — `proto/types/control/motor.proto`, `proto/types/devices/motor.proto` #p3
- [ ] Pin buf/protoc-gen-go versions (blocked on owner approval of the tool download) and regenerate all `.pb.go` with one version — `Makefile` `install-tools`, decisions.md #p3

## Inferred objectives (proposals)

- **Proposal:** one EasyRobot message contract for streams: `types.core.Frame` + `types.math.Tensor` (`data_bytes` for quantised data, explicit `scale`/`zero_point` fields to add), carried over dndm routes; `x/marshaller/proto` and the gocv sink both emit it.
- **Proposal:** a TinyGo-friendly dndm path (UART/SPI `io.ReadWriter` → `stream` → `remote` endpoint) proven first on the lidar firmware, then reused for the ESP32 recorder in [[W4-k230-feature-stream]].
