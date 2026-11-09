---
type: module
group: "Devices and apps"
status: partial
tags: [easyrobot, devices, hal]
workstream: [W1, W4, W8]
packages: [x/devices]
test_result: none
---
# Device I/O abstractions (x/devices core)

**Purpose:** TinyGo-shaped hardware interfaces (`Pin`, `I2C`, `SPI`, `Serial`, `PWM`/`PWMDevice`) so the same driver runs on an MCU (TinyGo `machine`) and on Linux SBCs (sysfs/ioctl), plus Windows serial for desktop tools.

**Packages:** `x/devices` (root package, no subpackages for the core):

| File | LOC | Tags | Content |
|---|---|---|---|
| `pin.go`, `i2c.go`, `spi.go`, `serial.go`, `pwm.go`, `errors.go`, `spectrometer.go` | 27/36/14/39/32/31/39 | — | interfaces, `SerialConfig`, shared errors, `Spectrometer` (see [[sensors]]) |
| `pin_linux.go` | 211 | `!tinygo && linux` | sysfs GPIO (`/sys/class/gpio`), epoll edge interrupts |
| `i2c_linux.go` | 180 | `!tinygo && linux` | `/dev/i2c-N` + `I2C_SLAVE` ioctl; target mode returns `ErrNotSupported` |
| `spi_linux.go` | 185 | `!tinygo && linux` | `/dev/spidevB.C` ioctls (mode, bits, speed, `SPI_IOC_MESSAGE`) |
| `pwm_linux.go` | 275 | `!tinygo && linux` | sysfs PWM (`/sys/class/pwm`), RPi GPIO→pwmchip table |
| `serial_linux.go` | 261 | `!tinygo && linux` | termios raw 8N1, custom baud via `BOTHER`, `ListSerialPorts` |
| `serial_windows.go` | 218 | `!tinygo && windows` | `CreateFile` COM ports, registry enumeration |
| `raspberrypi.go` | 156 | `!tinygo && linux` | RPi device-path helpers and enable checks |
| `*_tinygo.go` (pin/i2c/spi/serial) | 25/18/5/5 | `tinygo` | type aliases to `machine` constants; "use machine.X directly" |
| `*_stub.go` (pin/i2c/spi/serial) | 43/14/24/3 | `!tinygo && !linux` (serial: `&& !windows`) | constants + non-conforming stubs |
| `README.md`, `INTERFACES.md` | 214/247 | — | design docs |

## Public API

```go
type Pin interface { SetInterrupt(PinChange, func(Pin)) error; Get() bool; Set(bool); High(); Low() }
type I2C interface { ReadRegister(addr, reg uint8, buf []byte) error; WriteRegister(...); Tx(addr uint16, w, r []byte) error }
type SPI interface { Tx(w, r []byte) error; Transfer(byte) (byte, error) }
type Serial interface { io.Reader; io.Writer; Buffered() int }
type PWM interface { Set(duty float32) error; SetMicroseconds(us uint32) error; Stop() error }
type PWMDevice interface { Channel(Pin) (PWM, error); Configure(hz uint32) error; SetFrequency(hz uint32) error }
// linux: NewPin(n) (*LinuxPin, error), NewI2C("/dev/i2c-1"), NewSPI(...), NewPWMDevice(), NewSerialWithConfig(dev, SerialConfig{BaudRate, EnableBuffering}), ListSerialPorts()
```

Drivers take these interfaces (`pca9685.New(bus I2C, addr)`, `ld06.New(ctx, ser Serial, motor PWM, …)`), and decorators compose (`tca9548a.Router.Channel(n)` returns an `I2C`). See [[actuators]], [[sensors]], [[lidar]], [[nmea]], [[crsf-elrs]].

## Algorithms

<!-- vault:algorithms h=f56d515d3b10 -->

No algorithm notes belong to this module.

<!-- /vault:algorithms -->

No algorithm notes (glue code). Non-trivial pieces reviewed here: Linux epoll GPIO interrupts (`pin_linux.go:85-190`), termios setup (`serial_linux.go:33-99`), SPI ioctl transfer (`spi_linux.go:125-171`).

## Build targets and dependencies

- **linux** (any arch): builds (`GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build ./x/devices/...` OK, 2026-11-09). Uses `syscall` + `golang.org/x/sys/unix`, no CGO.
- **windows**: core builds; only `encoder`/`motor` fail (F1 — stub applies to windows too; cross-build 2026-11-09).
- **darwin**: root package builds but provides **no Serial, I2C, Pin or PWM implementation** — `serial_stub.go` is empty, `pin_stub.go` `stubPin` does not satisfy `Pin` (methods `High() error`, `Read()`, `SetInterrupt(…, func())`), `NewPin(uint8) *stubPin` has a different signature from linux `NewPin(int) (*LinuxPin, error)`. Hence F2 (`cmd/monitor`, `cmd/spectrometer/*` need `NewSerialWithConfig`/`ListSerialPorts`) and every serial app is linux/windows-only. `README.md:28` claims macOS hosts work.
- **TinyGo**: the interfaces are meant to be satisfied by `machine.*`, but `machine.Pin.SetInterrupt` takes `func(machine.Pin)` while `devices.Pin` requires `func(devices.Pin)`, so `machine.Pin` is **not** a `devices.Pin`; `NewTinyGoPin` (`pin_tinygo.go:23-25`) and `servo.pinNumberToPin` would not compile [INFERENCE: TinyGo signature from memory; TinyGo not run]. `machine.I2C`, `machine.SPI`, `machine.UART` match their interfaces [INFERENCE]. Board tags in the repo: `rp2040` (drivers/lidar, cmd/elrs-rx), `esp32` (cmd/elrs-rx), `sam && xiao` (x/devices/xiao, drivers/manipulator, cmd/elrs-rx).
- Allocation: Linux paths allocate per call (`fmt.Sprintf` paths, `make` in `ReadRegister`); fine on SBC.

## Tests and evidence

- No tests in `x/devices` root (baseline: build ok, vet ok, no test files). `go vet ./x/devices/...` darwin: only the encoder build error; linux vet: `encoder/example_test.go:11` "ExampleEncoder refers to unknown identifier: Encoder".
- Not run: anything touching `/dev`, `/sys`, COM ports.

## API notes (ease of use, composability)

- Good: small TinyGo-compatible interfaces; drivers are bus-agnostic and testable with fakes (probes used a 20-line recording `I2C`); decorator pattern for muxes.
- Friction: (1) **platform asymmetry** — constructors differ in name/signature per OS (`NewPin(int)` vs `NewPin(uint8)`), so apps need per-platform files; a `devices/host` package with uniform `Open*` functions and explicit `ErrNotSupported` on unsupported OSes would remove F2-style build breaks. (2) `Pin` mixes output and interrupt; no `Configure`/mode, so drivers needing pull-ups (encoder) cannot express it. (3) `Serial.Buffered()` is opt-in and returns 0 by default — drivers can't rely on it; `Read` returning `(0, nil)` when idle (`serial_linux.go:86-87` sets `VMIN=0, VTIME=0`; `:178-185`) makes every read loop spin (lidar, NMEA, CRSF). Prefer blocking reads with a timeout (`VTIME`) or a `SetReadTimeout`. (4) sysfs GPIO is deprecated and gone on recent kernels/RPi 5 (gpiochip character device is current) [INFERENCE]. (5) `PWMDevice.Channel` on Linux requires the concrete `*LinuxPin` (`pwm_linux.go:102`) — fine, but undocumented.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Add darwin serial (termios via x/sys/unix works on darwin) or a stub with `NewSerialWithConfig`/`ListSerialPorts` returning `ErrNotSupported` (fixes F2) — x/devices/serial_stub.go:1-3 #p1 #ER-025
- [ ] Make `stubPin` satisfy `Pin` and unify `NewPin` signature across platforms — x/devices/pin_stub.go:20-43 #p2
- [ ] Resolve `machine.Pin` vs `devices.Pin` callback mismatch (e.g. `SetInterrupt(PinChange, func())` or an adapter type) — x/devices/pin.go:26; x/devices/pin_tinygo.go:23-25 #p1 #ER-025
- [ ] Use blocking reads with timeout instead of `VMIN=0,VTIME=0` polling — x/devices/serial_linux.go:86-87 #p2
- [ ] Add pin mode/pull configuration to the abstraction — x/devices/pin.go #p2
- [ ] Replace sysfs GPIO with gpiochip (character device) — x/devices/pin_linux.go #p3
- [ ] Correct README macOS claim — x/devices/README.md:28 #p3
- [ ] Unit tests for termios baud mapping and Windows port-name parsing (pure functions) — x/devices/serial_linux.go:101; serial_windows.go:204 #p3

## Inferred objectives (proposals)

- **Proposal:** a `devices/fake` package (recording I2C/SPI, scripted Serial, toggled Pin, recording PWM) so every driver gets host unit tests — today 0 of 15 driver packages have value-checking hardware-path tests.
- **Proposal:** SPI slave/stream support on ESP32 for the K230 → ESP32 feature stream ([[W4-k230-feature-stream]]): the `SPI` interface is master-only.

Related: [[build-targets]], [[packages]], [[actuators]], [[sensors]], [[lidar]], [[nmea]], [[crsf-elrs]], [[apps]]
