---
type: algorithm
status: tested
tags: [easyrobot, devices, sensors, imu, i2c, tinygo]
workstream: W3
module: "[[sensors]]"
summary: "one-burst reads, rate/DLPF/ranges, SI units, data-ready IRQ, no alloc; fake-bus tests only"
source: [x/devices/mpu6050/mpu6050.go]
tests: [x/devices/mpu6050/mpu6050_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -race ./x/devices/mpu6050 pass (8 tests, fake register bus, literal datasheet addresses, error paths); 19 mutants (register addresses, accel/gyro swap, byte order, offsets, scales, DLPF rate, INT config, PLL, temperature sign and offset, validation, swallowed bus errors, leftover byte on error) all fail; review S1 (2026-11-12): a partial Configure records exactly the writes that landed, red first (1 g read 2.45 m/s² after a failed write 6), 7 record-order mutants fail (ER-SIMALGO-logs/s1-*.txt); tinygo 0.43 -target=tufty2040 builds, -print-allocs shows only New's Device. Hardware: none yet (the owner's badge MPU6050 is not fitted)."
---
# MPU6050 IMU driver

**What it does:** drives an InvenSense MPU-6050 over `devices.I2C`: WHO_AM_I probe, configuration (sample-rate divider, digital low-pass filter, accel and gyro full scale, data-ready interrupt), one 14-byte burst read of accel + temperature + gyro of the same sample, and conversion to SI units.

**Where:** `x/devices/mpu6050/mpu6050.go` — `Config`, `Config.SampleRate`, `Config.Scale` (:97-146), `Device`, `New` (:148-167), `Probe` (:170), `Configure` (:180-215), `DataReady` (:226), `ReadRaw` (:232-244), `Read` (:247), `read8` (:261).

**Concurrency:** a `Device`'s methods share one transfer buffer, so use it from one goroutine at a time or guard it with your own lock (review S3: `go test -race` flagged two goroutines reading at once). A lock inside would cost every read on the hot path.

**Inputs → outputs:** `New(bus devices.I2C, addr)` (0 → 0x68; AD0 high is 0x69). `Probe() (id, ok, err)`: `ok` iff WHO_AM_I is 0x68 (an MPU-6500 answers 0x70, an MPU-9250 0x71, some clones 0x98: `id` lets the caller decide; log it). `Configure(Config)`: out-of-range fields give `devices.ErrInvalidValue` before any write. The writes are not atomic: a bus error stops at that write, and `Config()` and the scaling then hold exactly the fields whose writes landed. The ranges in the chip and the scaling never disagree (review S1: they used to, and 1 g read 8× low after a failed later write). `ReadRaw(*Raw)` counts; `Read(*Sample)` m/s², rad/s, °C with the configured ranges. Every call returns the bus error; on an error `Probe` gives id 0 and `DataReady` false (never a byte left from an earlier transfer). The chip powers up asleep: reads before `Configure` return stale or zero data. `Configure` does not touch PWR_MGMT_2, USER_CTRL or FIFO_EN.

**Register use** (MPU-6000/6050 register map rev 4.2): `PWR_MGMT_1`=0x01 (awake, PLL on the X gyro, steadier than the internal oscillator); `SMPLRT_DIV`; `CONFIG.DLPF_CFG` 0–6 (0 = 260/256 Hz, gyro output 8 kHz; 1–6 = 184…5 Hz, gyro output 1 kHz; 7 refused); `GYRO_CONFIG.FS_SEL` and `ACCEL_CONFIG.AFS_SEL` bits 4:3; `INT_PIN_CFG`=0x30 (latched until any read) and `INT_ENABLE.DATA_RDY_EN` when `DataReadyInterrupt` is set, so an edge interrupt and a polled level both work; the burst reads 14 bytes from `ACCEL_XOUT_H` (0x3B).

**Scaling:** accel 16384/8192/4096/2048 LSB/g × 9.80665 m/s²; gyro 131/65.5/32.8/16.4 LSB/(°/s) × π/180; temperature counts × (1/340) + 36.53 °C. The reciprocal is a multiply: with `/340` TinyGo calls the soft-float `__divsf3` for every sample on the RP2040, with `× (1/340)` it does not (`ER-SIMALGO-logs/s4-fdiv.txt`). Sample rate = gyro output rate / (1 + divider).

**TinyGo:** the device holds one 15-byte buffer used by every transfer; reads fill caller values. `tinygo build -target=tufty2040 -print-allocs=mpu6050` reports only the `Device` from `New`. The Tufty 2040's Qw/ST port is `machine.I2C0` (GPIO4 SDA, GPIO5 SCL).

## Completeness review

- Complete: yes for the demo's needs (brief ER-SIMALGO §Owner MPU6050). Not covered: FIFO, self-test, gyro bias calibration, the auxiliary I2C master, motion interrupts.
- Tested: `mpu6050_test.go` on a fake register bus that records transactions; register addresses in the tests are literals from the register map, not the package's constants, and accel and gyro use different range codes:
  - `TestProbe`: 0x68, 0x70, and a bus error giving id 0.
  - `TestConfigure`: 0x6B=0x01, 0x19, 0x1A, 0x1B=±500 °/s, 0x1C=±8 g, 0x37=0x30, 0x38; the address; 100 Hz and 1 kHz; the 5 Hz filter (code 6, the last valid one) with divider 255 gives 1 kHz/256 (review S5); invalid fields refused before any write.
  - `TestBusErrorsPropagate`: `ReadRaw`, `Read` and `DataReady` return the error. A failure at each of `Configure`'s 7 writes stops there; `Config()` holds exactly the fields whose writes landed, and 4096 counts read as 1 g only once ±8 g has landed. No ready flag leaks into a failed read.
  - `TestReadRaw`: one write of 0x3B, one 14-byte read, signed big-endian decode incl. −32768 and 32767.
  - `TestScale`: every range; temperature +340 → 37.53 °C, −340 → 35.53 °C.
  - `TestReadScalesWithTheConfiguredRanges`, `TestDataReady` (0x3A), `TestNoAllocation`.
  - Run: `go test -race -count=1 ./x/devices/mpu6050/` pass, 2026-11-12.
- Correct: yes on the register map and scale factors (cross-checked by verify against `tinygo.org/x/drivers` mpu6050; 19 mutants killed, `ER-SIMALGO-logs/mpu6050-mutants-r2.txt`). Real-sensor behaviour is unverified until the badge's MPU6050 is fitted.

## TODOs

- [ ] Gyro bias calibration (average at rest) and self-test — x/devices/mpu6050/mpu6050.go:180 #p3
- [ ] Run on the badge's MPU6050 (log WHO_AM_I, 1 g at rest, data-ready at the configured rate, plausible temperature) — x/devices/mpu6050/mpu6050.go:232 #p2

## Links

- Module: [[sensors]] · Bus: [[device-io]] · Fusion: [[mahony-ahrs]]
