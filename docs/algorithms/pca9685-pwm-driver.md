---
type: algorithm
status: partial
tags: [easyrobot, devices, actuators, i2c]
workstream: [W3]
module: "[[actuators]]"
summary: "ON/OFF registers swapped"
source: [x/devices/pca9685/pca9685.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe (recording fake I2C, 2026-11-09): SetPWMRaw(0,1000) writes 06 e8 03 00 00 (OFF value into ON registers); SetFrequency(50) prescale 0x79 = 121 correct"
---
# PCA9685 16-channel PWM driver

**What it does:** I2C driver for the NXP PCA9685 (12-bit, 16-channel PWM, the usual hexapod/quadruped servo board): reset, set frequency via prescaler, set per-channel duty.

**Where:** `x/devices/pca9685/pca9685.go:43-52` (`New`, default address 0x40), `:54-65` (`Configure/Reset`), `:67-~100` (`SetFrequency`), `:~103` (`SetPWM` float), `:114-136` (`SetPWMRaw`), `write8/read8`.

**Inputs → outputs:** `devices.I2C`, channel 0–15, duty 0–1 (or raw 0–4095), invert flag → I2C writes.

## Completeness review

- Complete: **no.** No servo helper (µs → counts), no full-on/full-off bit (bit 4 of `ON_H`/`OFF_H`) so 100 % and 0 % are not exact, no all-channel write, no output-driver/OE configuration, no oscillator calibration (internal 25 MHz varies ±~5 % [INFERENCE]).
- Tested: none. Probe with a fake bus records the I2C bytes.
- Correct: **no.** `SetPWMRaw` writes `[OFF_L, OFF_H, ON_L, ON_H]` starting at `LED0_ON_L + 4·ch` (`pca9685.go:127-134`), but the register order is `ON_L, ON_H, OFF_L, OFF_H`. Probe: value 1000 → `06 e8 03 00 00`, i.e. ON=1000, OFF=0 — the output is high from count 1000 to wrap, an inverted duty of ≈(4096−1000)/4096 [INFERENCE: PCA9685 datasheet register map from memory]. `SetFrequency` prescale correct for 50 Hz (`0x79`) and follows sleep → prescale → restart sequence; `read8` allocates per call, `SetPWMRaw` allocates via `append`.

## TODOs

- [ ] Fix register order (ON first, then OFF) and add full-on/off bits — x/devices/pca9685/pca9685.go:127-134 #p1 #ER-025
- [ ] Add `SetMicroseconds(ch, us)` and implement `devices.PWMDevice` so `servo.ServoArray` can run on a PCA9685 (needed for 12/18-servo legged robots) — x/devices/pca9685/pca9685.go #p2
- [ ] Fake-bus unit tests asserting register bytes — x/devices/pca9685 #p2

## Links

- Module: [[actuators]] · Servo mapping: [[servo-angle-mapping-and-array]] · [[W3-legged-ik]]
