---
id: ER-025
title: "Device fixes: encoder, PCA9685, servo array, NMEA, serial"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: [W3, W8]
milestone: M-a
tags: [easyrobot, ticket, w3, w8, devices]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-025 — Device fixes: encoder, PCA9685, servo array, NMEA, serial

[[tickets/index|Ticket index]] · [[roadmap]] · ER-R2 (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · [[todo]]

## Goal and evidence

Device bugs on the M-a stage 2, M-d stage 2 and W8 paths ([[actuators]], [[nmea]], [[device-io]]):

- **Encoder RPM time units** — `x/devices/encoder/encoder.go:164` ([[quadrature-encoder-decoding]]). (The stub signature, F1, is [[ER-026-build-green|ER-026]].)
- **PCA9685 ON/OFF register order** and full-on/off bits — `x/devices/pca9685/pca9685.go:127-134` ([[pca9685-pwm-driver]]).
- **`ServoArray.Stop` deadlock** (calls the locking `set`) — `x/devices/servo/array.go:76-85` ([[servo-angle-mapping-and-array]]).
- **Motor ticker nil race** — `x/devices/motor/motor.go:237,257-266`.
- **NMEA**: 11-field RMC — `x/devices/nmea/data.go:115`; century — `:177-181`; `parseDMM` panic — `:205-213`; altitude merge — `:240-245` ([[nmea-gga-rmc-decoding]]).
- **Serial busy-spin**: blocking reads with timeout — `x/devices/serial_linux.go:86-87`.

## Dependencies and scope

Dependencies: None blocking for the host tests. Device verification needs the TinyGo stage of ER-026 and owner hardware.

Not in scope: XIAO PWM period/scope check (needs a scope), sensor drivers (ADNS3080, AS734x, VL53L0X) — backlog.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Failing test first: `ServoArray.Stop` returns within 100 ms with a running array (`-race`).
- [ ] Failing test first: PCA9685 writes ON registers before OFF, checked against a fake I²C bus; full-on/off bits set at 0 % and 100 %.
- [ ] Failing test first: encoder RPM from synthetic ticks at a known rate is correct within 1 %.
- [ ] Failing tests first: NMEA 2.0 RMC parses; date "010100" → 2000; short DMM returns an error; GGA altitude survives an RMC.
- [ ] Serial read on a pipe with no data blocks until timeout without spinning (CPU check by `testing` benchmark or a goroutine counter).
- [ ] Motor ticker race gone under `-race`.

## Effort uncertainty and revision trigger

Product's guess: **M**. Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
