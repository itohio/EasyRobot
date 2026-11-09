---
title: Rover
type: hub
tags: [rover, hub]
updated: 2026-10-30
---

# Rover

Four-wheel field robot. Firmware in Go (TinyGo) under this folder.

## Design notes

- [Architecture](docs/architecture.md)
- [Motor control](docs/motor-control.md), [IMU fusion](docs/imu-fusion.md), [Odometry](docs/odometry.md)
- [Lidar driver](docs/lidar-driver.md), [Telemetry link](docs/telemetry-link.md)
- [Attitude limiter](docs/attitude-limiter.md), [Safety watchdog](docs/safety-watchdog.md)
- [Bootloader](docs/bootloader.md), [Power budget](docs/power-budget.md)

## Packages

`example.com/fixture/rover/motor`, `.../imu`, `.../odometry`, `.../telemetry`, `.../safety`.
