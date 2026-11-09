---
title: IMU calibration
type: procedure
tags: [rover, imu]
updated: 2026-09-05
---

# IMU calibration

Keep the robot motionless for two seconds after power-on, average 400 gyro samples, store the mean as the bias and subtract it from every later sample.

Without this step the heading wanders by several degrees per minute even when parked ([[RV-004-imu-drift]]).

Code: `rover/imu/calibrate.go` (`Calibrate`).
