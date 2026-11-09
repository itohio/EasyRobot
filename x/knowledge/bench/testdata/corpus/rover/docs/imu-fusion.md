---
title: IMU fusion
type: design
tags: [rover, imu]
updated: 2026-05-20
---

# IMU fusion

A complementary filter blends integrated gyro rates with the accelerometer's gravity direction (alpha = 0.98).

Before fusion the gyro bias must be removed — see the [calibration procedure](old/imu-calibration.md).

Code: `rover/imu/imu.go`.
