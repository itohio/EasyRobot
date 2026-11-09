---
title: Odometry
type: design
tags: [rover, odometry]
updated: 2026-08-01
---

# Odometry

Pose is integrated from left and right encoder ticks with the differential-drive model. Wheel slip is flagged when the encoder yaw rate disagrees with the gyro by more than 0.3 rad/s ([[RV-007-odometry-slip]]).

Code: `rover/odometry/odometry.go`.
