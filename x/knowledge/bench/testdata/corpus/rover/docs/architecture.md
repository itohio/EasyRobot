---
title: Rover architecture
type: design
tags: [rover, design]
updated: 2026-06-12
---

# Rover architecture

The main loop runs at 200 Hz: read sensors, fuse attitude ([[imu-fusion]]), integrate wheel odometry ([[odometry]]), run the speed loops ([[motor-control]]) and send telemetry ([[telemetry-link]]).

A separate safety task watches the main loop ([[safety-watchdog]]) and clamps throttle ([[attitude-limiter]]).
