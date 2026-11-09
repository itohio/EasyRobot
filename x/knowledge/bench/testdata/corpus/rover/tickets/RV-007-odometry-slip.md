---
id: RV-007
title: Odometry over-reports distance on wet grass
type: ticket
status: open
tags: [rover, ticket]
updated: 2026-09-14
---

# RV-007 — Odometry over-reports distance on wet grass

Wheels spin without moving the body; encoder distance exceeds the true distance by 30 %. Detect slip by comparing encoder yaw rate with the gyro (`odometry.DetectSlip`). Design: [[rover/docs/odometry]].
