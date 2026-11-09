---
title: Telemetry link
type: design
tags: [rover, telemetry]
updated: 2026-07-18
---

# Telemetry link

Frames carry a sync byte, a length, a payload and a CRC16-CCITT checksum. The ground station drops frames whose checksum fails.

Radio choice is a roster decision ([[_roster/decisions]]). Mismatch bug: [[RV-015-telemetry-crc-mismatch]].
Code: `rover/telemetry/telemetry.go`.
