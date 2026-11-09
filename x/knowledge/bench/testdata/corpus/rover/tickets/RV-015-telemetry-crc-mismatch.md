---
id: RV-015
title: Ground station rejects every frame
type: ticket
status: open
tags: [rover, ticket]
updated: 2026-07-18
---

# RV-015 — Ground station rejects every frame

The ground station computes CRC16 over the sync byte too; the rover does not. See [[rover/docs/telemetry-link]].
