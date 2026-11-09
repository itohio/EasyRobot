---
id: RV-012
title: Watchdog reset loop after flashing
type: ticket
status: open
tags: [rover, ticket]
updated: 2026-10-01
---

# RV-012 — Watchdog reset loop after flashing

After a firmware update the MCU resets every 50 ms because flash erase blocks the main loop longer than the watchdog deadline. See [[rover/docs/safety-watchdog]] and [[rover/docs/bootloader]].
