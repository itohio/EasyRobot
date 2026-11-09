---
title: Safety watchdog
type: design
tags: [rover, safety]
updated: 2026-10-01
---

# Safety watchdog

The hardware watchdog is fed by the main loop every 5 ms; a missed deadline of 50 ms resets the MCU. The attitude limiter ([[attitude-limiter]]) runs inside the same loop.

Known issue: [[RV-012-watchdog-reset-loop]]. Power numbers: [[power-budget-draft]].
