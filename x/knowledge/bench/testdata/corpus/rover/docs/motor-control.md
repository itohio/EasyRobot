---
title: Motor control
type: design
tags: [rover, motor]
updated: 2026-07-03
---

# Motor control

Each wheel has a PID speed loop. The target comes from the planner in metres per second; the measurement from encoder ticks.

## Tuning

Gains were tuned on the bench: raise Kp until the wheel oscillates, halve it, then add Ki until steady-state error disappears. Kd stays zero because encoder noise is high.

Code: [motor.go](../motor/motor.go). Overheating driver: [[RV-001-motor-driver-overheat]].
