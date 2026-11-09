---
type: algorithm
status: partial
tags: [easyrobot, motion, trajectory, servo]
workstream: W3
module: "[[motion-and-gait]]"
summary: "partial (sign bug, limits violated)"
source: [x/math/control/motion/vaj1d.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "no tests; probe J (2026-11-09): positive moves stop up to 1 unit short, moves of +0.5 never start, |a| reaches 4403 with limit 100, 3.75 overshoot at dt=0.1"
---
# VAJ1D jerk-limited 1-D profile

**What it does:** Streams a position `Output` toward `Target` per `Update(dt)` with velocity, acceleration and jerk limits (S-curve).

**Where:** `motion/vaj1d.go:22-40` (`New1D`), `:52-76` (`Update`), `:80-109` (`calculateKinematics`), `:114-140` (`calculateJerk`), `:145-196` (`calculateStoppingDistance`). Package name `vaj` in dir `motion`.

**Inputs → outputs:** limits (maxV, maxA, maxJ), `Input/Target`, `dt` s → `Output`, `Velocity`, `Acceleration`. Used by the manipulator firmware (`drivers/manipulator/main.go:351-356`, degrees, `New1D(100,100,100)`, 100 ms tick) and by [[rigidbody-path-follower]].

## Completeness review

- Complete: no — direction handled by a sign flip of the remaining distance, state magnitudes are unsigned, so a target reversal mid-move snaps direction (probe: 0→100 for 0.5 s, then target 0: no deceleration past 1.90).
- Tested: no test file.
- Correct: **no**.
  - `if x1 < 1` (`:59`) should be `x1 < 0`: for a positive remaining distance below 1 unit the sign is flipped and the `x1 < .001` stop branch fires. Probe J: 0→100 ends at 99.319; 0→0.5 never moves; 0→1.5 rad (servo in radians) ends at 0.501; 0→10 with (1,1,5) ends at 9.012. Negative moves work (0→−0.5 ends at −0.5003). `motion/rigidbody/motion.go:427` (`delta < 1`) bypasses VAJ for short moves.
  - Limits not respected: 0→100 with a=100 limit shows finite-difference |a| up to 4403 (velocity snap when `v0` is forced to maxV, `:92-96`); at dt=0.1 s (manipulator tick) the move overshoots by 3.75.

## TODOs

- [ ] Fix sign test `x1 < 1` → `x1 < 0`; failing test first: 0→0.5 and 0→100 must end within 1e-3 (ER-018) — vaj1d.go:59 #p1 #ER-018
- [ ] Enforce |a| ≤ maxA and bounded jerk (no velocity snap); test with finite differences — vaj1d.go:92 #p1 #ER-018
- [ ] Signed velocity/acceleration so target reversals decelerate — vaj1d.go:57 #p2
- [ ] Rename package dir/name mismatch (`motion` dir, `vaj` package) — vaj1d.go:3 #p3

## Links

- Module: [[motion-and-gait]] · [[rigidbody-path-follower]] · [[ER-R2-ik-review]] · [[W3-legged-ik]]
