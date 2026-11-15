---
id: ER-058
title: "Sim S1: wheeled run, the house — a differential robot driven through `wheeled.Base` with logged results, teleop and kill criteria"
type: ticket
status: proposed
kind: measurement
owner: easyrobot
workstream: [W3, W6]
milestone: M-d
tags: [easyrobot, ticket, simulation, sim-s1]
created: 2026-11-10
updated: 2026-11-12
---
# ER-058 — Sim S1: wheeled run, the house — a differential robot driven through `wheeled.Base` with logged results, teleop and kill criteria

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; reviewed)

**Brick:** the first assembled loop, built only from the bricks ER-048 to ER-057, ER-059 and ER-060: description → MJCF → sidecar → client → `simbase` → `Base` and `Clock` → existing controller. It adds no new brick.

## Goal and evidence

Build the bricks first, then the house. First slice: wheeled. The note's S1d run was written for the hexapod; product adapts its kill logic to the wheeled base: if a kinematically sound controller fails in the sim, fix model parameters first (masses, friction, motor limits), and only then question the sim. A usability review (accepted): the loop speaks only `Base` and `Clock`; a teleop scenario; a one-line result; one command starts everything.

Decisions (2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule: primitives like the others (gait, matrix ops, filters), interface-based where performance allows, built on graphs, sets and other primitives; simple, solid, single responsibility; bricks first, then the house.

## Interface sketch

`cmd/sim-wheeled`:

```text
sim-wheeled --backend mujoco|fake|unix:///tmp/sim.sock --scenario straight|square|teleop [--viewer] [--log run.csv]
sim-wheeled --compare run-fake.csv run-mujoco.csv
  loop: Clock.Tick → Base.Encoders → odometry.Update → controller → twist → Base.SetTwist
  log per step: commanded twist, ground-truth pose (sim only), odometry pose, wheel saturation, sim speed
  final line: straight: 1.97 m (−1.5 %) heading 0.8° saturation none → PASS (≤ 5 %, ≤ 2°)
```

## Dependencies and scope

Dependencies: ER-050, ER-051, ER-053, ER-054, ER-056 (determinism known before results are trusted), ER-057, ER-059, ER-060; ER-038.

- **The loop uses only `Base` and `Clock`** (ER-059), never `simpb` types; ground truth for the log comes from ER-059's optional `Truth` interface, found by type assertion (implemented by `simbase` only; the column stays empty on hardware), so the same loop binary can drive the hardware base later.
- Controllers: a constant twist (straight) and the existing rigid-body path tracker for a 1 m square (`x/math/control/motion/rigidbody`, after ER-038 fixes its sign).
- **Teleop**: `--scenario teleop` reads the twist from the keyboard (arrow keys or WASD, a step per press, space stops), best with `--viewer`. **Keys go to the terminal**: with `--viewer`, the MuJoCo window can take focus, so the run prints `keys go to this terminal; click it if the viewer has focus` at start (reading keys from the viewer is a later option: MuJoCo's `launch_passive(…, key_callback=…)` is documented); the RC receiver (`cmd/elrs-rx`, the input the real robot will use) is a later input on the same path.
- **One command**: `--backend mujoco` starts ER-051's sidecar on a temp socket and stops it on exit; `--backend unix://…` keeps the two-process form for debugging; `--backend fake` runs ER-055 in-process.
- **Result at a glance**: each scripted run ends with one line in `simcheck`'s format (value, deviation, threshold, PASS/FAIL, and the kill action on FAIL); `--compare` prints fake and MuJoCo side by side (distance, heading, final error, odometry drift, sim speed).
- The same binary runs against the fake and the sidecar; results are compared.
- **What the sim offers vision navigation later (not a dependency):** the same loop can log body-camera frames, ground-truth poses and IMU streams once a camera brick exists; the owner's vision plan starts with bench and handheld data, no robot.

Not in scope: obstacle avoidance; camera; the hardware base; legged robots (the hexapod slice reuses the bricks later).

Source/test edits are approved once this ticket is scheduled. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] **Straight:** 0.2 m/s for 10 s on a plane: distance within 5 % of 2 m, heading drift ≤ 2°, no wheel saturation (the thresholds); the run ends with its PASS/FAIL line.
- [ ] **Square:** 1 m sides at 0.2 m/s with the path tracker: back within 10 cm and 5° of the start.
- [ ] Odometry vs ground truth after the square: error reported (no threshold; it is the baseline for later filters).
- [ ] `--compare` prints the fake and MuJoCo runs side by side; the table goes in the evidence.
- [ ] **Teleop:** with `--backend mujoco --viewer --scenario teleop`, scripted key presses (a test feeding stdin) move the robot forward, turn it and stop it; a person's check in the viewer is recorded once.
- [ ] `--backend mujoco` leaves no sidecar process behind after a normal exit or Ctrl+C.
- [ ] **The loop is its own package** (`x/robot/wheeled/loop`), used by `cmd/sim-wheeled`: a test runs `go list -deps` on it and finds no `types/sim` and no `x/sim/...`; a mutant that imports `simbase` into the loop package fails that test (moved here from ER-059).
- [ ] **Kill / rethink:** < 80 % of the commanded distance, heading drift > 10° over the straight run, or saturation at 0.2 m/s with a controller that works on the fake → tune model parameters first (friction, masses, motor limits from decision 7); if still failing, the sim choice is revisited.

## Open questions

- **Decision 6 (open):** a Cogent Lab `physics` spike behind the same `Sim` interface (needs approval to fetch the module); it would run this scenario as its first test.
- **Decision 7 (open):** motor part numbers; results are labelled with the placeholder values until then.

## Effort uncertainty and revision trigger

Size guess: **M** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Log

- 2026-11-12: review fixes (accepted with 11 fixes): the loop is its own package with a `go list -deps` test against `types/sim` and `x/sim` (moved from ER-059); MuJoCo's passive viewer documents `key_callback`, so viewer keys are a later option without the #verify.

## Implementation evidence

None. Proposed ticket; becomes `ready` after review.
