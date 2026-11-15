---
type: research
status: draft
owner: ideation
tags: [easyrobot, research, ideation, simulation, physics, mujoco, hexapod, legged, sensors]
workstream: [W3, W5]
created: 2026-11-10
updated: 2026-11-10
---

# A 3D robot simulator for EasyRobot: what Go offers, architectures, and a first slice

Related: [[W3-legged-ik]], [[2026-11-09-gaits-w3]], [[2026-11-10-vision-nav]] (the topological map), [[2026-11-09-on-device-learning]].

**Status:** investigation only. Nothing was downloaded, installed or built.

**Labels:** observed (read this session), reported (a primary source says so), inferred, `#verify`.

**Network (read-only):**
- 6 web searches: Go MuJoCo bindings ×2, pure-Go 3D physics, Go ODE bindings, Go Jolt bindings, cogentcore physics.
- 7 page reads:
  - `pkg.go.dev` ×3: cogentcore physics, cogentcore lab, jolt-go (these returned no body text);
  - `raw.githubusercontent.com` ×3: the cogentcore lab physics README, g3n `experimental/physics/doc.go`, the MuJoCo README;
  - `cogentcore.org/lab/physics`.
- About 18 `api.github.com` metadata requests: repository activity, licences and directory listings for g3n, cubez, cogentcore core and lab, cp, solid, ode, jolt-go, jolt-purego, MuJoCo (including the latest release), bullet3, JoltPhysics, gz-transport and webots.

**Requirement (2026-11-10):** EasyRobot likely needs a robot environment simulator for hexapods, bipeds and robots in general: a 3D environment with physics. Investigate whether existing Go software covers most of it; do not build it yet.

## Verdict

1. **Yes, EasyRobot needs a simulator.** Today it has only kinematic host tests: the feet stay put; no forces, no contacts, no sensors (observed, `walk_sim_test.go`). The kinematic tripod test `TestHexapodTripodSim` already reports hexapod joint speeds of about 8 rad/s against a 5.24 rad/s servo limit (recorded in the vault note `docs/algorithms/gait-foothold-planner.md:45`). A dynamic sim would expose that as falls, not as a table row.
2. **Nothing in Go covers legged contact dynamics well today.**
   - **The closest pure-Go engine** is **Cogent Lab's `physics`** (BSD-3, in `cogentcore/lab`: last pushed 2026-11-03, 18 stars, used in axon; XPBD solver ported from newton-physics, CPU or GPU, joints with position/velocity targets, first-person camera views through its `xyz` renderer).
   - **Its gaps:** primitive shapes only (no meshes or heightfields), "Sensors: TODO", known collision and stability bugs in its README, Y-up coordinates, no URDF/MJCF.
   - **The rest:**
     - g3n's physics is marked "experimental and incomplete";
     - cubez is archived (2016);
     - the Go ODE binding was last touched in 2016;
     - jolt-go and jolt-purego are short-lived bindings (a few days of commits each), with a handful of stars, unmaintained since 2025-11 and 2026-02;
     - cp is 2D;
     - **there is no Go binding for MuJoCo** in its own list of first- and third-party bindings (Swift, Java, Julia, Rust: yes; Go: no).
3. **Recommendation:** **MuJoCo as the reference simulator, driven from Go over protobuf** (architecture b).
   - **Start:** a small sidecar process that owns MuJoCo, with the step loop in lockstep with EasyRobot's Go controller.
   - **Later:** an in-process binding through `purego` (dlopen, no cgo) once the interface settles.
   - **Robot models:** generated from EasyRobot's `chain.Body` as MJCF.
   - **Pure-Go track:** keep Cogent Lab's `physics` as an optional fast CI and experiment backend behind the same Go interface, if a spike shows it stable for a hexapod.
4. **First slice:** a hexapod in MuJoCo driven by `walk.Walker.Step` (the existing IK and gait code), headless, lockstep, with servo torque and speed limits. Kill criteria in §4.

## 1. Needs

| Need | Why (EasyRobot) | Must-have now? |
|---|---|---|
| Articulated rigid bodies, revolute joints, **position-servo actuators with torque and speed limits** | hexapod/biped legs driven by hobby servos (`ServoMap`, 300°/s default limit, observed in `chain/legged.go:23`) | yes |
| **Contacts with friction**, stable under many simultaneous foot contacts | legged gaits; slip; falls | yes |
| Terrain: plane, steps, slopes, rough heightfield | gait robustness; `TerrainProvider` exists but is unused (observed) | plane now; heightfield soon |
| IMU (accel, gyro), joint encoders, foot contact/force | feedback for gait and balance; AHRS (`filter/ahrs`) | yes |
| **Camera rendering** (QVGA-class, body-mounted, known intrinsics) | the vision stack: frontend features, VO, place codes | soon |
| Lidar rays | the lidar driver and 2D SLAM (`filter/slam`) | later |
| Determinism and repeatability | regression tests; comparing controllers | yes |
| Faster than real time, headless | CI; parameter sweeps; later learning | yes |
| Viewer | debugging gaits | yes (desktop) |
| Driven from Go; same controller code as on the robot | hardware-in-the-loop parity | yes |
| Robot description from EasyRobot's types (DH chains, `chain.Body`) → URDF/MJCF | one source of truth for geometry | yes (exporter) |
| Wheeled robots (differential, mecanum, steer kinematics exist) | the wheeled avoider | soon |

## 2. Go options

| Option | Kind | Activity / licence | cgo? | Fit against the needs |
|---|---|---|---|---|
| **Cogent Lab `physics`** (`cogentcore.org/lab/physics`) | pure-Go 3D engine (GoSL: the same code on CPU or GPU) | `cogentcore/lab` (where `physics` lives) last pushed 2026-11-03, 18 stars, used in axon; BSD-3 (observed via API; the 2026-11-10 date belongs to `cogentcore/core`) | no for physics; its GPU path and `xyz` viewer use WebGPU (`#verify` how) | **XPBD** solver (newton-physics design); joints Revolute, Prismatic, Ball, Fixed, Free, Distance, D6, PlaneXZ with **target position/velocity + stiffness/damping** (a PD servo); contacts between primitives (Plane, Sphere, Capsule, Cylinder, Cone, Box); **parallel worlds**; first-person views from a body via `xyz`. **Gaps** (reported, its docs and README): no meshes or heightfields yet, "Sensors: TODO", sphere–sphere collision "definitely not right", 4+-link pendula unstable at some targets, restitution issues, Y-up, no URDF/MJCF; one shape per body |
| g3n `experimental/physics` | pure-Go engine inside a game engine | g3n pushed 2026-08, BSD-2; the package doc says "experimental and incomplete" (observed) | g3n's renderer uses OpenGL (cgo) | not usable for robotics |
| tbogdala/cubez | pure-Go 3D rigid bodies | **archived**, last push 2016 (observed) | no | dead |
| xavier268/solid | pure-Go rigid-body integration | 1 star, 2025 (observed) | no | no joints or contacts described |
| jakecoffman/cp | pure-Go Chipmunk2D port | active, MIT | no | **2D** only |
| ianremmler/ode | Go binding to ODE | last push 2016 (observed), MIT | yes | stale; ODE is old but capable |
| bbitechnologies/jolt-go; NilssonCreative/jolt-purego | bindings to **Jolt** | 7 and 2 stars; short-lived (a few days of commits each), unmaintained since 2025-11 and 2026-02 (observed) | jolt-go: cgo; jolt-purego: no (purego) | Jolt (MIT, very active) is a strong game engine: rigid bodies, constraints with motors, ragdolls. Not robotics-oriented: no sensors, no robot description format, no renderer |
| Bullet / PyBullet | C++ engine | bullet3 pushed 2026-10 (observed) | — | **no maintained Go binding found** |
| **MuJoCo** | C engine with a C API, MJCF models (imports URDF), built-in visualisation and offscreen rendering, multithreaded rollout | very active: **4.0.0 released 2026-11-06**, Apache-2.0 (observed) | a Go binding would be ours to write (cgo, or purego/dlopen) | **robotics-grade contacts** (soft-constraint convex optimisation), position actuators with gains and force ranges, IMU/touch/force sensors, cameras; first-party Python, JS/WASM, C#; third-party Swift, Java, Julia, Rust; **no Go** (observed in its README) |
| Gazebo (gz-sim) | external simulator | gz-transport pushed 2026-11 (observed), Apache-2.0 | — | rich sensors; its messages are protobuf over **gz-transport** (ZeroMQ + discovery), so a Go client would have to speak that protocol (`#verify` no Go library); a heavy install |
| Webots | external simulator | pushed 2026-11 (observed), Apache-2.0 | controller library is C | good sensor suite and robot library; extern controllers via its C library (cgo from Go); a heavy desktop app |
| Isaac Sim / Lab | external, NVIDIA GPU | — | — | overkill for our hardware and goals |

**Plainly:** no Go library covers legged contact dynamics with sensors well today. Cogent Lab's engine is the only pure-Go candidate worth a spike. MuJoCo is the engine that fits, but Go has no binding.

## 3. Architectures

| | (a) Pure-Go engine in EasyRobot | **(b) Go controller ↔ external simulator over protobuf** | (c) Go bindings via cgo or purego |
|---|---|---|---|
| Effort | low to start (Cogent Lab exists); **high** to reach legged fidelity (fix collisions, add sensors, meshes, heightfields) or to write our own | **low–medium**: a sidecar that owns MuJoCo + a step protocol + a Go client; the MJCF exporter | medium: bind the subset of MuJoCo's C API we use (`mj_loadXML`, `mj_step`, `mjData` fields, renderer); keep up with releases |
| Fidelity | XPBD trades some physical accuracy for robustness (reported, Cogent Lab docs); unproven for many-contact legged gaits | **high** (MuJoCo's contacts and actuators) | high (same engine) |
| Speed | fast in-process; GPU parallel worlds | one round trip per control step in lockstep; batch k physics substeps per control step (e.g. 2 ms physics, 20 ms control) | fastest (in-process) |
| Rendering for camera sensors | `xyz` first-person views (GPU) | MuJoCo offscreen rendering in the sidecar → frames over the link (`core/frame.proto` exists) | MuJoCo offscreen via its C API (OpenGL/EGL context management in Go is awkward) |
| Determinism | CPU mode probably deterministic, GPU parallel `#verify` | MuJoCo is deterministic for the same build and inputs (reported widely, `#verify` our build); lockstep keeps the controller in sync | same as (b) |
| Same controller code on sim and robot | yes (in-process) | **yes**: the Go controller (`walk.Walker.Step` → joint targets) runs unchanged; only the backend differs | yes |
| TinyGo / HIL | n/a (host only) | **HIL possible:** the real ESP32 controller talks to the sim over serial instead of to servos (the protocol is the same protobuf) | n/a |
| Deployment | `go build` only | a Python (or C) runtime beside the Go program; a download needing approval | a native `libmujoco` + (cgo) a C toolchain; purego needs only the shared library |

**How the Go side stays backend-neutral:** one Go interface, implemented by every backend:

```go
type Sim interface {
    Load(model []byte) error                                                    // MJCF generated from chain.Body
    Step(cmd *simpb.Command, out *simpb.State) error                            // lockstep: apply targets, advance k substeps
    Render(cam string, w, h int, buf []byte) (camera.Frame, error)
    Reset(seed uint64) error
}
```

- The existing **device seams** let a sim stand in for hardware at the driver layer too (observed):
  - `x/devices.PWMDevice` (servos via `ServoArray`);
  - `x/devices.I2C` (a fake MPU6050);
  - `drivers/camera.Backend`;
  - `x/devices/lidar.Device`.
- Servos, IMU and encoders are concrete structs today (no interfaces), so the first slice feeds joint targets directly, not through PWM.

**Protocol** (owner rule: protobuf schemas are the wire contract): a new `types/sim` package.
- **`Command`**: sim time, joint position targets (radians) or torques, a per-actuator gain override, an optional `render` request.
- **`State`**:
  - sim time;
  - joint positions and velocities;
  - **ground-truth body pose** and twist;
  - IMU accelerometer and gyro (with configurable noise);
  - foot contacts and normal forces;
  - actuator forces (saturation flags);
  - optional camera `Frame`.
- **Transport:** length-prefixed protobuf over TCP or a Unix socket at first. dndm once its route defects are fixed (`docs/architecture/dndm-and-protobuf.md:16-49` lists them, observed by the scout), so sim and robot share routes.

## 4. Recommendation and first slice

**Recommendation:**
- **(b) now:** MuJoCo in a sidecar, Go controller in lockstep, MJCF generated from EasyRobot's kinematic types.
- **(c) later:** a `purego` binding to `libmujoco` removes the sidecar once the `Sim` interface is stable. purego avoids cgo; jolt-purego shows the approach works for a physics library.
- **(a) as a spike:** Cogent Lab's `physics` behind the same interface for fast in-process CI checks, kept only if it walks the hexapod stably.

**First slice: S1, a hexapod walking in MuJoCo from EasyRobot's own gait code:**

| Step | Done when | Kill / rethink if |
|---|---|---|
| S1a MJCF exporter (Go) | `chain.Hexapod(...)` → MJCF: body box, legs as capsules from coxa/femur/tibia lengths, revolute joints with the DH axes and `Min/Max` limits, **position actuators** (`kp`, `forcerange` from the servo stall torque, `ctrlrange` from limits), foot spheres with friction, IMU site + `accelerometer`/`gyro` sensors, foot `touch` sensors. New fields on the Go types: link mass, servo stall torque and speed | joint frames don't match `chain` FK at zero pose (a test compares FK positions to MuJoCo site positions) |
| S1b Sidecar | loads the MJCF, serves `Command`/`State` in lockstep (control 50 Hz, physics 2 ms), headless, plus an optional viewer window | step round trip > 2 ms on the Mac (would make lockstep slower than real time at 50 Hz × 10 substeps) → batch or go to purego |
| S1c Go loop | `walk.Walker.Step(dt, Twist, …)` → `Walker.Joints()` radians → `Command`; the **servo speed limit** applied in Go (rate-limit targets at `MaxVel`) | — |
| S1d Run | tripod gait at 0.1 m/s straight on a plane for 60 s; logged: distance vs commanded, body roll/pitch, foot slip, actuator saturation, sim speed | **< 50 % of the commanded distance**, or a fall, or roll/pitch > 15°, with a gait that works kinematically → **model parameters first** (masses, gains), not the controller; if still failing after tuning, the sim choice is questioned |
| S1e Determinism | the same seed and inputs → bit-identical `State` streams on two runs | not bit-identical → document the tolerance or find the nondeterminism before using sims in CI |
| S1f Speed | headless ≥ 10× real time on the Mac | < 10× → batch steps per round trip |

**After S1:**
- **S2:** heightfield terrain + IMU-based body-level feedback.
- **S3:** a camera on the body (QVGA, OV2640/OV3660-like intrinsics) streaming frames to the vision stack.
- **S4:** wheeled robot (differential) with encoders.
- **S5:** biped (a different actuator model; harder balance).
- **Spike A:** the same hexapod in Cogent Lab `physics` (CPU mode), comparing distance and slip against MuJoCo.

### What the simulator must provide for topological navigation / SLAM

The owner will build these on EasyRobot's existing graph, traversal and search primitives (`x/math/graph`: generic graph, A*, Dijkstra, BFS/DFS, RRT*, kd-tree, loop/component detection, observed; known defects in ER-028).

| Provide | Why |
|---|---|
| **Body-camera frames** at QVGA/QQVGA with the real cameras' intrinsics and exposure-like noise | the frontend + heads run on sim frames first |
| **Ground-truth poses** per frame (and per node) | measure VO error (RPE/ATE), label place revisits for place-code training, check loop closures |
| **Place layouts:** a few rooms and corridors with distinct textures and objects; doors; repeated-looking places to test false matches; **lighting and texture randomisation** | place recognition, kidnapped-robot tests, map building |
| Scripted trajectories and teleop in sim | dataset generation that matches the owner's recording procedure |
| Wheel-encoder and IMU streams | scale for VO; odometry fusion |
| Lidar rays (later) | 2D SLAM (`filter/slam`) comparisons |
| Reset to a pose; a "teleport" command | kidnapped-robot tests |

MuJoCo provides textured geometry, lights and offscreen cameras (reported, its docs; texture quality is plain, `#verify` whether it's enough for CNN features). Photorealism isn't needed for the first map experiments, but the sim-to-real gap for the vision heads will be large. Treat sim frames as a pipeline test, not training data, until shown otherwise.

## Knowledge map

| Quadrant | Item | Next step, owner |
|---|---|---|
| Known knowns | EasyRobot has IK, gait, wheel kinematics, graph search and device seams, and no physics or 3D code (observed); MuJoCo has no Go binding; Cogent Lab `physics` is the only active pure-Go 3D engine with joints and contacts | — |
| Known unknowns | MuJoCo lockstep round-trip cost from Go over a socket; bit-determinism on our builds | S1b/S1e |
| Known unknowns | Cogent Lab `physics` stability for a hexapod (its README lists collision and multi-link issues) | Spike A |
| Known unknowns | Servo models: stall torque and no-load speed of the owner's servos | owner (part numbers) |
| Unknown knowns | Which robot is first in sim (hexapod per the roadmap, or the wheeled avoider for vision navigation) | owner |
| Unknown unknowns | Sim-to-real gap of gait parameters on the real hexapod | after S1 and the first hardware walk |

## Decisions for the owner

| # | Decision | Recommendation |
|---|---|---|
| 1 | Architecture: MuJoCo as the reference simulator, driven from Go over protobuf (sidecar first, purego binding later); a pure-Go engine only as an optional CI spike | Yes |
| 2 | Approve downloads when S1 starts: MuJoCo 4.x (release binary, or the Python package in a local uv venv) | Yes |
| 3 | A `types/sim` protobuf contract (`Command`, `State`, camera `Frame`) and a Go `Sim` interface; length-prefixed TCP first, dndm when its route defects are fixed | Yes |
| 4 | Robot descriptions stay Go structs; add link mass and servo torque/speed fields; generate MJCF (URDF export later if needed) | Yes |
| 5 | First slice S1 (hexapod tripod gait in lockstep, kill criteria above), then camera (S3) for the vision stack | Yes; or the wheeled avoider first if vision navigation leads |
| 6 | Spike A: Cogent Lab `physics` with the same hexapod (needs approval to fetch the module) | Optional, after S1 |
| 7 | Servo part numbers (stall torque, speed) for the actuator model | Owner to supply |
