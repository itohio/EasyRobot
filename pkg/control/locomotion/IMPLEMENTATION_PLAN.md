# Locomotion Package Implementation Plan (Go)

## Scope & Packaging

- Relocate the locomotion package to `pkg/control/locomotion` so imports read naturally (`EasyRobot/pkg/control/locomotion`).
- Mirror TinyGo, device-specific drivers under `EasyRobot/drivers/...`; keep this package device agnostic and controller centric.
- Move protobuf definitions to `proto/types/locomotion/types.proto`; regenerate bindings only after the schema stabilizes.
- Update documentation and tooling references to the new paths before executing code moves.

## Architectural Layers

| Layer | Responsibility | Primary Directory | Notes |
| --- | --- | --- | --- |
| `BodyCoordinator` | Whole-body command orchestration, safety interlocks, timing | `pkg/control/locomotion/body` | Hosts coordination loop, load distribution, stability logic |
| `SupportModules` | Locomotion modes (differential, omnipod, tracked, arm base) | `pkg/control/locomotion/support` | Each support converts body intents to actuator commands |
| `ActuatorModules` | Kinematics and constraints for individual effectors | `pkg/control/locomotion/actuator` | Wraps reusable kinematics, limits, calibration flows |

- Shared utilities (gait patterns, command routing, data validation) live under `pkg/control/locomotion/runtime`.
- Configuration lives in protobuf + Option patterns; runtime conversion produces strongly typed configs per module.
- dndm transport handles message pipelines between this package and TinyGo firmware or simulators. Control logic never assumes direct device access.

## Interface & Data Model Details

### Interfaces

- `RobotBody`  
  - Methods: `Configure(ctx context.Context, cfg BodyConfig) error`, `Request(ctx context.Context, target BodyTarget) (BodyReport, error)`, `Update(ctx context.Context, sensors SensorSuite, dt time.Duration) (BodyState, error)`  
  - Manages scheduling, safety, and support orchestration; accepts dependency injection for timing, telemetry, and logging.

- `Support`  
  - Methods: `BindActuators(actuators []Actuator) error`, `Plan(target SupportTarget, state BodyState) (SupportCommand, error)`, `Update(ctx context.Context, sensors SupportSensors, dt time.Duration) (SupportState, error)`  
  - Implements locomotion modes such as differential drive, omnipod gaiting, or stabilization platforms.

- `Actuator`  
  - Methods: `Apply(ctx context.Context, cmd ActuatorCommand) (ActuatorState, error)`, `SolveForward(jointAngles []float64, pose *Pose) error`, `SolveInverse(pose Pose, solution []float64) error`, `Limits() ActuatorLimits`  
  - Encapsulates DH parameters, calibration, and constraint enforcement. Replaces the previous “appendage” naming everywhere.

- `Gait`  
  - Methods: `Advance(ctx context.Context, params GaitParams, dt time.Duration, body BodyState) (PhaseState, error)`  
  - Supplies stance/swing distribution and timing for supports that require periodic modulation.

- `SensorSuite`  
  - Composition of typed sensor feeds (IMU, force, encoder, proximity). Provides typed accessors and validation states to prevent silent failures.

All interfaces avoid package-level state, pass dependencies explicitly, and are sized for single responsibility.

### Data Structures & Validation

- `BodyTarget` expresses desired pose, velocity, and support allocation with optional posture constraints. Validation includes coherence checks for achievable CoM and contact plans.
- `SupportTarget` wraps mode-specific intents (e.g., wheel velocities, leg footholds) plus safety envelopes.
- `ActuatorCommand` contains joint-level setpoints, torque limits, and trajectory hints. Always tied to `ActuatorLimits`.
- `SensorSuite` records timestamps, quality metrics, and stale-data flags. Supports incremental ingestion from dndm topics.
- Error handling wraps context with `fmt.Errorf("support %s: %w", id, err)` style messages to retain causality.

### Command & Telemetry Pipeline (dndm)

1. BodyCoordinator computes `SupportCommand` payloads per support.
2. Each Support serializes actuator commands via dndm publishers (`control.actuator.<id>.command`).
3. Firmware responses stream into `SensorSuite` through dndm subscriptions with bounded buffering and replay protection.
4. A pipeline registry maps logical actuators to transport endpoints, allowing simulation backends or real hardware selection at runtime.
5. Backpressure and timeout handling rely on context deadlines; supports emit degraded states when downstream is unavailable.

## Milestones

### Milestone A – Foundations (Weeks 1-3)

- Establish new directory layout (`body`, `support`, `actuator`, `runtime`, `docs`).
- Port existing core types into `locomotion.go`, renaming appendage → actuator.
- Define protobuf schemas for body/support/actuator configs in `proto/types/locomotion/types.proto`.
- Implement basic validation utilities and test scaffolding (`go test ./pkg/control/locomotion/...`).

### Milestone B – Core Supports (Weeks 4-6)

- Implement Differential Drive support module with forward/inverse kinematics, odometry, and validation.
- Stand up Omnipod support skeleton with configurable leg placements and gait hooks.
- Provide unit tests using table-driven cases for multiple robot geometries.
- Integrate sensor ingestion through dndm mocks to validate command/feedback wiring.

### Milestone C – Actuator Library (Weeks 7-9)

- Build DH-based actuator with constraint handler, destination-based math, and IK fallbacks.
- Add actuator calibration flows (zeroing, compliance) and persistence hooks.
- Provide conformance tests to ensure all actuators satisfy `Actuator` interface guarantees.
- Prepare reusable fixtures per robot class (leg, wheel pod, arm joint).

### Milestone D – Gaits & Motion Strategies (Weeks 10-12)

- Prototype gait algorithms (tripod, wave, ripple) against omnipod harness with pure simulation.
- Promote mature algorithms into production library with stability metrics, duty factor tuning, and context-aware transitions.
- Implement balance strategies for monopod/bipod supports leveraging CoM projection and sensor fusion.

### Milestone E – Multi-Support Coordination (Weeks 13-15)

- Implement BodyCoordinator scheduling, support arbitration, and load distribution.
- Add safety interlocks (emergency stop, velocity/accel limits, watchdogs).
- Provide health monitoring, diagnostics events, and timing budget enforcement.
- Validate heterogeneous setups (wheels + legs) in integration tests.

### Milestone F – Manipulation & Extended Modalities (Weeks 16-18)

- Introduce manipulator support module built atop actuator library with trajectory planning.
- Add tool-centric coordinate management and blending with base locomotion commands.
- Extend dndm pipeline mappings for new actuator types and confirm round-trip latency budgets.

### Milestone G – System Integration & Performance (Weeks 19-22)

- Assemble reference robot configurations (bipod balancer, quadruped walker, hexapod, wheeled manipulator).
- Profile runtime, optimize hot paths (matrix ops, allocs), and verify rate targets (200 Hz coordinator, 500 Hz support loops).
- Deliver comprehensive test suite: unit, integration, hardware-in-the-loop simulations.
- Prepare documentation, API references, and troubleshooting guides.

## Cross-Cutting Concerns

- **Safety & Diagnostics**: Centralized safety manager in `body` layer aggregates watchdogs, emergency stop triggers, and limit violations.
- **Timing Budgets**: Each module exposes `TickBudget()` metadata for runtime schedulers; violations raise structured diagnostics.
- **Configuration Management**: Protobuf configs converted via Options pattern; per-robot overrides stored in `configs/<robot>.yaml` referencing generated types.
- **Logging & Telemetry**: Use structured logging (zap/slog) via injected interface; metrics exported through dndm telemetry topics.

## Dependencies & Future Work

- Depends on existing kinematics (`pkg/math/kinematics`), linear algebra (`pkg/math/vec`, `pkg/math/mat`), and filters (currently under `pkg/core`, planned relocation).
- Requires dndm library for transport, storage, and pipeline composition; document its API touchpoints in module README files.
- Future refactors: dedicate planning layer (`pkg/control/planning`) and rebuild vision package to interface through planning and locomotion rather than directly.
- Out-of-scope placeholders: vision refactor, high-level action planner, TinyGo firmware updates (tracked in their respective SPEC/PLAN docs).

## Risk Mitigation

### Technical Risks

1. **Layer Coupling**: Accidental cross-layer dependencies.  
   _Mitigation_: Enforce dependency rules via linting, integration tests, and package-level documentation.
2. **Numerical Instability**: IK singularities or noisy sensors.  
   _Mitigation_: Damped least squares, sensor quality flags, fallback controllers.
3. **Transport Latency**: dndm pipeline delays impacting control loops.  
   _Mitigation_: Budgeted queues, priority messages for safety commands, offline profiling.
4. **Generalization Across Robots**: One-size abstractions may hinder specialized robots.  
   _Mitigation_: Use extension points (interfaces, options) and maintain reference configs per robot class.

### Schedule Risks

1. **Directory Move Complexity**: Large rename may disrupt imports.  
   _Mitigation_: Stage documentation updates first, script the move, run `go test ./...` to catch regressions.
2. **Support Explosion**: Adding many supports simultaneously.  
   _Mitigation_: Prioritize differential + omnipod before expanding to manipulator, tracked, or hybrid systems.
3. **Performance Tuning Delay**: Optimization left too late.  
   _Mitigation_: Instrument timing from Milestone B onward and track budgets in CI.

## Success Criteria

### Functional Requirements

- [ ] BodyCoordinator can coordinate ≥2 heterogeneous supports while maintaining stability margins.
- [ ] Omnipod support reconfigures for bipod, quadruped, and hexapod with shared code paths.
- [ ] Gait library provides tripod, wave, ripple, and balance strategies with documented parameters.
- [ ] Actuator library achieves <1 mm / <0.1° accuracy within workspace constraints.
- [ ] Differential drive support holds ±2 % kinematic accuracy with dead reckoning and IMU fusion.
- [ ] Manipulator support delivers 6 DOF trajectory execution with tooling awareness.

### Performance Requirements

- [ ] BodyCoordinator loop sustains 200 Hz with <5 ms jitter and proper degradation on overload.
- [ ] Support modules hit 500 Hz updates with documented alloc profiles.
- [ ] IK solves remain <5 ms for 6 DOF actuators; gait updates remain <1 ms.
- [ ] Memory usage <50 KB per support instance (excluding telemetry buffering).

### Quality Requirements

- [ ] >90 % test coverage across body, support, actuator, and gait packages.
- [ ] Interfaces documented with usage examples and error semantics.
- [ ] Comprehensive validation, logging, and recovery for configuration and runtime errors.
- [ ] Reference configs and tutorials for each robot archetype.
- [ ] Clearly defined extension points for new support or actuator types.
