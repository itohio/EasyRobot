# Locomotion Package Specification (Go)

## Overview

The locomotion subsystem lives at `EasyRobot/pkg/control/locomotion`. It provides device-independent body control for heterogeneous robots by coordinating supports (wheels, legs, manipulators, thrusters) and the actuators that power them. TinyGo firmware and device-specific logic remain inside `EasyRobot/drivers/...`; this package focuses on high-level control, motion planning, and diagnostics, communicating with firmware via dndm pipelines.

## Core Design Principles

- **Layered Control**: `BodyCoordinator` → `Support` → `Actuator`, with explicit interfaces and responsibilities at each layer.
- **Device Independence**: Modules receive data exclusively through typed `SensorSuite` feeds; hardware access is delegated to transport bindings.
- **Actuator-Centric Abstraction**: Every movable effector implements the `Actuator` interface (DH-based, constraint-aware, trajectory capable).
- **Explicit Context**: All operations accept `context.Context` for cancellation, deadlines, and structured logging hooks.
- **Destination-Based Math**: Numerical routines operate on caller-provided buffers to avoid allocations.
- **Validated Configurations**: Protobuf + Options pattern ensures strong typing and runtime validation before controllers run.
- **Extensibility**: Adding a new support or actuator type requires implementing a small interface (`Support`, `Actuator`) without touching existing modules.
- **Safety First**: Built-in limit enforcement, watchdogs, and emergency-stop propagation across layers.

## Architectural Layers

```
BodyCoordinator (pkg/control/locomotion/body)
   ├── Support Modules (pkg/control/locomotion/support)
   │      ├── Differential Drive
   │      ├── Omnipod (multi-leg)
   │      ├── Tracked / Skid-steer
   │      ├── Manipulator Base / Stabilizer
   └── Runtime Utilities (pkg/control/locomotion/runtime)
          ├── Gait Library
          ├── Command Routing & Diagnostics
          └── Validation / Safety helpers
Actuator Modules (pkg/control/locomotion/actuator)
   ├── DH Chains (legs, arms)
   ├── Wheel / Roller units
   └── Custom effectors (thrusters, winches)
```

- `proto/types/locomotion/types.proto` defines shared configuration types (`BodyConfig`, `SupportConfig`, `ActuatorConfig`, `GaitConfig`).
- `docs/` holds living documentation including this spec, design diagrams, and transport mapping tables.

### Robot Categorization Matrix

| Robot Class | Supports | Actuator Types | Notes |
| --- | --- | --- | --- |
| Differential wheeled | `support.Differential` | Wheel actuators | Emphasizes odometry, current limiting |
| Quadruped / Hexapod | `support.Omnipod` | Leg DH actuators | Requires gait scheduling, CoM stability |
| Balancing bipod/monopod | `support.Balance` (specialization of Omnipod) | Leg DH actuators | Heavy reliance on IMU and torque sensing |
| Manipulator on mobile base | `support.Composite` (drive + arm) | Wheel + arm actuators | Scheduler blends base and arm trajectories |
| Aerial / thruster platforms | `support.ThrustArray` | Rotor/Thruster actuators | PID/alloc control, momentum management |

## Interface Definitions

```go
package locomotion

type RobotBody interface {
    Configure(ctx context.Context, cfg BodyConfig, opts ...BodyOption) error
    RegisterSupport(id SupportID, support Support) error
    RegisterDiagnostics(sink DiagnosticsSink)
    Update(ctx context.Context, sensors SensorSuite, dt time.Duration) (BodyState, error)
    Request(ctx context.Context, target BodyTarget) (BodyReport, error)
}

type Support interface {
    ID() SupportID
    Configure(ctx context.Context, cfg SupportConfig, opts ...SupportOption) error
    BindActuators(register ActuatorRegistry) error
    Plan(ctx context.Context, target SupportTarget, body BodyState) (SupportCommand, error)
    Update(ctx context.Context, sensors SupportSensors, dt time.Duration) (SupportState, error)
    Diagnostics(ctx context.Context, sink DiagnosticsSink)
}

type Actuator interface {
    ID() ActuatorID
    Configure(ctx context.Context, cfg ActuatorConfig, opts ...ActuatorOption) error
    Apply(ctx context.Context, cmd ActuatorCommand) (ActuatorState, error)
    SolveForward(jointAngles []float64, pose *Pose) error
    SolveInverse(pose Pose, seed []float64, solution []float64) (int, error)
    Limits() ActuatorLimits
    Diagnostics(ctx context.Context, sink DiagnosticsSink)
}

type Gait interface {
    Configure(params GaitParams) error
    Advance(ctx context.Context, dt time.Duration, body BodyState, supports []SupportState) (PhaseState, error)
    SupportsInContact() []bool
}

type SensorSuite interface {
    Timestamp() time.Time
    Quality() SensorQuality
    IMU() (IMUData, bool)
    Force(id SupportID) (ForceData, bool)
    Encoder(id ActuatorID) (EncoderData, bool)
    Contact(id SupportID) (ContactData, bool)
}
```

- `SupportID`/`ActuatorID` are strongly typed strings to avoid accidental cross-referencing.
- `ActuatorRegistry` maps logical actuators to transport endpoints (dndm topics) and optionally simulation adapters.
- Interfaces reject implicit dependencies; each module receives explicit configuration + injected collaborators.

## Data Model

### Targets & Commands

- `BodyTarget`: pose, velocity, force envelopes, and desired support allocation. Includes stability hints (e.g., keep CoM within polygon) and optional motion constraints (max jerk, posture bounds).
- `SupportTarget`: mode-specific goals (wheel velocities, foot placements, thrust vectors) plus safety caps.
- `ActuatorCommand`: joint trajectories (position/velocity/torque), limit masks, feed-forward terms, sequencing metadata.
- `BodyReport` and `SupportCommand` capture predicted motion, safety status, and requested actuator commands for telemetry.

### States

- `BodyState`: fused pose, velocity, support load distribution, fault flags, timing budget usage.
- `SupportState`: contact status, slip indicators, energy usage, actuator readiness.
- `ActuatorState`: joint measurements, effort, temperature, calibration offsets, fault classification.

### Validation & Errors

- Configuration validation occurs via `Validate()` methods on each proto-derived struct, producing aggregated error lists.
- Runtime errors are wrapped with `%w` and include subsystem identifiers (e.g., `fmt.Errorf("support %s plan: %w", id, err)`).
- Sensor ingestion enforces timestamp monotonicity; stale or out-of-window data flips the appropriate quality bit.

## Module Specifications

### Body Coordinator (`pkg/control/locomotion/body`)

**Responsibilities**
- Manage main coordination loop with deterministic timing.
- Distribute body-level targets to supports based on weighting strategy (e.g., load sharing matrix).
- Aggregate feedback, run stability checks, and trigger safety responses (emergency stop, degraded modes).
- Interface with diagnostics, logging, and metrics systems.

**Key Components**
- `Scheduler`: orchestrates update order, including asynchronous supports.
- `SafetyManager`: monitors limits (velocity, acceleration, torque) and handles watchdog resets.
- `LoadBalancer`: solves optimization problem allocating forces across supports.
- `TransportAdapter`: registers dndm topics and maps support/actuator IDs to publishers/subscribers.

### Support Modules (`pkg/control/locomotion/support`)

1. **Differential Drive**
   - Forward/inverse kinematics, odometry integration, IMU fusion.
   - Velocity and acceleration limiting, anti-slip control, current limiting.
   - Emits diagnostic metrics (wheel speed, encoder drift, skid detection).

2. **Omnipod (Legged)**
   - Configurable leg count, per-leg coordinate frames.
   - Integrates with gait library for stance/swing scheduling.
   - Balances body pose adjustments with leg workspace constraints.
   - Supports balance-only mode for stationary stabilization.

3. **Composite / Hybrid**
   - Wraps multiple supports (e.g., drive + manipulator) into cohesive unit.
   - Provides arbitration to avoid conflicting commands (e.g., base motion during manipulation).

4. **Thruster / Rotor Arrays** (future milestone)
   - Accepts thrust vectors, enforces momentum/energy budget.
   - Emphasizes fast feedback via dndm prioritized topics.

Each support exposes a README documenting assumptions, required sensors, and sample configs.

### Actuator Modules (`pkg/control/locomotion/actuator`)

- **DHChain**: general kinematic chain with Damped Least Squares IK, joint limit enforcement, workspace boundary checking, and seeds for multiple solutions.
- **WheelUnit**: converts between wheel angular velocity and chassis motion, handles rolling radius calibration.
- **TorqueControlledJoint**: manipulator joints with torque limits, compliance, and temperature derating.

Shared utilities include:
- Constraint evaluators (`WithinJointLimits`, `WithinWorkspace`).
- Calibration flows (zeroing, backlash measurement) with persistent storage hooks.
- Profile generators for minimum-jerk and trapezoidal trajectories.

### Runtime Utilities (`pkg/control/locomotion/runtime`)

- **Gait Library**: tripod, wave, ripple, balance; returns `PhaseState` with contact schedule, desired footholds, and timing diagnostics.
- **Command Router**: maps `SupportCommand` to actuator-level dndm messages, handles retries and acknowledgements.
- **Diagnostics Hub**: central place to aggregate health events and publish to monitoring/telemetry.

## Transport & dndm Integration

- Command topics: `control.actuator.<actuator-id>.command`
- Feedback topics: `telemetry.actuator.<actuator-id>.state`, `telemetry.support.<support-id>.state`
- Control loop publishes `control.body.state` and `control.body.alerts` for system monitoring.
- Pipelines include bounded queues, optional lossy sampling for high-rate sensors, and priority channels for emergency stops.
- dndm storage is used for persistent configuration snapshots and for log replay in simulation environments.

## Configuration Strategy

1. Author configuration in protobuf (`proto/types/locomotion/types.proto`).
2. Generate Go structs with validation helpers.
3. Provide Options wrappers to customize defaults per robot without deep copies.
4. Store per-robot overrides in `configs/<robot>.yaml`, referencing generated types via dndm's config loader.
5. Validate configs in CI using table-driven tests to ensure constraints remain satisfied.

## Testing & Verification

- **Unit Tests**: Table-driven tests for each support/actuator verifying kinematics, constraint enforcement, and error paths.
- **Integration Tests**: Compose BodyCoordinator with multiple supports using fake dndm transports and sensor feeds.
- **Hardware-in-the-Loop**: Connect to TinyGo firmware via dndm to verify latency budgets and safety signaling.
- **Performance Benchmarks**: Track allocation counts, timing (200 Hz + 500 Hz loops), and end-to-end pipeline latency.
- **Simulation Assets**: Provide Gazebo/Isaac adapters for regression scenarios.

## Safety & Diagnostics

- Emergency stop signals propagate through BodyCoordinator to supports and actuators within one control tick.
- Watchdog monitors for missed actuator acknowledgements; triggers degraded mode before full stop.
- Diagnostics include stability margin, slip detection, actuator saturation, and transport health.
- Structured logging (zap/slog) with per-module trace IDs; integrates with dndm telemetry exporters.

## Future Work & Out-of-Scope Items

- Planning layer (`pkg/control/planning`) for high-level motion/action sequencing.
- Vision package refactor to feed perception outputs through planning to locomotion.
- Transport-level QoS tuning for aerial robots requiring sub-millisecond command paths.
- Additional support types: tracked vehicles, snake robots, underwater thruster arrays.

## Glossary

- **Actuator**: Physical effector implementing motion (joint, wheel, thruster) with its constraints and calibration data.
- **Support**: Logical coordination unit translating body intents to actuator commands (e.g., leg group, wheelbase).
- **BodyCoordinator**: Top-level orchestrator ensuring overall stability, safety, and mission execution.
- **dndm**: Library handling distributed messaging, storage, and pipeline management for EasyRobot components.
