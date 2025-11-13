# Gait Module Implementation Plan

## Scope
Stand up the gait module structure outlined in `DESIGN.md`, comprising shared contracts (`gait/types`), the support endpoint planner (`gait/support`), and rigid-body gait-aware kinematics (`gait/rigidbody`). Each package must compile independently, consume explicit time inputs, and integrate with terrain-awareness modes.

## Package Breakdown
1. **gait/types**
   - Define planner interfaces (`SupportEndpointPlanner`, `RigidBodyPlanner`, `GaitScheduler` hooks).
   - Provide core data structures (`PhaseState`, `EndpointPose`, `FootState`, `GaitTemplate`, `TerrainMode`, capability flags).
   - Document supported features via Go doc comments; export option structs.
2. **gait/support**
   - Implement `SupportEndpointPlanner` using parametric sub-phase trajectories.
   - Remain agnostic to terrain capability; branch on `TerrainMode`.
   - Expose constructor with dependency injection for terrain sampling, swing profiles, clocks, and per-phase path shapes (`WithPhaseShape`).
3. **gait/rigidbody**
   - Implement rigid-body forward/backward kinematics adaptable to multi-leg gaits.
   - Consume body linear/angular velocity plus endpoint states to update pose estimates.
   - Publish leg adjustment vectors back to support planner via `SupportCommand`.

## Goals
- Deterministic, time-dependent endpoint generation based on gait phase progression.
- Smooth transitions across sub-phases (lift, swing, touchdown, retreat) with configurable durations.
- Interfaces that integrate with `GaitScheduler` presets and the body planner.
- Unit tests covering canonical phase progressions and edge cases (phase resets, partial leg disable).

## Goals
- Deterministic, time-dependent endpoint generation based on gait phase progression.
- Smooth transitions across sub-phases (lift, swing, touchdown, retreat) with configurable durations.
- Interfaces that integrate with `GaitScheduler` presets and the body planner.
- Unit tests covering canonical phase progressions and edge cases (phase resets, partial leg disable).

## Non-Goals
- Full dynamic simulation of the leg or body (handled elsewhere).
- Terrain sampling logic (use injected `TerrainSampler`).
- Optimization-based trajectory solving; rely on parametric curves.

## Architecture
1. **Contracts (`gait/types`)**
   - Interfaces declare `Update` signatures accepting `Context`, `UpdateRequest` (with `Timestamp` + `Delta`).
   - Enumerations for `PhaseMode`, `TerrainMode`, `PlannerFeature`.
   - Structs for presets, bindings, transition policies; zero-value safe defaults.
2. **SupportPathPlanner (`gait/support`)**
   - Dependencies injected via options: terrain sampler, clock/timestep override, swing profile provider.
   - Stores per-leg state machines tracking sub-phase (Lift, SwingForward, Touchdown, SupportBack).
   - Exposes `Update(ctx, req)` -> `FootState` and `Preview(ctx, legID, horizon)` for lookahead.
   - Handles terrain capability downgrades seamlessly.
   - Default path shapes: support phase linear, swing phases arc; allow overrides through constructor.
3. **RigidBodyPlanner (`gait/rigidbody`)**
   - Maintains body pose estimate, integrates linear/angular velocities over variable timestep.
   - Solves for endpoint adjustments to keep CoM within support polygon, leveraging support planner outputs.
   - Provides backward mapping: given endpoint states, infer body twist for feedback controllers.

## Data Flow
`Update` sequence per leg:
1. Resolve active gait instance and phase ratio.
2. Determine sub-phase durations (from template or overrides).
3. Interpolate trajectory segment given elapsed sub-phase time.
4. Query terrain (if available) to adjust touchdown/stance height. Support three observation modes:
   - No terrain knowledge: assume nominal flat reference plane.
   - Height-aware: use sampler response to offset target pose.
   - Contact-only: refine touchdown height retrospectively based on sensed contact.
5. Compose `FootState` with position, velocity, and contact flag.

## Testing Strategy
- Table-driven tests for each sub-phase verifying expected positions/velocities.
- Tests for mid-step template switch ensuring continuity.
- Terrain mock returning variable heights to confirm touchdown adjusts accordingly.
- Time-step variation tests to ensure stability across dt changes.
- Rigid-body tests verifying CoM stays inside support polygon and pose integration matches analytic expectations.

## Open Questions / Risks
- How to expose per-leg state for debugging? Proposal: optional observer callback.
- Should planner buffer future samples for preview mode? Possibly reuse generator to emit `Preview(n)`.
- Need to confirm integration contract with body planner regarding support/backward displacement reference frame.
- Validate rigid-body planner numerical stability under fast timestep changes; may need configurable integrator.

