# Legged Gait Planning Design

## Goal
Support gait planning for legged robots with an arbitrary number of legs by decomposing motion planning into modular planners that can be composed for different locomotion strategies (crawl, trot, bound, etc.). The design must integrate with existing motion planning and kinematics packages, expose clear interfaces for downstream planners, and remain agnostic to specific robot morphology beyond configurable leg metadata.

## Principles
- Composition-first: planners remain independently testable and swappable.
- Single responsibility: each planner handles one concern (endpoint paths, body motion, terrain awareness).
- Deterministic outputs: identical inputs yield identical trajectories.
- Real-time friendly: planners avoid global optimization loops and prefer incremental, streaming APIs.
- Explicit time base: all planner APIs accept timestamps or time deltas; no implicit fixed-step assumptions.

## Module Layout
- `gait/types`: Contracts and data structures shared across gait planners.
  - Planner interfaces (`SupportEndpointPlanner`, `RigidBodyPlanner`, etc.).
  - Gait descriptors (`GaitTemplate`, `GaitInstance`, `TransitionPolicy`).
  - Capability flags documenting supported features (terrain awareness modes, preset switching, variable timestep).
- `gait/support`: Concrete implementation of the support endpoint path planner driven by phase state and endpoint requests. Depends only on `gait/types` contracts.
- `gait/rigidbody`: Rigid body kinematics planner that maps body linear/angular velocity into leg endpoint adjustments and back-estimates body pose from endpoint measurements.

## High-Level Architecture
1. **Leg Endpoint Planner**
   - Generates per-leg Cartesian trajectories for foot endpoints in world coordinates.
   - Supports switching path orientation mid-motion (e.g., change swing direction) without discontinuities.
   - Consumes desired gait phase schedule and leg contact templates.
   - Publishes desired foot positions/velocities/phase state per planning tick.
2. **Body Planner**
   - Receives desired body pose/velocity targets and terrain-aware foot placements.
   - Computes center-of-mass (CoM) motion and distributes adjustments to each leg motion vector.
   - Ensures dynamic consistency (e.g., maintains support polygon, balance).
3. **Terrain Awareness Module**
   - Supplies ground height map or implicit plane beneath each leg.
   - Provides surface normals and friction estimates when available.
   - Offers query API for both leg endpoint planner (landing placement) and body planner (CoM adjustments).

The planners communicate through typed channels or callback interfaces, enabling chaining inside the overall motion planner pipeline (`motion/planner`).

## Data Model
- **LegDescriptor**: Unique leg id, default attachment pose (body frame), stroke limits, preferred swing plane vector.
- **GaitPhase**: Phase offset, duty factor, contact schedule.
- **FootState**: Position, velocity, contact flag, phase progress, orientation frame.
- **BodyState**: Pose, velocity, support polygon vertices, CoM projection.
- **TerrainSample**: Position, height, normal vector, confidence.

All structures must have zero-value defaults that are safe (e.g., zero height => flat terrain).
Every time-dependent structure must track the time base (absolute timestamp and/or delta) to maintain deterministic progression.

## Preset Gait Library & Configuration
- **Goals**
  - Allow authoring gait presets independent of robot topology or planner internals.
  - Enable smooth transitions between presets (walk ↔ run ↔ jump) with mid-step blending.
  - Provide declarative format that higher-level behaviors can manipulate at runtime.
- **Representation**
  - `GaitTemplate` is an immutable description containing:
    - `Id` (string): canonical name (`"walk"`, `"trot"`, `"bound"`).
    - `CycleTime` (seconds).
    - `PhaseMap`: map of logical leg roles (`"front_left"`, `"hind_right"`, etc.) to normalized phase offsets `[0,1)`.
    - `DutyCycles`: map of leg roles to stance ratio `[0,1]`.
    - `ContactSequence`: ordered list of contact events referencing leg roles with optional tags (`support`, `launch`, `land`).
    - `SwingProfile`: parameters for spline shape (e.g., apex height, lateral bias) abstracted from absolute coordinates.
    - `TransitionHints`: metadata for how to blend to/from other templates (e.g., cross-fade window, momentum requirements).
  - `LegRoleBinding`: runtime mapping of leg roles defined in template to actual `LegDescriptor` ids for robots with different topologies. Supports many-to-one (paired legs share role) and one-to-many (split role across legs) via weighting factors.
  - `GaitInstance`: template plus resolved bindings, phase timers, and runtime overrides (speed scaling, stride length).
- **Selection & Transition**
  - `GaitScheduler` maintains active and target `GaitInstance`s.
  - Mid-step transitions blend phase progress using normalized time; stance legs honor current contact commitments while swing legs adopt new template parameters at configurable blend ratio.
  - Provide `TransitionPolicy` objects (e.g., `Immediate`, `CompleteCurrentCycle`, `AtNextSupport`) to control swap semantics without encoding robot-specific rules.
- **Configuration Format**
  - Store presets in declarative formats (YAML/TOML/JSON) under `motion/gait/presets/`.
  - Parser populates `GaitTemplate`; validation ensures role coverage and consistency (phase offsets sorted, duty cycles within bounds).
  - Avoid referencing implementation classes; rely solely on above data structures.

## Leg Endpoint Planner
- **Inputs**
  - `LegDescriptor` list (arbitrary length).
  - Desired gait definition (phase timing, sequence).
  - Waypoint queue or stream representing target footholds/body direction.
  - `TerrainSampler` interface for height queries.
  - Optional orientation override requests (e.g., rotate swing path).
- **Outputs**
  - Per-leg `FootState` sequences at planner timestep.
- **Responsibilities**
  - Maintain per-leg finite state machine (swing vs stance).
  - Interpolate swing trajectories using bezier/spline segments parameterized by phase.
  - Apply orientation switch mid-swing by blending between orientation frames.
  - Enforce kinematic reach and velocity limits; surface adaptation by adjusting target touchdown height using terrain.
- **Key Considerations**
  - Provide `Preview(nSteps)` to anticipate foot positions for body planner.
  - Ensure continuity when modifying orientation: use slerp or rotation interpolation around vertical axis.
  - Support partial leg failure by marking leg inactive and redistributing phase timing.
  - Expose phase-specific path shapes with defaults (`Support` linear, `Lift/Swing/Touchdown` arc) and allow overrides via constructor options.

## Body Planner
- **Inputs**
  - Desired body pose/velocity command (from higher-level motion planner).
  - Support polygon predicted from leg endpoint planner (active stance legs).
  - Terrain samples under each stance leg.
- **Outputs**
  - Body trajectory (`BodyState` timeline).
  - Adjustment vectors for each leg (offsets to combine with endpoint planner outputs).
- **Responsibilities**
  - Maintain CoM within support polygon margin; compute corrective translations/rotations.
  - Solve small optimization (quadratic or weighted least squares) to distribute body command across legs respecting constraints.
  - Update leg motion vectors to compensate for body rotation/translation during stance.
  - Provide differential commands (velocity/acceleration) to interface with `motion/planner`.
- **Key Considerations**
  - Return early when command infeasible (e.g., support polygon degenerate); signal via structured error.
  - Use context-aware APIs for cancellation/timeouts (e.g., streaming planner loop).
  - Keep solver modular to allow trotting vs walking heuristics.

## Terrain Awareness Module
- **Interface**
  - `SampleFootprint(pos)` -> `TerrainSample`.
  - `BatchSample(positions []Vec3)` for efficiency.
  - `EstimatePlane(footIds)` -> plane fit for stance legs.
- **Operation Modes**
  - `ModeNone`: no terrain knowledge; planners assume nominal flat ground and rely on default touchdown height.
  - `ModeHeights`: sampler returns heights (and optionally normals) within kinematic reach for predictive adjustment.
  - `ModeContactOnly`: system observes only contact/no-contact events at runtime with no explicit heights.
- **Responsibilities**
  - Cache recent samples, expose TTL to avoid stale data.
  - Allow plugging different backends (height map, point cloud, on-board sensors).
- **Planner Agnosticism**
  - Leg endpoint planner treats terrain data as optional; fallback heuristics maintain continuity regardless of mode.
  - Body planner adjusts support polygon margins based on reported confidence; degrades gracefully when information sparse.
  - Contact-only mode feeds touchdown events back into gait scheduler to refine stance height estimates without altering APIs.
- **Integration**
  - Leg endpoint planner queries touchdown height before swing end.
  - Body planner fetches normals to adjust body orientation relative to terrain slope.

## Planner Coordination
- Central `GaitController` orchestrates data flow:
  - Calls leg endpoint planner to advance phases.
  - Fetches predicted support polygon.
  - Invokes body planner with desired command and support info.
  - Applies terrain corrections.
- `GaitScheduler` exposes APIs to query/modify active template:
  - `SetTargetGait(id, policy)` initiates transition using configured policy.
  - `OverrideSwingProfile(legId, params, duration)` for dynamic adjustments (e.g., obstacle avoidance) while retaining template defaults.
  - `BlendFactor()` returns current interpolation progress for observers (e.g., body planner).
- Exposes two modes:
  - **Preview**: produce `k` future steps for planning.
  - **Realtime**: single-step incremental update suitable for control loop.
- Provide instrumentation hooks (metrics, debug traces) for testing.
 - Time management: controller owns monotonic clock and forwards `time.Time` or `time.Duration` to each planner to prevent drift and support variable step simulation.

## Configuration & Extensibility
- Use option structs for planner constructors (`WithTerrainSampler`, `WithPhaseOffsets`, etc.).
- Support dynamic reconfiguration (e.g., gait switching) by atomically swapping gait definitions between cycles.
- Preserve separation between templates and bindings so the same preset works for quadrupeds, hexapods, etc.
- Provide default preset registry; allow injection of custom registries through dependency injection.
- Document default parameters in SPEC once implementation begins.

## Error Handling
- All planners return typed errors with context labels (`ErrSupportPolygonDegenerate`, `ErrLegOutOfReach`, etc.).
- Terrain module returns best-effort sample with confidence; planners decide fallback (e.g., assume flat ground).
- Missing leg descriptors or inconsistent gait definitions must panic during construction (fail fast).

## Testing Strategy
- Unit tests: table-driven scenarios for individual planners with synthetic terrain maps.
- Property-based tests: ensure orientation switching preserves continuity (no position jumps).
- Integration tests: simulate multi-leg robots (4, 6, 8 legs) executing different gaits; verify CoM remains inside support polygon.
- Fuzz tests for terrain queries to ensure planner handles sparse/noisy data.

## Open Questions
- How to incorporate dynamic effects (inertia, compliance) beyond quasi-static assumption?
- Should body planner provide momentum references for downstream controllers?
- What is the minimal API to integrate with existing `motion/planner` loop without tight coupling?

