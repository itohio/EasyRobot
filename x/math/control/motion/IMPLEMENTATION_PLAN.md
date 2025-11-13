# Generic Motion Planner – Implementation Plan

## Goals
- Allow the motion planner to accept any kinematics model that satisfies `kinematics/types.Bidirectional`.
- Provide a unified optimization pipeline that can plan control trajectories for arbitrary model dimensions and constraints.
- Maintain backwards compatibility for existing planner entry points while enabling incremental adoption.

## Deliverables
1. Planner core refactor with kinematics adapters and optimization loop scaffolding.
2. Path abstraction capable of expressing pose, velocity, and orientation targets in a model-independent manner.
3. Cost/constraint module supporting configurable penalties and automatic constraint enforcement using model metadata.
4. Example integration demonstrating the generic planner driving at least two different kinematics implementations (e.g., rigid body + differential drive).

## Work Breakdown

### 1. Abstractions & Interfaces
- Define `PlannerProblem` struct bundling:
  - `Kinematics` (`Bidirectional`)
  - `StateAdapter` / `ControlAdapter` functions for converting app state into matrix form.
  - `PathSampler` interface returning desired pose/velocity along arc length or time.
  - `CostTerms` slice and `ConstraintHandler`.
- Document expected state/control shapes (6×1 default) and adapter responsibilities.
- Update planner design docs (`SPEC.md`, future `DESIGN.md`).

### 2. Path Module
- Introduce `path` subpackage with:
  - `Sampler` interface providing position, orientation, derivatives.
  - Implementations for piecewise-linear and parametric paths.
  - Utility helpers for arc length, curvature, reparameterization.

### 3. Optimization Pipeline
- Implement iterative solver skeleton:
  - Warm-start controls from previous iteration.
  - Predict future state via `Kinematics.Forward`.
  - Evaluate tracking cost, control effort, constraint slack.
  - Apply projected gradient or SQP update respecting `ConstraintSet`.
- Provide hooks for termination criteria (max iterations, tolerance).

### 4. Integration & Examples
- Wire new planner core into rigid-body toy simulation.
- Add second example (e.g., differential drive) to validate adapters.
- Supply unit tests covering:
  - Adapter conversion correctness.
  - Optimization convergence on simple paths.
  - Constraint satisfaction checks (forces/torques within bounds).

### 5. Migration & Cleanup
- Deprecate legacy planner entry points once the generic planner stabilizes.
- Update documentation and roadmap for future enhancements (stochastic costs, multi-horizon planning).

## Risks / Considerations
- Optimizer stability across diverse kinematics; may require multiple solver strategies.
- Performance impact of generic matrix operations; ensure fixed-size types are used where possible.
- Adapter complexity for high-DOF models—provide guidelines/examples.

## Timeline (Rough)
1. Abstractions & docs – 1-2 days.
2. Path module – 1 day.
3. Optimization core – 3-4 days (including tuning).
4. Integrations/tests – 2 days.
5. Cleanup/documentation – 1 day.

Total estimate: ~8-10 working days (adjust as complexity evolves).

