# Rigid Body Motion Planner Specification

## Overview
- Component: `pkg/core/math/control/motion/rigidbody`
- Purpose: generate short-horizon trajectory previews and command updates that honour vehicle kinematic limits.
- Consumes: current state estimate (`state` matrix), optional previously issued controls (`controls` matrix), path definition via `SetPath` (positions only) or `SetWaypointMatrix` (positions/orientations/velocity hints).
- Produces: next state target (`destination` matrix) and updated controls (same matrix as input).

## Interfaces
- Implements `kinematics/types.Bidirectional` (`Dimensions`, `Capabilities`, `ConstraintSet`, `Forward`, `Backward`).
- Matrix layout:
  - State: `[x, y, z, yaw, speed, timestamp]^T`.
  - Controls: `[linear, angular, effort...]^T` where effort slice is optional and round-tripped without modification.
- `SetPath([]vec.Vector3D)` or `SetWaypointMatrix(mat.Matrix)` must be invoked before `Forward`; these construct an internal planner using position-only waypoints, pose waypoints (position + quaternion), or optional linear/angular velocity hints.

## Behaviour
- `Forward`:
  - Projects current position onto cached path, advances VAJ1D profile toward lookahead target.
  - Limits velocity with jerk/acceleration constraints and curvature-based lateral limits.
  - Aligns yaw with local tangent while respecting turn-rate and turn-acceleration bounds.
- `Backward`:
  - PID controllers (speed, heading, lateral) adjust commands toward desired state.
  - Applies acceleration and turn-rate clamps plus lateral acceleration torque limits.
  - Returns updated control vector while preserving any effort components supplied.

## Constraints Metadata
- `Dimensions.ControlSize` tracks active control vector rows (>=2) and propagates to constraint bounds.
- `ConstraintSet` exposes symmetric velocity/turn-rate limits and rate-of-change constraints.

## Testing Expectations
- Table-driven validation for parameter/constraint checking.
- Forward path following on straight and curved paths (progress, yaw alignment, speed clamps).
- Backward command clamping under aggressive targets.
- Metadata retrieval (`Dimensions`, `Capabilities`, `ConstraintSet`) sanity checks.

