# Rigid Body Kinematics Design

## Goal
Provide a lightweight kinematics module that maps between body-frame linear/angular velocities and actuator-level thrust/torque requirements for a rigid body with known mass and inertia. The module must integrate with `kinematics/types` interfaces so motion planners can query capabilities and invoke forward/backward mappings.

## Responsibilities
- Accept desired body-frame velocities (linear `vx, vy, vz` and angular `wx, wy, wz`).
- Produce aggregate wrench (force + torque) required to achieve those velocities given simple drag or timescale assumptions.
- Backward mapping: solve for actuator force/torque commands required to reach target velocities from current controls.
- Expose metadata consistent with other kinematics models (`Dimensions`, `Constraints`, `Capabilities`).

## Assumptions
- Rigid body mass is constant, inertia tensor is represented as a full `mat.Matrix3x3` (not limited to diagonal form).
- Control vector comprises thrust XYZ and torque XYZ (6 DOF).
- Mapping uses proportional gain parameters to convert velocity error into force/torque demands (quasi-static model suitable for simulation toy tests).
- Environmental effects (drag, gravity) ignored; focus on mapping rather than dynamics integration.

## Interfaces
- Constructor `NewModel(mass float32, inertia mat.Matrix3x3, gains)` returns `kinematics/types.Bidirectional` using fixed-size matrices to avoid heap churn. Inertia tensors must be positive definite (diagonal > 0) to avoid singular inverses.
- State vector stores current velocities (6x1) in the order `[vx, vy, vz, wx, wy, wz]`. Callers must supply matrices of that shape; nil is treated as zero velocity.
- Control vector stores thrust/torque (6x1). When nil, `Forward`/`Backward` skip population.
- `Forward`: given velocities, outputs required controls (force/torque).
- `Backward`: given desired velocities in destination, outputs controls achieving them.
- `AngularAcceleration(torque)` exposes convenience method for dynamics integration.

## Testing
- Verify forward mapping produces expected proportional outputs for canonical inputs.
- Validate backward computes symmetric result.
- Ensure inertia helper functions return positive definite tensors and rotation preserves values for identity orientation.
- Ensure constraints metadata matches configured limits.

