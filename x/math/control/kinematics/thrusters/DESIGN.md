# Thruster-Based Kinematics Module

## Scope

- Model rigid bodies actuated by multiple thrusters (e.g. multi-directional drones, underwater vehicles).
- Support forward kinematics: given thruster outputs, compute resulting force/torque on the body.
- Support inverse kinematics/control allocation: given desired force/torque, compute thruster commands under limits.
- Capture body properties (mass, inertia tensor) for completeness and future dynamics coupling.

## Concepts

- **Thruster**: rigidly mounted actuator defined by:
  - Position vector `p` in body coordinates.
  - Direction unit vector `d` indicating thrust orientation.
  - Maximum/minimum thrust `T_min`, `T_max`.
  - Maximum/minimum torque bias `τ_min`, `τ_max` (to model rotor drag or gimbal torque).
- **BodyState**: linear force `F` and torque `τ` about the body origin. Optional velocity components can be added later.
- **Configuration**: collection of thrusters with current thrust and torque commands.

## Public API (planned)

- `type Thruster struct { Position, Direction vec.Vector3D; Thrust struct { Min, Max float32 }; Torque struct { Min, Max float32 } }`
- `type Body struct { Mass float32; Inertia mat.Matrix3x3 }`
- `Model` implements `kinematics/types.Bidirectional` allowing destination-based forward/backward calls.
  - `ThrusterCommand` couples a Thruster with applied thrust/torque values, clamped to limits.
- `Inverse(body Body, thrusters []Thruster) (Allocator, error)`
  - Allocator exposes `Solve(desired BodyState) ([]ThrusterCommand, error)` performing constrained allocation.

## Forward Model

- Resultant force `F = Σ (thrust_i * direction_i)`
- Resultant torque `τ = Σ ( r_i × (thrust_i * direction_i) + appliedTorque_i )`
- Validate commands within limits before summation.

## Inverse Model

- Formulate as constrained least squares:
  - Decision vector `u = [t₁ … t_n, τ₁ … τ_n]^T`
  - Map to wrench via matrix `W` (6 × 2n) built from thruster geometry.
  - Solve `W u = desired_wrench` with box constraints on each entry.
- Initial implementation:
  - Use pseudo-inverse for unconstrained solution.
  - Clamp to limits and iterate with projected gradient or simple active-set (acceptable due to small n).
  - Provide deterministic fallback when solution infeasible (return error and saturate).

## Helper Structures

- `Allocator` caches matrix `W` and its pseudo-inverse using `mat` package.
- Reuse existing `mat.Matrix3x3` and `vec.Vector3D` utilities.
- Keep helper functions private within package for clarity.

## Testing Strategy

- Table-driven tests covering:
  - Symmetric thruster arrangement producing pure translation.
  - Thrusters positioned to produce rotation around axes.
  - Inverse solving small systems (2–4 thrusters).
  - Saturation scenarios verifying clamping and infeasibility detection.
- Toy craft scenarios (e.g. quadcopter) that validate both forward and inverse allocation using realistic geometry.
- Regression for matrix/vector semantics: ensure helper maths operate on returned values rather than implicit mutation.

## Future Extensions

- Add gimballed thrusters (direction control).
- Integrate with dynamics to produce accelerations using `F = m a`, `τ = I α`.
- Add weighting matrix for optimal control allocation.

