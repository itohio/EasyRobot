# Kinematics Specification

## Overview

`pkg/core/math/control/kinematics` hosts kinematic models for joints, wheeled bases, and thruster arrays.  Each model exposes a consistent matrix-based API so callers can:

- propagate actuator state vectors into chassis or end-effector wrenches (`Forward`)
- solve for actuator commands that realise a desired chassis/end-effector target (`Backward`)

The package does **not** generate motion profiles (the older VAJ filter has moved elsewhere).  Instead it provides the geometric mapping layer used by higher-level planners.

## Shared Interfaces (`types` package)

```go
// Config describes immutable model parameters (geometry, mass properties, etc.).
type Config struct {
    Name             string
    DegreesOfFreedom int
    ActuatorCount    int
    StateDimension   int
    ControlDimension int
    Mass             float32
    Metadata         map[string]float32
}

// Dimensions advertises the shape of state/control vectors.
type Dimensions struct {
    StateRows    int // expected rows in state matrix
    StateCols    int // (almost always 1)
    ControlSize  int // flat size written by Backward
    ActuatorSize int // number of physical actuators
}

// Model exposes metadata shared by all concrete kinematics.
type Model interface {
    Dimensions() Dimensions
    Capabilities() Capabilities
    ConstraintSet() Constraints
}

// ForwardKinematics consumes a state column matrix and writes a destination column.
type ForwardKinematics interface {
    Model
    Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error
}

// BackwardKinematics mirrors Forward but writes actuator controls.
type BackwardKinematics interface {
    Model
    Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error
}

// Bidirectional models implement both forward and backward propagation.
type Bidirectional interface {
    ForwardKinematics
    BackwardKinematics
}
```

All matrices come from `pkg/core/math/mat`.  Column vectors are represented as `rows × 1` matrices; callers may reuse buffers across calls.

### Conventions

- Inputs are not copied: models read directly from the supplied `state`/`destination` matrix.  Callers must ensure the shapes advertised in `Dimensions()`.
- Return values are written into the supplied `destination` or `controls` matrices.  Optional arguments may be `nil` if not needed (e.g. passing `nil` for `controls` in forward propagation when no buffer reuse is required).
- Units follow SI conventions.  Steering angles are expressed in radians.

## Package Inventory

The table below summarises the column layouts used by each bidirectional model.

| Package / Model | Forward expects (`state`) | Forward writes (`destination`) | Backward writes (`controls`) |
|-----------------|----------------------------|--------------------------------|------------------------------|
| `joints/dh` | `[θ₀ … θₙ₋₁]ᵀ` (DOF×1 joint parameters) | `[x, y, z, qx, qy, qz, qw]ᵀ` | `[θ₀ … θₙ₋₁]ᵀ` |
| `joints/planar.New2DOF` | `[a₀, a₁]ᵀ` | `[x, y, z, roll, pitch, yaw]ᵀ` (orientation slots zeroed) | `[a₀, a₁]ᵀ` |
| `joints/planar.New3DOF` | `[a₀, a₁, a₂]ᵀ` | `[x, y, z, roll, pitch, yaw]ᵀ` | `[a₀, a₁, a₂]ᵀ` |
| `wheels/differential` | `[ω_L, ω_R]ᵀ` | `[v, ω]ᵀ` | `[ω_L, ω_R]ᵀ` |
| `wheels/mecanum` | `[ω_fl, ω_fr, ω_rl, ω_rr]ᵀ` | `[v_x, v_y, ω]ᵀ` | `[ω_fl, ω_fr, ω_rl, ω_rr]ᵀ` |
| `wheels/steer4` | `[ω_fl, ω_fr, ω_rl, ω_rr, δ_fl, δ_fr]ᵀ` | `[v, ω, δ_fl, δ_fr]ᵀ` | same as state |
| `wheels/steer4dual` | `[ω_fl, ω_fr, ω_rl, ω_rr, δ_fl, δ_fr, δ_rl, δ_rr]ᵀ` | `[v, ω, δ_fl, δ_fr, δ_rl, δ_rr]ᵀ` | same as state |
| `wheels/steer6` | `[ω_fl, ω_fr, ω_ml, ω_mr, ω_rl, ω_rr, δ_fl, δ_fr, δ_rl, δ_rr]ᵀ` | `[v, ω, δ_fl, δ_fr, δ_rl, δ_rr]ᵀ` | same as state |
| `thrusters.Model` | `[thrust₀, torque₀, …]ᵀ` (2×thruster count) | `[F_x, F_y, F_z, τ_x, τ_y, τ_z]ᵀ` | `[thrust₀, torque₀, …]ᵀ` |

### Error Semantics

- `kintypes.ErrInvalidDimensions`: matrix shape mismatch.
- `kintypes.ErrUnsupportedOperation`: configuration cannot satisfy the request (e.g. unsupported joint type, singular Jacobian).
- Model-specific errors:
  - `dh.ErrNoConvergence`: inverse solver failed to reach tolerance.
  - `thrusters.ErrInfeasible`: wrench cannot be produced within command limits.
  - `thrusters.ErrCommandLimit`: requested command exceeds actuator bounds.

## Package Notes

### Joints (`joints/dh`, `joints/planar`)

- `dh.New` constructs a Denavit–Hartenberg chain with arbitrary DOF.  Forward performs FK by chaining `Matrix4x4` transforms, while Backward runs a Jacobian pseudo-inverse loop (position only).
- `planar.New2DOF`/`New3DOF` provide analytical planar arm solvers that map joint angles to Cartesian positions and back.

### Wheels (`wheels/*`)

All wheeled models share helper utilities from `wheels/internal/rigid`.  Forward solves for chassis velocity using wheel speeds (and steering angles) while Backward computes the wheel rates required to achieve a desired twist.

### Thrusters (`thrusters`)

`thrusters.NewModel` wraps a `Body` and thruster array, exposing the shared interface.  Forward aggregates force/torque contributions; Backward solves a damped least-squares allocation problem to honour thrust/torque limits.

## Usage Examples

### 1. Denavit–Hartenberg Forward/Backward

```go
cfg := []dh.Config{{R: 1, Index: 0}, {R: 1, Index: 0}}
arm := dh.New(1e-4, 50, cfg...)

state := mat.New(len(cfg), 1)
state[0][0], state[1][0] = math32.Pi/4, math32.Pi/6
destination := mat.New(7, 1)

if err := arm.Forward(state, destination, nil); err != nil {
    log.Fatal(err)
}
fmt.Printf("end effector: %v\n", destination.View())

controls := mat.New(len(cfg), 1)
desired := mat.New(7, 1)
desired.CopyFrom(destination.View())
desired[0][0] += 0.1 // shift x by 10 cm
if err := arm.Backward(nil, desired, controls); err != nil {
    log.Fatal(err)
}
```

### 2. Mecanum Drive Twist Mapping

```go
drive := mecanum.New(wheelRadius, baseX, baseY)
wheelRates := mat.New(4, 1)
wheelRates[0][0], wheelRates[1][0] = 4, 4
wheelRates[2][0], wheelRates[3][0] = 4, 4

chassis := mat.New(3, 1)
if err := drive.Forward(wheelRates, chassis, nil); err != nil {
    log.Fatal(err)
}
fmt.Printf("vx=%.2f vy=%.2f omega=%.2f\n",
    chassis[0][0], chassis[1][0], chassis[2][0])

commands := mat.New(4, 1)
desired := mat.New(3, 1)
desired[0][0] = 1.0  // 1 m/s forward
desired[2][0] = 0.5 // 0.5 rad/s yaw
if err := drive.Backward(nil, desired, commands); err != nil {
    log.Fatal(err)
}
```

### 3. Thruster Allocation

```go
body := thrusters.Body{Mass: 1.2, Inertia: mat.FromDiagonal3x3(0.05, 0.05, 0.08)}
th := [...]thrusters.Thruster{ /* position/direction/limits */ }
model, err := thrusters.NewModel(body, th[:])
if err != nil {
    log.Fatal(err)
}

state := mat.New(len(th)*2, 1)
commands := mat.New(len(th)*2, 1)
desired := mat.New(6, 1)
desired[2][0] = 9.81   // hover thrust
desired[5][0] = 0.15  // small yaw torque

if err := model.Backward(nil, desired, commands); err != nil {
    log.Fatal(err)
}
fmt.Println("thruster commands:", commands.View())
```

These patterns generalise across all models: **Forward** consumes the current actuator column vector and writes the resulting chassis/end-effector column vector; **Backward** does the reverse, populating an actuator column that approximates the requested `destination` within each model’s capabilities.

