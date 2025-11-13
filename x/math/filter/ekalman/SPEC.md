# Extended Kalman Filter Specification

## Overview

The Extended Kalman Filter (EKF) package provides state estimation for nonlinear dynamic systems with Gaussian noise. It extends the standard Kalman filter by linearizing nonlinear state transition and measurement functions using their Jacobian matrices.

## Components

### Extended Kalman Filter (`ekalman.go`)

**Purpose**: Extended Kalman filter for nonlinear state estimation

**Description**: Implements the Extended Kalman Filter algorithm for estimating state of a nonlinear dynamic system from noisy measurements. Uses first-order Taylor series expansion to linearize nonlinear functions around the current state estimate.

**Mathematical Model**:

Nonlinear state transition:
- `x(k+1) = f(x(k), u(k)) + w(k)` (with control input)
- `x(k+1) = f(x(k)) + w(k)` (without control input)

Nonlinear measurement:
- `z(k) = h(x(k)) + v(k)`

Linearization (using Jacobians):
- `F = ∂f/∂x` (state transition Jacobian)
- `H = ∂h/∂x` (measurement Jacobian)

Prediction:
- `x_pred = f(x, u)` (nonlinear prediction)
- `P_pred = F * P * F^T + Q` (linearized covariance prediction)

Update:
- `y = z - h(x_pred)` (innovation/residual)
- `S = H * P_pred * H^T + R` (innovation covariance)
- `K = P_pred * H^T * S^-1` (Kalman gain)
- `x = x_pred + K * y` (updated state)
- `P = (I - K * H) * P_pred` (updated covariance)

**Type Definition**:
```go
type EKF struct {
    // State
    x vec.Vector // State vector (n dimensions)
    
    // Covariances
    P mat.Matrix // State covariance (n x n)
    Q mat.Matrix // Process noise covariance (n x n)
    R mat.Matrix // Measurement noise covariance (m x m)
    
    // Nonlinear functions
    FFunc func(x, u vec.Vector, dt float32) vec.Vector // State transition function
    HFunc func(x vec.Vector) vec.Vector                // Measurement function
    
    // Jacobian computation
    // Option 1: User provides Jacobian functions
    FJacobian func(x, u vec.Vector, dt float32) mat.Matrix // State transition Jacobian
    HJacobian func(x vec.Vector) mat.Matrix                // Measurement Jacobian
    
    // Option 2: Numerical Jacobian (if user functions not provided)
    // Computed numerically using finite differences
    
    // Temporary matrices for computations
    K mat.Matrix  // Kalman gain (n x m)
    S mat.Matrix  // Innovation covariance (m x m)
    F mat.Matrix  // State transition Jacobian (n x n)
    H mat.Matrix  // Measurement Jacobian (m x n)
    tempP mat.Matrix  // Temporary for P_pred (n x n)
    tempM mat.Matrix  // Temporary for matrix operations (n x n)
    tempHP mat.Matrix // Temporary for H * P (m x n)
    tempHT mat.Matrix // Temporary for Hᵀ (n x m)
    tempPH mat.Matrix // Temporary for P * Hᵀ (n x m)
    tempN mat.Matrix  // Temporary for matrix operations
    tempN2 mat.Matrix // Temporary for matrix operations (n x n)
    tempV vec.Vector  // Temporary for vector operations
    tempV2 vec.Vector // Temporary for vector operations
    tempV3 vec.Vector // Temporary for vector operations
    
    // Dimensions
    n int // State dimension
    m int // Measurement dimension
    k int // Control input dimension (0 if not used)
    
    // Numerical Jacobian settings
    JacobianEpsilon float32 // Step size for numerical differentiation
    
    // Filter interface
    Input vec.Vector   // Measurement input
    Output vec.Vector  // Estimated state output
    Target vec.Vector  // Target state (optional)
}
```

**Operations**:

1. **Initialization**:
   - `New(n, m int, fFunc, hFunc JacobianFuncs, Q, R mat.Matrix) *EKF`: Create new EKF with Jacobian functions
   - `NewWithControl(n, m, k int, fFunc, hFunc JacobianFuncs, Q, R mat.Matrix) *EKF`: Create EKF with control input
   - `NewWithNumericalJacobian(n, m int, fFunc, hFunc NonlinearFuncs, Q, R mat.Matrix) *EKF`: Create EKF using numerical Jacobians

2. **Prediction**:
   - `Predict(dt float32) *EKF`: Predict state and covariance (no control)
   - `PredictWithControl(u vec.Vector, dt float32) *EKF`: Predict with control input

3. **Update**:
   - `Update(z vec.Vector) *EKF`: Update state with measurement
   - `UpdateTimestep(dt float32) *EKF`: Update with time step (implements Filter interface)

4. **State Management**:
   - `Reset() *EKF`: Reset filter state
   - `SetState(x vec.Vector) *EKF`: Set initial state
   - `SetCovariance(P mat.Matrix) *EKF`: Set initial covariance

5. **Filter Interface**:
   - `GetInput() vec.Vector`: Get measurement input
   - `GetOutput() vec.Vector`: Get estimated state
   - `GetTarget() vec.Vector`: Get target state (if applicable)

**Function Types**:
```go
// State transition function: x_next = f(x, u, dt)
type StateTransitionFunc func(x, u vec.Vector, dt float32) vec.Vector

// Measurement function: z = h(x)
type MeasurementFunc func(x vec.Vector) vec.Vector

// State transition Jacobian: F = ∂f/∂x
type StateJacobianFunc func(x, u vec.Vector, dt float32) mat.Matrix

// Measurement Jacobian: H = ∂h/∂x
type MeasurementJacobianFunc func(x vec.Vector) mat.Matrix
```

**Characteristics**:
- Handles nonlinear systems by linearization
- Supports user-provided Jacobians (more accurate)
- Supports numerical Jacobians (easier to use, less accurate)
- Uses `mat.Matrix` and `vec.Vector` for all computations
- In-place operations where possible
- `float32` precision for embedded systems compatibility

**Algorithm Steps**:

1. **Initialize**:
   - Set initial state `x` and covariance `P`
   - Validate dimensions and function signatures
   - Allocate temporary matrices

2. **Predict** (with nonlinear function):
   - Compute predicted state: `x_pred = f(x, u, dt)` (nonlinear)
   - Compute Jacobian: `F = ∂f/∂x` (at current state)
   - Compute predicted covariance: `P_pred = F * P * F^T + Q` (linearized)
   - Store `x_pred` in `x`, `P_pred` in `tempP`

3. **Update** (with nonlinear function):
   - Compute predicted measurement: `z_pred = h(x_pred)`
   - Compute Jacobian: `H = ∂h/∂x` (at predicted state)
   - Compute innovation: `y = z - z_pred`
   - Compute innovation covariance: `S = H * P_pred * H^T + R`
   - Compute Kalman gain: `K = P_pred * H^T * S^-1`
   - Update state: `x = x_pred + K * y`
   - Update covariance: `P = (I - K * H) * P_pred`

**Questions**:

1. Should we support Unscented Kalman Filter (UKF) as an alternative?
2. How to handle Jacobian computation failures?
3. Should we support second-order EKF (Hessian-based)?
4. How to optimize numerical Jacobian computation?
5. Should we support adaptive Jacobian step sizes?
6. How to handle singular Jacobians?
7. Should we support multiple measurement functions with different update rates?
8. How to handle discontinuous functions?
9. Should we support automatic differentiation?
10. How to validate user-provided functions?

## Design Decisions

### Architecture

1. **Function Signatures**:
   - Functions take current state, control input, and time step
   - Functions return new state or measurement
   - Jacobian functions compute derivatives at given state

2. **Jacobian Computation**:
   - Primary: User-provided analytical Jacobians (most accurate)
   - Fallback: Numerical Jacobians using finite differences
   - Step size configurable via `JacobianEpsilon`

3. **Numerical Jacobian**:
   - Central difference: `f'(x) ≈ (f(x+ε) - f(x-ε)) / (2ε)`
   - Forward difference as fallback
   - Step size ε configurable (default: 1e-5)

### Performance

1. **Jacobian Computation**:
   - User-provided Jacobians: O(1) per call
   - Numerical Jacobians: O(n²) evaluations of f/h

2. **Memory**:
   - Pre-allocate all temporary matrices
   - Store Jacobians F and H for reuse

### Compatibility

1. **Embedded Systems**:
   - `float32` precision
   - Minimal allocations
   - Deterministic execution time

2. **Testing**:
   - Unit tests with known nonlinear systems
   - Comparison with analytical solutions
   - Numerical stability tests

## Implementation Notes

### Current Implementation

- Extended Kalman Filter
- Support for user-provided or numerical Jacobians
- Optional control input
- Generic matrix/vector implementation

### Missing Features

- Unscented Kalman Filter (UKF)
- Second-order EKF
- Automatic differentiation
- Adaptive step sizes
- Discontinuous function handling

## Usage Example

```go
// Nonlinear system: pendulum
// State: [angle, angular_velocity]
// Measurement: angle (nonlinear: angle measurement)

// State transition: θ' = θ + ω*dt, ω' = ω - (g/L)*sin(θ)*dt
fFunc := func(x, u vec.Vector, dt float32) vec.Vector {
    theta := x[0]
    omega := x[1]
    g := float32(9.81)
    L := float32(1.0)
    
    next := vec.New(2)
    next[0] = theta + omega*dt
    next[1] = omega - (g/L)*math32.Sin(theta)*dt
    return next
}

// Measurement: h(x) = angle
hFunc := func(x vec.Vector) vec.Vector {
    z := vec.New(1)
    z[0] = x[0]
    return z
}

// Jacobians
fJacobian := func(x, u vec.Vector, dt float32) mat.Matrix {
    theta := x[0]
    g := float32(9.81)
    L := float32(1.0)
    
    F := mat.New(2, 2)
    F[0][0] = 1
    F[0][1] = dt
    F[1][0] = -(g/L)*math32.Cos(theta)*dt
    F[1][1] = 1
    return F
}

hJacobian := func(x vec.Vector) mat.Matrix {
    H := mat.New(1, 2)
    H[0][0] = 1  // measure angle
    H[0][1] = 0
    return H
}

// Create EKF
ekf := ekalman.New(2, 1, 
    ekalman.StateTransitionFunc(fFunc),
    ekalman.MeasurementFunc(hFunc),
    ekalman.StateJacobianFunc(fJacobian),
    ekalman.MeasurementJacobianFunc(hJacobian),
    Q, R,
)

ekf.SetState(vec.NewFrom(0, 0))
ekf.SetCovariance(initialP)

// Predict
ekf.Predict(0.01)

// Update
measurement := vec.NewFrom(angle_measured)
ekf.Update(measurement)
```

