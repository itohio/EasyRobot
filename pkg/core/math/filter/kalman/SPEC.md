# Kalman Filter Specification

## Overview

The Kalman filter package provides optimal state estimation for linear dynamic systems with Gaussian noise. It combines predictions from a dynamic model with measurements to provide optimal estimates of system state.

## Components

### Kalman Filter (`kalman.go`)

**Purpose**: Linear Kalman filter for state estimation

**Description**: Implements the standard Kalman filter algorithm for estimating state of a linear dynamic system from noisy measurements. Suitable for tracking, sensor fusion, and state estimation in robotics applications.

**Mathematical Model**:

State prediction:
- `x_pred = F * x + B * u` (with control input)
- `x_pred = F * x` (without control input)
- `P_pred = F * P * F^T + Q` (covariance prediction)

Measurement update:
- `y = z - H * x_pred` (innovation/residual)
- `S = H * P_pred * H^T + R` (innovation covariance)
- `K = P_pred * H^T * S^-1` (Kalman gain)
- `x = x_pred + K * y` (updated state)
- `P = (I - K * H) * P_pred` (updated covariance)

**Type Definition**:
```go
type Kalman struct {
    // State
    x vec.Vector  // State vector (n dimensions)
    
    // Covariances
    P mat.Matrix  // State covariance (n x n)
    Q mat.Matrix  // Process noise covariance (n x n)
    R mat.Matrix  // Measurement noise covariance (m x m)
    
    // Matrices
    F mat.Matrix  // State transition matrix (n x n)
    H mat.Matrix  // Measurement matrix (m x n)
    B mat.Matrix  // Control input matrix (n x k) - optional
    
    // Temporary matrices for computations
    K mat.Matrix  // Kalman gain (n x m)
    S mat.Matrix  // Innovation covariance (m x m)
    tempP mat.Matrix  // Temporary for P_pred
    tempM mat.Matrix  // Temporary for matrix operations
    
    // Dimensions
    n int  // State dimension
    m int  // Measurement dimension
    k int  // Control input dimension (0 if not used)
    
    // Filter interface
    Input vec.Vector   // Measurement input
    Output vec.Vector  // Estimated state output
    Target vec.Vector  // Target state (if applicable)
}
```

**Operations**:

1. **Initialization**:
   - `New(n, m int, F, H, Q, R mat.Matrix) *Kalman`: Create new filter
   - `NewWithControl(n, m, k int, F, H, B, Q, R mat.Matrix) *Kalman`: Create filter with control input

2. **Prediction**:
   - `Predict() *Kalman`: Predict state and covariance (no control)
   - `PredictWithControl(u vec.Vector) *Kalman`: Predict with control input

3. **Update**:
   - `Update(z vec.Vector) *Kalman`: Update state with measurement
   - `UpdateTimestep(dt float32) *Kalman`: Update with time step (implements Filter interface)

4. **State Management**:
   - `Reset() *Kalman`: Reset filter state
   - `SetState(x vec.Vector) *Kalman`: Set initial state
   - `SetCovariance(P mat.Matrix) *Kalman`: Set initial covariance

5. **Filter Interface**:
   - `GetInput() vec.Vector`: Get measurement input
   - `GetOutput() vec.Vector`: Get estimated state
   - `GetTarget() vec.Vector`: Get target state (if applicable)

**Characteristics**:
- Generic implementation using `mat.Matrix` and `vec.Vector`
- In-place operations where possible to minimize allocations
- `float32` precision for embedded systems compatibility
- Supports optional control input
- Reuses temporary matrices to reduce allocations

**Algorithm Steps**:

1. **Initialize**:
   - Set initial state `x` and covariance `P`
   - Validate matrix dimensions
   - Allocate temporary matrices

2. **Predict**:
   - Compute predicted state: `x_pred = F * x` (or `F * x + B * u`)
   - Compute predicted covariance: `P_pred = F * P * F^T + Q`
   - Store `x_pred` in `x`, `P_pred` in `tempP`

3. **Update**:
   - Compute innovation: `y = z - H * x_pred`
   - Compute innovation covariance: `S = H * P_pred * H^T + R`
   - Compute Kalman gain: `K = P_pred * H^T * S^-1`
   - Update state: `x = x_pred + K * y`
   - Update covariance: `P = (I - K * H) * P_pred`

**Questions**:

1. Should we support Extended Kalman Filter (EKF) for non-linear systems?
2. Should we support Unscented Kalman Filter (UKF) for non-linear systems?
3. How to handle numerical stability (e.g., Joseph form for covariance update)?
4. Should we support square-root Kalman filter for better numerical stability?
5. How to handle matrix inversion failures (singular S matrix)?
6. Should we support adaptive noise estimation?
7. How to handle missing measurements?
8. Should we support multiple measurement types with different update rates?
9. How to optimize for embedded systems (fixed-point arithmetic)?
10. Should we provide specialized implementations for common cases (1D, 2D, 3D)?

## Design Decisions

### Architecture

1. **Matrix Operations**:
   - Use generic `mat.Matrix` and `vec.Vector` for flexibility
   - All operations in-place where possible
   - Reuse temporary matrices to reduce allocations

2. **Control Input**:
   - Optional control input support via separate constructor
   - `B` matrix is nil if control not used
   - Control dimension `k` is 0 if not used

3. **Filter Interface**:
   - Implement `Filter` interface for consistency
   - `Update(timestep)` calls `Predict()` then expects measurement in `Input`
   - Measurement update happens when `Update()` is called with measurement

### Performance

1. **Memory Allocation**:
   - Pre-allocate all temporary matrices in constructor
   - Reuse matrices across iterations
   - Minimize allocations in hot path

2. **Numerical Stability**:
   - Use Cholesky decomposition for S inversion if needed
   - Consider Joseph form for covariance update (more stable)
   - Validate matrix dimensions and singularity

### Compatibility

1. **Embedded Systems**:
   - `float32` precision
   - Minimal allocations
   - Deterministic execution time

2. **Testing**:
   - Unit tests for prediction and update steps
   - Integration tests with known trajectories
   - Numerical stability tests

## Implementation Notes

### Current Implementation

- Linear Kalman filter only
- Optional control input support
- Generic matrix/vector implementation
- In-place operations

### Missing Features

- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Square-root Kalman filter
- Adaptive noise estimation
- Missing measurement handling
- Specialized implementations for common cases

## Usage Example

```go
// Create filter for 2D position tracking (state: [x, y, vx, vy])
n, m := 4, 2  // 4 state dims, 2 measurement dims
F := mat.New(4, 4)  // State transition matrix
H := mat.New(2, 4)  // Measurement matrix
Q := mat.New(4, 4)  // Process noise
R := mat.New(2, 2)  // Measurement noise

// Initialize matrices...
filter := kalman.New(n, m, F, H, Q, R)

// Set initial state
filter.SetState(vec.NewFrom(0, 0, 0, 0))
filter.SetCovariance(initialP)

// Prediction
filter.Predict()

// Update with measurement
measurement := vec.NewFrom(x_measured, y_measured)
filter.Update(measurement)

// Get estimated state
state := filter.GetOutput()
```

