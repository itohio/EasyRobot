# SLAM Filter Optimizations

## Overview

This document outlines optimizations for the SLAM filter to improve performance on embedded devices and add online map building capabilities.

## Clarifications Addressed

### 1. Raycasting Optimization for Embedded

**Current Issues**:
- Step-based ray casting allocates memory in hot path
- Repeated trigonometric computations (cos/sin) for each ray
- Inefficient grid traversal

**Optimizations**:
1. **Use Bresenham algorithm as default**: More efficient discrete grid traversal
2. **Pre-compute ray directions**: Cache cos/sin values for ray angles
3. **Avoid allocations in hot path**: Reuse temporary vectors/matrices
4. **Optimize coordinate conversion**: Inline calculations where possible
5. **Early termination**: Stop as soon as obstacle is found

**Implementation**:
- Pre-compute `cos(θ_i)` and `sin(θ_i)` for all ray angles at initialization
- Use Bresenham's line algorithm instead of step-based approach
- Reuse temporary variables instead of allocating new ones
- Inline coordinate conversions

### 2. Online Map Building (Configurable)

**Requirements**:
- Update occupancy grid from measurements
- Use inverse sensor model for occupancy updates
- Log-odds representation for numerical stability
- Configurable flag to enable/disable

**Inverse Sensor Model**:
```
For each ray:
  If measurement < maxRange:
    - Cells along ray (0 to measurement): P_free
    - Cell at measurement: P_occupied
    - Cells beyond measurement: P_prior (no update)
  If measurement >= maxRange:
    - Cells along ray (0 to maxRange): P_free
    - Cell at maxRange: P_prior (no update)
```

**Log-Odds Representation**:
```
log_odds = log(p / (1 - p))
p = 1 / (1 + exp(-log_odds))
```

**Benefits**:
- Numerical stability
- Additive updates (faster)
- Avoids saturation

### 3. Other Optimizations

**Ray Direction Caching**:
- Pre-compute cos/sin for all ray angles at initialization
- Store in vectors for reuse

**Reduce Jacobian Computations**:
- Cache Jacobian when pose doesn't change significantly
- Use analytical Jacobian approximation for small pose changes
- Adaptive Jacobian update frequency

**Sparse Operations**:
- Use sparse matrices if map is mostly empty
- Update only cells along ray path (not entire map)

**Memory Pooling**:
- Pre-allocate all temporary vectors/matrices
- Reuse buffers for ray casting
- Minimize garbage collection

**Embedded-Specific**:
- Avoid dynamic allocations
- Use fixed-point arithmetic option (future)
- Reduce floating-point operations where possible

### 4. Kalman Filter vs Extended Kalman Filter

**Question**: Can we substitute EKF with KF?

**Answer**: **No, we need EKF** because:

1. **Measurement function is nonlinear**: 
   - Ray casting distance `d = h(x, θ, M)` is nonlinear in pose `x`
   - Depends on `px`, `py`, and `heading` in a nonlinear way

2. **Why EKF is needed**:
   - EKF linearizes measurement function around current pose estimate
   - Updates linearization point as pose changes
   - Handles nonlinearity by computing Jacobian `H = ∂h/∂x`

3. **Why KF wouldn't work**:
   - KF assumes linear measurement model: `z = H * x`
   - Ray casting cannot be expressed as linear function of pose
   - Would require fixed linearization point (less accurate)

**However**, we can provide an option to use:
- **EKF** (default): More accurate, handles nonlinearity
- **KF with fixed linearization**: Faster but less accurate, only if pose changes slowly

**Implementation**: Add option to choose filter type.

## Performance Targets

### Raycasting
- **Target**: < 1ms for 360 rays on embedded device (ARM Cortex-M4)
- **Current**: ~5-10ms (step-based, unoptimized)
- **Optimized**: < 1ms (Bresenham, pre-computed, no allocations)

### Map Building
- **Target**: < 0.5ms for updating cells along ray path
- **Current**: Not implemented
- **Optimized**: < 0.5ms (log-odds, sparse updates)

### Overall Update
- **Target**: < 2ms total (ray casting + EKF update)
- **Current**: ~10-15ms
- **Optimized**: < 2ms

## Implementation Plan

### Phase 1: Raycasting Optimization
1. Pre-compute ray directions (cos/sin)
2. Use Bresenham algorithm as default
3. Avoid allocations in hot path
4. Reuse temporary variables

### Phase 2: Online Map Building
1. Implement log-odds representation
2. Implement inverse sensor model
3. Add configurable flag
4. Add sparse map updates

### Phase 3: Additional Optimizations
1. Cache Jacobians when possible
2. Memory pooling
3. Reduce floating-point operations
4. Add option for KF vs EKF

### Phase 4: Embedded-Specific
1. Fixed-point arithmetic option
2. Architecture-specific optimizations
3. Memory-mapped map storage
4. Reduced precision where acceptable

