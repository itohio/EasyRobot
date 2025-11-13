# Why Extended Kalman Filter (EKF) is Required for SLAM

## Question

Can we substitute Extended Kalman Filter (EKF) with standard Kalman Filter (KF)?

## Answer: **No, EKF is required**

### Why EKF is Necessary

The measurement function in SLAM is **nonlinear** in the robot pose:

```
z_i = h_i(x) = rayCastDistance(px, py, heading, θ_i, M)
```

Where:
- `z_i`: Distance measurement for ray `i`
- `x = [px, py, heading]`: Robot pose
- `θ_i`: Ray angle relative to robot heading
- `M`: Occupancy grid map

### Nonlinearity Analysis

**Standard Kalman Filter** requires:
```
z = H * x + v
```
where `H` is a constant matrix.

**Ray casting distance** cannot be expressed as:
```
d = h(px, py, heading, θ, M) ≠ H * [px, py, heading]
```

**Why it's nonlinear**:

1. **Position (px, py)**:
   - Affects which grid cells the ray traverses
   - Non-linear: Different positions → different cell paths → different distances
   - Example: Ray from (0, 0) vs (1, 0) hits different obstacles at different distances

2. **Orientation (heading)**:
   - Affects ray direction
   - Non-linear: `rayAngle = heading + θ`, distance = `rayCast(px, py, rayAngle, M)`
   - Rotating robot by small angle can cause ray to miss/hit different obstacles

3. **Map interaction**:
   - Ray traversal path depends on pose
   - Distance = function of which cells ray traverses before hitting obstacle
   - Cannot be linearized with constant matrix

### Mathematical Justification

**KF measurement model**:
```
z = H * x + v
```
where `H` is constant (doesn't depend on `x`).

**EKF measurement model**:
```
z = h(x) + v
H = ∂h/∂x (Jacobian, depends on x)
```

**For ray casting**:
- `h(x)` is nonlinear function of `x`
- Jacobian `H = ∂h/∂x` must be computed at current pose estimate
- Jacobian changes as pose changes (different linearization point)

### Example

Consider a simple case: ray at angle 0 (forward) from pose (px, py, heading).

**Distance measurement**:
```
d = rayCast(px, py, heading, 0, M)
```

If robot moves from (px, py) to (px+δx, py+δy):
- Ray path changes
- Distance may change non-linearly
- Cannot predict with constant `H * [δx, δy]`

**EKF linearization**:
```
H[0] = ∂d/∂px  (computed numerically at current pose)
H[1] = ∂d/∂py
H[2] = ∂d/∂heading
```

These partial derivatives depend on current pose and map structure.

### When Could KF Work?

KF could only work if:
1. **Measurement function is linear**: `h(x) = H * x`
2. **Ray casting was linear**: Distance = constant matrix times pose
3. **Map was irrelevant**: Distance independent of map structure

None of these are true for ray-based SLAM.

### Performance Comparison

**KF** (if it worked):
- `H` computed once (constant)
- Update: `O(n*m)` where n=state dim, m=measurement dim
- Faster but **incorrect** for nonlinear system

**EKF** (required):
- `H` computed at each step (numerical or analytical Jacobian)
- Update: `O(n*m)` + Jacobian computation `O(n*m*rayCast_time)`
- Slower but **correct** for nonlinear system

**Optimization**: Use optimized ray casting to reduce Jacobian computation time.

### Conclusion

**We must use EKF** because:
1. ✅ Measurement function is nonlinear
2. ✅ Ray casting distance cannot be expressed as `H * x`
3. ✅ Jacobian depends on current pose
4. ✅ EKF handles nonlinearity correctly

**KF cannot be substituted** because:
1. ❌ Assumes linear measurement model
2. ❌ Requires constant `H` matrix
3. ❌ Would give incorrect results for ray-based SLAM

### Future Optimizations

While we can't substitute EKF with KF, we can:
1. **Optimize Jacobian computation**: Cache when pose changes little
2. **Use analytical Jacobians**: Faster than numerical (future work)
3. **Adaptive Jacobian update**: Update less frequently if pose changes slowly
4. **Approximate Jacobians**: Use simplified models for speed (trade accuracy)

But the fundamental algorithm must be EKF (or other nonlinear filter like UKF, particle filter).

