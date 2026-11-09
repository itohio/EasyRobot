# Denavit-Hartenberg Kinematics Specification

## Overview

The DH (Denavit-Hartenberg) package provides forward and inverse kinematics solvers for serial manipulators using the standard DH parameterization. This is the most general approach for robot manipulators and supports arbitrary numbers of joints.

## Components

### Configuration (`config.go`)

**Purpose**: DH parameter configuration for a single joint

**Structure**:
```go
type Config struct {
    Min   float32  // Minimum joint limit
    Max   float32  // Maximum joint limit
    Theta float32  // Joint angle offset
    Alpha float32  // Link twist angle
    R     float32  // Link length (along X)
    D     float32  // Link offset (along Z)
    Index int      // Parameter index (0=theta, 1=alpha, 2=R, 3=D)
}
```

**DH Parameters**:
- **Theta (θ)**: Rotation around Z axis (previous frame)
- **Alpha (α)**: Rotation around X axis (previous frame)
- **R (r)**: Displacement along X axis
- **D (d)**: Displacement along Z axis

**Parameter Index**:
- `0`: Theta is variable (revolute joint)
- `1`: Alpha is variable (revolute about the link's x axis)
- `2`: R is variable (prismatic joint along X)
- `3`: D is variable (prismatic joint along Z)

**Joint Types**:
- **Revolute Joint**: Index = 0 (θ) or 1 (α); angles wrap modulo 2π into the limits during IK
- **Prismatic Joint**: Index = 2 or 3, R or D is variable

**Validation**: `Config.Validate()` rejects non-finite fields, `Min >= Max` (so the zero value, which would pin the joint at 0, is rejected) and `Index ∉ {0,1,2,3}`; errors wrap `kintypes.ErrInvalidConfig`. Units: radians and metres.

**Questions**:
1. Should we support more joint types (cylindrical, spherical)?
2. How to handle joint type validation?
3. Should we support joint coupling (e.g., parallel joints)?
4. How to handle joint singularities?

### DenavitHartenberg (`denavithartenberg.go`)

**Purpose**: Forward and inverse kinematics solver for arbitrary DOF manipulators

**Current Implementation**:
- ✅ Forward Kinematics (FK): Destination-based matrix API implemented
- ✅ Inverse Kinematics (IK): Iterative Jacobian-based solver implemented (position only)

**Interface**:
```go
type DenavitHartenberg struct {
    c             []Config      // Joint configurations
    eps           float32       // Convergence tolerance
    maxIterations int           // Maximum IK iterations
    params        []float32     // Joint parameters (input/output)
    pos           [7]float32    // End-effector pose (x, y, z, qx, qy, qz, qw)
    H0i           []mat.Matrix4x4  // Transformation matrices
}

func (p *DenavitHartenberg) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error
func (p *DenavitHartenberg) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error
```

**Forward Kinematics (FK)**:
1. Validate that `state` is a DOF×1 joint parameter column vector and `destination` is at least 7×1.
2. Initialize identity transformation matrix.
3. For each joint:
   - Clamp the joint value to configuration limits.
   - Calculate the joint transformation matrix using DH parameters.
   - Multiply with the cumulative transform.
4. Extract position (x, y, z) from the final transform and write to `destination`.
5. Extract orientation (quaternion) from the final transform and write to rows 3–6 of `destination`.

**Constructor**: `New(eps float32, maxIterations int, cfg ...Config) (*DenavitHartenberg, error)` validates every joint, `eps > 0`, `maxIterations >= 0`, at least one joint, and preallocates all solver buffers.

**Inverse Kinematics (IK)** (position only; orientation rows of `destination` are ignored):
- Damped least squares `Δq = S·J'ᵀ(J'J'ᵀ + λ²I)⁻¹e` on the column-scaled Jacobian `J' = J·S` (S = 1 for θ/α joints, `scale` for r/d joints, so every column is a length and λ is unit-consistent), 3×3 Cholesky, any DOF incl. n > 3. Adaptive λ (Levenberg–Marquardt accept/reject; floor `3e-3·scale` acts as the relative σ cutoff), Cartesian error clamped to `0.2·scale` per step, per-joint step cap (0.35 rad / `0.1·scale`), revolute angles wrapped into the limits, limit-aware active set (joints on a limit pushed outward are removed from the step). `scale` = Σ|(R, D)| per joint, so tuning is unit-independent (gantry and SCARA give identical results in m and mm).
- An attempt ends early when it crawls along a limit (< 10 % distance gain per 8 iterations with a joint blocked) or stalls. Up to 4 restarts share the `maxIterations` budget. For the most proximal blocked angular joint: flip the elbow (negate the next parallel-axis joint, re-aim the blocked one at the target), else re-aim it at the target, else aim it away from the target (a yaw whose foot must fold behind its axis); a stall with no blocked joint restarts nudged by ±0.2 rad.
- Jacobian columns: θ `zᵢ × (pₙ − oᵢ)`, α `xᵢ₊₁ × (pₙ − oᵢ₊₁)`, r `xᵢ₊₁`, d `zᵢ`.
- Input errors (dimensions; `kintypes.ErrNonFinite` for a NaN/Inf target or seed) return before solving and leave `controls` untouched. Otherwise `controls` is always written with the best in-limit solution of this call (initialised from the seed, so a residual that overflows float32 returns the seed). Errors: `kintypes.ErrUnreachable` (target beyond Σ|(R, D)|, or stalled at a residual minimum with no joint blocked); for a blocked failure up to 30 extra iterations run with the limits lifted: converging in-limit returns that solution, converging (or getting ≥ 4× closer) only past a limit → `kintypes.ErrJointLimit`, ending at a stationary point with a nonzero residual (`|Jᵀe| ≤ 1e-2·‖J‖·|e|`, e.g. inside the inner workspace hole) → `kintypes.ErrUnreachable`, otherwise `kintypes.ErrNoConvergence`. This is a local test: another in-limit branch may exist (seen in ≤ 3/2000 random-seed cases).
- Not safe for concurrent use (preallocated workspaces): one instance per goroutine / leg.
- No allocation per call, including failure and restart paths (`testing.AllocsPerRun` = 0 when callers pass pre-boxed matrices).
- Measured (hexapod leg 39/45/100 mm, eps 1e-4 m, 100 iterations): review probe R workspace nominal seed 500/500 (iterations p50/p90/max 5/8/27), zero seed 500/500, warm 500/500; full-limit targets (half knee-up) from the nominal seed 2000/2000; 4-link planar and 6-DOF PUMA-like 300/300.

**Recommended IK Approach**:
- **For 2-3 DOF**: Analytical solutions (geometric)
- **For 4-6 DOF**: Analytical solutions where possible (e.g., spherical wrist)
- **For 7+ DOF**: Numerical methods (Jacobian pseudo-inverse, damped least squares)
- **Arbitrary DOF**: Numerical methods with redundancy resolution

**IK Algorithm Options**:

1. **Jacobian Pseudo-Inverse**:
   - Calculate Jacobian matrix
   - Compute pseudo-inverse
   - Update joint angles iteratively
   - Handle singularities with damping

2. **Damped Least Squares (DLS)**:
   - Similar to pseudo-inverse but with damping
   - Better singularity handling
   - Slower convergence near singularities

3. **Levenberg-Marquardt**:
   - Adaptive damping
   - Faster convergence
   - Better for far-from-target cases

4. **Analytical Methods**:
   - For specific configurations (e.g., 6-DOF with spherical wrist)
   - Solve geometric constraints directly
   - Much faster when applicable

**Questions**:
1. **IK Implementation**:
   - Should we implement analytical IK for specific configurations (2, 3, 6 DOF)?
   - Should we implement numerical IK (Jacobian-based) for arbitrary DOF?
   - Should we support multiple IK algorithms (user-selectable)?
   - How to handle IK singularities?
   - How to handle multiple IK solutions?

2. **Joint Limits**:
   - How to enforce joint limits in IK?
   - Should we use constrained optimization?
   - How to handle unreachable targets (outside workspace)?

3. **Convergence**:
   - How to determine convergence criteria?
   - What tolerance (eps) is appropriate?
   - How to handle non-convergence?

4. **Performance**:
   - How to optimize FK for real-time?
   - Should we cache transformation matrices?
   - How to optimize IK iterations?

5. **Redundancy**:
   - How to handle redundant manipulators (7+ DOF)?
   - Should we support redundancy resolution (optimize secondary criteria)?
   - How to handle self-collision avoidance?

## Implementation Requirements

### Forward Kinematics (Current)

**Status**: ✅ Implemented

**Algorithm**:
1. Initialize `H0i[0]` to identity matrix
2. For each joint `i`:
   - Calculate transformation matrix `H` using `Config.CalculateTransform()`
   - Multiply: `H0i[i+1] = H0i[i] * H`
3. Extract position from `H0i[len(c)].Col(3)` → `pos[0:3]`
4. Extract quaternion from `H0i[len(c)].Quaternion()` → `pos[3:7]`
5. **Matrix Semantics**: Fixed-size matrix operations now return new values; callers must capture the return value rather than relying on in-place mutation.

**Issues**:
- `H0i` slice not initialized in `New()` - potential panic
- Quaternion extraction assumes `Quaternion()` method exists on `Matrix4x4`

### Inverse Kinematics

**Status**: ✅ Implemented (position-only DLS, see above)

**Requirements**:
1. Support arbitrary number of joints (2, 3, 4, 5, 6, 7+)
2. Handle joint limits
3. Handle singularities
4. Support multiple solutions (when available)
5. Converge within `maxIterations`
6. Respect `eps` tolerance

**Recommended Implementation**:

#### For 2-3 DOF (Analytical)
- Geometric solutions
- Direct trigonometric calculations
- Multiple solutions possible

#### For 6 DOF (Analytical for Spherical Wrist)
- Decompose into position and orientation IK
- Solve position IK for first 3 joints
- Solve orientation IK for wrist (3 joints)
- Multiple solutions possible

#### For Arbitrary DOF (Numerical)
- Jacobian-based iterative method
- Pseudo-inverse or DLS
- Redundancy resolution for 7+ DOF

**Algorithm Pseudocode**:
```
function InverseKinematics(target_pos, target_quat):
    for iteration in 0..maxIterations:
        current_pos = Forward(params)
        error = target_pos - current_pos
        if |error| < eps:
            return true
        
        J = CalculateJacobian(params)
        dq = J_pseudo_inverse * error
        params = params + dq
        params = ApplyJointLimits(params)
    
    return false  // Did not converge
```

## Design Questions

### Architecture

1. **IK Algorithm Selection**:
   - Should we detect configuration and use analytical IK when possible?
   - Should we allow user to specify IK algorithm?
   - Should we support fallback (analytical → numerical)?

2. **Multiple Solutions**:
   - How to handle multiple IK solutions?
   - Should we return all solutions?
   - Should we select "best" solution based on criteria?

3. **Singularity Handling**:
   - How to detect singularities?
   - Should we use damped least squares?
   - How to handle near-singular configurations?

### Performance

4. **Real-Time Constraints**:
   - How to ensure deterministic execution time?
   - Should we limit IK iterations?
   - How to optimize for embedded systems?

5. **Optimization**:
   - Should we cache Jacobian matrices?
   - Should we use optimized matrix operations?
   - How to minimize floating-point operations?

### Compatibility

6. **Platform Support**:
   - How to handle TinyGo limitations?
   - Should we provide platform-specific optimizations?
   - How to test on embedded platforms?

## Known Issues

1. **Orientation targets ignored**: IK is position only.
2. **Quaternion output**: `Forward` rows 3–6 come from `Matrix.Quaternion`, which returns the inverse rotation (tracked separately).
3. **Local method**: from a cold seed some targets end at a limit (`ErrJointLimit` with best effort); seed from the previous solution when tracking.
4. **No Multiple Solutions**: one solution per call, chosen by the seed.

## Potential Improvements

1. **Complete IK Implementation**: Analytical and numerical methods
2. **Multiple Solutions**: Return all valid IK solutions
3. **Singularity Handling**: Robust singularity detection and handling
4. **Redundancy Resolution**: Support for redundant manipulators
5. **Trajectory Planning**: Integration with motion planning
6. **Self-Collision Avoidance**: Check and avoid self-collisions
7. **Testing**: Comprehensive test suite
8. **Documentation**: Complete algorithm documentation
9. **Optimization**: Performance optimizations

## Reference Implementation

You mentioned adding temporary C code for reference. The IK implementation should handle:

1. **2 DOF**: Analytical solution (simple geometric)
2. **3 DOF**: Analytical solution (planar or spatial)
3. **6 DOF**: Analytical solution if spherical wrist, otherwise numerical
4. **Arbitrary DOF**: Numerical solution (Jacobian-based)

Common IK algorithms in C libraries:
- IKFast (analytical IK code generator)
- KDL (Kinematics and Dynamics Library) - numerical IK
- Orocos KDL - similar to KDL

The C reference should demonstrate:
- Jacobian calculation
- Pseudo-inverse computation
- Iterative update
- Joint limit handling
- Singularity handling

