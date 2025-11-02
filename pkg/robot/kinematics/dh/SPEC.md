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
- `1`: Alpha is variable (prismatic joint with twist)
- `2`: R is variable (prismatic joint along X)
- `3`: D is variable (prismatic joint along Z)

**Joint Types**:
- **Revolute Joint**: Index = 0, Theta is variable
- **Prismatic Joint**: Index = 2 or 3, R or D is variable

**Questions**:
1. Should we support more joint types (cylindrical, spherical)?
2. How to handle joint type validation?
3. Should we support joint coupling (e.g., parallel joints)?
4. How to handle joint singularities?

### DenavitHartenberg (`denavithartenberg.go`)

**Purpose**: Forward and inverse kinematics solver for arbitrary DOF manipulators

**Current Implementation**:
- ✅ Forward Kinematics (FK): Fully implemented
- ❌ Inverse Kinematics (IK): Not implemented (returns false)

**Interface**:
```go
type DenavitHartenberg struct {
    c             []Config      // Joint configurations
    eps           float32       // Convergence tolerance
    maxIterations int           // Maximum IK iterations
    params        []float32     // Joint parameters (input/output)
    pos           [7]float32    // End-effector pose (x, y, z, qw, qx, qy, qz)
    H0i           []mat.Matrix4x4  // Transformation matrices
}

func (p *DenavitHartenberg) Forward() bool   // FK: params → pos
func (p *DenavitHartenberg) Inverse() bool  // IK: pos → params (NOT IMPLEMENTED)
```

**Forward Kinematics (FK)**:
1. Initialize identity transformation matrix
2. For each joint:
   - Calculate joint transformation matrix using DH parameters
   - Multiply with previous transformation
3. Extract position (x, y, z) from transformation matrix
4. Extract orientation (quaternion) from transformation matrix

**Inverse Kinematics (IK)** - TO BE IMPLEMENTED:
The IK solver needs to be implemented. Common approaches:
1. **Analytical IK**: Closed-form solutions (only for specific configurations)
2. **Numerical IK**: Iterative methods (Jacobian-based)
3. **Hybrid IK**: Analytical where possible, numerical otherwise

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

**Issues**:
- `H0i` slice not initialized in `New()` - potential panic
- Quaternion extraction assumes `Quaternion()` method exists on `Matrix4x4`

### Inverse Kinematics (To Be Implemented)

**Status**: ❌ Not Implemented

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

1. **IK Not Implemented**: Inverse kinematics returns false
2. **H0i Not Initialized**: Slice not allocated in constructor
3. **No Joint Limit Enforcement in IK**: (will be needed)
4. **No Singularity Handling**: (will be needed)
5. **No Multiple Solutions**: (will be needed)
6. **Limited Testing**: Missing comprehensive tests

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

