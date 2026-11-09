# Planar Kinematics Specification

## Overview

The planar kinematics package provides forward and inverse kinematics solvers for planar manipulators (manipulators where all joints lie in a single plane). This is a simplified case compared to general DH parameterization, allowing for efficient analytical solutions.

## Components

### Configuration (`config.go`)

**Purpose**: Configuration for a single planar joint

**Structure**:
```go
type Config struct {
    Min    float32  // Minimum angle limit
    Max    float32  // Maximum angle limit
    Length float32  // Link length
}
```

**Characteristics**:
- Joint rotation around Z-axis (perpendicular to plane)
- Link length in XZ plane
- Angle limits for joint constraints (radians; lengths in metres)

**Validation**: `Config.Validate()` rejects non-finite fields, `Min >= Max` (the zero value would pin the joint at 0) and negative lengths; errors wrap `kintypes.ErrInvalidConfig`. `New2DOF([2]Config) (*p2d, error)` and `New3DOF([3]Config) (*p3d, error)` validate; `New3DOF` also requires femur and tibia lengths > 0.

**Questions**:
1. Should we support prismatic joints in planar kinematics?
2. How to handle joint limits in IK?
3. Should we support different joint configurations?

### Planar 2DOF (`planar2dof.go`)

**Purpose**: Forward and inverse kinematics for 2-degree-of-freedom planar manipulator

**Current Implementation**:
- ✅ Forward Kinematics (FK): Fully implemented
- ✅ Inverse Kinematics (IK): Fully implemented

**Structure**:
```go
type p2d struct {
    c      [2]Config      // Joint configurations
    params [2]float32     // Joint angles (input/output)
    pos    [6]float32     // End-effector pose (x, y, z, orientation, ...)
}

func (p *p2d) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error
func (p *p2d) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error
```

**Forward Kinematics**:
1. Apply joint angle limits
2. Calculate position in XZ plane:
   - `x = l0 + l1*cos(a1)`
   - `z = l1*sin(a1)`
3. Rotate around Z-axis by `a0`:
   - `x' = x*cos(a0)`
   - `y' = x*sin(a0)`
   - `z' = z`
4. Store position and orientation angles in the provided `destination` matrix (rows 0-5, column 0)

**Inverse Kinematics**:
1. Calculate distance from base to target:
   - `x_prime = sqrt(x^2 + y^2) - l0`
2. Calculate joint angles:
   - `a0 = atan2(y, x)`  // Base rotation
   - `a1 = atan2(z, x_prime)`  // Elbow angle
3. Apply joint limits and write results to the `controls` matrix

**Questions**:
1. Should we support multiple IK solutions (elbow up/down)?
2. How to handle unreachable targets (outside workspace)?
3. Should we validate IK solutions against joint limits?
4. How to optimize for real-time constraints?

### Planar 3DOF (`planar3dof.go`)

**Purpose**: Forward and inverse kinematics for 3-degree-of-freedom planar manipulator

**Current Implementation**:
- ✅ Forward Kinematics (FK): Fully implemented
- ✅ Inverse Kinematics (IK): Fully implemented

**Structure**:
```go
type p3d struct {
    c      [3]Config      // Joint configurations
    params [3]float32     // Joint angles (input/output)
    pos    [6]float32     // End-effector pose (x, y, z, orientation, ...)
}

func (p *p3d) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error
func (p *p3d) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error
```

**Forward Kinematics**:
1. Apply joint angle limits
2. Calculate intermediate angles:
   - `a2_total = a2 + a1`  // Second link angle relative to first
3. Calculate position in XZ plane:
   - `x = l0 + l1*cos(a1) + l2*cos(a2_total)`
   - `z = l1*sin(a1) + l2*sin(a2_total)`
4. Rotate around Z-axis by `a0`
5. Store position and orientation angles in the provided `destination` matrix

**Inverse Kinematics** (analytic, exact, no allocation):
1. Signed reach: for coxa `a0 = atan2(y, x)` the reach is `s = +sqrt(x^2 + y^2)`; for `a0 + π` (foot behind the yaw axis) `s = -sqrt(x^2 + y^2)`. `x_prime = s - l0`, `r^2 = x_prime^2 + z^2`.
2. If `r` is outside `[|l1 - l2|, l1 + l2]` (relative slack 1e-5) the direction is unreachable.
3. Knee: `a2 = ∓acos((r^2 - l1^2 - l2^2) / (2*l1*l2))` (knee down, `a2 ≤ 0`, first); femur `a1 = atan2(z, x_prime) - atan2(l2*sin(a2), l1 + l2*cos(a2))`.
4. Every angle is wrapped modulo 2π into its limits; candidates are tried in the order (coxa forward, knee down), (forward, knee up), (reversed, knee down), (reversed, knee up); the first in-limit one is returned with `nil`.
5. Otherwise `controls` still receives a best-effort in-limit solution and the error is `kintypes.ErrUnreachable` (no direction reachable: leg stretched/folded towards the target) or `kintypes.ErrJointLimit` (the nearest of: for the coxa clamped towards the foot and reversed, each tibia candidate {Min, Max, 0, π} with the femur re-aimed at the target, and each femur limit with the tibia re-aimed; all clamped).
- `p2d` and `p3d` reject a NaN/Inf target with `kintypes.ErrNonFinite`, `controls` untouched. Instances hold scratch state: one per goroutine / leg.
- Verified: review probe R workspace 500/500 exact (78 feet behind the yaw axis), ±π limits 2000/2000 exact (`FK(IK(t))` within 1e-4 m).

**Questions**:
1. Should we support multiple IK solutions (elbow up/down)?
2. How to handle workspace boundaries?
3. Should we validate IK solutions geometrically?
4. How to handle singular configurations?
5. Should we support redundancy resolution (3DOF for 2D positioning)?

### Arbitrary DOF Planar Kinematics (To Be Implemented)

**Purpose**: Forward and inverse kinematics for N-degree-of-freedom planar manipulator

**Status**: ❌ Not Implemented

**Requirements**:
1. Support arbitrary number of joints (N > 3)
2. Forward kinematics using recursive transformations
3. Inverse kinematics using numerical or analytical methods
4. Handle joint limits
5. Handle workspace constraints

**Recommended Implementation**:

#### Forward Kinematics (Arbitrary DOF)
```
function Forward(params):
    x = l0
    z = 0
    angle_sum = 0
    
    for i in 0..len(joints):
        angle_sum += params[i]
        x += lengths[i] * cos(angle_sum)
        z += lengths[i] * sin(angle_sum)
    
    // Rotate around Z-axis
    x' = x * cos(params[0])
    y' = x * sin(params[0])
    z' = z
    
    return (x', y', z')
```

#### Inverse Kinematics (Arbitrary DOF)

**For N ≤ 3**: Use analytical solutions (already implemented)

**For N > 3**: Use numerical methods (Jacobian-based)

**Approach**:
1. **Position IK** (first N-1 joints for 2D positioning):
   - Use Jacobian pseudo-inverse
   - Or use geometric iterative method
2. **Orientation IK** (last joint or set of joints):
   - Analytical or numerical

**Alternative**: Use geometric approach similar to 3DOF but extended

**Questions**:
1. **IK Algorithm**:
   - Should we use analytical methods for specific N values?
   - Should we use numerical methods (Jacobian-based) for arbitrary N?
   - Should we support hybrid approaches?

2. **Redundancy**:
   - How to handle redundant planar manipulators (N > 3 for 2D)?
   - Should we support redundancy resolution?
   - How to optimize secondary criteria (joint angles, manipulability)?

3. **Workspace**:
   - How to calculate workspace boundaries?
   - How to handle unreachable targets?
   - Should we support workspace visualization?

## Design Questions

### Architecture

1. **Arbitrary DOF Implementation**:
   - Should we create a generic `Planar` struct for arbitrary DOF?
   - Or should we keep separate implementations (2DOF, 3DOF, NDOF)?
   - How to handle type safety with arbitrary DOF?

2. **Solution Selection**:
   - How to handle multiple IK solutions?
   - Should we return all solutions?
   - Should we select "best" solution (e.g., minimum joint movement)?

3. **Configuration Validation**:
   - How to validate joint configurations?
   - Should we check for degenerate configurations?
   - How to handle invalid link lengths (e.g., negative)?

### Performance

4. **Real-Time Constraints**:
   - How to ensure deterministic execution time?
   - Should we limit IK iterations for numerical methods?
   - How to optimize for embedded systems?

5. **Optimization**:
   - Should we cache intermediate calculations?
   - Should we use optimized trigonometric functions?
   - How to minimize floating-point operations?

### Compatibility

6. **Platform Support**:
   - How to handle TinyGo limitations?
   - Should we provide platform-specific optimizations?
   - How to test on embedded platforms?

## Implementation Requirements

### Current Implementation

**2DOF and 3DOF**: ✅ Fully implemented

**Issues**:
- No validation of IK solutions
- No multiple solution support
- No workspace boundary checking

### To Be Implemented

**Arbitrary DOF**:
1. Generic `Planar` struct for N DOF
2. Forward kinematics using recursive approach
3. Inverse kinematics:
   - Analytical for N ≤ 3
   - Numerical for N > 3 (Jacobian-based)
4. Workspace calculation
5. Multiple solution support
6. Joint limit enforcement

**Recommended Structure**:
```go
type Planar struct {
    c      []Config
    params []float32
    pos    [6]float32
}

func NewPlanar(cfg []Config) kinematics.Kinematics {
    return &Planar{
        c:      cfg,
        params: make([]float32, len(cfg)),
    }
}
```

## Known Issues

1. **Arbitrary DOF Not Implemented**: Only 2DOF and 3DOF supported
2. **No Multiple Solutions**: IK returns single solution
3. **2DOF IK**: targets off the reachable surface and limit violations are still clamped silently (3DOF reports typed errors).

## Potential Improvements

1. **Arbitrary DOF Support**: Generic implementation for N joints
2. **Multiple Solutions**: Return all valid IK solutions
3. **Workspace Calculation**: Calculate and validate workspace boundaries
4. **Solution Selection**: Select best solution based on criteria
5. **Trajectory Planning**: Integration with motion planning
6. **Optimization**: Performance optimizations
7. **Testing**: Comprehensive test suite
8. **Documentation**: Complete algorithm documentation

## Reference Implementation

You mentioned adding temporary C code for reference. The implementation should demonstrate:

1. **2 DOF**: ✅ Already implemented in Go
2. **3 DOF**: ✅ Already implemented in Go
3. **Arbitrary DOF**: Need implementation

For arbitrary DOF, the C reference should show:
- Recursive forward kinematics
- Numerical inverse kinematics (Jacobian-based)
- Or analytical extension of 2/3 DOF methods

Common planar IK algorithms:
- Geometric approach (for N ≤ 3)
- Jacobian pseudo-inverse (for N > 3)
- Hybrid approach (analytical where possible)

The C reference will help understand:
- Efficient trigonometric calculations
- Jacobian computation for planar case
- Numerical iteration for IK
- Workspace boundary calculation

