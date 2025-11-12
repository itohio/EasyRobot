# Denavit-Hartenberg Inverse Kinematics Implementation Plan

## Overview

This document outlines the implementation plan for the `Inverse()` method in the Denavit-Hartenberg kinematics package. The implementation will support arbitrary DOF manipulators with multiple algorithm strategies.

## Current State

### Issues to Fix First

1. **H0i Not Initialized**: `H0i` slice not allocated in constructor
   - **Fix**: Allocate `H0i` slice in `New()` function
   - **Size**: `len(cfg) + 1` (one matrix per joint plus base frame)

2. **Quaternion Method**: Ensure `Matrix4x4.Quaternion()` exists
   - **Check**: Verify method exists in `mat` package
   - **Fix**: If missing, implement quaternion extraction from rotation matrix

## Implementation Strategy

### Algorithm Selection

Based on DOF and configuration:
1. **2-3 DOF**: Analytical solutions (geometric)
2. **6 DOF with Spherical Wrist**: Analytical decomposition (position + orientation)
3. **Arbitrary DOF**: Numerical methods (Jacobian-based)

### Recommended Approach: Hybrid

Use a hybrid approach that:
- Attempts analytical IK when applicable (2, 3, 6 DOF)
- Falls back to numerical IK for arbitrary DOF or when analytical fails
- Supports user-specified algorithm preference

## Implementation Phases

### Phase 1: Fix Existing Issues and Add Helper Functions

**Tasks**:
1. Fix `H0i` initialization in `New()`
2. Verify/implement quaternion extraction
3. Implement helper functions for error calculation
4. Implement joint limit application

**Files to Modify**:
- `denavithartenberg.go`

**New Helper Functions**:
```go
// Apply joint limits to parameters
func (p *DenavitHartenberg) applyJointLimits() {
    for i, cfg := range p.c {
        p.params[i] = cfg.Limit(p.params[i])
    }
}

// Calculate position error
func (p *DenavitHartenberg) positionError(targetPos [3]float32) float32 {
    dx := targetPos[0] - p.pos[0]
    dy := targetPos[1] - p.pos[1]
    dz := targetPos[2] - p.pos[2]
    return math32.Sqrt(dx*dx + dy*dy + dz*dz)
}

// Calculate orientation error (angle between quaternions)
func (p *DenavitHartenberg) orientationError(targetQuat [4]float32) float32 {
    // Calculate angle between quaternions
    // Using quaternion dot product
    dot := p.pos[3]*targetQuat[3] + p.pos[4]*targetQuat[4] + 
           p.pos[5]*targetQuat[5] + p.pos[6]*targetQuat[6]
    if dot > 1.0 {
        dot = 1.0
    }
    if dot < -1.0 {
        dot = -1.0
    }
    return math32.Acos(math32.Abs(dot))
}
```

### Phase 2: Implement Jacobian Calculation

**Purpose**: Calculate geometric Jacobian for numerical IK methods

**Tasks**:
1. Implement `calculateJacobian()` method
2. Calculate linear velocity Jacobian (3×N)
3. Calculate angular velocity Jacobian (3×N)
4. Combine into 6×N Jacobian matrix

**Algorithm**:
For each joint `i`:
1. Get transformation matrix from base to joint `i` (`H0i[i]`)
2. Extract Z-axis direction (rotation axis)
3. Calculate end-effector position relative to joint `i`
4. Calculate Jacobian column:
   - Linear part: `Z_i × (p_ee - p_i)` for revolute joints
   - Angular part: `Z_i` for revolute joints
   - For prismatic: translation along joint axis

**Implementation**:
```go
func (p *DenavitHartenberg) calculateJacobian() (mat.Matrix, error) {
    dof := len(p.c)
    // 6 rows: 3 for position, 3 for orientation
    J := mat.New(6, dof)
    
    // Get end-effector position
    eePos := [3]float32{p.pos[0], p.pos[1], p.pos[2]}
    
    for i := 0; i < dof; i++ {
        // Get joint position and axis from H0i[i]
        jointPos := [3]float32{}
        p.H0i[i].Col(3, jointPos[:])
        
        // Get Z-axis (rotation/translation axis)
        zAxis := [3]float32{}
        p.H0i[i].Col(2, zAxis[:])  // Assuming Z-axis is in column 2
        
        cfg := p.c[i]
        
        if cfg.Index == 0 {  // Revolute joint
            // Linear velocity: cross product
            r := [3]float32{
                eePos[0] - jointPos[0],
                eePos[1] - jointPos[1],
                eePos[2] - jointPos[2],
            }
            linear := vec.Cross3D(zAxis, r)
            
            // Set Jacobian columns
            J.SetCol(i, 0, linear[:])
            J.SetCol(i, 3, zAxis[:])
        } else if cfg.Index == 2 || cfg.Index == 3 {  // Prismatic joint
            // Linear velocity: translation axis
            J.SetCol(i, 0, zAxis[:])
            // Angular velocity: zero
        }
    }
    
    return J, nil
}
```

**Notes**:
- Need to verify matrix column setting methods exist
- May need to create helper functions for cross product
- Need to handle quaternion vs axis-angle for angular velocity

### Phase 3: Implement Pseudo-Inverse and DLS

**Purpose**: Implement matrix pseudo-inverse and damped least squares for numerical IK

**Tasks**:
1. Implement `pseudoInverse()` helper
2. Implement `dampedLeastSquares()` helper
3. Add damping parameter configuration

**Pseudo-Inverse (Moore-Penrose)**:
```
J+ = J^T * (J * J^T)^(-1)  // If J has more rows than columns
J+ = (J^T * J)^(-1) * J^T  // If J has more columns than rows
```

**Damped Least Squares (DLS)**:
```
J+ = J^T * (J * J^T + λ^2 * I)^(-1)
```
Where `λ` is damping factor.

**Implementation Considerations**:
- Need matrix inverse (may need to implement)
- Need SVD for robust pseudo-inverse (future enhancement)
- Damping factor selection (adaptive vs fixed)

### Phase 4: Implement Numerical IK Method

**Purpose**: Iterative Jacobian-based IK solver

**Tasks**:
1. Implement `inverseNumerical()` method
2. Support position-only, orientation-only, or both
3. Handle convergence checking
4. Handle joint limits

**Algorithm**:
```go
func (p *DenavitHartenberg) inverseNumerical(targetPos [3]float32, targetQuat [4]float32) bool {
    // Extract target
    // Initialize if needed
    
    for iteration := 0; iteration < p.maxIterations; iteration++ {
        // Forward kinematics
        if !p.Forward() {
            return false
        }
        
        // Calculate error
        posErr := p.positionError(targetPos)
        oriErr := p.orientationError(targetQuat)
        
        // Check convergence
        if posErr < p.eps && oriErr < p.eps {
            return true
        }
        
        // Calculate Jacobian
        J, err := p.calculateJacobian()
        if err != nil {
            return false
        }
        
        // Calculate error vector (6D: 3 position + 3 orientation)
        errorVec := mat.New(6, 1)
        // Position error
        errorVec[0][0] = targetPos[0] - p.pos[0]
        errorVec[1][0] = targetPos[1] - p.pos[1]
        errorVec[2][0] = targetPos[2] - p.pos[2]
        // Orientation error (convert quaternion error to axis-angle)
        // ... orientation error calculation
        
        // Calculate delta joint angles
        Jplus := pseudoInverse(J)  // or DLS
        deltaParams := Jplus.MulVec(errorVec)
        
        // Update parameters
        for i := range p.params {
            p.params[i] += deltaParams[i]
        }
        
        // Apply joint limits
        p.applyJointLimits()
    }
    
    return false  // Did not converge
}
```

**Notes**:
- Need to handle orientation error (quaternion to axis-angle conversion)
- May need to scale error vector for position vs orientation
- Need adaptive step size for better convergence

### Phase 5: Implement Analytical IK for Specific Cases (Optional)

**Purpose**: Fast analytical solutions for 2, 3, and 6 DOF cases

**Tasks**:
1. Detect special configurations (spherical wrist, planar)
2. Implement analytical IK for 2 DOF
3. Implement analytical IK for 3 DOF
4. Implement analytical IK for 6 DOF with spherical wrist

**2 DOF Analytical IK**:
- Simple geometric solution
- Two solutions (elbow up/down)
- Can use law of cosines

**3 DOF Analytical IK**:
- Extend 2 DOF approach
- May need to solve for intersection of spheres
- Multiple solutions possible

**6 DOF with Spherical Wrist**:
- Decompose into position IK (first 3 joints) and orientation IK (wrist)
- Position IK solves for wrist center
- Orientation IK solves for wrist angles

**Implementation Strategy**:
- Check if analytical IK is applicable
- If yes, try analytical first
- If fails or not applicable, fall back to numerical

### Phase 6: Implement Main Inverse() Method

**Purpose**: Main entry point that orchestrates IK solution

**Tasks**:
1. Extract target from `pos` field
2. Select IK algorithm (analytical vs numerical)
3. Call appropriate IK method
4. Handle errors and convergence

**Implementation**:
```go
func (p *DenavitHartenberg) Inverse() bool {
    // Check initialization
    if len(p.c) == 0 {
        return false
    }
    
    // Extract target from pos (assumes pos contains desired pose)
    targetPos := [3]float32{p.pos[0], p.pos[1], p.pos[2]}
    targetQuat := [4]float32{p.pos[3], p.pos[4], p.pos[5], p.pos[6]}
    
    // Select IK method based on DOF
    dof := len(p.c)
    
    switch {
    case dof <= 3:
        // Try analytical IK
        // If fails, fall back to numerical
        return p.inverseNumerical(targetPos, targetQuat)
    
    case dof == 6 && p.hasSphericalWrist():
        // Try analytical IK with spherical wrist decomposition
        // If fails, fall back to numerical
        if p.inverseAnalytical6DOF(targetPos, targetQuat) {
            return true
        }
        return p.inverseNumerical(targetPos, targetQuat)
    
    default:
        // Use numerical IK
        return p.inverseNumerical(targetPos, targetQuat)
    }
}
```

### Phase 7: Add Configuration and Optimization

**Tasks**:
1. Add IK algorithm selection option
2. Add damping factor configuration
3. Add step size configuration
4. Add orientation weight configuration
5. Add singularity detection and handling

**Configuration Options**:
```go
type IKOptions struct {
    Algorithm    string   // "auto", "analytical", "numerical"
    Method       string   // "pseudo_inverse", "dls", "lm"
    Damping      float32  // For DLS
    StepSize     float32  // Adaptive step size
    OrientWeight float32  // Weight for orientation vs position
    MaxIterations int     // Override default
    Eps          float32  // Override default
}
```

## Testing Plan

### Unit Tests

1. **Forward Kinematics Validation**:
   - Test FK with known configurations
   - Verify position and orientation output
   - Test with different joint limits

2. **Jacobian Calculation**:
   - Test Jacobian for simple 2 DOF manipulator
   - Test Jacobian for 3 DOF manipulator
   - Verify Jacobian dimensions
   - Test numerical vs analytical Jacobian (if available)

3. **Pseudo-Inverse**:
   - Test pseudo-inverse calculation
   - Test with singular matrices
   - Test with rectangular matrices

4. **IK Convergence**:
   - Test IK with reachable targets
   - Test IK with unreachable targets
   - Test IK with joint limits
   - Test IK convergence rate

5. **Error Calculation**:
   - Test position error calculation
   - Test orientation error calculation
   - Test combined error

### Integration Tests

1. **Full IK Workflow**:
   - Set target pose
   - Solve IK
   - Verify FK matches target
   - Test with different DOF (2, 3, 6, 7+)

2. **Real-World Scenarios**:
   - Test with common robot configurations
   - Test trajectory following
   - Test with obstacles (joint limits)

### Performance Tests

1. **Convergence Speed**:
   - Measure iterations to convergence
   - Test with different initial conditions
   - Test with different tolerances

2. **Computation Time**:
   - Benchmark IK for different DOF
   - Compare analytical vs numerical
   - Test on embedded targets (if applicable)

## Implementation Questions to Resolve

1. **Target Input Format**:
   - How should target pose be specified?
   - Should we add separate target methods (SetTargetPos, SetTargetOri)?
   - Or use pos field as both input and output?

2. **Orientation Representation**:
   - Should we support Euler angles in addition to quaternions?
   - How to handle orientation error calculation?
   - Should we use axis-angle for error?

3. **Multiple Solutions**:
   - Should IK return multiple solutions?
   - How to select best solution?
   - Should we support solution selection criteria?

4. **Singularity Handling**:
   - How to detect singularities?
   - What damping factor to use for DLS?
   - Should damping be adaptive?

5. **Convergence Criteria**:
   - Should position and orientation have separate tolerances?
   - Should we use relative or absolute error?
   - How to handle slow convergence?

6. **Performance vs Accuracy**:
   - Should we optimize for speed or accuracy?
   - Should we support different quality levels?
   - How to balance real-time constraints?

## Dependencies

### Required Math Functions

1. **Matrix Operations**:
   - Matrix multiplication
   - Matrix inverse (for pseudo-inverse)
   - Matrix transpose
   - Matrix determinant (for singularity detection)

2. **Vector Operations**:
   - Cross product (3D)
   - Dot product
   - Vector magnitude
   - Vector normalization

3. **Quaternion Operations**:
   - Quaternion multiplication
   - Quaternion to axis-angle
   - Axis-angle to quaternion
   - Quaternion SLERP (for orientation interpolation)

4. **Numerical Methods**:
   - Pseudo-inverse computation
   - SVD (for robust pseudo-inverse, optional)
   - Eigenvalue calculation (for singularity detection, optional)

### Missing Implementations to Create

1. Matrix inverse (if not in `mat` package)
2. Matrix pseudo-inverse
3. Damped least squares
4. Quaternion error calculation
5. Jacobian calculation
6. Singularity detection

## File Structure

```
pkg/robot/kinematics/dh/
├── denavithartenberg.go     # Main struct and Forward/Inverse methods
├── config.go                # Config struct and transform calculation
├── jacobian.go             # NEW: Jacobian calculation
├── pseudo_inverse.go       # NEW: Pseudo-inverse and DLS
├── inverse_numerical.go    # NEW: Numerical IK implementation
├── inverse_analytical.go    # NEW: Analytical IK implementations (optional)
├── helpers.go              # NEW: Helper functions (errors, limits, etc.)
├── SPEC.md                 # Specification
└── IMPLEMENTATION_PLAN.md  # This file
```

## Timeline Estimate

- **Phase 1**: 1-2 days (fixes and helpers)
- **Phase 2**: 2-3 days (Jacobian calculation)
- **Phase 3**: 2-3 days (pseudo-inverse and DLS)
- **Phase 4**: 3-4 days (numerical IK)
- **Phase 5**: 3-5 days (analytical IK, optional)
- **Phase 6**: 1 day (main Inverse method)
- **Phase 7**: 1-2 days (configuration)
- **Testing**: 3-5 days

**Total**: ~16-25 days (depending on optional features and testing depth)

## Success Criteria

1. ✅ IK converges for reachable targets
2. ✅ IK respects joint limits
3. ✅ IK handles singularities gracefully
4. ✅ IK works for 2, 3, 6, and arbitrary DOF
5. ✅ IK converges within maxIterations
6. ✅ FK after IK matches target within tolerance
7. ✅ Comprehensive test coverage
8. ✅ Performance acceptable for real-time use

## References

1. **Robotics Textbook**: Craig, "Introduction to Robotics: Mechanics and Control"
2. **IKFast**: OpenRAVE IKFast (analytical IK generator)
3. **KDL**: Orocos KDL (Kinematics and Dynamics Library)
4. **Textbooks**: 
   - Spong et al., "Robot Modeling and Control"
   - Siciliano et al., "Robotics: Modelling, Planning and Control"

