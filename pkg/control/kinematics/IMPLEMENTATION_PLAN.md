# Kinematics Implementation Plan

## Overview

This plan addresses:
1. Updating Go planar kinematics code to match tested C reference
2. Implementing DH inverse kinematics
3. Moving Jacobian-related math to `pkg/core/math`
4. Ensuring compatibility with tested C implementations

## Analysis: C Reference vs Go Implementation

### Planar 3DOF Comparison

**C Reference (_PlanarKinematics3DOF.h)**:
- ✅ Forward kinematics matches Go implementation
- ✅ Inverse kinematics calculation matches Go implementation
- ⚠️ **Difference**: C code calls `forward()` after IK to verify actual position
- C code stores result in `actual` parameter (not just `param_arr`)

**Current Go Implementation**:
- Forward kinematics: ✅ Correct
- Inverse kinematics: ✅ Calculation correct
- ❌ **Missing**: Does not call Forward() after IK to verify/update actual position

### Required Fixes

1. **Planar 3DOF**: Add Forward() call after IK to match C reference
2. **Planar 2DOF**: Verify and potentially add Forward() call after IK
3. **DH**: Complete inverse kinematics implementation
4. **Math Library**: Add Jacobian, pseudo-inverse, DLS methods to `pkg/core/math`

## Implementation Phases

### Phase 1: Fix Planar Kinematics to Match C Reference

**Files to Modify**:
- `pkg/robot/kinematics/planar/planar2dof.go`
- `pkg/robot/kinematics/planar/planar3dof.go`

**Changes Required**:

#### Planar 3DOF Fix
```go
func (p *p3d) Inverse() bool {
    // ... existing calculation ...
    
    p.params[0] = p.c[0].Limit(math32.Atan2(p.pos[1], p.pos[0]))
    p.params[1] = gamma + alpha
    p.params[2] = beta - math32.Pi
    
    // ADD: Call Forward to verify/update actual position (matches C reference)
    if !p.Forward() {
        return false
    }
    
    return true
}
```

#### Planar 2DOF Fix
```go
func (p *p2d) Inverse() bool {
    // ... existing calculation ...
    
    p.params[0] = p.c[0].Limit(math32.Atan2(p.pos[1], p.pos[0]))
    p.params[1] = p.c[1].Limit(math32.Atan2(p.pos[2], x_prime))
    
    // ADD: Call Forward to verify/update actual position (matches C reference)
    if !p.Forward() {
        return false
    }
    
    return true
}
```

**Rationale**: C reference code explicitly calls `forward()` after IK to:
1. Verify the calculated parameters produce correct result
2. Update actual position (accounting for joint limits)
3. Ensure consistency between IK and FK

### Phase 2: Add Core Math Functions for Jacobian

**Files to Create/Modify**:
- `pkg/core/math/mat/jacobian.go` (NEW)
- `pkg/core/math/mat/pseudo_inverse.go` (NEW)
- `pkg/core/math/mat/inverse.go` (if not exists)

**Functions to Implement**:

#### 1. Matrix Inverse (`pkg/core/math/mat/inverse.go`)
```go
// Inverse calculates the inverse of a square matrix
// Returns error if matrix is singular (determinant ≈ 0)
func (m Matrix) Inverse(dst Matrix) error

// Inverse calculates the inverse of a Matrix4x4
func (m *Matrix4x4) Inverse(dst *Matrix4x4) error

// Same for Matrix3x3, Matrix2x2
```

**Algorithm**: Use LU decomposition or Gauss-Jordan elimination

#### 2. Pseudo-Inverse (`pkg/core/math/mat/pseudo_inverse.go`)
```go
// PseudoInverse calculates Moore-Penrose pseudo-inverse
// J+ = J^T * (J * J^T)^(-1)  // If rows >= columns
// J+ = (J^T * J)^(-1) * J^T  // If rows < columns
func (m Matrix) PseudoInverse(dst Matrix) error

// DampedLeastSquares calculates damped least squares
// J+ = J^T * (J * J^T + λ^2 * I)^(-1)
func (m Matrix) DampedLeastSquares(lambda float32, dst Matrix) error
```

**Algorithm**:
- For overdetermined (rows > cols): `J+ = (J^T * J)^(-1) * J^T`
- For underdetermined (rows < cols): `J+ = J^T * (J * J^T)^(-1)`
- DLS: Add damping term to handle singularities

#### 3. Jacobian Calculation (`pkg/core/math/mat/jacobian.go`)
```go
// JacobianColumn represents a single column of Jacobian matrix
type JacobianColumn struct {
    Linear  [3]float32  // Linear velocity component
    Angular [3]float32  // Angular velocity component
}

// CalculateJacobianColumn calculates one column of geometric Jacobian
// For revolute joint: Linear = Z_i × (p_ee - p_i), Angular = Z_i
// For prismatic joint: Linear = Z_i, Angular = [0,0,0]
func CalculateJacobianColumn(
    jointPos [3]float32,      // Joint position
    jointAxis [3]float32,     // Joint axis (Z for revolute, translation for prismatic)
    eePos [3]float32,         // End-effector position
    isRevolute bool,          // Joint type
) JacobianColumn
```

**Rationale**: Separate general Jacobian math from kinematics-specific code

### Phase 3: Fix DH Forward Kinematics Issues

**Files to Modify**:
- `pkg/robot/kinematics/dh/denavithartenberg.go`

**Issues to Fix**:

1. **H0i Not Initialized**:
```go
func New(eps float32, maxIterations int, cfg ...Config) kinematics.Kinematics {
    return &DenavitHartenberg{
        eps:           eps,
        maxIterations: maxIterations,
        c:             cfg,
        params:        make([]float32, len(cfg)),
        H0i:           make([]mat.Matrix4x4, len(cfg)+1),  // ADD: Initialize H0i
    }
}
```

2. **Verify Quaternion Extraction**:
   - Already exists: `Matrix4x4.Quaternion()` ✅
   - Verify it returns correct quaternion format

### Phase 4: Implement DH Inverse Kinematics

**Files to Create/Modify**:
- `pkg/robot/kinematics/dh/denavithartenberg.go`
- `pkg/robot/kinematics/dh/jacobian.go` (NEW - kinematics-specific)
- `pkg/robot/kinematics/dh/inverse.go` (NEW - optional, for analytical IK)

**Implementation Strategy**:

#### 4.1 Helper Functions (`denavithartenberg.go`)
```go
// applyJointLimits applies joint limits to parameters
func (p *DenavitHartenberg) applyJointLimits()

// positionError calculates position error
func (p *DenavitHartenberg) positionError(target [3]float32) float32

// orientationError calculates orientation error (angle between quaternions)
func (p *DenavitHartenberg) orientationError(target [4]float32) float32

// setTarget sets target pose from pos field
func (p *DenavitHartenberg) setTarget(targetPos *[3]float32, targetQuat *[4]float32)
```

#### 4.2 Jacobian Calculation (`jacobian.go`)
```go
// calculateJacobian calculates geometric Jacobian for DH manipulator
func (p *DenavitHartenberg) calculateJacobian() (mat.Matrix, error) {
    dof := len(p.c)
    J := mat.New(6, dof)  // 6 rows: 3 linear + 3 angular
    
    eePos := [3]float32{p.pos[0], p.pos[1], p.pos[2]}
    
    for i := 0; i < dof; i++ {
        // Get joint position from H0i[i]
        jointPos := [3]float32{}
        p.H0i[i].Col(3, jointPos[:])
        
        // Get Z-axis (rotation/translation axis)
        zAxis := [3]float32{}
        p.H0i[i].Col(2, zAxis[:])
        
        // Calculate Jacobian column using math library
        col := mat.CalculateJacobianColumn(
            jointPos,
            zAxis,
            eePos,
            p.c[i].Index == 0,  // Revolute if Index == 0
        )
        
        // Set Jacobian columns
        J.SetCol(i, 0, col.Linear[:])
        J.SetCol(i, 3, col.Angular[:])
    }
    
    return J, nil
}
```

**Uses**: `mat.CalculateJacobianColumn()` from Phase 2

#### 4.3 Numerical IK Method (`denavithartenberg.go`)
```go
// inverseNumerical implements iterative Jacobian-based IK
func (p *DenavitHartenberg) inverseNumerical(
    targetPos [3]float32,
    targetQuat [4]float32,
    method string,  // "pseudo_inverse" or "dls"
) bool {
    const defaultDamping float32 = 0.1  // For DLS
    
    for iteration := 0; iteration < p.maxIterations; iteration++ {
        // Forward kinematics
        if !p.Forward() {
            return false
        }
        
        // Check convergence
        posErr := p.positionError(targetPos)
        oriErr := p.orientationError(targetQuat)
        
        if posErr < p.eps && oriErr < p.eps {
            return true  // Converged
        }
        
        // Calculate Jacobian
        J, err := p.calculateJacobian()
        if err != nil {
            return false
        }
        
        // Calculate error vector (6D)
        errorVec := mat.New(6, 1)
        errorVec[0][0] = targetPos[0] - p.pos[0]
        errorVec[1][0] = targetPos[1] - p.pos[1]
        errorVec[2][0] = targetPos[2] - p.pos[2]
        
        // Orientation error (convert quaternion error to axis-angle)
        // TODO: Implement quaternion to axis-angle conversion
        
        // Calculate pseudo-inverse or DLS
        var Jplus mat.Matrix
        switch method {
        case "dls":
            Jplus, err = J.DampedLeastSquares(defaultDamping)
        default:
            Jplus, err = J.PseudoInverse()
        }
        if err != nil {
            return false  // Singularity or other error
        }
        
        // Calculate delta joint angles
        deltaParams := Jplus.MulVec(errorVec, nil)
        
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

#### 4.4 Main Inverse Method (`denavithartenberg.go`)
```go
func (p *DenavitHartenberg) Inverse() bool {
    // Extract target from pos field
    targetPos := [3]float32{p.pos[0], p.pos[1], p.pos[2]}
    targetQuat := [4]float32{p.pos[3], p.pos[4], p.pos[5], p.pos[6]}
    
    // For now, use numerical IK
    // TODO: Add analytical IK for 2, 3, 6 DOF when applicable
    return p.inverseNumerical(targetPos, targetQuat, "dls")
}
```

**Algorithm Selection** (Future Enhancement):
- 2-3 DOF: Try analytical first, fall back to numerical
- 6 DOF: Try analytical if spherical wrist, otherwise numerical
- Arbitrary DOF: Numerical only

### Phase 5: Testing and Validation

**Test Cases**:

1. **Planar Kinematics**:
   - Test FK → IK → FK round trip
   - Verify position matches within tolerance
   - Test with joint limits
   - Compare with C reference results

2. **DH Kinematics**:
   - Test FK with known configurations
   - Test IK convergence for reachable targets
   - Test IK with joint limits
   - Test IK with singularities
   - Test with different DOF (2, 3, 6, 7+)

3. **Math Functions**:
   - Test matrix inverse
   - Test pseudo-inverse
   - Test DLS
   - Test Jacobian column calculation

## File Structure

### New Files to Create

```
pkg/core/math/mat/
├── inverse.go          # Matrix inverse
├── pseudo_inverse.go   # Pseudo-inverse and DLS
└── jacobian.go         # Jacobian column calculation

pkg/robot/kinematics/dh/
├── jacobian.go         # DH-specific Jacobian calculation
└── helpers.go          # Helper functions (errors, limits)
```

### Files to Modify

```
pkg/robot/kinematics/planar/
├── planar2dof.go       # Add Forward() after IK
└── planar3dof.go       # Add Forward() after IK

pkg/robot/kinematics/dh/
├── denavithartenberg.go  # Fix H0i init, implement Inverse()
└── config.go            # (No changes)
```

## Implementation Order

### Week 1: Core Math Functions

1. **Day 1-2**: Matrix inverse (`mat/inverse.go`)
   - Implement for generic Matrix
   - Implement for Matrix4x4, Matrix3x3, Matrix2x2
   - Test with known matrices

2. **Day 3-4**: Pseudo-inverse and DLS (`mat/pseudo_inverse.go`)
   - Implement Moore-Penrose pseudo-inverse
   - Implement damped least squares
   - Test with various matrix sizes

3. **Day 5**: Jacobian column calculation (`mat/jacobian.go`)
   - Implement `CalculateJacobianColumn`
   - Test with simple cases

### Week 2: Planar Kinematics Fixes

4. **Day 1**: Fix Planar 2DOF and 3DOF
   - Add Forward() call after IK
   - Test round-trip (FK → IK → FK)
   - Compare with C reference

5. **Day 2**: Planar testing and validation
   - Comprehensive test suite
   - Edge cases (joint limits, workspace boundaries)

### Week 3: DH Kinematics Implementation

6. **Day 1**: Fix DH Forward Kinematics
   - Initialize H0i in constructor
   - Verify Forward() works correctly
   - Test with various configurations

7. **Day 2-3**: DH Jacobian Calculation
   - Implement `calculateJacobian()` using math library
   - Test with simple 2 DOF, 3 DOF manipulators
   - Verify Jacobian dimensions

8. **Day 4-5**: DH Inverse Kinematics
   - Implement helper functions
   - Implement `inverseNumerical()`
   - Implement main `Inverse()` method
   - Test convergence

### Week 4: Testing and Refinement

9. **Day 1-2**: Comprehensive testing
   - Unit tests for all components
   - Integration tests (full IK workflow)
   - Performance tests

10. **Day 3-4**: Bug fixes and optimization
    - Fix any issues found during testing
    - Optimize for performance if needed
    - Add error handling improvements

11. **Day 5**: Documentation and cleanup
    - Update documentation
    - Code review
    - Final validation

## Key Design Decisions

### 1. Jacobian Math Location

**Decision**: General Jacobian math in `pkg/core/math`, kinematics-specific in `pkg/robot/kinematics`

**Rationale**:
- `mat.CalculateJacobianColumn()` is general (works for any manipulator)
- `dh.calculateJacobian()` is DH-specific (uses H0i, DH parameters)
- Separation of concerns: math library is general, kinematics uses it

### 2. Algorithm Selection

**Decision**: Start with numerical IK only, add analytical later if needed

**Rationale**:
- Numerical IK works for all DOF
- Easier to implement and test
- Can add analytical IK as optimization later

### 3. Error Handling

**Decision**: Return boolean for Inverse(), add error logging internally

**Rationale**:
- Matches existing Kinematics interface
- Keeps API simple
- Can log detailed errors for debugging

### 4. Orientation Error Calculation

**Decision**: Use quaternion dot product to calculate angle

**Rationale**:
- Simple and efficient
- Works well with existing quaternion representation
- May need axis-angle conversion for error vector

## Success Criteria

1. ✅ Planar kinematics matches C reference behavior
2. ✅ FK → IK → FK round-trip works correctly
3. ✅ DH IK converges for reachable targets
4. ✅ DH IK respects joint limits
5. ✅ Math functions work correctly (inverse, pseudo-inverse, Jacobian)
6. ✅ Comprehensive test coverage
7. ✅ Performance acceptable for real-time use
8. ✅ Code follows repository conventions

## References

1. **C Reference Code**: `temporary_for_reference_only/locomotion/src/kinematics/`
2. **Robotics Textbook**: Craig, "Introduction to Robotics"
3. **Math References**:
   - Matrix inverse: LU decomposition, Gauss-Jordan
   - Pseudo-inverse: Moore-Penrose definition
   - DLS: Damped Least Squares method

## Notes

- C code is thoroughly tested, so matching its behavior is priority
- Math functions should be general and reusable
- Kinematics-specific code should use math library functions
- Test coverage should be comprehensive before considering complete

