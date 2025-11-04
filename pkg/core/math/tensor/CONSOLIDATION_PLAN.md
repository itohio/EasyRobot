# Tensor and FP32 Consolidation Plan

## Overview

This document outlines the consolidation of operations within the `fp32` package to eliminate duplicates and ensure the `tensor` package consistently uses `fp32` BLAS primitives. The goal is to create a clean, efficient API similar to TensorFlow Lite with proper dependency hierarchy: tensor → fp32 (BLAS primitives).

**Scope**: `pkg/core/math/tensor` and `pkg/core/math/primitive/fp32` packages only.
**Dependency Rule**: Tensor package uses fp32 primitives; fp32 does not depend on tensor.

**Goals**:
- Remove 11 deprecated functions from fp32 package
- Eliminate duplicate implementations between array.go and tensor_elementwise.go
- Maintain proper dependency hierarchy: tensor → fp32 (not circular)
- Ensure no performance regression when deprecated functions are removed

## Current State Analysis

### Duplicate Operations Identified

| Category | fp32 Functions | Recommended Migration | Status |
|----------|----------------|------------------|--------|
| **Element-wise Add** | `SumArr` (array.go) vs `ElemAdd` (tensor_elementwise.go) | Use `ElemAdd` (tensor_elementwise.go) | 🚫 DEPRECATED |
| **Element-wise Sub** | `DiffArr` (array.go) vs `ElemSub` (tensor_elementwise.go) | Use `ElemSub` (tensor_elementwise.go) | 🚫 DEPRECATED |
| **Element-wise Mul** | `MulArr` (array.go), `HadamardProduct` (vector.go) vs `ElemMul` (tensor_elementwise.go) | Use `ElemMul` (tensor_elementwise.go) | 🚫 DEPRECATED |
| **Element-wise Div** | `DivArr` (array.go) vs `ElemDiv` (tensor_elementwise.go) | Use `ElemDiv` (tensor_elementwise.go) | 🚫 DEPRECATED |
| **Dot Product** | `Dot` (level1.go) vs `DotProduct` (vector.go) | Use `Dot` (BLAS Level 1) | 🚫 DEPRECATED |
| **Scaling** | `Scal` (level1.go) vs `MulArrInPlace` (vector.go) | Use `Scal` (BLAS Level 1) | 🚫 DEPRECATED |
| **Convolution** | `Conv2D` (tensor.go) vs `Convolve2DAdd` (conv.go) | Use `Conv2D` (tensor.go) | 🚫 DEPRECATED |
| **Vector Add (AXPY)** | `Axpy` (level1.go) vs `SumArrAdd`/`MulArrAdd` (array.go) | Use `Axpy` (BLAS Level 1) | 🚫 DEPRECATED |

### BLAS Usage Analysis

**Well-integrated with BLAS (Tensor Package):**
- `tensor.Add` → `fp32.Axpy` ✓
- `tensor.Sub` → `fp32.Axpy` ✓
- `tensor.Scale` → `fp32.Scal` ✓
- `tensor.MatMul` → `fp32.Gemm_*` variants ✓
- `tensor.Conv2D` → `fp32.Conv2D` ✓
- `tensor.Dot` → `fp32.Dot` ✓
- `tensor.Sum` (vector case) → `fp32.Asum` ✓
- `tensor.ArgMax` → `fp32.Iamax` ✓
- `tensor.Norm` → `fp32.Nrm2`/`fp32.Asum` ✓

**Status**: ✅ **Tensor package already uses 100% recommended fp32 functions - no changes needed**

**Issues in fp32 Package (To Be Fixed):**
- 11 deprecated functions with duplicate implementations
- Element-wise operations duplicated between `array.go` and `tensor_elementwise.go`
- Non-BLAS operations that should be removed or consolidated

## Consolidation Plan

### Consolidation Status: ✅ COMPLETE

**✅ COMPLETED**: Tensor package already uses 100% recommended fp32 functions
- All tensor operations correctly delegate to BLAS primitives
- Proper dependency hierarchy: tensor → fp32 (not circular)

**✅ COMPLETED**: Removed 11 deprecated functions from fp32 package + ElemMask wrapper
- Migrated all usage to recommended alternatives (including ElemMask → ElemMul)
- Updated documentation and removed test functions
- All tests pass with no regressions

### Phase 1: Final Validation and Migration (1-2 weeks)

**Objective**: Ensure all deprecated functions are properly deprecated and migrate any remaining internal usage.

**Actions**:
1. Search codebase for any remaining usage of deprecated functions
2. Update any internal code to use recommended alternatives
3. Ensure all deprecated functions have proper deprecation warnings
4. Verify comprehensive test coverage for recommended functions

**Deprecated Functions to Remove** (11 total):
- **Element-wise (array.go)**: `SumArr`, `DiffArr`, `MulArr`, `DivArr` → `ElemAdd/Sub/Mul/Div`
- **Vector (vector.go)**: `HadamardProduct`, `DotProduct`, `MulArrInPlace` → `ElemMul`, `Dot`, `Scal`
- **Utility (array.go)**: `SumArrAdd`, `MulArrAdd`, `DivArrInPlace` → `Axpy`, `Scal`
- **Convolution (conv.go)**: `Convolve2DAdd` → `Conv2D`

### Phase 2: Function Removal and Cleanup (3-4 weeks)

**Objective**: Remove deprecated functions and update documentation.

**Actions**:
1. Remove all 11 deprecated functions from fp32 package
2. Update fp32/OPS.md to remove deprecated function documentation
3. Run full test suite to ensure no regressions
4. Update any documentation references

**Files to Modify**:
- `fp32/array.go`: Remove 7 deprecated functions
- `fp32/vector.go`: Remove 3 deprecated functions
- `fp32/conv.go`: Remove 1 deprecated function
- `fp32/OPS.md`: Update operation reference

### Phase 3: Final Validation (5-6 weeks)

**Objective**: Ensure consolidation is complete and no regressions introduced.

**Validation Steps**:
1. Run performance benchmarks to ensure no regression
2. Verify all tensor operations still work correctly
3. Check that fp32 package only contains recommended functions
4. Validate dependency hierarchy is maintained

**Success Criteria**:
- All 11 deprecated functions removed
- No performance regression (>95% of current performance)
- Clean codebase with no duplicate implementations
- Proper dependency hierarchy maintained

## Implementation Timeline

| Phase | Duration | Key Activities | Success Criteria |
|-------|----------|----------------|------------------|
| **Phase 1: Validation** | 1-2 weeks | Check for usage, ensure deprecation warnings | All deprecated functions properly marked |
| **Phase 2: Removal** | 3-4 weeks | Remove deprecated functions, update docs | 11 functions removed, tests pass |
| **Phase 3: Validation** | 5-6 weeks | Performance testing, final checks | No regression, clean codebase |

## Risk Mitigation

1. **Comprehensive Testing**: Full test suite pass required before and after removal
2. **Performance Benchmarks**: Automated benchmarks to detect regressions
3. **Gradual Migration**: Deprecation warnings allow time for external code updates
4. **Backup Strategy**: Keep git history for rollback if needed
5. **Documentation Updates**: Clear migration guides maintained

## Success Metrics ✅ ACHIEVED

- **✅ COMPLETED**: Tensor package uses 100% recommended fp32 functions
- **✅ COMPLETED**: Removed 11 deprecated functions from fp32 package + 1 wrapper function
- **✅ MAINTAINED**: Proper dependency hierarchy: tensor → fp32 (not circular)
- **✅ VERIFIED**: No performance regression on existing workloads
- **✅ ACHIEVED**: Clean codebase with no duplicate implementations

## Summary

The tensor and fp32 consolidation is now complete. The codebase has been successfully streamlined with:

- **11 deprecated functions removed** from fp32 package + 1 wrapper function eliminated
- **All usage migrated** to recommended BLAS and tensor_elementwise functions
- **Documentation updated** to reflect the clean API
- **Tests pass** with no regressions
- **Proper architecture maintained** with tensor package using fp32 primitives

## Final File Organization ✅ CLEAN

```
pkg/core/math/
├── primitive/fp32/           # BLAS primitives (foundation layer) ✅ CLEAN
│   ├── level1.go            # BLAS Level 1 (vectors) - AXPY, DOT, SCAL, etc.
│   ├── level2.go            # BLAS Level 2 (matrix-vector) - GEMV, etc.
│   ├── level3.go            # BLAS Level 3 (matrix-matrix) - GEMM, etc.
│   ├── batched.go           # Batched BLAS operations
│   ├── tensor_elementwise.go # Element-wise ops with strides ✅ RECOMMENDED
│   ├── tensor.go            # Tensor-specific ops (Conv2D, etc.)
│   ├── activations.go       # Activation functions
│   ├── la.go                # LAPACK operations
│   ├── array.go             # ✅ CLEANED - Utility functions only
│   ├── vector.go            # ✅ CLEANED - Utility functions only
│   ├── conv.go              # ✅ CLEANED - 1D convolution only
│   └── OPS.md               # ✅ UPDATED - Clean operation reference
└── tensor/                   # High-level tensor API ✅ CORRECTLY USES FP32
    ├── dense.go             # Core tensor structure
    ├── tensor_math.go       # Element-wise and reductions (calls fp32)
    ├── tensor_linalg.go     # Linear algebra operations (calls fp32 BLAS)
    ├── tensor_conv.go       # Convolution operations (calls fp32)
    ├── tensor_linalg_helpers.go  # Helper functions
    ├── SPEC.md              # API specification
    └── CONSOLIDATION_PLAN.md # This file
```

**Dependency Flow**: tensor package → fp32 primitives (BLAS) ✅ MAINTAINED
**Status**: ✅ CONSOLIDATION COMPLETE - Clean, efficient codebase with proper architecture

