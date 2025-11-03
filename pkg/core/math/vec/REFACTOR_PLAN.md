# Vector Package Refactoring Plan

## Goal

Refactor `vec` package methods to use existing BLAS Level 1 primitives from `primitive` package with minimal overhead, while maintaining the current API and performance characteristics.

## Current State Analysis

### Vector Type (Variable Length)

Currently uses deprecated/non-BLAS primitive functions:

| Current Method | Current Primitive | Target BLAS Level 1 | Overhead Issue |
|----------------|-------------------|---------------------|----------------|
| `Sum()` | `primitive.Sum()` | ✅ Already optimal | None |
| `SumSqr()` | `primitive.SqrSum()` | ✅ Already optimal | None |
| `Add()` | `primitive.SumArrAdd()` | `primitive.Axpy()` | Extra function call + potential scalar add |
| `Sub()` | `primitive.DiffArr()` | `primitive.Axpy()` with alpha=-1 | Extra copy operation (dst != v) |
| `Neg()` | `primitive.MulArrInPlace()` | `primitive.Scal()` with alpha=-1 | Uses deprecated function |
| `MulC()` | `primitive.MulArrInPlace()` | `primitive.Scal()` | Uses deprecated function |
| `DivC()` | `primitive.DivArrInPlace()` | `primitive.Scal()` with alpha=1/c | Uses deprecated function |
| `AddC()` | `primitive.SumArrInPlace()` | Custom loop (no BLAS equivalent) | Minimal - BLAS doesn't have scalar add |
| `SubC()` | `primitive.DiffArrInPlace()` | Custom loop (no BLAS equivalent) | Minimal - BLAS doesn't have scalar sub |
| `MulCAdd()` | `primitive.MulArrAdd()` | `primitive.Axpy()` | Uses deprecated function |
| `MulCSub()` | `primitive.MulArrAdd()` with -c | `primitive.Axpy()` with alpha=-c | Uses deprecated function |
| `Dot()` | `primitive.DotProduct()` | `primitive.Dot()` | Deprecated wrapper |
| `Multiply()` | `primitive.HadamardProduct()` | ✅ Already optimal | None |
| `Normal()` | `primitive.Nrm2()` + `primitive.Scal()` | ✅ Already optimal | None |

### Fixed-Size Vectors (Vector2D, Vector3D, Vector4D, Quaternion)

Currently use **inline loops** instead of primitives:
- **Keep as-is**: Small fixed sizes (2-4 elements) benefit from inline loops
- Function call overhead > loop overhead for small sizes
- Compiler can optimize inline loops better (unrolling, SIMD)

## Refactoring Strategy

### Phase 1: Direct BLAS Level 1 Replacements

Replace deprecated primitive functions with direct BLAS Level 1 calls for `Vector` type.

#### 1.1 Replace `MulArrInPlace()` → `Scal()`

**Current:**
```go
func (v Vector) Neg() Vector {
    primitive.MulArrInPlace(v, -1, len(v))
    return v
}

func (v Vector) MulC(c float32) Vector {
    primitive.MulArrInPlace(v, c, len(v))
    return v
}
```

**Target:**
```go
func (v Vector) Neg() Vector {
    if len(v) == 0 {
        return v
    }
    primitive.Scal(v, 1, len(v), -1.0)
    return v
}

func (v Vector) MulC(c float32) Vector {
    if len(v) == 0 {
        return v
    }
    primitive.Scal(v, 1, len(v), c)
    return v
}
```

**Overhead**: None - direct replacement, same performance.

#### 1.2 Replace `DivArrInPlace()` → `Scal()`

**Current:**
```go
func (v Vector) DivC(c float32) Vector {
    if c == 0 {
        return v
    }
    primitive.DivArrInPlace(v, c, len(v))
    return v
}
```

**Target:**
```go
func (v Vector) DivC(c float32) Vector {
    if len(v) == 0 || c == 0 {
        return v
    }
    primitive.Scal(v, 1, len(v), 1.0/c)
    return v
}
```

**Overhead**: None - direct replacement, same performance.

#### 1.3 Replace `MulArrAdd()` → `Axpy()`

**Current:**
```go
func (v Vector) MulCAdd(c float32, v1 Vector) Vector {
    primitive.MulArrAdd(v, v1, c, len(v), 1)
    return v
}

func (v Vector) MulCSub(c float32, v1 Vector) Vector {
    primitive.MulArrAdd(v, v1, -c, len(v), 1)
    return v
}
```

**Target:**
```go
func (v Vector) MulCAdd(c float32, v1 Vector) Vector {
    if len(v) == 0 {
        return v
    }
    primitive.Axpy(v, v1, 1, 1, len(v), c)
    return v
}

func (v Vector) MulCSub(c float32, v1 Vector) Vector {
    if len(v) == 0 {
        return v
    }
    primitive.Axpy(v, v1, 1, 1, len(v), -c)
    return v
}
```

**Overhead**: None - `MulArrAdd()` already calls `Axpy()` internally.

#### 1.4 Replace `DotProduct()` → `Dot()`

**Current:**
```go
func (v Vector) Dot(v1 Vector) float32 {
    return primitive.DotProduct(v, v1, len(v), 1, 1)
}
```

**Target:**
```go
func (v Vector) Dot(v1 Vector) float32 {
    if len(v) == 0 || len(v1) == 0 || len(v) != len(v1) {
        return 0
    }
    return primitive.Dot(v, v1, 1, 1, len(v))
}
```

**Overhead**: None - `DotProduct()` already calls `Dot()` internally.

### Phase 2: Optimize In-Place Operations

Optimize operations that currently do extra work for in-place operations.

#### 2.1 Optimize `Add()` to use `Axpy()` directly

**Current:**
```go
func (v Vector) Add(v1 Vector) Vector {
    primitive.SumArrAdd(v, v1, 0, len(v), 1)
    return v
}
```

**Current `SumArrAdd()` implementation:**
```go
func SumArrAdd(dst, src []float32, c float32, num int, stride int) {
    Axpy(dst, src, stride, stride, num, 1.0)
    if c != 0 {
        SumArrInPlace(dst, c, num)
    }
}
```

**Target:**
```go
func (v Vector) Add(v1 Vector) Vector {
    if len(v) == 0 {
        return v
    }
    primitive.Axpy(v, v1, 1, 1, len(v), 1.0)
    return v
}
```

**Overhead Eliminated**: 
- Removed extra function call (`SumArrAdd()` → direct `Axpy()`)
- Removed unnecessary check for `c != 0`
- **Performance Gain**: ~1-2 function call overhead eliminated

#### 2.2 Optimize `Sub()` to use `Axpy()` directly

**Current:**
```go
func (v Vector) Sub(v1 Vector) Vector {
    primitive.DiffArr(v, v, v1, len(v), 1, 1)
    return v
}
```

**Issue**: `DiffArr()` computes `dst = a - b`, but we need `v = v - v1` (in-place).

**Target:**
```go
func (v Vector) Sub(v1 Vector) Vector {
    if len(v) == 0 {
        return v
    }
    // v = v - v1  =>  v = v + (-1.0) * v1
    primitive.Axpy(v, v1, 1, 1, len(v), -1.0)
    return v
}
```

**Overhead Eliminated**: 
- Direct in-place operation (no temporary copies)
- Uses optimized `Axpy()` instead of generic `DiffArr()`
- **Performance Gain**: Better cache locality, single pass instead of read-write

### Phase 3: Keep Fixed-Size Vectors Inline

**Decision**: Keep `Vector2D`, `Vector3D`, `Vector4D`, `Quaternion` using inline loops.

**Rationale**:
1. **Function Call Overhead**: For 2-4 elements, function call overhead (~10-20ns) > loop overhead (~5-10ns)
2. **Compiler Optimizations**: Inline loops can be unrolled and vectorized by compiler
3. **Cache Locality**: Inline operations stay in CPU registers
4. **Code Size**: Small inline loops don't bloat code significantly

**Exception**: Consider using primitives for `Dot()` and `Magnitude()` even for small sizes, as they require fewer operations and benefit from optimized implementations.

**Example - Keep Inline for Add/Sub/MulC:**
```go
// Keep as-is - inline is faster
func (v *Vector2D) Add(v1 Vector2D) *Vector2D {
    for i := range v {
        v[i] += v1[i]
    }
    return v
}
```

**Example - Consider Primitives for Dot/Magnitude:**
```go
// Could use primitive.Dot() but inline might still be faster
func (v *Vector2D) Dot(v1 Vector2D) float32 {
    return v[0]*v1[0] + v[1]*v1[1]  // Inline, compiler optimizes
}
```

## Implementation Plan

### Step 1: Update Vector Type Methods

1. Replace `MulArrInPlace()` → `Scal()` in `Neg()` and `MulC()`
2. Replace `DivArrInPlace()` → `Scal()` in `DivC()`
3. Replace `MulArrAdd()` → `Axpy()` in `MulCAdd()` and `MulCSub()`
4. Replace `DotProduct()` → `Dot()` in `Dot()`
5. Replace `SumArrAdd()` → `Axpy()` in `Add()`
6. Replace `DiffArr()` → `Axpy()` in `Sub()`

### Step 2: Verify Performance

1. Run benchmarks before and after changes
2. Ensure no performance regression for common operations
3. Verify correctness with existing tests

### Step 3: Clean Up Deprecated Functions

1. Remove or mark as deprecated:
   - `primitive.MulArrInPlace()` (already deprecated)
   - `primitive.DivArrInPlace()` (already deprecated)
   - `primitive.MulArrAdd()` (already deprecated)
   - `primitive.DotProduct()` (already deprecated)
   - `primitive.SumArrAdd()` (already deprecated)

### Step 4: Documentation

1. Update SPEC.md with BLAS Level 1 mappings
2. Document performance characteristics
3. Add comments indicating BLAS operations used

## Performance Impact

### Expected Improvements

| Operation | Current Overhead | After Refactor | Improvement |
|-----------|-----------------|----------------|-------------|
| `Add()` | Extra function call + scalar check | Direct `Axpy()` | ~1-2ns (eliminated call) |
| `Sub()` | Temporary copy via `DiffArr()` | Direct `Axpy()` | ~5-10ns (eliminated copy pass) |
| `Neg()` | Deprecated function | Direct `Scal()` | ~0ns (same performance) |
| `MulC()` | Deprecated function | Direct `Scal()` | ~0ns (same performance) |
| `DivC()` | Deprecated function | Direct `Scal()` | ~0ns (same performance) |

### Minimal Overhead Considerations

1. **Function Call Overhead**: ~10-20ns per call (negligible for vectors > 10 elements)
2. **Stride Parameters**: Always `stride = 1` for contiguous vectors (no overhead)
3. **Length Checks**: Minimal overhead (branch prediction)

## Code Examples

### Before Refactoring

```go
func (v Vector) Add(v1 Vector) Vector {
    if len(v) == 0 {
        return v
    }
    // Extra function call: SumArrAdd() -> Axpy() + scalar check
    primitive.SumArrAdd(v, v1, 0, len(v), 1)
    return v
}

func (v Vector) Sub(v1 Vector) Vector {
    if len(v) == 0 {
        return v
    }
    // Extra copy: DiffArr() creates dst, but we need in-place
    primitive.DiffArr(v, v, v1, len(v), 1, 1)
    return v
}
```

### After Refactoring

```go
func (v Vector) Add(v1 Vector) Vector {
    if len(v) == 0 {
        return v
    }
    // Direct BLAS Level 1: y = alpha*x + y  with alpha=1.0
    primitive.Axpy(v, v1, 1, 1, len(v), 1.0)
    return v
}

func (v Vector) Sub(v1 Vector) Vector {
    if len(v) == 0 {
        return v
    }
    // Direct BLAS Level 1: y = alpha*x + y  with alpha=-1.0
    primitive.Axpy(v, v1, 1, 1, len(v), -1.0)
    return v
}
```

## Testing Strategy

1. **Unit Tests**: Verify correctness of refactored methods
2. **Benchmark Tests**: Compare performance before/after
3. **Regression Tests**: Ensure existing code still works
4. **Edge Cases**: Test empty vectors, zero division, etc.

## Migration Checklist

- [ ] Update `Vector.Neg()` to use `primitive.Scal()`
- [ ] Update `Vector.MulC()` to use `primitive.Scal()`
- [ ] Update `Vector.DivC()` to use `primitive.Scal()`
- [ ] Update `Vector.MulCAdd()` to use `primitive.Axpy()`
- [ ] Update `Vector.MulCSub()` to use `primitive.Axpy()`
- [ ] Update `Vector.Dot()` to use `primitive.Dot()`
- [ ] Update `Vector.Add()` to use `primitive.Axpy()`
- [ ] Update `Vector.Sub()` to use `primitive.Axpy()`
- [ ] Run all tests
- [ ] Run benchmarks
- [ ] Update documentation
- [ ] Verify no performance regression

## Summary

The refactoring will:
1. ✅ **Eliminate deprecated function calls** - Use direct BLAS Level 1 operations
2. ✅ **Reduce function call overhead** - Direct calls instead of wrapper functions
3. ✅ **Optimize in-place operations** - Use `Axpy()` directly for Add/Sub
4. ✅ **Maintain API compatibility** - No breaking changes
5. ✅ **Keep fixed-size vectors inline** - Preserve performance for small vectors
6. ✅ **Zero performance regression** - Maintain or improve performance

**Estimated Performance Gain**: 5-15% for `Vector` type operations, especially `Add()` and `Sub()`.

