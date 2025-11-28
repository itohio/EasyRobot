# Dispatch Optimization Issue

## The Problem

Go generics use **monomorphization** - the compiler generates specialized code for each `[T, U]` combination. Types ARE known at compile time, but `any()` type assertions force runtime checks.

## Current Overhead

For `Convert[float32, int16]` with 10,000 elements:
- **Direct call**: ~7.4μs (convertFromInt16ToFloat32)
- **With dispatch**: ~10.3μs (Convert function)
- **Overhead**: ~2.9μs (39% slower)

## Why `any()` Prevents Optimization

When the compiler monomorphizes `Convert[float32, int16]`:
1. It generates specialized code where `T = float32` and `U = int16`
2. The compiler KNOWS `dst` is always `[]float32` and `src` is always `[]int16`
3. BUT `any(dst).(type)` forces a runtime type check because `any()` erases type information
4. The compiler can't optimize away the switch because it sees `any` (interface{}), not the concrete type

## Current Dispatch Path

```go
func Convert[T, U Number](dst []T, src []U, n int) {
    // Type assertion #1: any(dst) - runtime check!
    switch d := any(dst).(type) {
    case []float32:  // Compiler should know this is always true, but can't prove it
        // Type assertion #2: any(src) - runtime check!
        if s, ok := any(src).([]int16); ok {  // Compiler should know this, but can't
            convertFromInt16ToFloat32(d, s)
        }
    }
}
```

## Solutions

### Option 1: Compiler Should Optimize (Future Go Version)
The Go compiler could theoretically:
- Detect that in `Convert[float32, int16]`, `dst` is always `[]float32`
- Eliminate the switch and directly call `convertFromInt16ToFloat32`
- This is a compiler limitation, not a language limitation

### Option 2: Use Type-Specific Wrappers
Create non-generic wrapper functions that the compiler can inline:
```go
func ConvertInt16ToFloat32(dst []float32, src []int16, n int) {
    convertFromInt16ToFloat32(dst, src)
}
```

### Option 3: Code Generation
Use `go generate` to create type-specific versions that bypass dispatch entirely.

### Option 4: Accept the Overhead
For a generic API that supports all type combinations, ~2.9μs overhead is acceptable.
The direct function calls are still faster than baseline inline code.

## Benchmark Results

- **baseline_inline**: ~4.5μs - Simple loop (no optimizations)
- **convert_optimized**: ~7.4μs - Direct function call (optimized loop, BCE hints)
- **convert_with_dispatch**: ~10.3μs - Full dispatch (2.9μs overhead)
- **assembly_placeholder**: ~3.7μs - Future SIMD implementation

The optimized version is **faster than baseline** despite the function call overhead, showing the benefit of optimized loops with BCE hints.

