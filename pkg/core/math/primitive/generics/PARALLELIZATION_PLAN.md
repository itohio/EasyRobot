# Parallelization Plan for Generic Operations

**Created:** November 6, 2025  
**Package:** `github.com/itohio/EasyRobot/pkg/core/math/primitive/generics`

## Overview

This document outlines the plan for parallelizing generic operations to improve performance on multi-core systems. The parallelization strategy uses build tags to provide both single-threaded and multi-threaded implementations, allowing users to choose based on their needs.

## Core Concepts

### Build Tags
- **`use_mt`**: When provided, enables multi-threaded implementations (`*_mt.go` files)
- **Default (no tag)**: Single-threaded implementations (`*_st.go` files)

### File Organization
- **`*_st.go`**: Single-threaded implementations (default)
- **`*_mt.go`**: Multi-threaded implementations (requires `use_mt` build tag)
- **`multithreaded.go`**: Goroutine pool implementation (only built with `use_mt` tag)

### Goroutine Pool
- Initialized in `init()` when `use_mt` tag is present
- Number of goroutines = `runtime.NumCPU()`
- Uses work-stealing pattern for load balancing
- Reuses goroutines to minimize allocation overhead

## Target Operations for Parallelization

Based on benchmark analysis, the following operations are prioritized for parallelization:

### High Priority (Easy Wins)

1. **Elements Iterator** (`iterators.go`)
   - **Current Performance**: 94,316 ns/op (+1448.9% vs baseline)
   - **Rationale**: Highest overhead, iterates over multi-dimensional indices
   - **Parallelization Strategy**: Split iteration space across goroutines
   - **Files**: `iterators_st.go`, `iterators_mt.go`

2. **ElemConvertStrided** (`convert.go`)
   - **Current Performance**: 118,101 ns/op (+607.7% vs baseline)
   - **Rationale**: Very slow, processes elements independently
   - **Parallelization Strategy**: Divide shape into chunks, process in parallel
   - **Files**: `convert_st.go`, `convert_mt.go`

3. **ElemApplyBinaryStrided** (`apply_tensor.go`)
   - **Current Performance**: 32,459 ns/op (+59.4% vs baseline)
   - **Rationale**: Moderate overhead, independent element operations
   - **Parallelization Strategy**: Chunk-based parallel processing
   - **Files**: `apply_tensor_st.go`, `apply_tensor_mt.go`

4. **ElemApplyUnaryScalarStrided** (`apply_tensor.go`)
   - **Current Performance**: 40,485 ns/op (+62.9% vs baseline)
   - **Rationale**: Moderate overhead, independent element operations
   - **Parallelization Strategy**: Chunk-based parallel processing
   - **Files**: `apply_tensor_st.go`, `apply_tensor_mt.go`

5. **ElemMatApplyStrided** (`apply_matrix.go`)
   - **Current Performance**: 34,117 ns/op (+135.8% vs baseline)
   - **Rationale**: High overhead, row-based parallelization natural
   - **Parallelization Strategy**: Process rows in parallel
   - **Files**: `apply_matrix_st.go`, `apply_matrix_mt.go`

6. **ElemVecApplyStrided** (`apply_vector.go`)
   - **Current Performance**: 25,536 ns/op (+59.5% vs baseline)
   - **Rationale**: Moderate overhead, independent element operations
   - **Parallelization Strategy**: Chunk-based parallel processing
   - **Files**: `apply_vector_st.go`, `apply_vector_mt.go`

### Medium Priority

7. **Comparison Operations** (`comparison.go`)
   - **Current Performance**: 14,884-23,024 ns/op
   - **Rationale**: Already fast, but can benefit for large arrays
   - **Parallelization Strategy**: Chunk-based parallel processing
   - **Files**: `comparison_st.go`, `comparison_mt.go`
   - **Operations**: `ElemGreaterThan`, `ElemEqual`, `ElemLess`, `ElemNotEqual`, `ElemLessEqual`, `ElemGreaterEqual` (and Strided variants)

8. **ElementsVecStrided** (`iterators.go`)
   - **Current Performance**: 8,390 ns/op (+105.8% vs baseline)
   - **Rationale**: Moderate overhead, simple iteration
   - **Parallelization Strategy**: Split range across goroutines
   - **Files**: `iterators_st.go`, `iterators_mt.go`

9. **ElemApplyUnary** (`apply_tensor.go`)
   - **Current Performance**: 10,690 ns/op (-17.0% vs baseline)
   - **Rationale**: Already fast, but can benefit for large arrays
   - **Parallelization Strategy**: Chunk-based parallel processing
   - **Files**: `apply_tensor_st.go`, `apply_tensor_mt.go`

10. **ElemApplyTernary** variants (`apply_tensor.go`, `apply_vector.go`, `apply_matrix.go`)
    - **Rationale**: Independent element operations, can benefit from parallelization
    - **Parallelization Strategy**: Chunk-based parallel processing
    - **Files**: `apply_tensor_st.go`, `apply_tensor_mt.go`, `apply_vector_st.go`, `apply_vector_mt.go`, `apply_matrix_st.go`, `apply_matrix_mt.go`

### Low Priority (Already Optimized)

- **Copy Operations**: Already highly optimized (1,543-1,724 ns/op), parallelization overhead may not be worth it
- **Swap Operations**: Already fast (9,891-11,270 ns/op)
- **Unary Operations** (Sign, Negative): Already fast (9,208-18,361 ns/op)
- **ElementsVec/Mat**: Already very fast (3,954-5,217 ns/op), minimal benefit

## Implementation Strategy

### Phase 1: Infrastructure (Foundation)

1. **Create `multithreaded.go`** (with `//go:build use_mt` tag)
   - Goroutine pool implementation
   - Work queue with work-stealing
   - Initialization in `init()`
   - Helper functions for parallel chunk processing

2. **Create `multithreaded_stub.go`** (without build tag, default)
   - Empty stub file to ensure package compiles
   - Or use build constraints to exclude `multithreaded.go` when `use_mt` is not set

### Phase 2: High Priority Operations

3. **Refactor `iterators.go` → `iterators_st.go`**
   - Move all iterator functions to `iterators_st.go`
   - Create `iterators_mt.go` with parallel implementations
   - Focus on `Elements` and `ElementsStrided` first

4. **Refactor `convert.go` → `convert_st.go`**
   - Move conversion functions to `convert_st.go`
   - Create `convert_mt.go` with parallel `ElemConvertStrided`
   - Keep `ElemConvert` (contiguous) single-threaded initially (already fast)

5. **Refactor `apply_tensor.go` → `apply_tensor_st.go`**
   - Move all apply functions to `apply_tensor_st.go`
   - Create `apply_tensor_mt.go` with parallel strided variants
   - Focus on `ElemApplyBinaryStrided` and `ElemApplyUnaryScalarStrided`

6. **Refactor `apply_matrix.go` → `apply_matrix_st.go`**
   - Move matrix apply functions to `apply_matrix_st.go`
   - Create `apply_matrix_mt.go` with parallel row-based processing

7. **Refactor `apply_vector.go` → `apply_vector_st.go`**
   - Move vector apply functions to `apply_vector_st.go`
   - Create `apply_vector_mt.go` with parallel chunk-based processing

### Phase 3: Medium Priority Operations

8. **Refactor `comparison.go` → `comparison_st.go`**
   - Move comparison functions to `comparison_st.go`
   - Create `comparison_mt.go` with parallel implementations
   - Focus on strided variants for large arrays

9. **Extend iterators with parallel variants**
   - Add parallel `ElementsVecStrided` in `iterators_mt.go`

10. **Extend apply operations**
    - Add parallel `ElemApplyUnary` and `ElemApplyTernary` variants

## Parallelization Patterns

### Pattern 1: Chunk-Based Processing (Most Common)

For operations that process independent elements:

```go
// Split work into chunks
chunkSize := (n + numWorkers - 1) / numWorkers
for i := 0; i < numWorkers; i++ {
    start := i * chunkSize
    end := start + chunkSize
    if end > n {
        end = n
    }
    // Process chunk [start:end] in parallel
}
```

### Pattern 2: Row-Based Processing (Matrices)

For matrix operations, process rows in parallel:

```go
// Each goroutine processes a subset of rows
for i := 0; i < rows; i++ {
    // Process row i in parallel
}
```

### Pattern 3: Multi-Dimensional Chunking (Tensors)

For strided tensor operations, divide the iteration space:

```go
// Divide shape into chunks along first dimension
// Each goroutine processes a subset of the first dimension
// Then iterates over remaining dimensions
```

### Pattern 4: Iterator Parallelization

For iterators, split the iteration space:

```go
// Divide total size into chunks
// Each goroutine iterates over its chunk
// Use work-stealing for load balancing
```

## Thresholds for Parallelization

To avoid overhead for small arrays, only parallelize when:

- **Size threshold**: `n >= 1000` elements (configurable)
- **CPU threshold**: `runtime.NumCPU() > 1`
- **Strided operations**: Always consider parallelization (already have overhead)

## Implementation Details

### Goroutine Pool Structure

```go
type workerPool struct {
    workers    int
    taskQueue  chan task
    wg         sync.WaitGroup
}

type task func()

func (p *workerPool) submit(t task) {
    p.taskQueue <- t
}

func (p *workerPool) shutdown() {
    close(p.taskQueue)
    p.wg.Wait()
}
```

### Chunk Processing Helper

```go
func parallelChunks(n int, fn func(start, end int)) {
    numWorkers := runtime.NumCPU()
    if n < minParallelSize || numWorkers == 1 {
        fn(0, n)
        return
    }
    
    chunkSize := (n + numWorkers - 1) / numWorkers
    var wg sync.WaitGroup
    for i := 0; i < numWorkers; i++ {
        start := i * chunkSize
        end := start + chunkSize
        if end > n {
            end = n
        }
        if start >= end {
            break
        }
        wg.Add(1)
        go func(s, e int) {
            defer wg.Done()
            fn(s, e)
        }(start, end)
    }
    wg.Wait()
}
```

## Testing Strategy

1. **Unit Tests**: Ensure parallel implementations produce same results as single-threaded
2. **Benchmark Tests**: Compare performance with/without `use_mt` tag
3. **Race Detection**: Run tests with `-race` flag
4. **Size Threshold Tests**: Verify parallelization only activates for large arrays

## Migration Path

1. **Backward Compatibility**: Default behavior remains single-threaded
2. **Opt-in**: Users enable parallelization with `-tags use_mt`
3. **Gradual Rollout**: Implement high-priority operations first
4. **Performance Monitoring**: Track benchmark improvements

## Expected Performance Improvements

Based on benchmark analysis:

- **Elements**: 50-70% improvement expected (from 94,316 ns/op)
- **ElemConvertStrided**: 60-80% improvement expected (from 118,101 ns/op)
- **ElemApplyBinaryStrided**: 40-60% improvement expected (from 32,459 ns/op)
- **ElemMatApplyStrided**: 50-70% improvement expected (from 34,117 ns/op)
- **Comparison Operations**: 30-50% improvement for large arrays

## Risks and Considerations

1. **Overhead**: Parallelization has overhead (goroutine creation, synchronization)
   - **Mitigation**: Only parallelize for arrays above threshold size
   - **Mitigation**: Use goroutine pool to reuse workers

2. **Cache Coherency**: False sharing can degrade performance
   - **Mitigation**: Align chunk boundaries to cache lines
   - **Mitigation**: Process chunks that don't overlap in memory

3. **Memory Allocations**: Parallel processing may increase allocations
   - **Mitigation**: Reuse buffers in goroutine pool
   - **Mitigation**: Pre-allocate work queues

4. **Build Complexity**: Managing build tags adds complexity
   - **Mitigation**: Clear documentation and examples
   - **Mitigation**: Provide both builds in CI/CD

## Success Criteria

1. **Correctness**: Parallel implementations produce identical results
2. **Performance**: 30%+ improvement for target operations on multi-core systems
3. **No Regressions**: Single-threaded performance remains unchanged
4. **Maintainability**: Code remains readable and testable

## Timeline

- **Phase 1 (Infrastructure)**: 1-2 days
- **Phase 2 (High Priority)**: 3-5 days
- **Phase 3 (Medium Priority)**: 2-3 days
- **Testing & Validation**: 2-3 days

**Total Estimated Time**: 8-13 days

## Next Steps

1. Review and approve this plan
2. Create infrastructure (`multithreaded.go`, build tags)
3. Implement high-priority operations
4. Benchmark and validate improvements
5. Iterate based on results

