# FP32 Benchmark Report

**Generated:** November 10, 2025  
**Platform:** Linux amd64  
**CPU:** Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz  
**Package:** `github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32`

**Benchmark command:** `go test -bench . -run ^$ ./pkg/core/math/primitive/fp32`

## Summary

This report records the current baseline for the core fp32 primitives. Each benchmark was executed once (Go 1.22) immediately before this report was generated. All runs completed with zero allocations and reuse pooled buffers where applicable.

| Benchmark | ns/op | B/op | allocs/op |
|-----------|-------|------|-----------|
| `BenchmarkElemAdd/contiguous-10` | 81,606 | 0 | 0 |
| `BenchmarkElemAdd/strided-10` | 102,573 | 0 | 0 |
| `BenchmarkReduceSum/contiguous-10` | 284,318 | 0 | 0 |
| `BenchmarkReduceSum/strided-10` | 226,869 | 0 | 0 |
| `BenchmarkHadamardProduct/contiguous-10` | 4,519 | 0 | 0 |
| `BenchmarkHadamardProduct/strided-10` | 6,098 | 0 | 0 |
| `BenchmarkConvolve1D/forward-10` | 21,600 | 0 | 0 |
| `BenchmarkConvolve1D/transposed-10` | 19,590 | 0 | 0 |
| `BenchmarkNormalizeVec/contiguous-10` | 1,494 | 0 | 0 |
| `BenchmarkNormalizeVec/strided-10` | 1,538 | 0 | 0 |
| `BenchmarkSumArrScalar/contiguous-10` | 3,465 | 0 | 0 |
| `BenchmarkSumArrScalar/strided-10` | 4,811 | 0 | 0 |
| `BenchmarkDiffArrScalar/contiguous-10` | 4,199 | 0 | 0 |
| `BenchmarkDiffArrScalar/strided-10` | 4,503 | 0 | 0 |
| `BenchmarkElemScale/contiguous-10` | 87,064 | 0 | 0 |
| `BenchmarkElemScale/strided-10` | 87,852 | 0 | 0 |

## Notable Observations

1. **Vector apply gains.** The single-threaded `st` helpers now unroll contiguous loops and batch stride updates. Compared to the previous run (November 10, 2025 12:00 UTC), contiguous vector ops improved by 18–26% (`ElemAdd` 99.9µs → 81.6µs, `HadamardProduct` 6.1µs → 4.5µs) while strided variants saw 35–43% speedups (`ElemAdd` 180µs → 103µs, `NormalizeVec` 2.47µs → 1.54µs).
2. **Matrix wrappers benefit automatically.** The matrix apply routines reuse the vector helpers row-by-row; `ReduceSum/strided` dropped from 310µs to 227µs (~27% faster) without changing the public API.
3. **Convolution helpers pick up improvements.** `Convolve1D` forward/transposed now run 15–17% faster thanks to the tighter scalar and binary apply loops they depend on.

## Action Items

- Propagate the same unrolled strided logic to the multi-threaded (`mt`) variants once we retune chunk sizing. Re-benchmark with `-tags use_mt` afterwards to ensure the threshold logic does not negate the contiguous gains.
- Integrate these benchmarks into `make bench-fp32` so we can track both the November 10 baseline (pre-optimization) and this improved run automatically.
