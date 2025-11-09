# FP32 Primitive Operations Performance Report

**Generated:** 2025-11-09  
**Platform:** Linux (amd64)  
**CPU:** Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz

## Benchmark Configuration

All measurements use Go's benchmarking framework with `-benchmem` and a 1 second bench time per case.

```
# Go fallback
GOFLAGS='-tags=use_go' go test -bench="BenchmarkTensor(Conv|MatMul|Dot|BroadcastTo)" -benchmem -benchtime=1s -run=^$

# Default (assembly-accelerated) path
go test -bench="BenchmarkTensor(Conv|MatMul|Dot|BroadcastTo)" -benchmem -benchtime=1s -run=^$
```

Raw outputs are archived in `/tmp/bench_go_focused.txt` and `/tmp/bench_asm_focused.txt` for this run.

## Summary

- **Conv2D remains the main hotspot.** The Go fallback is still **12–57%** slower than the assembly path, depending on kernel size and stride, largely due to naïve `Im2Col` + GEMM implementations and repeated allocations.
- **Conv2DTransposed parity.** The Go version now matches or slightly beats the assembly path (<1.5% difference) because both sides are dominated by large GEMM calls and identical data movement.
- **MatMul shows mixed results.** The Go fallback wins on the largest square-ish 512×256 cases (≈22–27% faster) but is slower on smaller matrices (up to 30% slower) where cache blocking in assembly helps more.
- **Conv1D and Dot improvements.** For the large batched Conv1D and all Dot cases the Go code is up to 20% faster thanks to tighter loops, but small/no-batch Conv1D still suffers from per-sample overhead.

## Detailed Results

_Δ (Go vs Assembly) is computed as **(Go − Assembly) / Assembly**. Negative values mean the Go fallback is faster._

### Convolution

| Operation | Case | Assembly ns/op | Go ns/op | Go Δ | Notes |
|-----------|------|----------------|----------|------|-------|
| Conv2D    | batch32_in64_out128_k3x3_s1x1_p1x1 | 21,169,627,334 | 25,925,348,629 | +22.5% | Go done via naive GEMM; large temporary buffers dominate |
| Conv2D    | batch16_in32_out64_k5x5_s2x2_p2x2 | 7,359,683,109 | 11,516,238,784 | +56.5% | Wider kernels amplify Im2Col cost |
| Conv2D    | batch8_in16_out32_k7x7_s1x1_p3x3  | 31,677,146,133 | 35,621,166,383 | +12.4% | Memory traffic dominated |
| Conv2D    | batch1_in3_out64_k3x3_s1x1_p1x1   | 201,887,419 | 228,453,168 | +13.2% | Single image still slower in Go |
| Conv2Dᵀ   | batch32_in64_out128_k3x3_s1x1_p1x1 | 42,229,424,084 | 41,910,532,620 | −0.8% | Parity between paths |
| Conv2Dᵀ   | batch16_in32_out64_k5x5_s2x2_p2x2 | 52,905,319,090 | 52,830,986,587 | −0.1% | Practically identical |
| Conv2Dᵀ   | batch8_in16_out32_k7x7_s1x1_p3x3  | 50,108,993,741 | 49,503,895,013 | −1.2% | Go slightly ahead |
| Conv2Dᵀ   | batch1_in3_out64_k3x3_s1x1_p1x1   | 367,041,190 | 362,395,591 | −1.3% | Minor win for Go |
| Conv1D    | batch32_in64_out128_k3_s1_p1      | 261,051,320 | 209,347,819 | −19.8% | Go avoids some overhead inside `Convolve1DAdd` |
| Conv1D    | batch16_in32_out64_k5_s2_p2       | 58,025,461 | 48,335,946 | −16.7% | Larger stride remains Go-friendly |
| Conv1D    | batch8_in16_out32_k7_s1_p3        | 33,380,333 | 39,370,885 | +17.9% | Smaller batch exposes per-call allocations |
| Conv1D    | no_batch_in64_out128_k3_s1_p1     | 5,997,895  | 7,627,064 | +27.2% | Single vector hit hardest in Go |

### Matrix Multiplication

| Operation | Case | Assembly ns/op | Go ns/op | Go Δ | Notes |
|-----------|------|----------------|----------|------|-------|
| MatMul    | 2D_100x50_50x100 (in-place)  | 914,917   | 1,102,623  | +20.5% | Assembly caching wins on smaller GEMM |
| MatMul    | 2D_256x128_128x256          | 16,994,598 | 22,171,992 | +30.4% | Go version lacks blocking |
| MatMul    | 2D_512x256_256x512          | 239,589,418 | 188,002,891 | −21.6% | Go loops benefit from better cache on large aspect |
| MatMul    | batched_32x64x128_32x128x64 | 31,437,179 | 33,009,510 | +5.0% | Minor regression |
| MatMul→dst| 2D_100x50_50x100            | 891,580   | 938,201    | +5.2% | Small gap |
| MatMul→dst| 2D_256x128_128x256          | 18,936,277 | 17,116,833 | −9.6% | Go faster |
| MatMul→dst| 2D_512x256_256x512          | 258,414,826 | 188,126,337 | −27.2% | Go wins big |
| MatMul→dst| batched_32x64x128_32x128x64 | 32,330,298 | 31,852,184 | −1.5% | Essentially tied |
| MatMulᵀ   | 2D_100x50_50x100_NN         | 895,309   | 914,282    | +2.1% | Close |
| MatMulᵀ   | 2D_100x50_100x50_NT         | 1,081,509 | 870,954    | −19.5% | Go faster |
| MatMulᵀ   | 2D_50x100_50x100_TN         | 1,290,059 | 1,157,696  | −10.3% | Go faster |
| MatMulᵀ   | 2D_50x100_100x50_TT         | 1,196,590 | 969,007    | −19.0% | Go faster |

### Dot Product & Broadcast

| Operation | Case | Assembly ns/op | Go ns/op | Go Δ |
|-----------|------|----------------|----------|------|
| Dot       | 1D_1000    | 1,664 | 1,325 | −20.4% |
| Dot       | 1D_10000   | 14,848 | 11,647 | −21.5% |
| Dot       | 2D_100x100 | 14,784 | 12,993 | −12.1% |
| Dot       | 2D_256x256 | 93,334 | 78,388 | −16.0% |
| Broadcast | 1x50x100→10x50x100 | 370,653 | 398,958 | +7.6% |
| Broadcast | 1x1→100x100        | 78,152  | 82,692  | +5.8% |

## Methodology Notes

- Measurements stop each benchmark once 1 second of timing is gathered. Large convolution cases therefore report only one or two iterations.
- Allocation counts include both user data and temporary buffers created inside the primitives; the Go fallback currently re-allocates large workspaces on each call.

## Recommended Next Steps

1. **Reduce Conv2D allocations.** `Conv2D` (see `pkg/core/math/primitive/fp32/conv_complex.go`) allocates `im2col` and `gemmOutput` slices every call. Introducing a scratch-tensor cache or caller-provided workspace would remove ~page-sized allocations and reduce GC pressure.
2. **Improve Go GEMM micro-kernel.** `Gemm_N*` functions in `level3_go.go` are straight triple loops. Adding tiling (e.g. 8×8×K blocks), pointer prefetching, and simple SIMD-friendly unrolling would slash the 30–60% regression observed on medium-sized matrices and Conv2D.
3. **Batch Conv1D better.** `Convolve1DAdd` in `conv_go.go` repeatedly calls `Dot`, recomputes offsets, and handles padding inside tight loops. Specialising for contiguous kernels, fusing padding checks, and operating on multiple output positions per pass would eliminate the 18–27% losses for smaller batches.
4. **Shared stride utilities.** Broadcasting penalties stem from repeated stride recomputation and bounds checks. Hoisting stride calculation and reusing scratch slices would bring the Go fallback closer to assembly for common shapes.

## Shared Hot Paths Worth Assemblifying

Even though the current focus is on Go-level tuning, the following routines are used by many operations and remain excellent candidates for future micro-kernel work when the assembly toolchain stabilises:

- `Gemm_NN`, `Gemm_NT`, `Gemm_TN`, `Gemm_TT` (`level3_go.go`): power both MatMul APIs and Conv2D/Conv2DTransposed.
- `Im2Col` / `Col2Im` (`pooling_go.go`): feed every convolution forward/backward pass and appear in pooling.
- `Convolve1DAdd` (`conv_go.go`): shared by forward, transposed, and kernel-gradient 1D convolutions.
- `Dot` (`dot_go.go` via `Convolve1DAdd` and Dot benchmarks): the inner product is sprinkled across optimisers, layer maths, and broadcast comparisons.

Optimising any of these code paths benefits multiple high-level tensor operations simultaneously.

