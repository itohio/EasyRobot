---
type: algorithm
status: stub
tags: [easyrobot, types, conversion, asm, simd]
workstream: [W2, W9]
module: "[[type-conversion]]"
summary: "asm writes the constant 3.05e-5 into every element (checked under Rosetta); declared only in a…"
source: [x/types/st/convert_asm_amd64.s, x/types/st/convert_dither.go, x/types/st/CONVERT_PLAN.md, x/types/st/dispatch_overhead_analysis.md, x/types/st/dispatch_optimization_issue.md]
tests: [x/types/st/convert_bench_test.go]
test_result: build-fail
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09): x/types/st build-fail darwin/arm64 (convert_bench_test.go:583 missing function body); probe InfraNotes/asmprobe (GOARCH=amd64 via Rosetta, 2026-11-09): output = constant 3.0517578e-05 for every element"
---
# amd64 "SIMD" int16→float32 conversion (x/types/st)

**What it does:** Intended: vectorised `int16 → float32 × (1/32767)` (audio sample conversion). Actual: an amd64 assembly loop that **stores the scale constant into every destination element** and never converts the input.

**Where:** `x/types/st/convert_asm_amd64.s:24-107` `TEXT ·convertInt16ToFloat32Assembly`; Go declaration only in a test file: `x/types/st/convert_bench_test.go:583` (`func convertInt16ToFloat32Assembly(dst []float32, src []int16)`), used only by the sub-benchmark `assembly_simd_placeholder` (`convert_bench_test.go:567-575`). Not called from any production code. Dither PRNG used by the real Go path: `x/types/st/convert_dither.go:5-38` (xorshift64*, TPDF = u1−u2).

**Inputs → outputs:** `dst []float32, src []int16`; n = min(len). Output (observed): `dst[i] = 3.0517578e-05` for all i.

## Completeness review

- Complete: **no** — placeholder. Loop body `convert_asm_amd64.s:43-100` loads `src` into AX/X0 and then executes `MOVSS X7, (DI)` (`:95`, comment "Placeholder: proper implementation needs CGO"). The scale constant `0x38000000` (`:7`) is `2^-15 = 3.0517578e-05`, not `1/32767 = 3.0518509e-05` as the comment says. The comment's premise that Go's assembler lacks int→float conversion is false (Go asm has `CVTSL2SS`, `CVTPL2PS`, and AVX2 `VCVTDQ2PS`; arm64 has `SCVTF` / NEON `SCVTF Vd.4S`).
- Tested: no correctness test; benchmark only. On darwin/arm64 the whole `x/types/st` test binary fails to build (`convert_bench_test.go:583: missing function body`) because the declaration is in an untagged `_test.go` file. Under `GOARCH=amd64` (Rosetta) `go test ./x/types/st` passes (the benchmark is not run).
- Correct: **no** — probe copying the `.s` into `tmp/easyrobot/tensor/InfraNotes/asmprobe`: input `[0 100 -32767 32767]` → output `[3.0517578e-05 ×4]`.

**Declared (repo docs, claims only):** `CONVERT_PLAN.md` plans per-type kernels + SIMD; `dispatch_overhead_analysis.md` / `dispatch_optimization_issue.md` discuss type-switch dispatch cost in `Convert` (declared, not verified here).

**arm64 / Xtensa gap:** no NEON or generic fallback exists; nothing in production depends on it, so the only effect is the arm64 test build break. Xtensa (ESP32) has no SIMD for this; K230 (RISC-V, RVV) would need its own path or rely on the Go compiler.

**Embedded suitability:** irrelevant until real; the Go fallback `convertFromInt16ToFloat32` (`x/types/st/convert_ops.go:347`) is allocation-free scalar float32 code.

## TODOs

- [ ] Delete the placeholder asm or implement real `CVTPL2PS`-based kernel with Go fallback and a value test — x/types/st/convert_asm_amd64.s:95 #p1 #ER-026
- [ ] Fix constant to 1/32767 (0x38000100) or document 2^-15 scaling — x/types/st/convert_asm_amd64.s:7 #p3
- [ ] If SIMD is wanted for W2: NEON (arm64) and RVV (K230) variants behind build tags, benchmark vs pure Go — proposal #p3

## Links

- Module: [[type-conversion]]
- Related: [[numeric-cast-and-clamp]], [[bce-loop-experiments]], [[matmul-asm-experiment]]
