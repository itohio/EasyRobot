---
type: algorithm
status: partial
tags: [easyrobot, nn, layers, transformer, llm]
workstream: W9
module: "[[nn]]"
summary: "2-D path correct; 1-D path panics or overwrites the table (`gpt_embedding.go:145-147`)"
source: [x/math/nn/layers/gpt_embedding.go]
tests: [x/math/nn/layers/gpt_embed_test.go, x/math/learn/gpt_embedding_test.go]
test_result: fail
complete: yes
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → TestGPTEmbedding_BasicFunctionality pass, TestGPTEmbedding_ErrorCases FAIL (stale assertion); learn TestGPTEmbeddingLearning pass (go test -count=1 -short ./x/math/learn/, 2026-11-09); probe (2026-11-09): 1D path panics or corrupts the table"
---
# GPT token embedding layer

**What it does:** looks up rows of a learnable `[vocab, D]` table for integer token ids stored in a float tensor. Backward scatter-adds into the table gradient.

**Where:** `x/math/nn/layers/gpt_embedding.go`: `NewGPTEmbedding` :23-52, `Init` :56-88, `Forward` :93-125, `lookupEmbeddings` :129-173, `Backward` :178-216, `accumulateEmbeddingGradients` :220-256, `OutputShape` :261-275, `EmbeddingTable/SetEmbeddingTable` :278-315.

**Inputs → outputs:** ids `[T]` → `[T,D]`, or `[B,T]` → `[B,T,D]`. Ids are float values cast with `int(input.At(i))` (:138, :157), so FP32 represents ids exactly only up to 2²⁴. That is fine for Qwen's 151 936 vocabulary. Table = `ParamWeights`, Xavier-init (:48). Backward returns `nil` gradInput (:215).

## Completeness review

- Complete: yes in scope (lookup plus table gradient).
- Tested:
  - `TestGPTEmbedding_BasicFunctionality` checks every output element equals the table row on a `[2,3]` batch (2D path only). Pass.
  - **`TestGPTEmbedding_ErrorCases` FAIL** (gpt_embed_test.go:82): the impl error is `"GPTEmbedding.Init: input must be 1D or 2D, got 3D"` (gpt_embedding.go:63) but the test expects the substring `"input must be 2D"`. Stale assertion: the impl was widened to accept 1D. Out-of-range id → error is also covered there.
  - learn `TestGPTEmbeddingLearning`: embeddings get non-zero gradients and similar tokens cluster (2D only). Pass.
- Correct: **no for the 1D path**; the 2D path is value-tested. Probe `embed1d` (vocab 3–4, D=2, table row r = [10r, 10r+1]):
  - A: ids `[0]` → output `[0 0]` (expected `[0 1]`); **table row 0 overwritten to `[0 0]`**.
  - B: ids `[2,0,1]` → PANIC `tensor.Slice: start+length (2+3=5) exceeds dimension size 3`.
  - C: ids `[1]` (vocab 4) → PANIC `tensor.Copy: shape mismatch: dst [2 2] vs src [1 2]`.
  - D: Backward ids `[0]` → grad row0 = `[1 1]` ✓ (only id 0 works).
  - E: Backward ids `[1]` → PANIC `tensor.Add: shape mismatch: [2 2] vs [1 2]`.
  - F: same ids as 2D `[[2,0,1]]` → `[20 21 0 1 10 11]` ✓.

## Defects

- 1D forward: `embeddings.Slice(nil, 0, tokenID, tokenID+1)` passes `tokenID+1` as a LENGTH (`Slice(dst,dim,start,length)`, eager/tensor.go:592), and `embedSlice.Copy(outputSlice)` copies output **into the table** (`Copy(src)` copies into the receiver, eager/tensor.go:300) (gpt_embedding.go:145-147). Result: a panic for most ids, and table corruption plus zero output for id 0. #p1
- 1D backward: the same LENGTH bug in `gradEmbeddings.Slice(nil,0,tokenID,tokenID+1)` and `gradOutput.Slice(nil,0,i,i+1)` (:231-233). #p1
- The 2D path is correct but scalar: `At`/`SetAt` per element with variadic index slices (:164-167, :246-250). O(B·T·D) interface calls, which allocate [INFERENCE].
- The gradient *accumulates* (+=), while Dense/Conv *overwrite*. TrainStep calls ZeroGrad first, so it works, but semantics are inconsistent across layers (see [[dense-layer]]).
- Stale test assertion (gpt_embed_test.go:82). #p3

## Qwen2 readiness

- Present: the table lookup.
- Missing:
  - integer id tensors (ids pass through FP32)
  - FP16/BF16/int8 table storage (Qwen2-0.5B table is 151936×896 ≈ 136M params = 545 MB FP32)
  - row-gather kernel (`Gather`)
  - tied LM-head matmul helper
  - mmap/lazy loading

## TODOs

- [ ] Fix 1D lookup: `embeddings.Slice(nil,0,tokenID,1)` and `outputSlice.Copy(embedSlice)` (direction) — gpt_embedding.go:145-147 #p1 #ER-029
- [ ] Fix 1D gradient slices (length 1) — gpt_embedding.go:231-233 #p1 #ER-029
- [ ] Update test to expect "input must be 1D or 2D" and add 1D value + backward tests — gpt_embed_test.go:80-82 #p3
- [ ] Replace per-element At/SetAt with a row copy (copy on contiguous data) — gpt_embedding.go:155-169,240-252 #p2
- [ ] Support quantised/half table and integer ids for LLM inference — gpt_embedding.go:43-49 #p2

## Embedded suitability

Lookup is memory-bound. The scalar path costs an interface call per element. The table is FP32 only, so large vocabularies cannot fit on ESP32. No goroutines.

## Links

- Module: [[nn]] · kernels: [[tensor]], [[type-conversion]] · related: [[gpt-positional-encoding-layer]], [[gpt-attention-layer]], [[learn]]
