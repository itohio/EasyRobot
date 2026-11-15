---
type: algorithm
status: partial
tags: [easyrobot, nn, layers, transformer, llm]
workstream: W9
module: "[[nn]]"
summary: "panics for seqLen ≥ 2; reversed `Copy` (`gpt_encoding.go:193,209,271`)"
source: [x/math/nn/layers/gpt_encoding.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "no GPTEncoding test (go test -count=1 ./x/... 2026-11-09); probe (2026-11-09): panics for seqLen≥2, zeroes its input, sinusoid frequencies non-standard"
---
# GPT positional encoding layer (absolute: sinusoidal or learned)

**What it does (declared):** adds absolute positional vectors to token embeddings. The table is either fixed sinusoidal (`learned=false`) or learnable `[maxSeqLen,D]` (`learned=true`). **There is no RoPE.**

**Where:** `x/math/nn/layers/gpt_encoding.go`: `NewGPTEncoding` :26-68, `createSinusoidalEncodings` :74-94, `Init` :98-144, `Forward` :149-181, `addPositionalEncodings` :185-226, `Backward` :231-275, `accumulatePositionalGradients` :279-309, `OutputShape` :313-334, `PositionalEncodings/SetPositionalEncodings` :337-378.

**Inputs → outputs:** `[T,D]` or `[B,T,D]` → same shape, `T ≤ maxSeqLen`. The table is stored at `ParamWeights` (`RequiresGrad=false` for sinusoidal, :63).

## Completeness review

- Complete: no. Only absolute encodings. No RoPE / ALiBi, no position offset for incremental decoding (always starts at position 0).
- Tested: none.
- Correct: **no**. Probe `gradcheck` (maxSeqLen 4, D=4, input all ones):
  - `[3,4]` → PANIC `tensor.Slice: start+length (2+3=5) exceeds dimension size 4`.
  - `[2,2,4]` → PANIC `start+length (1+2=3) exceeds dimension size 2`.
  - `[1,4]` → output `[0 1 0 1]` (expected `1+PE(0)` = `[1 2 1 2]`); **input after Forward = `[0 0 0 0]`**.
  - Sinusoid table vs standard Vaswani `PE(p,2k)=sin(p/10000^(2k/D))`: maxAbs 0.030 at D=4. The impl gives column 2 = 0.0001/0.0002/0.0003, the standard gives 0.01/0.02/0.03.

## Defects

- `input.Copy(output)` copies output **into the input** (`Copy(src)` copies into the receiver, eager/tensor.go:300) (:193, :209). It destroys the caller's embeddings (the embedding layer's output buffer) and drops the token embedding from the result. From the second call on, the output accumulates PE onto the previous output [INFERENCE from code path; first call verified]. #p1
- Slice LENGTH misuse: `posEncodings.Slice(nil,0,pos,pos+1)` / `output.Slice(nil,0,pos,pos+1)` / `.Slice(nil,0,b,b+1)` (:197, :200, :214, :218), so any T≥2 or B≥2 panics or mis-adds rows. #p1
- Backward: `gradOutput.Copy(gradInput)` is reversed. It zeroes the incoming gradient and returns zeros (:271). Gradient accumulation has the same Slice-length bug (:288-289, :300-301). #p1
- Sinusoid exponent uses `2*i/D` where `i` already steps by 2 (:79-81). Frequencies decay twice as fast as the standard, which does not match "standard GPT formula" (:71-73) or pretrained models. #p2
- Learned mode is Xavier-initialised with fanIn=maxSeqLen (:55). GPT-2 uses N(0,0.02) [INFERENCE: convention only]. #p3

## Qwen2 readiness

Qwen2 uses **RoPE** (rotary on q/k inside attention, θ=1e6) and no additive positional table. This layer does not apply to it. RoPE belongs in [[gpt-attention-layer]]; it needs cos/sin caches and a position offset for KV-cache decoding. None exists anywhere in `x/math` (grep "rotary/rope" → 0 hits).

## TODOs

- [ ] Fix Copy direction (`output.Copy(input)`) — gpt_encoding.go:193,209,271 #p1 #ER-029
- [ ] Fix Slice lengths (use 1) or vectorise as `output[b] += pe[0:T]` — gpt_encoding.go:197,200,214,218,288-289,300-301 #p1 #ER-029
- [ ] Correct the sinusoid exponent to `i/D` for even index i — gpt_encoding.go:81 #p2
- [ ] Add forward/backward value tests — (no test) #p1 #ER-029
- [ ] Implement RoPE (+ position offset) for Qwen2-style models — x/math/nn/layers (absent) #p2

## Embedded suitability

Table is `maxSeqLen×D` FP32. Uses float64 `math.Pow/Sin/Cos` at construction only. Every position creates `Slice`/`Reshape` view headers in Forward. No goroutines.

## Links

- Module: [[nn]] · kernels: [[tensor]] · related: [[gpt-token-embedding-layer]], [[gpt-attention-layer]]
