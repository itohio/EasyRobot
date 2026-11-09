---
id: ER-029
title: "NN layer fixes and reference-fixture harness"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W9
milestone: M-f
tags: [easyrobot, ticket, w9, nn]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-029 — NN layer fixes and reference-fixture harness

[[tickets/index|Ticket index]] · [[roadmap]] · [[ER-R2]] · [[todo]]

## Goal and evidence

Fix existing layers and build the reference-fixture harness that lifts ER-R2 decision 9's freeze on new layers (O5) ([[nn]]):

- **GPT attention**: QKV split lengths, head reshape, 2-D output shape — `x/math/nn/layers/gpt_attention.go:163`, `:279-296`, `:370-386`; placeholder backward — `:408-423`.
- **GPT positional encoding**: `Copy` direction, `Slice` lengths — `gpt_encoding.go:193-218`, `:271`, `:288-301`.
- **GPT embedding** 1-D lookup and gradients — `gpt_embedding.go:145-147`, `:231-233`; stale assertion `gpt_embed_test.go:82`.
- **LSTM** gates break on views — `x/math/tensor/eager/activations.go:52-73` (root cause in ER-020).
- **BatchNorm2D** is instance norm — `layers/activations.go:669-690`.
- **Conv1D kernel gradient** layout — `x/math/primitive/fp32/tensor.go:345-372`; Conv2D stride-tail input grad — `conv2d.go:238-257`.
- **`Sequential.Parameters`** collides across layers — `models/model.go:187-214`.
- **Dense numerical gradient check** uses forward differences — `dense_comprehensive_test.go:297,325,370`.
- **Harness**: per-layer reference values (inputs, weights, outputs, gradients) as committed fixtures, generated once from a pinned reference implementation.

## Dependencies and scope

Dependencies: After [[ER-002-cpu-graph-semantics|ER-002]], [[ER-020-eager-views-newas-norm-gradients|ER-020]], [[ER-028-graph-fixes|ER-028]] (O5). Feeds ER-011…014 (M-f stage 2) and M-c's Go port.

Not in scope: No new layers (frozen until the harness exists), no decoder blocks (ER-012/013).

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Harness exists and loads fixtures offline; each fixed layer has a forward and backward fixture test that failed first.
- [ ] Central-difference gradient check passes for Dense, Conv1D (multi-channel, stride 2) and Conv2D (stride tail).
- [ ] `Sequential.Parameters` returns one entry per layer parameter.
- [ ] `go test -count=1 ./x/math/nn/...` passes.

## Effort uncertainty and revision trigger

Product's guess: part of M-f stage 1 (**L** for the stage); this ticket ≈ **M–L**. Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
