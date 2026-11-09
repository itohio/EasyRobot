---
id: ER-020
title: "Eager tensor views, NewAs aliasing and norm gradients"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W9
milestone: M-f
tags: [easyrobot, ticket, w9, tensor]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-020 — Eager tensor views, NewAs aliasing and norm gradients

[[tickets/index|Ticket index]] · [[roadmap]] · [[ER-R2]] · [[todo]]

## Goal and evidence

Make eager tensors a trustworthy reference ([[tensor]], [[primitive-kernels]], [[tensor-views-strides]]):

- `Data()` ignores a view's offset — `x/math/tensor/eager/tensor.go:103-111`; `x/iter/st/apply_tensor.go:53` drops the inner stride, so ops on `Slice`/transposed views are wrong (breaks LSTM gate slicing and attention heads).
- `NewAs` aliases its input — `x/math/tensor/eager/tensor.go:42-44`.
- γ-weighted RMSNorm/LayerNorm gradients wrong — `x/math/primitive/fp32/normalization.go:513-596`.

## Dependencies and scope

Dependencies: Runs with [[ER-002-cpu-graph-semantics|ER-002]] (its item 4 is expected to expose the view and `NewAs` defects) and [[ER-028-graph-fixes|ER-028]]. Followed by [[ER-029-nn-layers-fixture-harness|ER-029]] (O5).

Not in scope: No new ops, no performance work, no Gorgonia changes.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Failing test first: element-wise op on a `Slice` view with non-zero offset and on a transposed view equals the same op on a contiguous copy.
- [ ] Failing test first: mutating the result of `NewAs(x)` leaves `x` unchanged.
- [ ] Failing test first: RMSNorm and LayerNorm γ and input gradients match central differences within 1e-3.
- [ ] `go test -count=1 ./x/math/tensor/eager ./x/math/primitive/fp32 ./x/iter/st` passes.

## Effort uncertainty and revision trigger

Product's guess: part of M-f stage 1 (**L** for the stage); this ticket ≈ **M**. Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
