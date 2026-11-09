---
id: ER-027
title: "Marshallers: proto small ints, json/yaml ints, graph node data"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W4
milestone: M-b
tags: [easyrobot, ticket, w4, marshallers]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-027 — Marshallers: proto small ints, json/yaml ints, graph node data

[[tickets/index|Ticket index]] · [[roadmap]] · ER-R2 (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · [[todo]]

## Goal and evidence

Gates M-b stage 1 (O7: before any M-b work) ([[marshallers]], [[dndm-and-protobuf]]):

- **proto** drops INT8/INT16 data and panics on UINT8; emits empty data for unsupported dtypes — `x/marshaller/proto/convert.go:12-121` ([[marshaller-tensor-proto-conversion]]). Accepted decision 8: `types.math.Tensor` with `data_bytes`, scale, zero-point, layout.
- **json/yaml** decode via `SetAt` panics on integer tensors; `uint8` missing from dtype maps — `x/marshaller/json/unmarshaller.go:162`, `x/marshaller/yaml/unmarshaller.go:159`, `json/internal.go:101-153`, `yaml/internal.go:101-153`.
- **graph marshaller** drops node data — `x/marshaller/graph/graph_capture.go:8-10`, `marshaller.go:106`; `GraphTransaction` methods silently no-op — `graph.go:120-187`.

## Dependencies and scope

Dependencies: None blocking. Precedes [[ER-031-spi-slave-feature-framing|ER-031]] and the M-b host recorder.

Not in scope: The `easyrobot.*` proto package rename and the feature-frame wire contract are separate M-b work (no ticket yet).

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Failing test first: INT8/INT16/UINT8/FP16 tensors round-trip bit-exact through proto; unsupported dtypes return an error.
- [ ] Failing test first: json and yaml round-trip INT8/INT16/INT32/UINT8 tensors without panic.
- [ ] Failing test first: graph marshaller round trip preserves node data; `GraphTransaction.Commit` returns `errors.ErrUnsupported` until implemented.
- [ ] `go test -count=1 ./x/marshaller/...` passes.

## Effort uncertainty and revision trigger

Product's guess: **S**. Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
