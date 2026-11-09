---
id: ER-045
title: "Native int8 conv/dot/GEMV benchmark on one K210 hart"
type: ticket
status: blocked
kind: measurement
owner: k210
workstream: K210-nn
milestone: K210-M5
tags: [easyrobot, ticket, k210, tinygo]
created: 2026-11-09
updated: 2026-11-09
source_revision: db9eb8f2a50e0c84dc7f32a9109148dd13516d9e
---
# ER-045 — Native int8 conv/dot/GEMV benchmark on one K210 hart (K210 M5)

[[tickets/index|Ticket index]] · [[2026-11-09-k210-camera-nn|K210 plan]] · [[build-targets]]

## Goal

Native TinyGo inference covers heads and post-processing; the K210 is the first real target of the flat kernel layer (ER-research-1 decision 4).

## Scope

- int8 dot, GEMV, 3×3 conv 32→32 on 40×30 (host-tested Go reference).
- Example `drivers/unitv/examples/bench`: MAC/s on one hart.

## Acceptance

Board outputs bit-identical to the host Go reference (checksums over UART); MAC/s per kernel recorded.

## Dependencies

ER-041; int8 kernels in `x/math/primitive/qi8` (currently empty: `int8.go` has no functions). Roster task ER-K210; one milestone per round, each with `~/src/docs/evidence/ER-K210-M5.md`.

## Evidence

—
