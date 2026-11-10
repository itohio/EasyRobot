---
id: ER-046
title: "KPU kmodel v3 run through a pure-Go runner"
type: ticket
status: ready
kind: implementation
owner: k210
workstream: K210-nn
milestone: K210-M6
tags: [easyrobot, ticket, k210, tinygo]
created: 2026-11-09
updated: 2026-11-10
source_revision: db9eb8f2a50e0c84dc7f32a9109148dd13516d9e
---
# ER-046 — KPU kmodel v3 run through a pure-Go runner (K210 M6)

[[tickets/index|Ticket index]] · [[2026-11-09-k210-camera-nn|K210 plan]] · [[build-targets]]

## Goal

Hybrid NN core (plan decision 3): KPU backbone via kmodel v3, nncase only as the host compiler.

## Scope

- `drivers/kpu/kmodel`: kmodel v3 parser (host-testable).
- `drivers/kpu/kpu_k210.go`: layer-argument FIFO writes, AI SRAM I/O, CALC_DONE wait.
- A tiny test net compiled by nncase v0.1.0-rc5.

## Acceptance

The test kmodel runs on the board; output within quantisation tolerance of nncase's simulator; ms/frame logged.

## Dependencies

ER-041; nncase v0.1.0-rc5 host install (owner approval). Roster task ER-K210; one milestone per round, each with `~/src/docs/evidence/ER-K210-M6.md`.

## Evidence

Verify accepted K8b (roster verification ER-K210-K8b, under Leader's per-layer bar): the pure-Go runner (`drivers/kpu`, `drivers/kpu/kmodel`) runs nncase v0.1.0-rc5 kmodels on the UnitV; a single conv layer agrees with nncase's simulator within 1 LSB everywhere, and deeper models differ by up to 4 LSB as the KPU's ±1 per layer compounds (evidence: `~/src/docs/evidence/ER-K210-K8.md` §K8b, outside the repo). Layer format: [[k210-kpu]].
