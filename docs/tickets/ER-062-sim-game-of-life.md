---
id: ER-062
title: "Game of Life, bit-packed (port from a badge demo) — x/math/automata/life"
type: ticket
status: blocked
kind: implementation
owner: easyrobot
workstream:
milestone: none
tags: [easyrobot, ticket, simulation, tufty]
created: 2026-11-12
updated: 2026-11-12
---
# ER-062 — Game of Life, bit-packed (port from a badge demo) — x/math/automata/life

[[tickets/index|Ticket index]] · [[simulation-algorithms]] (the liquid engine belongs in EasyRobot; paths and RNG decided in the brief)

**Brick:** Conway's Life on a torus, bit-packed (32 cells per word, bit-sliced adder), keeping the previous generation so a renderer can show births and deaths, with seed patterns.

## Goal and evidence

**Source:** the badge demo's bit-packed Life engine with its tests, **pinned to the demo revision that lands it — pending that revision**; port nothing until the revision is named here. In the demo's working tree as of 2026-11-12 it is a 320×240 torus, cells as bits 32 to a word, a generation computed 32 cells at a time by full adders over the eight shifted neighbour bitboards, and the previous generation kept (born bright, just-died dim). Tests there: `TestLifeStepMatchesTheRules`, `TestLifeGliderGunFires`. The older byte-per-cell engine with ages 0–7 (an older demo revision, 64×64 step 1.8 ms on the badge) is not ported; ages were dropped by the demo.

## Dependencies and scope

Package `x/math/automata/life`: `Board` (W×H, W a multiple of 32; current and previous generation as words), `Step`, `Seed(pattern)` with a private xorshift32, `Alive`, `WasAlive`, `Population`. Not in scope: the view widget (demo).

Every piece: check written first (red), tests, host benchmark, `tinygo build -target=tufty2040` with no allocation in the package, check, then commit. No hardware or flashing is authorised by this ticket; the badge run happens outside it.

## Acceptance evidence required later

- [ ] The pinned commit's two tests pass on the port, red against an empty package first.
- [ ] Against a plain byte-per-cell reference step: identical generations for 1,000 random boards (incl. widths 32, 64, 320 and live cells on every word edge and the torus seams).
- [ ] Blinker has period 2; a glider moves by (1, 1) in 4 generations and wraps on the torus; block and beehive are still lifes; `WasAlive` is the previous generation.
- [ ] 0 allocations per `Step`; host benchmark at the pinned commit's board size; tufty2040 build; step time compared with the demo's on the badge.

## Effort uncertainty and revision trigger

Estimate: **S**. Revisit when the badge's frame budget or the scope changes.

## Implementation evidence

None.
