---
id: ER-040
title: "UnitV board target for TinyGo (build only)"
type: ticket
status: done
kind: implementation
owner: k210
workstream: K210-camera
milestone: K210-M0
tags: [easyrobot, ticket, k210, tinygo]
created: 2026-11-09
updated: 2026-11-09
source_revision: db9eb8f2a50e0c84dc7f32a9109148dd13516d9e
---
# ER-040 — UnitV board target for TinyGo (build only) (K210 M0)

[[tickets/index|Ticket index]] · [[2026-11-09-k210-camera-nn|K210 plan]] · [[build-targets]]

## Goal

TinyGo 0.43 knows the K210 only through the MAix Bit board (`maixbit`), whose LED constants (D12–D14, discrete RGB) do not exist on the M5Stack UnitV. Plan decision 4: an EasyRobot `unitv` target, not a TinyGo patch.

## Scope

- `drivers/unitv/target/unitv.json` (inherits `maixbit`, tag `unitv`) and `drivers/unitv/target/unitv.ld` (16 KiB stack) — moved from the first-draft `targets/` in M3 after acceptance of the `drivers/` layout.
- `drivers/unitv/`: pin map (WS2812 G8, buttons G19/G18, Grove G34/G35, UARTHS G4/G5, camera IO40–47); examples `blink`, `hello`.

## Acceptance

`tinygo build -target=drivers/unitv/target/unitv.json` of `blink` and `hello` succeeds from the repo root; sizes recorded; `_stack_size` = 0x4000 in the ELF; host `go build`/`go vet` of `drivers/unitv/...` unaffected.

## Dependencies

none. One milestone per round, each with its evidence record.

## Evidence

Verified; review fixes applied.
