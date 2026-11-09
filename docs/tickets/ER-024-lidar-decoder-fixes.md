---
id: ER-024
title: "Lidar decoder fixes"
type: ticket
status: blocked
kind: implementation
owner: easyrobot
workstream: W8
milestone: W8
tags: [easyrobot, ticket, w8, lidar]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-024 — Lidar decoder fixes

[[tickets/index|Ticket index]] · [[roadmap]] · ER-R2 (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · [[todo]]

## Goal and evidence

Fix the decoder matching the owner's module and park the other ([[lidar]], [[W8-lidar]]). Ideation suggests the module is the Mini-360 sold as "LD06" ([[2026-11-09-lidar-w8]]); XWPFTB is the alternative.

- **LD06**: byte 1 is VerLen — `x/devices/lidar/ld06/device.go:209`, `decoder.go:64`; speed units — `device.go:274-279`; uint16 angles — `device.go:234,239`, `decoder.go:71,84`; per-point rotation wrap — `device.go:250-257,290-330`; test builder emits unreal frames — `decoder_test.go:45-109` ([[ld06-packet-decoder-and-scan-assembly]]).
- **Motor PID sign** (duty ↑ ⇒ points ↓) — `ld06/device.go:377-405`.
- **XWPFTB**: test skips `Configure` and uses the wrong frame format — `xwpftb/decoder_test.go:39-75`; scaling — `device.go:348-355,407-408`.
- **RP2040 firmware compile**: duplicate consts — `drivers/lidar/main_rp2040.go:41-42`; route path — `main_rp2040.go:79`; desktop driver's missing intent helpers — `drivers/lidar/main.go:52,93`. (The `!tinygo` tags on `drivers/lidar/{config,device_factory,display}.go` are [[ER-026-build-green|ER-026]].)

## Dependencies and scope

Dependencies: **Blocked** on the owner: a label photo and a 5 s raw capture into `x/devices/lidar/<model>/testdata/` (held ER-R2 #6). The RP2040 build check also needs the TinyGo stage of ER-026.

Not in scope: No SLAM adapter (W8 L-sized follow-up), no hardware run.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Failing test first: the recorded capture decodes to monotone angles, ≈ 360° per rotation, plausible distances; point count per packet = 12.
- [ ] Failing test first: a simulated motor converges to the target point count with the corrected PID sign.
- [ ] Datasheet-shaped packet test (12 points, ~9° spans, ≥ 2 rotations) with value checks.
- [ ] The unused driver's test is skipped with a reason, not deleted.
- [ ] `go test -count=1 ./x/devices/lidar/...` passes.

## Effort uncertainty and revision trigger

Product's guess: **M** (decoder). Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
