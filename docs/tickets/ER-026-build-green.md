---
id: ER-026
title: "Build green: host part, then TinyGo stage"
type: ticket
status: ready
kind: implementation
owner: easyrobot
workstream: all
milestone: all
tags: [easyrobot, ticket, build, tinygo]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-026 — Build green: host part, then TinyGo stage

[[tickets/index|Ticket index]] · [[roadmap]] · ER-R2 (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · [[todo]]

## Goal and evidence

One ticket, **two stages** (O6). `status: ready` covers stage 1 only; stage 2 is blocked until the owner approves TinyGo. Findings F1–F7 from baseline §4 (evidence: `~/src/docs/evidence/baseline-2026-11-08.md#4. Findings (project code, darwin/arm64)`, outside the repo); [[build-targets]].

**Stage 1 — host (no approval needed):**
- amd64-only asm without fallback (F4) — `x/iter/loop_experiments/matmul.go:5`, `bce_experiments.go:317,318,374,375`, `x/types/st/convert_bench_test.go:583`.
- Encoder/motor stub returns nothing (F1) — `x/devices/encoder/stub.go:7`.
- Int-size build tags by GOARCH list — `x/types/st/convert_i32.go:1`, `x/types/st/cast.go:82` (use `strconv.IntSize`).
- `!tinygo` tags on desktop files — `cmd/elrs-rx/platform_desktop.go:1` (F3, also add darwin), `drivers/lidar/config.go:1`, `device_factory.go:1`, `display.go:1`.
- darwin serial (F2) — `x/devices/serial_stub.go:1-3`.
- vet failures: `x/buffer` `WriteByte`/`ReadByte` clash with `io.ByteWriter`/`ByteReader` (F7, `bit_buffer.go:144,210`). The F5 malformed test names lived in `x/store` and `x/pipeline/steps`, which ER-cleanup (evidence: `~/src/docs/evidence/ER-cleanup.md`, outside the repo) deleted. F6 (`tc` test port) is ER-AUDIO-1's first step (O3), not here.

**Stage 2 — TinyGo (blocked: owner approval of TinyGo install/run):** the first `tinygo build` (`cmd/elrs-rx` for an ESP32 board, no flashing), CI/Makefile matrix. Held decision #13 is resolved: ER-cleanup (evidence: `~/src/docs/evidence/ER-cleanup.md`, outside the repo) removed the committed 9.3 MB ELF and replaced the `go:generate tinygo flash` line with a confirm-first `make flash-manipulator` target.

## Dependencies and scope

Dependencies: Stage 1: none. Stage 2: owner TinyGo approval; opens every device stage (M-a 2, M-b 2, M-d 2, M-f 3, W8).

Not in scope: No flashing, no `go generate`, no deletions under held decisions.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Stage 1: `go build ./...` and `go vet ./...` on darwin/arm64 report no F1–F5/F7 failures (OpenCV-dependent packages excepted and listed); a test for each fallback asm function compares it with the amd64 path's expected values.
- [ ] Stage 1: `GOOS=js GOARCH=wasm go build ./x/types/st` succeeds.
- [ ] Stage 2: `tinygo build -target=<esp32 board> ./cmd/elrs-rx` succeeds, command and version recorded.

## Effort uncertainty and revision trigger

Product's guess: **M** (both stages). Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
