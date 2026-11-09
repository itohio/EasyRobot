---
id: ER-001
title: Discover a reproducible CPU reference environment
type: ticket
status: done
kind: discovery
owner: easyrobot
created: 2026-11-08
workstream: all
milestone: all
tags: [easyrobot, ticket]
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-001 — Discover a reproducible CPU reference environment

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

Pin host architecture, Go version, source/dependency revisions and native-library requirements. Identify the narrow CPU package/import boundary; do not assume a root import is CGO-free.

Source: go.mod, Makefile, x/math/tensor/types/, x/math/tensor/eager/, and the sibling dndm replacement. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: None. Executed 2026-11-08 under roster task R1-05 (owner instruction: "properly update dndm, easyrobot projects so i can start improving on them").

Not in scope: No broad dependency upgrades or accelerator implementation.

Execution limits for R1-05: no installs (OpenCV, TFLite, buf, TinyGo), no model, no hardware or serial access, no source edits.

## Acceptance evidence

- [x] A manifest records source, sibling dependency, compiler, platform, flags and native dependency versions or explicit missing prerequisites. → [[baseline-2026-11-08#1. Environment manifest|baseline §1]].
- [x] A package-level CPU check approved by Leader task R1-05 has captured command, result and limitations; unavailable dependencies are reported, not bypassed. → [[baseline-2026-11-08#2. Commands|baseline §2–6]]. OpenCV-dependent packages are reported as missing-prerequisite, not stubbed.
- [x] No target hardware or model is selected implicitly. None was selected.

Scope of `done`: host darwin/arm64 only. Vision (GoCV/OpenCV), TFLite, V4L2, TinyGo and linux execution remain unverified; reopen or add a ticket if the owner wants one of them covered.

## Effort uncertainty and revision trigger

Revisit when go.mod's Go version, the dndm revision or a native dependency changes, or when a second host (linux) is needed.

## Implementation evidence

2026-11-08, `abb5c9cca7f756c632548d9c7be2436b46dc2e4d`, dndm `c7b5b61f79ea07d3359d8963ac81b967e3e544ec`, go1.27.2 darwin/arm64 (Apple M5, macOS 26.3, Apple clang 17.0.0), `GOTOOLCHAIN=local`. Full record: [[baseline-2026-11-08|baseline-2026-11-08.md]].

| Check | Outcome |
|---|---|
| Module fetch | empty module cache: 32 module zips downloaded from proxy.golang.org into `~/go/pkg/mod/cache/download` at three points: 28 during the first `go list` (19:40), 1 during `go vet ./...` (19:43), 3 during the linux cross-build (19:46). Leader ruling: disclosed minor boundary note; later runs use `GOPROXY=off`. |
| `go mod verify` | all modules verified |
| `go build ./...` (per package, 197) | 155 ok; 33 need OpenCV (`pkg-config opencv4` missing); 6 fail on darwin platform-file gaps (F1–F3); 1 on the arm64 amd64-only-assembly gap (F4); 2 test-only |
| `go vet ./...` (per package) | 154 ok; 33 OpenCV; 6 darwin gaps; 4 vet findings (F5–F7) |
| `go test -count=1` on the 164 non-GoCV packages | 56 pass, 16 with failing tests, 11 test-build failures, 81 no test files; 1633 / 40 / 2 top-level pass/fail/skip |
| `go test -race` on 12 CPU-core packages | 10 pass; `x/math/nn/layers` same 2 failures; `x/math/graph` flaky `TestRRTStar_Simple2DGrid` (test sampler uses global `rand.Float32()`, `rrtstar_test.go:24–25`; 14/30 and 16/30 failures at `-count=30`); no data races |
| `CGO_ENABLED=0` darwin/arm64, linux/amd64, linux/arm64 builds | CPU core is CGO-free; CGO only via GoCV, go-tflite, go4vl, malgo, tarm/serial (darwin) |

CPU reference boundary: `x/math/tensor/{types,eager,gorgonia}`, `x/math/primitive/fp32`, `x/math/graph`, `x/math/mat`, `x/math/vec`, `x/iter`, `x/workers`; none import dndm, GoCV or TFLite.

Hazard found: `drivers/manipulator/main.go:12` has `//go:generate tinygo flash -target=xiao`, so `make gen` (`go generate ./...`) would try to flash a board.

Not run: TinyGo, tagged heavy tests (`mnist_test`, `xor_test`), `use_mt` build, benchmarks, Go 1.25 (go.mod version).
