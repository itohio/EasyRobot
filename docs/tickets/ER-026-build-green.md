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
updated: 2026-11-19
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-026 — Build green: host part, then TinyGo stage

[[tickets/index|Ticket index]] · [[roadmap]] · [[todo]]

## Goal and evidence

One ticket, **two stages** (O6). `status: ready` covers stage 1 only; stage 2 is blocked until the owner approves TinyGo. Findings F1–F7 from baseline §4; [[build-targets]].

**Stage 1 — host (no approval needed):**
- amd64-only asm without fallback (F4) — `x/iter/loop_experiments/matmul.go:5`, `bce_experiments.go:317,318,374,375`, `x/types/st/convert_bench_test.go:583`.
- Encoder/motor stub returns nothing (F1) — `x/devices/encoder/stub.go:7`.
- Int-size build tags by GOARCH list — `x/types/st/convert_i32.go:1`, `x/types/st/cast.go:82` (use `strconv.IntSize`).
- `!tinygo` tags on desktop files — `cmd/elrs-rx/platform_desktop.go:1` (F3, also add darwin), `drivers/lidar/config.go:1`, `device_factory.go:1`, `display.go:1`.
- darwin serial (F2) — `x/devices/serial_stub.go:1-3`.
- vet failures: `x/buffer` `WriteByte`/`ReadByte` clash with `io.ByteWriter`/`ByteReader` (F7, `bit_buffer.go:144,210`). The F5 malformed test names lived in `x/store` and `x/pipeline/steps`, which ER-cleanup deleted. F6 (`tc` test port) is the audio composition fixes' first step (O3), not here.

**Stage 2 — TinyGo (blocked: owner approval of TinyGo install/run):** the first `tinygo build` (`cmd/elrs-rx` for an ESP32 board, no flashing), CI/Makefile matrix. Held decision #13 is resolved: ER-cleanup removed the committed 9.3 MB ELF and replaced the `go:generate tinygo flash` line with a confirm-first `make flash-manipulator` target.

## Dependencies and scope

Dependencies: Stage 1: none. Stage 2: owner TinyGo approval; opens every device stage (M-a 2, M-b 2, M-d 2, M-f 3, W8).

Not in scope: No flashing, no `go generate`, no deletions under held decisions.

Source/test edits are approved; scheduling is decided separately. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Stage 1: `go build ./...` and `go vet ./...` on darwin/arm64 report no F1–F5/F7 failures (OpenCV-dependent packages excepted and listed); a test for each fallback asm function compares it with the amd64 path's expected values.
- [ ] Stage 1: `GOOS=js GOARCH=wasm go build ./x/types/st` succeeds.
- [ ] Stage 2: `tinygo build -target=<esp32 board> ./cmd/elrs-rx` succeeds, command and version recorded.

## Effort uncertainty and revision trigger

Size guess: **M** (both stages). Sizes are rough guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

## Host test baseline (2026-11-19, `main` `9230292a`, darwin/arm64, Go 1.27.2, no OpenCV installed)

`go test -count=1 ./...` from the repository root: **89 packages ok, 56 fail** (42 do not build, 14 have failing tests), 69 without test files. The failing set is the same with and without the newest code (the UnitV YOLO example passes). Two classes:

### A. Environment (not a defect in the code): 33 packages

They import `gocv.io/x/gocv`, whose cgo build needs OpenCV 4 (`pkg-config opencv4`), which this host does not have: `cmd/calibrate`, `cmd/cr30`, `cmd/display` (+ `destination`, `source`), `cmd/spectrometer` (+ `cameras`, `internal/calibrate`, `internal/config`, `internal/render`, `measure`), `drivers/lidar`, `x/marshaller/gocv`, `x/math/tensor/gocv`, and these `x/vision` packages: `detect/{blobs,corners,features}`, `extract/{blobs,features}`, `match/features`, `odometry/{phase,pose}`, `stereo/{depth,matching}`, `transform/{affine,background,color,crop,format,histogram,rectify,undistort}`, `types`. The YOLO packages (`x/vision/detect/{nms,yolox}`) do not import OpenCV and pass.

Also environmental: in a `git worktree` outside the main checkout, `replace github.com/itohio/dndm => ../dndm` points at a missing directory, so `cmd/manipulator` (and anything else importing `dndm`) fails to load there. In the main checkout, beside a `dndm` checkout, it builds.

### B. Real failures (in the code or in the test; each needs a look): 23 packages

Does not build on darwin/arm64 (9):

| Package | Cause | Ticket |
|---|---|---|
| `cmd/elrs-rx` | the desktop platform file is excluded on darwin (`initHardware`, `getStream`, … undefined) | F3 above |
| `cmd/monitor`, `cmd/spectrometer/ports`, `cmd/spectrometer/internal/acquirer` | no darwin serial implementation (`devices.ListSerialPorts`, `NewSerialWithConfig` undefined) | F2 above |
| `x/devices/encoder`, `x/devices/motor` | `configurePins` stub returns nothing but is used as a value (`encoder.go:77`) | F1 above |
| `x/iter/loop_experiments`, `x/types/st` (test build) | amd64-only assembly declared without a fallback (`missing function body`) | F4 above |
| `x/math/dsp/modulation/tc` | the test calls an old `mpsk` and `tc` API | F6 (voice-modem composition fixes) |

Tests fail (14):

| Package | Failing tests (first assertion) | Ticket |
|---|---|---|
| `x/math/dsp/modulation/{dsss,mask,mpsk,qam}` | round trips: measured SNR far from the target, empty decodes | [[ER-022-audio-modem-acquisition\|ER-022]] |
| `x/math/dsp/coding/{crc,four5b}`, `x/math/protocol/peg` | encoder/decoder sizes, 4B5B round trip, PEG offsets and emits | [[ER-023-coding-fixes\|ER-023]] |
| `x/math/filter/pll` | frequency tracking (CFO off by 1 kHz), rotation | [[ER-019-filter-correctness\|ER-019]] |
| `x/devices/lidar/{ld06,xwpftb}` | full-rotation scan never assembled (timeout) | [[ER-024-lidar-decoder-fixes\|ER-024]] |
| `x/devices/nmea` | `ParseRMC` returns `nmea: invalid sentence` for the test's RMC sentence | [[ER-025-device-fixes\|ER-025]] |
| `x/math/nn/layers` | dense numerical gradient off (0.047 vs < 0.005); GPT embedding error text | [[ER-020-eager-views-newas-norm-gradients\|ER-020]], [[ER-029-nn-layers-fixture-harness\|ER-029]] |
| `x/math/gps` | the north-pole ECEF test expects the equatorial radius (6,378,137 m) where WGS-84 gives the polar radius the code returns (6,356,752 m), so that test looks wrong; trilateration and multilateration results are off (15 m to 1 km) | none yet |
| `x/math/colorscience` | custom illuminant, XYZ, a panic on mismatched lengths | none yet |

`x/math/dsp/modulation/mfsk` is flaky: it passed in this full run, and `Test16FSK_Noise` failed when the package ran alone the same day (cause not yet looked at). The ratchet below should treat it as "may fail" until it is fixed.

### Proposal: make `go test ./...` meaningful without hiding a failure

1. **`make test-host`:** runs `go test` on `go list ./...` minus the packages whose dependencies include `gocv.io/x/gocv` when `pkg-config --exists opencv4` fails, and prints the skipped list. Class A then disappears on hosts without OpenCV and comes back by itself where OpenCV is installed. A few Makefile lines; no source change.
2. **Known-failure ratchet:** `make test-host` compares the failing packages with a checked-in list (class B above, one package per line). A failure outside the list is a new regression; a listed package that passes leaves the list. Real defects stay red and visible, nothing is skipped in code, and a reviewer reads one line: "no new failures".
3. **Worktrees:** run the tests in the main checkout, or point the `dndm` replace at a checkout with `-modfile` (documented beside the target).

Not proposed: `t.Skip` on the class B tests (it would hide real defects), or build tags across the 33 OpenCV packages (large and churny).
