---
type: evidence
status: review
tags: [easyrobot, evidence]
task: ER-R2
objective: O-005
date: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-R2 — knowledge vault, algorithm catalog, IK review

**Result:** the vault is built in `EasyRobot/docs/`: 6 architecture notes, 36 module notes, 258 algorithm notes, 9 workstream notes, a TODO roll-up (1401 open, 467 `#p1`), an IK/Jacobian review, and 14 proposed fix tickets (ER-017…ER-030).

Algorithm status:

| Status | Count |
|---|---|
| tested | 56 |
| implemented | 37 |
| partial | 138 |
| stub | 14 |
| unverified | 13 |

On the IK question: FK and the Jacobian are correct. The IK solvers are where the problems are.

Home: [[index]]. IK detail: [[ER-R2-ik-review]].

## 1. Scope and limits

- **Task:** roster task ER-R2 (objective O-005, workstreams W1–W9).
- **Writes:** `EasyRobot/docs/**` and throwaway probes under `~/src/tmp/easyrobot/`. No repo source/test edits, no commits.
- **Environment:** offline (`GOTOOLCHAIN=local GOPROXY=off`, `GOFLAGS=-mod=readonly` for every Go command inside the repo). No hardware, TinyGo, OpenCV, downloads or web.
- **Network contacts this round:** none.
- **How the work ran:** 8 parallel slice agents wrote disjoint files under a fixed note-name registry, plus their own sub-agents. I set the layout, conventions and templates; ran an independent IK check; spot-checked defect claims; wrote `index.md`, `todo.md`, W1 and the ticket/ER-002 updates; and validated frontmatter and links.

## 2. Vault layout

| Path | Content | Writer |
|---|---|---|
| `index.md` | home map: workstreams, modules with status counts, headline findings | lead |
| `templates/` | [[conventions]], algorithm/module/workstream templates (Obsidian `templates` folder from the pre-existing `.obsidian/templates.json`) | lead |
| `architecture/` | [[overview]], [[packages]], [[data-flow]], [[dndm-and-protobuf]], [[build-targets]], `source-inventory.md` (moved from root) | ArchWorkstreams; inventory from R1 |
| `modules/` (36) | maps of content per package family | slice agents |
| `algorithms/` (258) | one note per algorithm with completeness review | slice agents and sub-agents |
| `workstreams/` (9) | W1 (lead), W2/W9 (TensorNN), W3 (KinematicsIK), W4/W5 (ArchWorkstreams), W6 (FiltersGraph), W7 (DSPAudio), W8 (DevicesLidar) | as listed |
| `todo.md` | generated roll-up | lead (script) |
| `planning/` | `roadmap.md`, `known-unknown.md` (moved from root) | R1 |
| `evidence/` | `baseline-2026-11-08.md` (moved), `ER-R2-ik-review.md`, this report | R1, KinematicsIK, lead |
| `tickets/` | ER-001…016, index with the ER-R2 proposals | R1, lead |

- **Moves:** I moved four files into the new layout and rewrote their relative links mechanically, from each file's old location to its new one; a link check passes.
- **The rename was not mine:** midway through the round, Leader renamed `EasyRobot/docs/` to `EasyRobot/docs/` following the owner's decision ("repository docs should be docs, not _docs"). Leader also edited `.gitignore` to ignore `docs/.obsidian/workspace*.json`. All later writes went to `docs/`. I removed the empty `docs/` leftovers; it held no files.

**Validation** (python check over all notes):
- 338 files at the time (337 unique names; the two `index` notes share a name), all with parseable frontmatter. **Correction (verify, ER-docs-fix):** 21 pre-existing notes (source inventory, baseline, two planning notes, 17 tickets) had no `type`/`tags`, and my check skipped `templates/`, where `templates/workstream.md:13` held two unresolved wikilinks. ER-docs-fix gave 20 of the 21 notes `type`/`tags`. The 21st is `planning/roadmap.md`, product's file, so product is adding its frontmatter. ER-docs-fix also removed the template links, and `make docs-check` now enforces both rules for notes easyrobot owns.
- Every algorithm note has `status`, `test_result`, `complete`, `correct`, `evidence` and `module`.
- No note is `tested` without `test_result: pass`.
- Eight notes have `correct: yes` with no repo tests. Each cites a probe and is marked `implemented`.
- Every wikilink and relative link resolves.
- One duplicate note name: `index` (root and `tickets/`). Wikilinks avoid it; links to it use relative paths.

## 3. Results by slice

The numbers come from each agent's offline test runs (2026-11-09) and its probes; full detail is in the module notes.

| Slice | Notes | Algorithm status (t/i/p/s/u) | Main defects (file:line in notes) |
|---|---|---|---|
| Kinematics/IK | 3 modules, 21 algorithms, W3, IK review | 6/5/8/2/0 | IK solvers (§4); VAJ1D `x1 < 1` sign test (`motion/vaj1d.go:59`); path-follower steering sign; swing height; CLI degrees/mm into a radians API |
| DSP coding | 3 modules, 22 algorithms | 4/2/12/4/0 | RS is a placeholder (`rs/rs_library.go:21-123`); CRC non-standard; HDB3 panic (`hdb3_encoder.go:106`); 4B5B nibble swap; LFSR taps; PEG pattern changes itself while matching (29 data races) |
| DSP/audio | 4 modules, 37 algorithms, W7 | 5/8/21/2/1 | Acquisition and synchronisation, not symbol decisions, cause the 25–30 dB gap; uniform noise and wideband SNR in tests; float32 oscillator phase drift; unused sync blocks broken |
| Filters/graph | 7 modules, 70 algorithms, W6 | 21/13/34/2/0 | Kalman/EKF `P` aliasing (`kalman.go:338`, `ekalman.go:459`); Mahony sign; IIR/FIR coefficients; PLL wrap; AGC; SLAM localisation static; GPS trilateration z; expression-graph cycle check and detached handles |
| Tensor/NN | 7 modules, 56 algorithms, W2, W9 | 13/5/34/2/2 | Eager view offset (`eager/tensor.go:103-111`); `NewAs` aliasing; norm gradients; GPT attention/encoding panics; LSTM gates; quaternion and 4×4 inverse in mat |
| Devices/lidar | 7 modules, 25 algorithms, W8 | 4/2/16/0/3 | LD06 speed units, int16 angles and wrap test; XWPFTB test/format mismatch; RP2040 firmware redeclared constants; motor PID sign; encoder µs/ns; PCA9685 registers; NMEA; serial busy-spin |
| IO/vision | 5 modules, 27 algorithms | 3/2/13/2/7 | `chain.Adapter` crash (`x/chain/adaper.go:111,138`) used by `cmd/audio`; graph marshaller drops node data; dtype matrix across formats; phase correlation without conjugate (`odometry/phase/processor.go:198`); 11/20 vision packages would not compile even with OpenCV |
| Architecture | 5 notes, W4, W5 | — | TinyGo selects `linux \|\| windows` desktop files; `@` in a dndm route path (`drivers/lidar/main_rp2040.go:79`); proto tensor INT8 loss; orphaned `types/core/types.pb.go`; proto package `types.core` shared with dndm |

Test runs repeated the 2026-11-08 baseline's failures. One new flaky case: `mfsk` `Test16FSK_Noise` failed at 25 dB this time; the baseline saw 22 dB. RRT* failed 7/20 and 6/20 at `-count=20`.

## 4. IK question (owner: "not sure if i solve jacobian properly")

From [[ER-R2-ik-review]] (KinematicsIK), plus my independent cross-check.

- **FK positions are correct; the pose quaternion is inverted** (see "Other IK defects" below). KinematicsIK: positions match hand computation to 1.5e-8 m. My check, a 3-DOF hexapod leg in mm against a float64 standard-DH reference over 1000 random poses: max |err| 2.79e-5 mm; zero pose (180, 0, 0) as expected.
- **The Jacobian is correct.** It is the standard-DH geometric position Jacobian, column zᵢ₋₁ × (oₙ − oᵢ₋₁); prismatic columns are zᵢ₋₁. KinematicsIK: matches a float64 analytic version to 4.4e-7 and central differences to 5.5e-5 (h = 1e-2), over 7 chains × 1000 poses. My check, the code's formula in float64 against central FD (h = 1e-6): 3.38e-8 mm/rad.
- **The IK solver is the problem** (`dh/denavithartenberg.go:150-229`): an undamped full Newton step with pseudo-inverse, while `forwardInternal` silently clamps every joint to its limits on each iteration (`:126`, `config.go:49`). My check, the same leg with targets = FK(random q in ±1 rad) and seed q = 0.1:

  | Run | Converged |
  |---|---|
  | Code, limits ±π, any of eps 0.01/0.1/1 mm, 100 or 1000 iterations | 29/500 |
  | Code, limits widened to ±1000 rad | 490/500 |
  | Same Newton method in float64 with an exact 3×3 solve | 494/500 |
  | `J · mat.PseudoInverse(J)` vs I at q = (0.5, −0.4, 0.7) | 1.0e-6 error |

  So the pseudo-inverse and Jacobian are fine. Full Newton steps overshoot past ±π, the clamp pins them, and the solver stalls without saying why.

  KinematicsIK measured 37–81 % from a cold start. Seeded with the true solution ± 0.2 rad it reached 100 % on the hexapod leg and 491–499/500 on the other chains; seeded from the previous target's answer, 191/500 (verify's correction). It notes that widening limits helps or hurts depending on the seed and the knee branch. Both results point to the same fix: wrap revolute angles before clamping, take damped and step-limited steps (`Matrix.DampedLeastSquares` already exists, unused), make the solver aware of limits, and return a best-effort answer with a reason.
- **Other IK defects:**
  - Chains with more than 3 joints always fail, because the SVD rejects 3×n with n > 3 (`la.go:760`).
  - The DH pose quaternion is the inverse rotation (`mat.go:626-648`, confirmed by reading: x = m12 − m21 at `mat.go:643`).
  - The analytic leg IK (`planar3dof.go:122,142-144`) returns wrong poses with `err = nil` behind the yaw axis, out of reach, or past a limit.
- **Helpers and gaits:** joint-configuration helper API sketch in review §5. Gait gap in §6: phase scheduler, strides, body→leg transforms, stability margin and servo timing are not written; the gait templates are interfaces only. The review's 15 proposed tickets (§7) are filed as [[ER-017-dh-planar-ik-robustness|ER-017]], [[ER-018-leg-helpers-gait-scheduler|ER-018]], ER-021, ER-037, ER-038 and ER-039; the mapping is in the IK review §7 and the [[tickets/index|ticket index]].

Probe: `~/src/tmp/easyrobot/lead-ikcheck/main.go`. Run: `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod go run .`

## 5. Lead spot checks of agent claims (read the cited lines)

| Claim | Check | Verdict |
|---|---|---|
| LD06 speed treated as 0.01 Hz, int16 angles, `targetPoints` redeclared for rp2040 | `ld06/device.go:209-279`, `drivers/lidar/main_rp2040.go:40-42` vs `main_ld06.go:13-15` | confirmed |
| Kalman `P` aliasing | `kalman.go:338` `copy(k.P, k.tempM)` on `[][]float32` copies row headers | confirmed |
| Mahony error sign | `mahony.go:224-228` computes v × a | consistent with claim |
| RS placeholder, HDB3 shift | `rs_library.go:21-40` XOR "parity"; `hdb3_encoder.go:106` `bit-1` at bit 0 | confirmed |
| VAJ1D sign test | `vaj1d.go:59` `if x1 < 1` | confirmed |
| Quaternion inverse | `mat.go:643` x = m12 − m21 | confirmed |
| Eager `Data()` offset, `NewAs` aliasing | `eager/tensor.go:42-44,103-111` | confirmed |
| Committed 9.3 MB ELF `cmd/manipulator/manipulator` | `git ls-files -s`, `file` → ELF x86-64, commit `1e9921d` 2025-11-15 | confirmed (upstream, not ours) |

## 6. Commands (lead)

- `git -C EasyRobot rev-parse HEAD` → `abb5c9c…`. `git status --short` at the end → ` M .gitignore`, ` M go.mod`, `?? docs/`. Neither modification is mine (see §7).
- Scratch probe recipe check: `go run .` in `tmp/easyrobot/recipe` importing `dh` printed `dh.Config`.
- IK cross-check as in §4.
- Vault validation and `todo.md` generation: `python3 ~/src/tmp/easyrobot/vault-tools/rollup.py` → `todo.md: 1401 items, p1 467, p2 612, p3 322`.
- Agents' commands and outputs are cited in each note's `evidence` field and in their module notes.

## 7. Incident: `EasyRobot/go.mod` modified at 01:25:16

- **What changed:** `malgo`, `go-audio/audio`, `go-audio/wav` and `spf13/cobra` moved from indirect to direct requires (the result `go mod tidy` would give). `go.sum` is unchanged.
- **Who:** every slice agent and sub-agent ran its in-repo Go commands with `-mod=readonly` (verify confirmed this from all 19 transcripts), and `-mod=mod` ran only in scratch modules. My own scratch runs: `recipe/` at 01:22:06 and the first `go run` in `lead-ikcheck/` at 01:25:57 (01:25:31 was when I copied `go.mod`/`go.sum` there). None of these can write the repo's `go.mod`.
- **Cause (found by verify, recorded in DND-R5):** the ux session ran `GOFLAGS=-mod=mod … go build … ./cmd/audio` inside EasyRobot at 23:25:15.9Z (01:25:15.9 local), one second before the `go.mod` mtime. The earlier pointer at Leader's folder move was wrong and is withdrawn: that rename happened at 01:21–01:23 and the `.gitignore` edit at 01:33:32.
- **Action:** not reverted. Leader decided to keep it and commit it on its own (`chore(deps)`) with the first gated round.
- **Undisclosed leftovers (verify):** sub-agents left `/tmp/crsf_test.json` and an empty `/tmp/x.go`. Both were deleted in ER-docs-fix.

## 8. Decisions for the owner

1. **Leg IK method.**
   - Options: analytic 3-DOF leg solver plus a damped, step-limited, limit-aware solver for longer chains; or a damped solver everywhere.
   - **Recommend:** analytic plus damped (ER-017).
2. **Gait first step.**
   - Options: a minimal phase scheduler with tripod/wave/ripple and walk/trot presets; or the DESIGN's general template machinery.
   - **Recommend:** the minimal scheduler (ER-018).
3. **Units and quaternion layout.**
   - **Recommend:** radians and metres on the wire, degrees only in the servo mapping; `[x y z w]` in Go, converted at the proto boundary.
4. **Voice modem baseline.**
   - Options: a voice-band 4/8-FSK with a chirp preamble, CRC and interleaved FEC first; or fix PSK/QAM acquisition first.
   - **Recommend:** FSK first, with BER-vs-Eb/N0 tests on a channel model in place of "SNR within ±8 dB" (ER-022).
   - The target channel (speaker→mic, phone codec, radio) needs your answer.
5. **Reed–Solomon.**
   - Options: in-repo GF(256) with erasures; an external pure-Go library; or a convolutional code with Viterbi.
   - **Recommend:** in-repo (TinyGo-friendly).
   - Also delete the wireline placeholders (AMI/B8ZS/HDB3/RLL/PPM) unless you want them.
6. **Lidar.**
   - Identify the module first (label photo and a 5 s capture).
   - **Recommend:** if it is an LD06, fix ld06 and park xwpftb with its test skipped (ER-024).
7. **Vision stack.**
   - Options: OpenCV upgrade, pure Go, or hybrid.
   - **Recommend:** hybrid — OpenCV for calibration and datasets, pure Go for anything that runs on the device (ER-030).
8. **Feature-stream wire contract.**
   - **Recommend:** `core.Frame` + `types.math.Tensor` with `data_bytes` plus scale, zero-point and layout fields; delete the orphaned `types/core/types.pb.go`; rename EasyRobot proto packages to `easyrobot.*` to avoid future clashes with dndm's `types.core`.
9. **NN framework.**
   - **Recommend:**
     - Freeze new layer work until a reference-fixture harness exists.
     - Make eager the trusted reference after fixing views (ER-020).
     - Run Qwen 0.5B on the host or K230 Linux, and use a micro-LLM for the ESP32 demo. At int4 Qwen 0.5B is ≈0.28 GB, 31× the 8 MB PSRAM.
     - Train W5 in Python notebooks first.
10. **Decision trees.**
    - **Recommend:** a data-driven tree type alongside the closure trees, learned offline from driving logs first.
11. **Deprecated cluster** (`x/pipeline`, plugin, store, options, logger, `internal/concurrency`).
    - **Recommend:** delete it once dndm covers the transport. Nothing outside `x/pipeline` uses it.
12. **`go.mod` change (§7)**, made by the ux session's `-mod=mod` build of `./cmd/audio`.
    - **Recommend:** keep it. It is the tidy-correct direct/indirect marking. Commit it on its own with the first gated round. *Decided by Leader: keep, commit separately.*
13. **Repo hygiene.**
    - Remove the committed ELF `cmd/manipulator/manipulator`, and the `//go:generate tinygo flash` line, or move it behind a Makefile target that asks first.
    - **Recommend:** both, in ER-026.

## 9. Research questions for ideation

- **W3:**
  - Phase scheduler vs central pattern generator for gait timing on an ESP32.
  - Foothold choice when turning in place or crab walking with 3-joint legs.
  - Swing curve for slow hobby servos (Bezier, cycloid, minimum-jerk).
  - Support polygon without an IMU for quadruped walk.
  - Servo zero auto-calibration from a few measured foot positions.
  - Decision-tree gait selection.
- **W4:**
  - Which int8 backbone the K230 runs through nncase, and on which runtime (CanMV, Linux SDK, Go on linux/riscv64).
  - ESP32 SPI slave max reliable clock, and whether TinyGo supports slave mode.
  - SPI frame layout (sync, sequence, length, CRC, IRQ back-pressure).
  - Sustained ESP32 SD write rate. Budget: 7×7×1280 int8 @ 10 fps = 5.02 Mbit/s, 2.26 GB/h.
- **W5:**
  - P5 alone vs P3/P4 for a YOLO head.
  - Visual words vs a global descriptor for place recognition.
  - Feature-metric loss at 7×7/10×10 stride.
  - Notebook export path into Go.
- **W6:**
  - Best online tree learner per KB on ESP32.
  - Reliable self-supervised labels (bumper, lidar near-range, stall).
  - Int8 MLP vs tree policy.
  - Pooled K230 features enough for free-space classification.
- **W7:**
  - Which "voice" channel is meant.
  - FSK vs OFDM with pilots vs chirp spread spectrum in 300–3400 Hz, and the ESP32 decode cost of each.
  - Best joint acquisition preamble (chirp, Zadoff–Chu, Barker).
  - FEC feasible at audio rates on ESP32 (soft Viterbi, short LDPC, interleaved RS).
  - Rates that survive speech codecs.
  - How to carry soft/erasure info through byte-only coder interfaces.
- **W8:**
  - Which lidar this is (LD06/LD19 0x54, Delta-2G 0xAA, or the `55 AA 23 10` 60-byte protocol).
  - ICP scan matching vs the repo's EKF-against-map on TinyGo.
  - Raw polar scans vs fixed-ray bins on the bus.
  - Lidar sector minima as decision-tree features.
  - Whether TinyGo `UART.Read` also returns immediately when empty.
- **W9:**
  - Can TinyGo on Xtensa reach the ESP32-S3 vector instructions, or does it need ESP-NN through C.
  - Real ESP32-S3 int8 GEMV throughput.
  - Which 1–5 M-parameter LM fits 8 MB PSRAM plus flash.
  - Calling the K230 NPU runtime from Go without CGO.
  - safetensors vs GGUF for a pure-Go loader.

## 10. Unverified

- Anything on hardware, TinyGo builds (only tag selection and type-checks with the normal compiler), GoCV/TFLite/V4L2 runtime, Linux execution, `use_mt` builds, heavy tagged tests (mnist/xor).
- Datasheet facts marked `[INFERENCE]` in notes. Benchmark numbers come from a shared machine.
- The repo built against dndm's current working tree (uncommitted edits by the dndm peer), not a pinned dndm commit.

## 11. Proposed mb learnings (inbox format)

```
- 2026-11-09 [code] [high] A Jacobian-pseudoinverse IK that clamps joints inside FK each iteration stalls when undamped Newton steps overshoot the limits (EasyRobot DH leg: 29/500 converged with ±π limits vs 490/500 with limits widened); wrap angles, damp and step-limit before clamping — ER-R2 lead probe
- 2026-11-09 [code] [high] In Go, copy(dst, src) on [][]float32 matrices copies row headers, not values: the destination aliases the source rows (EasyRobot Kalman/EKF P froze after one update) — ER-R2 FiltersGraph
- 2026-11-09 [embedded] [medium] TinyGo targets set GOOS=linux and add the linux tag, so files tagged `linux || windows` for desktop are compiled into firmware unless they also say `!tinygo` — ER-R2 ArchWorkstreams (read from TinyGo 0.43 target files, not run)
- 2026-11-09 [process] [medium] For a large catalog, fix a vault-unique note-name registry and frontmatter contract before fanning out parallel writers; then validate links and frontmatter by script — ER-R2
```

## 12. Next action

Leader routes this report to verify. After it is accepted, I propose this implementation order: ER-002 together with ER-020 and ER-028 (graph/tensor semantics, failing tests first), then ER-017/ER-018 (IK and gait), then ER-026 (build green). ER-R3 (skill drafts) comes after this catalog, as briefed.
