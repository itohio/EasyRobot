---
type: architecture
status: active
tags: [easyrobot, architecture, packages]
workstream: W1
source: [go.mod]
evidence: "GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go list -e -f '{{.ImportPath}}|{{join .Imports \" \"}}' ./... → 197 packages (2026-11-09); status column from baseline-2026-11-08 (evidence: `~/src/docs/evidence/baseline-2026-11-08.md`, outside the repo) §3–5"
---
# Packages

Every Go directory in `github.com/itohio/EasyRobot` at `abb5c9c`, mapped to a family, its module note and its darwin/arm64 build/test status. Layers and the dependency diagram: [[overview]]. Tensor/graph internals: [[source-inventory|source inventory]].

**Counts.** 200 directories hold `.go` files; `go list ./...` on darwin/arm64 returns 197 (command in frontmatter). The 3 others are excluded by build tags: `drivers/manipulator` (`sam && xiao`, TinyGo), `x/marshaller/tflite` and `x/marshaller/v4l` (`linux`). Non-test Go: 175,182 lines, of which 15,571 are generated protobuf (`types/**`); test Go: 89,853 lines (`wc -l` over `*.go` per directory, `/docs/` excluded).

**Since the snapshot.** ER-cleanup (evidence: `~/src/docs/evidence/ER-cleanup.md`, outside the repo) deleted `x/math/dsp/coding/{ami,b8zs,hdb3,rll,ppm}`, `x/pipeline` (with `steps`, `steps/fps` and the three examples), `x/plugin`, `x/store`, `x/options`, `x/logger` and `internal/concurrency`; their rows below are kept as the `abb5c9c` record. [[ER-018-leg-helpers-gait-scheduler|ER-018]] added `x/math/control/kinematics/chain`, `x/math/control/motion/gait/scheduler` and `x/math/control/motion/gait/walk` ([[kinematics]], [[motion-and-gait]]).

**Status legend** (all from the 2026-11-08 baseline, same HEAD; not rerun here): `build ok` = `go build -o /dev/null <pkg>` passed on darwin/arm64; `not built (OpenCV missing)` = depends on GoCV, OpenCV not installed, compile status **unknown** (not proof that it compiles); `FAIL darwin gap` = missing darwin platform file; `FAIL arm64 asm gap` = Go declarations backed only by `*_amd64.s`; `test-only pkg` = only `_test.go` files. Test counts are top-level tests.

**Mapping rules.** Family = module note from the vault registry. Packages with no registry note are listed as `—`: `x/logger`, `x/options`, `x/crypto`, `internal/concurrency` (122 LOC together; used only by `x/pipeline*`, `x/plugin`, `x/store`; `x/crypto` has no importer). `x/math/tensor/gocv` sits in [[tensor]] but is the OpenCV bridge used by [[camera-io]] and [[vision]]. `x/math/shape` is grouped with [[tensor]] although 20 packages across [[iter-workers]], [[type-conversion]], [[primitive-kernels]], [[dsp-core]], [[nn]] and [[marshallers]] import it; that grouping is the only reason the family graph in [[overview]] shows cycles (Go packages themselves cannot cycle).

## Families

| Module note | Layer | Packages | LOC (non-test) | Test files | `go build` ok on darwin/arm64 |
|---|---|---|---|---|---|
| [[dndm-and-protobuf]] | L0 schemas | 10 | 15571 | 0 | 10/10 |
| — | L0 infra | 4 | 122 | 0 | 4/4 |
| [[bit-buffer]] | L1 core | 1 | 1665 | 6 | 1/1 |
| [[type-conversion]] | L1 core | 3 | 6768 | 13 | 3/3 |
| [[iter-workers]] | L1 core | 6 | 6838 | 14 | 5/6 |
| [[primitive-kernels]] | L1 core | 9 | 8510 | 19 | 9/9 |
| [[mat-vec]] | L1 core | 5 | 6072 | 24 | 5/5 |
| [[tensor]] | L2 numeric | 7 | 14561 | 28 | 5/7 |
| [[nn]] | L2 numeric | 4 | 7119 | 14 | 4/4 |
| [[learn]] | L2 numeric | 2 | 798 | 14 | 2/2 |
| [[tflite]] | L4 IO | 1 | 596 | 1 | 0/1 |
| [[filters]] | L3 algorithms | 13 | 3537 | 8 | 13/13 |
| [[dsp-core]] | L3 algorithms | 1 | 4751 | 10 | 1/1 |
| [[modulation]] | L3 algorithms | 11 | 9996 | 23 | 11/11 |
| [[line-coding-and-fec]] | L3 algorithms | 17 | 6003 | 16 | 17/17 |
| [[peg-protocol-parser]] | L3 algorithms | 1 | 3085 | 10 | 1/1 |
| [[kinematics]] | L3 algorithms | 12 | 2503 | 7 | 12/12 |
| [[motion-and-gait]] | L3 algorithms | 5 | 2462 | 2 | 5/5 |
| [[pid-control]] | L3 algorithms | 1 | 88 | 1 | 1/1 |
| [[graph-and-planning]] | L3 algorithms | 1 | 6588 | 13 | 1/1 |
| [[grid-maps]] | L3 algorithms | 1 | 1844 | 3 | 1/1 |
| [[interpolation]] | L3 algorithms | 1 | 950 | 5 | 1/1 |
| [[gps-geodesy]] | L3 algorithms | 1 | 291 | 2 | 1/1 |
| [[colorscience]] | L3 algorithms | 1 | 3761 | 2 | 1/1 |
| [[pipeline-and-chain]] | L4 composition | 7 | 1959 | 10 | 7/7 |
| [[store-and-plugin]] | L4 composition | 2 | 1149 | 4 | 2/2 |
| [[marshallers]] | L4 IO | 10 | 12409 | 20 | 9/10 |
| [[camera-io]] | L4 IO | 2 | 5896 | 5 | 0/2 |
| [[audio-io]] | L4 IO | 1 | 7022 | 2 | 1/1 |
| [[vision]] | L4 IO | 20 | 3996 | 5 | 1/20 |
| [[device-io]] | L5 devices | 4 | 2427 | 1 | 4/4 |
| [[sensors]] | L5 devices | 6 | 2962 | 1 | 5/6 |
| [[actuators]] | L5 devices | 3 | 978 | 0 | 2/3 |
| [[lidar]] | L5 devices | 4 | 1546 | 2 | 4/4 |
| [[crsf-elrs]] | L5 devices | 4 | 2329 | 3 | 3/4 |
| [[nmea]] | L5 devices | 1 | 555 | 1 | 1/1 |
| [[audio-modem-cli]] | L6 apps | 1 | 6091 | 0 | 1/1 |
| [[apps]] | L6 apps | 17 | 11384 | 0 | 1/17 |

`go build ok` counts exclude test-only packages (`x/marshaller`, `x/math/tensor/gorgonia/gorgonia_xor_test`), which build their tests fine (baseline §5).

## All packages

### [[dndm-and-protobuf]] — L0 schemas

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `types/control` | 4/0 | 1700 |  | [[dndm-and-protobuf]] | build ok; no tests |
| `types/control/kinematics` | 5/0 | 2292 |  | [[dndm-and-protobuf]] | build ok; no tests |
| `types/core` | 3/0 | 1267 |  | [[dndm-and-protobuf]] | build ok; no tests |
| `types/devices` | 4/0 | 2041 |  | [[dndm-and-protobuf]] | build ok; no tests |
| `types/filter` | 1/0 | 458 |  | [[dndm-and-protobuf]] | build ok; no tests |
| `types/interaction` | 3/0 | 2018 |  | [[dndm-and-protobuf]] | build ok; no tests |
| `types/marshaller` | 2/0 | 923 |  | [[dndm-and-protobuf]] | build ok; no tests |
| `types/math` | 3/0 | 1533 |  | [[dndm-and-protobuf]] | build ok; no tests |
| `types/nn` | 1/0 | 369 |  | [[dndm-and-protobuf]] | build ok; no tests |
| `types/spectrometer` | 2/0 | 2970 |  | [[dndm-and-protobuf]] | build ok; no tests |

### infra (no module note) — L0 infra

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `internal/concurrency` | 2/0 | 15 |  | — | build ok; no tests |
| `x/crypto` | 1/0 | 55 |  | — | build ok; no tests |
| `x/logger` | 2/0 | 42 |  | — | build ok; no tests |
| `x/options` | 1/0 | 10 |  | — | build ok; no tests |

### [[bit-buffer]] — L1 core

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/buffer` | 5/6 | 1665 |  | [[bit-buffer]] / [[pipeline-and-chain]] | build ok; vet: FAIL stdmethods; test: ok (89 pass) |

### [[type-conversion]] — L1 core

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/types` | 18/10 | 2840 | `!use_mt`; `386 \|\| arm \|\| mips \|\| mipsle`; `amd64 \|\| arm64 \|\| ppc64 \|\| ppc64le \|\| mips64 \|\| mips64le \|\| riscv64 \|\| s390x`; `use_mt` | [[type-conversion]] | build ok; test: ok (27 pass) |
| `x/types/internal` | 3/0 | 69 | `386 \|\| arm \|\| mips \|\| mipsle`; `amd64 \|\| arm64 \|\| ppc64 \|\| ppc64le \|\| mips64 \|\| mips64le \|\| riscv64 \|\| s390x` | [[type-conversion]] | build ok; no tests |
| `x/types/st` | 10/3 | 3859 | `386 \|\| arm \|\| mips \|\| mipsle`; `amd64 \|\| arm64 \|\| ppc64 \|\| ppc64le \|\| mips64 \|\| mips64le \|\| riscv64 \|\| s390x` | [[type-conversion]] | build ok; test: FAIL build |

### [[iter-workers]] — L1 core

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/iter` | 15/11 | 1592 | `!use_mt`; `use_mt` | [[iter-workers]] | build ok; test: ok (57 pass) |
| `x/iter/loop_experiments` | 2/2 | 894 |  | [[iter-workers]] | FAIL arm64 asm gap; test: FAIL build |
| `x/iter/mt` | 5/0 | 1635 |  | [[iter-workers]] | build ok; no tests |
| `x/iter/st` | 5/0 | 2075 |  | [[iter-workers]] | build ok; no tests |
| `x/workers` | 3/1 | 559 | `!use_mt`; `use_mt` | [[iter-workers]] | build ok; test: ok (19 pass) |
| `x/workers/mt` | 1/0 | 83 |  | [[iter-workers]] | build ok; no tests |

### [[primitive-kernels]] — L1 core

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/primitive/bytes` | 1/0 | 7 |  | [[primitive-kernels]] | build ok; no tests |
| `x/math/primitive/c64` | 5/4 | 767 |  | [[primitive-kernels]] | build ok; test: ok (30 pass) |
| `x/math/primitive/fp32` | 18/15 | 7694 |  | [[primitive-kernels]] | build ok; test: ok (97 pass) |
| `x/math/primitive/qi` | 1/0 | 7 |  | [[primitive-kernels]] | build ok; no tests |
| `x/math/primitive/qi16` | 1/0 | 7 |  | [[primitive-kernels]] | build ok; no tests |
| `x/math/primitive/qi32` | 1/0 | 7 |  | [[primitive-kernels]] | build ok; no tests |
| `x/math/primitive/qi64` | 1/0 | 7 |  | [[primitive-kernels]] | build ok; no tests |
| `x/math/primitive/qi8` | 1/0 | 7 |  | [[primitive-kernels]] | build ok; no tests |
| `x/math/primitive/qu8` | 1/0 | 7 |  | [[primitive-kernels]] | build ok; no tests |

### [[mat-vec]] — L1 core

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math` | 1/1 | 117 |  | [[mat-vec]] | build ok; test: ok (4 pass) |
| `x/math/mat` | 17/17 | 3665 |  | [[mat-vec]] | build ok; test: ok (145 pass) |
| `x/math/mat/types` | 1/0 | 118 |  | [[mat-vec]] | build ok; no tests |
| `x/math/vec` | 6/6 | 2060 |  | [[mat-vec]] | build ok; test: ok (61 pass) |
| `x/math/vec/types` | 1/0 | 112 |  | [[mat-vec]] | build ok; no tests |

### [[tensor]] — L2 numeric

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/shape` | 3/6 | 348 |  | [[tensor]] | build ok; test: ok (18 pass) |
| `x/math/tensor` | 2/1 | 281 |  | [[tensor]] | build ok; test: ok (no Test funcs) |
| `x/math/tensor/eager` | 8/12 | 7926 |  | [[tensor]] | build ok; test: ok (112 pass) |
| `x/math/tensor/gocv` | 14/0 | 1446 |  | [[tensor]] / [[camera-io]] | not built (OpenCV missing) |
| `x/math/tensor/gorgonia` | 6/5 | 3391 |  | [[tensor]] | build ok; test: ok (30 pass, 1 skip) |
| `x/math/tensor/gorgonia/gorgonia_xor_test` | 0/1 | 0 |  | [[tensor]] | test-only pkg; test: ok (1 pass) |
| `x/math/tensor/types` | 12/3 | 1169 |  | [[tensor]] | build ok; test: ok (8 pass) |

### [[nn]] — L2 numeric

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/nn` | 4/0 | 493 |  | [[nn]] | build ok; no tests |
| `x/math/nn/layers` | 13/13 | 6075 |  | [[nn]] | build ok; test: FAIL (91 pass, 2 fail) |
| `x/math/nn/models` | 1/0 | 335 |  | [[nn]] | build ok; no tests |
| `x/math/nn/types` | 3/1 | 216 |  | [[nn]] | build ok; test: ok (4 pass) |

### [[learn]] — L2 numeric

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/learn` | 3/14 | 704 |  | [[learn]] | build ok; test: ok (26 pass, 1 skip) |
| `x/math/learn/datasets/mnist` | 1/0 | 94 |  | [[learn]] | build ok; no tests |

### [[tflite]] — L4 IO

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/marshaller/tflite` | 2/1 | 596 | `linux` | [[tflite]] | not in darwin list (`linux`); not built |

### [[filters]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/filter` | 1/0 | 21 |  | [[filters]] | build ok; no tests |
| `x/math/filter/agc` | 1/0 | 58 |  | [[filters]] | build ok; no tests |
| `x/math/filter/ahrs` | 3/1 | 596 |  | [[filters]] | build ok; test: ok (9 pass) |
| `x/math/filter/ekalman` | 1/1 | 505 |  | [[filters]] | build ok; test: ok (6 pass) |
| `x/math/filter/fir` | 1/1 | 259 |  | [[filters]] | build ok; test: ok (8 pass) |
| `x/math/filter/gaussian` | 1/0 | 139 |  | [[filters]] | build ok; no tests |
| `x/math/filter/iir` | 1/1 | 283 |  | [[filters]] | build ok; test: ok (13 pass) |
| `x/math/filter/kalman` | 1/1 | 384 |  | [[filters]] | build ok; test: ok (6 pass) |
| `x/math/filter/ma` | 1/1 | 129 |  | [[filters]] | build ok; test: ok (8 pass) |
| `x/math/filter/median` | 1/0 | 101 |  | [[filters]] | build ok; no tests |
| `x/math/filter/pll` | 1/1 | 109 |  | [[filters]] | build ok; test: FAIL (9 pass, 2 fail) |
| `x/math/filter/savgol` | 1/0 | 200 |  | [[filters]] | build ok; no tests |
| `x/math/filter/slam` | 2/1 | 753 |  | [[filters]] | build ok; test: ok (12 pass) |

### [[dsp-core]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/dsp` | 11/10 | 4751 |  | [[dsp-core]] | build ok; test: ok (71 pass) |

### [[modulation]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/dsp/modulation` | 1/0 | 110 |  | [[modulation]] | build ok; no tests |
| `x/math/dsp/modulation/adaptive` | 1/1 | 98 |  | [[modulation]] | build ok; test: ok (4 pass) |
| `x/math/dsp/modulation/dsss` | 4/1 | 1043 |  | [[modulation]] | build ok; test: FAIL (1 pass, 3 fail) |
| `x/math/dsp/modulation/internal` | 14/0 | 1691 |  | [[modulation]] | build ok; no tests |
| `x/math/dsp/modulation/iq` | 3/1 | 530 |  | [[modulation]] | build ok; test: ok (13 pass) |
| `x/math/dsp/modulation/mask` | 6/1 | 1329 |  | [[modulation]] | build ok; test: FAIL (4 pass, 2 fail) |
| `x/math/dsp/modulation/mfsk` | 5/3 | 1039 |  | [[modulation]] | build ok; test: FAIL (6 pass, 1 fail) |
| `x/math/dsp/modulation/mpsk` | 5/6 | 1497 |  | [[modulation]] | build ok; test: FAIL (9 pass, 3 fail) |
| `x/math/dsp/modulation/qam` | 5/5 | 1392 |  | [[modulation]] | build ok; test: FAIL (9 pass, 4 fail) |
| `x/math/dsp/modulation/tc` | 4/5 | 844 |  | [[modulation]] | build ok; vet: FAIL stale test API; test: FAIL build |
| `x/math/dsp/modulation/test_utils` | 3/0 | 423 |  | [[modulation]] | build ok; no tests |

### [[line-coding-and-fec]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/dsp/coding/ami` | 2/0 | 247 |  | [[line-coding-and-fec]] | build ok; no tests |
| `x/math/dsp/coding/b8zs` | 2/0 | 250 |  | [[line-coding-and-fec]] | build ok; no tests |
| `x/math/dsp/coding/biphase_i` | 2/0 | 308 |  | [[line-coding-and-fec]] | build ok; no tests |
| `x/math/dsp/coding/biphase_m` | 2/0 | 321 |  | [[line-coding-and-fec]] | build ok; no tests |
| `x/math/dsp/coding/crc` | 2/3 | 559 |  | [[line-coding-and-fec]] | build ok; test: FAIL (16 pass, 5 fail) |
| `x/math/dsp/coding/eight10b` | 3/0 | 366 |  | [[line-coding-and-fec]] | build ok; no tests |
| `x/math/dsp/coding/four5b` | 2/1 | 324 |  | [[line-coding-and-fec]] | build ok; test: FAIL (0 pass, 1 fail) |
| `x/math/dsp/coding/gray` | 2/1 | 290 |  | [[line-coding-and-fec]] | build ok; test: ok (5 pass) |
| `x/math/dsp/coding/hamming` | 2/4 | 349 |  | [[line-coding-and-fec]] | build ok; test: ok (20 pass) |
| `x/math/dsp/coding/hdb3` | 2/0 | 262 |  | [[line-coding-and-fec]] | build ok; no tests |
| `x/math/dsp/coding/lfsr` | 2/0 | 311 |  | [[line-coding-and-fec]] | build ok; no tests |
| `x/math/dsp/coding/manchester` | 3/1 | 611 |  | [[line-coding-and-fec]] | build ok; test: ok (15 pass) |
| `x/math/dsp/coding/miller` | 2/0 | 330 |  | [[line-coding-and-fec]] | build ok; no tests |
| `x/math/dsp/coding/nrzi` | 2/1 | 242 |  | [[line-coding-and-fec]] | build ok; test: ok (2 pass) |
| `x/math/dsp/coding/ppm` | 2/0 | 246 |  | [[line-coding-and-fec]] | build ok; no tests |
| `x/math/dsp/coding/rll` | 2/0 | 226 |  | [[line-coding-and-fec]] | build ok; no tests |
| `x/math/dsp/coding/rs` | 4/5 | 761 |  | [[line-coding-and-fec]] | build ok; test: ok (36 pass) |

### [[peg-protocol-parser]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/protocol/peg` | 9/10 | 3085 |  | [[peg-protocol-parser]] | build ok; test: FAIL (58 pass, 5 fail) |

### [[kinematics]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/control/kinematics/joints/dh` | 2/1 | 350 |  | [[kinematics]] | build ok; test: ok (7 pass) |
| `x/math/control/kinematics/joints/planar` | 3/2 | 319 |  | [[kinematics]] | build ok; test: ok (9 pass) |
| `x/math/control/kinematics/rigidbody` | 2/2 | 243 |  | [[kinematics]] | build ok; test: ok (3 pass) |
| `x/math/control/kinematics/thrusters` | 6/1 | 543 |  | [[kinematics]] | build ok; test: ok (3 pass) |
| `x/math/control/kinematics/types` | 1/0 | 85 |  | [[kinematics]] | build ok; no tests |
| `x/math/control/kinematics/wheels` | 1/0 | 4 |  | [[kinematics]] | build ok; no tests |
| `x/math/control/kinematics/wheels/differential` | 2/0 | 137 |  | [[kinematics]] | build ok; no tests |
| `x/math/control/kinematics/wheels/internal/rigid` | 1/0 | 47 |  | [[kinematics]] | build ok; no tests |
| `x/math/control/kinematics/wheels/mecanum` | 2/1 | 166 |  | [[kinematics]] | build ok; test: ok (3 pass) |
| `x/math/control/kinematics/wheels/steer4` | 2/0 | 195 |  | [[kinematics]] | build ok; no tests |
| `x/math/control/kinematics/wheels/steer4dual` | 2/0 | 195 |  | [[kinematics]] | build ok; no tests |
| `x/math/control/kinematics/wheels/steer6` | 2/0 | 219 |  | [[kinematics]] | build ok; no tests |

### [[motion-and-gait]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/control/motion` | 1/0 | 196 |  | [[motion-and-gait]] | build ok; no tests |
| `x/math/control/motion/gait/rigidbody` | 1/0 | 91 |  | [[motion-and-gait]] | build ok; no tests |
| `x/math/control/motion/gait/support` | 1/0 | 420 |  | [[motion-and-gait]] | build ok; no tests |
| `x/math/control/motion/gait/types` | 1/0 | 196 |  | [[motion-and-gait]] | build ok; no tests |
| `x/math/control/motion/rigidbody` | 9/2 | 1559 |  | [[motion-and-gait]] | build ok; test: ok (10 pass) |

### [[pid-control]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/control/pid` | 2/1 | 88 |  | [[pid-control]] | build ok; test: ok (8 pass) |

### [[graph-and-planning]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/graph` | 21/13 | 6588 |  | [[graph-and-planning]] | build ok; test: ok (87 pass) |

### [[grid-maps]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/grid` | 7/3 | 1844 |  | [[grid-maps]] | build ok; test: ok (64 pass) |

### [[interpolation]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/interpolation` | 8/5 | 950 |  | [[interpolation]] | build ok; test: ok (27 pass) |

### [[gps-geodesy]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/gps` | 3/2 | 291 |  | [[gps-geodesy]] | build ok; test: FAIL (16 pass, 5 fail) |

### [[colorscience]] — L3 algorithms

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/math/colorscience` | 16/2 | 3761 |  | [[colorscience]] | build ok; test: FAIL (28 pass, 3 fail) |

### [[pipeline-and-chain]] — L4 composition

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/chain` | 4/0 | 390 |  | [[pipeline-and-chain]] | build ok; no tests |
| `x/pipeline` | 3/3 | 252 |  | [[pipeline-and-chain]] | build ok; test: ok (17 pass) |
| `x/pipeline/example.processor` | 1/0 | 45 |  | [[pipeline-and-chain]] | build ok; no tests |
| `x/pipeline/example.processor.embed` | 2/0 | 71 |  | [[pipeline-and-chain]] | build ok; no tests |
| `x/pipeline/example.step` | 2/0 | 81 |  | [[pipeline-and-chain]] | build ok; no tests |
| `x/pipeline/steps` | 7/7 | 998 |  | [[pipeline-and-chain]] | build ok; vet: FAIL test names, unkeyed; test: FAIL build |
| `x/pipeline/steps/fps` | 2/0 | 122 |  | [[pipeline-and-chain]] | build ok; no tests |

### [[store-and-plugin]] — L4 composition

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/plugin` | 3/3 | 380 |  | [[store-and-plugin]] | build ok; test: FAIL (8 pass, 1 fail) |
| `x/store` | 4/1 | 769 |  | [[store-and-plugin]] | build ok; vet: FAIL test names; test: FAIL build |

### [[marshallers]] — L4 IO

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/marshaller` | 0/3 | 0 |  | [[marshallers]] | test-only pkg; test: ok (6 pass) |
| `x/marshaller/csv` | 3/1 | 1338 |  | [[marshallers]] | build ok; test: ok (48 pass) |
| `x/marshaller/gob` | 4/2 | 537 |  | [[marshallers]] | build ok; test: ok (4 pass) |
| `x/marshaller/graph` | 21/3 | 3750 |  | [[marshallers]] | build ok; test: ok (11 pass) |
| `x/marshaller/json` | 3/2 | 1551 |  | [[marshallers]] | build ok; test: ok (8 pass) |
| `x/marshaller/proto` | 4/2 | 686 |  | [[marshallers]] | build ok; test: ok (4 pass) |
| `x/marshaller/storage` | 9/3 | 1845 | `!tinygo`; `!tinygo && !unix && !windows`; `!tinygo && unix`; `!tinygo && windows`; `tinygo` | [[marshallers]] | build ok; test: ok (18 pass) |
| `x/marshaller/text` | 1/1 | 287 |  | [[marshallers]] | build ok; test: ok (2 pass) |
| `x/marshaller/types` | 4/1 | 973 |  | [[marshallers]] | build ok; test: ok (5 pass) |
| `x/marshaller/yaml` | 3/2 | 1442 |  | [[marshallers]] | build ok; test: ok (8 pass) |

### [[camera-io]] — L4 IO

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/marshaller/gocv` | 19/4 | 3702 |  | [[camera-io]] | not built (OpenCV missing) |
| `x/marshaller/v4l` | 7/1 | 2194 | `linux` | [[camera-io]] | not in darwin list (`linux`); not built |

### [[audio-io]] — L4 IO

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/marshaller/audio` | 17/2 | 7022 | `!wasm` | [[audio-io]] | build ok; test: ok (21 pass) |

### [[vision]] — L4 IO

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/vision/detect/blobs` | 1/1 | 230 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/detect/corners` | 1/1 | 312 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/detect/features` | 1/1 | 209 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/extract/blobs` | 1/0 | 191 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/extract/features` | 1/1 | 188 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/match/corners` | 1/0 | 52 |  | [[vision]] | build ok; no tests |
| `x/vision/match/features` | 1/0 | 87 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/odometry/phase` | 1/1 | 262 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/odometry/pose` | 1/0 | 205 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/stereo/depth` | 2/0 | 256 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/stereo/matching` | 1/0 | 120 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/transform/affine` | 1/0 | 122 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/transform/background` | 1/0 | 218 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/transform/color` | 1/0 | 112 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/transform/crop` | 1/0 | 123 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/transform/format` | 4/0 | 267 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/transform/histogram` | 1/0 | 295 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/transform/rectify` | 1/0 | 126 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/transform/undistort` | 1/0 | 144 |  | [[vision]] | not built (OpenCV missing) |
| `x/vision/types` | 4/0 | 477 |  | [[vision]] | not built (OpenCV missing) |

### [[device-io]] — L5 devices

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/devices` | 22/0 | 1841 | `!tinygo && !linux`; `!tinygo && !linux && !windows`; `!tinygo && linux`; `!tinygo && windows`; `tinygo` | [[device-io]] | build ok; no tests |
| `x/devices/pcf8574` | 1/0 | 84 |  | [[device-io]] | build ok; no tests |
| `x/devices/tca9548a` | 2/1 | 184 |  | [[device-io]] | build ok; no tests |
| `x/devices/xiao` | 2/0 | 318 | `!sam \|\| !xiao`; `sam && xiao` | [[device-io]] | build ok; no tests |

### [[sensors]] — L5 devices

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/devices/adns3080` | 1/0 | 277 |  | [[sensors]] | build ok; no tests |
| `x/devices/as734x` | 9/0 | 987 |  | [[sensors]] | build ok; no tests |
| `x/devices/cr30` | 2/0 | 877 |  | [[sensors]] | build ok; no tests |
| `x/devices/encoder` | 4/1 | 232 | `!tinygo && !linux`; `!tinygo && linux`; `tinygo` | [[sensors]] | FAIL darwin gap; test: FAIL build |
| `x/devices/mpu6050` | 1/0 | 166 |  | [[sensors]] | build ok; no tests |
| `x/devices/vl53l0x` | 2/0 | 423 |  | [[sensors]] | build ok; no tests |

### [[actuators]] — L5 devices

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/devices/motor` | 3/0 | 568 |  | [[actuators]] | FAIL darwin gap; test: FAIL build |
| `x/devices/pca9685` | 1/0 | 146 |  | [[actuators]] | build ok; no tests |
| `x/devices/servo` | 4/0 | 264 | `!tinygo`; `tinygo` | [[actuators]] | build ok; no tests |

### [[lidar]] — L5 devices

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/devices/lidar` | 1/0 | 36 |  | [[lidar]] | build ok; no tests |
| `x/devices/lidar/ld06` | 2/1 | 602 |  | [[lidar]] | build ok; test: FAIL (1 pass, 1 fail) |
| `x/devices/lidar/vl53l0x` | 1/0 | 243 |  | [[lidar]] | build ok; no tests |
| `x/devices/lidar/xwpftb` | 2/1 | 665 |  | [[lidar]] | build ok; test: FAIL (1 pass, 1 fail) |

### [[crsf-elrs]] — L5 devices

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `cmd/elrs-rx` | 5/0 | 288 | `esp32`; `linux \|\| windows`; `rp2040`; `sam && xiao` | [[crsf-elrs]] | FAIL darwin gap; test: FAIL build |
| `x/devices/crsf` | 11/2 | 1546 |  | [[crsf-elrs]] | build ok; test: ok (38 pass) |
| `x/devices/crsf/rx` | 2/1 | 275 |  | [[crsf-elrs]] | build ok; test: ok (4 pass) |
| `x/devices/crsf/tx` | 1/0 | 220 |  | [[crsf-elrs]] | build ok; no tests |

### [[nmea]] — L5 devices

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `x/devices/nmea` | 3/1 | 555 |  | [[nmea]] | build ok; test: FAIL (5 pass, 1 fail) |

### [[audio-modem-cli]] — L6 apps

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `cmd/audio` | 22/0 | 6091 |  | [[audio-modem-cli]] | build ok; no tests |

### [[apps]] — L6 apps

| Package | src/test files | LOC (non-test) | build tags (first `//go:build` per file) | Module note | darwin/arm64 status (baseline §5) |
|---|---|---|---|---|---|
| `cmd/calibrate` | 7/0 | 1906 |  | [[apps]] | not built (OpenCV missing) |
| `cmd/cr30` | 1/0 | 1035 |  | [[apps]] | not built (OpenCV missing) |
| `cmd/display` | 1/0 | 226 |  | [[apps]] | not built (OpenCV missing) |
| `cmd/display/destination` | 6/0 | 615 |  | [[apps]] | not built (OpenCV missing) |
| `cmd/display/source` | 7/0 | 1150 |  | [[apps]] | not built (OpenCV missing) |
| `cmd/manipulator` | 1/0 | 473 |  | [[apps]] / [[kinematics]] | build ok; no tests |
| `cmd/monitor` | 3/0 | 685 |  | [[apps]] | FAIL darwin gap; test: FAIL build |
| `cmd/spectrometer` | 1/0 | 266 |  | [[apps]] | not built (OpenCV missing); also darwin gap |
| `cmd/spectrometer/cameras` | 1/0 | 14 |  | [[apps]] | not built (OpenCV missing) |
| `cmd/spectrometer/internal/acquirer` | 2/0 | 380 |  | [[apps]] | FAIL darwin gap; test: FAIL build |
| `cmd/spectrometer/internal/calibrate` | 3/0 | 987 |  | [[apps]] | not built (OpenCV missing) |
| `cmd/spectrometer/internal/config` | 4/0 | 831 |  | [[apps]] | not built (OpenCV missing) |
| `cmd/spectrometer/internal/render` | 3/0 | 542 |  | [[apps]] | not built (OpenCV missing) |
| `cmd/spectrometer/measure` | 2/0 | 811 |  | [[apps]] | not built (OpenCV missing) |
| `cmd/spectrometer/ports` | 1/0 | 28 |  | [[apps]] | FAIL darwin gap; test: FAIL build |
| `drivers/lidar` | 7/0 | 883 | `!rp2040 && cgo`; `ld06 && rp2040`; `linux \|\| windows`; `rp2040`; `xwpftb && rp2040` | [[apps]] / [[lidar]] | not built (OpenCV missing) |
| `drivers/manipulator` | 5/0 | 552 | `sam && xiao`; `sam && xiao && !planar && !dh`; `sam && xiao && dh && !planar`; `sam && xiao && planar && !dh` | [[apps]] / [[kinematics]] | not in darwin list (TinyGo `sam && xiao`); not built |

## Observations

- **Unused packages:** `x/crypto` (ed25519 + base58 keys) and `x/math/primitive/bytes` have no importer in the tree (reverse-import scan of the `go list` output).
- **Library-only algorithm families.** No `cmd/`, `drivers/` or device package imports [[filters]] (`ahrs`, `kalman`, `ekalman`, `slam`, `iir`, …), [[kinematics]] (`dh`, `planar`, wheels), [[nn]]/[[learn]], [[gps-geodesy]] or [[nmea]]; the only in-tree users of a filter are `x/math/filter/slam` (uses `ekalman`, `x/math/grid`) and `x/math/dsp/modulation/internal` (uses `fir`). [[pid-control]] is used inside `x/devices/motor`, both lidar decoders and `x/math/control/motion/rigidbody`. See [[data-flow]].
- **Stray binary:** `cmd/manipulator/manipulator` is a committed 9.3 MB ELF x86-64 executable (`file`, `git ls-files`).
- **Per-package design docs:** many packages carry their own `SPEC.md`/`PLAN*.md`/`DESIGN.md` (e.g. `x/marshaller/PLAN_*.md`, `x/marshaller/gocv/{DESIGN,SPEC,SOLID_PLAN}.md`, `cmd/spectrometer/{PLAN,REVIEW,SPEC,WORK}.md`); treated as untrusted claims, not reconciled here.

## TODOs

- [ ] Delete the committed binary `cmd/manipulator/manipulator` and ignore build outputs — `git ls-files cmd/manipulator/manipulator` #p3
- [ ] Decide the fate of unused `x/crypto` and `x/math/primitive/bytes` (delete, or wire into dndm auth per the owner's zone/auth decisions) — reverse-import scan #p3
- [ ] Give `x/math/shape` its own family or move it under `x/math/tensor/types` so the family graph has no cycles — [[overview]] #p3
