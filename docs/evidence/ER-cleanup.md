---
type: evidence
status: review
tags: [easyrobot, evidence, cleanup]
task: ER-cleanup
date: 2026-11-09
source_revision: db9eb8f2a50e0c84dc7f32a9109148dd13516d9e
---
# ER-cleanup — owner-approved deletions

**Result:** all three approved deletions are done in the working tree. They were made together and are split into three commits (§5), each gated on its own staged tree, `make docs-check` included. Uncommitted: local commits wait for review and Leader's go. Owner approval: "easyrobot deletions - ok." (roster task `ER-cleanup`). Nothing outside the deleted code imported it; the builds and tests that passed before still pass. The vault is updated and regenerated per commit.

## 1. Wireline placeholders (AMI, B8ZS, HDB3, RLL, PPM)

- **Deleted:** `x/math/dsp/coding/{ami,b8zs,hdb3,rll,ppm}/` — 10 files, 1 231 lines, no tests. The probe in [[line-coding-and-fec]] had them as identity, lossy, panicking, identity and lossy.
- **Importer check:** grep over all `.go` files for the import paths, plus the bare names as quoted strings or whole words in `cmd/` and `x/math/dsp/coding` (to catch CLI encoding names, registries and switch tables): none outside the deleted directories.
- **Gates:** `go build -o /dev/null ./cmd/audio` ok. `go test -count=1 ./x/math/dsp/coding/...`: gray, hamming, manchester, nrzi and rs ok. `crc` (5 tests) and `four5b` (`TestFourB5B_Roundtrip`) fail exactly as in the 2026-11-09 baseline (stale CRC-8 expectations, nibble swap — [[ER-023-coding-fixes|ER-023]]). Neither imports the deleted code.
- **Vault:** 5 algorithm notes deleted (`ami-bipolar-line-code`, `b8zs-line-code`, `hdb3-line-code`, `rll-run-length-limited-code`, `ppm-pulse-position-code`). [[line-coding-and-fec]] packages, table, streaming summary and TODOs updated; the "decide fate" TODO is ticked. The HDB3 item is dropped from [[ER-023-coding-fixes|ER-023]].
- **Not touched:** `x/math/dsp/coding/SPEC.md` and `PLAN.md` are already stale design docs (old `ecc.*` API); their rewrite is an existing p3 TODO.

## 2. Committed ELF and the flash `go:generate`

- **Deleted from the working tree:** `cmd/manipulator/manipulator`, a 9.3 MB x86-64 ELF tracked since `1e9921d` ("chore: port manipulator control."). The commit will record the deletion; the blob stays in history (no rewrite).
- **`.gitignore`:** one exact line per command directory, `/cmd/<dir>/<dir>`, for audio, calibrate, cr30, display, elrs-rx, manipulator, monitor and spectrometer (what `go build` inside `cmd/<dir>` writes). No source file matches.
- **`drivers/manipulator/main.go`:** the `//go:generate tinygo flash -target=xiao -tags logless` line removed, so `make gen` / `go generate ./...` can no longer flash a board.
- **`Makefile`:** new `flash-manipulator` target, listed in `make help`. It echoes the exact command, asks `Flash connected XIAO? [y/N]`, and runs `tinygo flash -target=xiao ./drivers/manipulator` only on `y`/`Y`. Not run. `make -n flash-manipulator` shows the echo, the prompt and the guarded call. The no-op `-tags logless` (no EasyRobot or dndm file reads it) is dropped here, in `drivers/manipulator/main.go`'s usage header and in `drivers/manipulator/SPEC.md`.
- **Check:** `grep -rn 'go:generate' --include=*.go . | grep -i flash` → nothing.
- **Vault:** [[apps]], [[build-targets]], [[kinematics]] TODOs ticked; [[ER-026-build-green|ER-026]] marks held decision #13 resolved.

## 3. Deprecated cluster: `x/pipeline`, `x/plugin`, `x/store`, `x/options`, `x/logger`, `internal/concurrency`

- **Precondition 1 — no outside importers:** grep for `itohio/EasyRobot/<pkg>` over all `.go` files, excluding matches inside the cluster: empty for all six. `options` and `logger` were imported only by `plugin`/`store`/`pipeline`; `internal/concurrency` by nothing. `x/chain` (used by `cmd/audio`, the coders and vision) does not depend on the cluster. No `.go`, Makefile or YAML file outside `docs/` names the deleted paths.
- **Precondition 2 — dndm covers the transport:** `x/pipeline/DEPRECATED.md` says "Pipeline, Plugin, Store will be replaced by dndm package and associated patterns". The live device apps (`drivers/lidar`, `drivers/manipulator`, `cmd/manipulator`, `cmd/display/destination`) already move data over dndm ([[dndm-and-protobuf]]); in-process composition stays with `x/chain`. The step graph had no users.
- **Deleted:** 47 files, 6 314 lines, including `x/pipeline/steps`, `steps/fps`, the three examples, `SPEC.md` and the three `DEPRECATED.md` files. The `ants`/`logless` build-tag variants went with them; `x/iter` and `x/workers` (`use_mt`) are untouched. Remaining mentions fixed: `DESIGN.md` logger section, [[multi-source-frame-sync]].
- **Gates:** before the change, only `cmd/audio` of the `cmd/*` mains built offline on darwin (the others fail on OpenCV or darwin serial, baseline F2/F3); after it, `cmd/audio` still builds. `go vet ./x/chain/... ./cmd/audio` clean.
- **`go.mod`:** `github.com/rs/zerolog` was required only by `x/logger`; `go mod why -m github.com/rs/zerolog` → "main module does not need module". Its require line and two `go.sum` lines are removed (other stale `go.sum` lines are left for a future online `go mod tidy`; offline tidy cannot run). On top of `db9eb8f` (protobuf bump) with dndm's committed HEAD: `go mod verify` → all modules verified; read-only `go list ./...` ok; `go build -o /dev/null ./cmd/audio` ok.
- **Defects removed with the code:** FQDN code collision, ignored patterns in `store.ForEach` and `Registry.ForEach`, a builder deadlock under the registry lock, a broken timestamp sync in `steps/sync.go`, stale `TestDefaultOptions`, and the malformed `Test*` names in `x/store` and `x/pipeline/steps` that vet rejected (baseline F5).
- **Vault and README:** `pipeline-step-graph` deleted; [[pipeline-and-chain]] now covers `x/chain` only; [[store-and-plugin]] keeps only `x/crypto` (unused, broken key helpers — its TODO says fix or delete); [[overview]], [[packages]] ("Since the snapshot" paragraph; the `abb5c9c` rows stay as the record), [[source-inventory]], [[build-targets]] (the `// +build` TODO is now moot) and [[ER-026-build-green|ER-026]] (F5 gone) updated. `README.md` no longer advertises the plugin/store/pipeline frameworks or the zerolog logger; it points to `x/chain` and dndm.

**Review** (reviewer agent `CleanupReview`): verdict correct; no importer anywhere (incl. build-tagged and test files, `../dndm`); `.gitignore` rules match only build outputs (`git ls-files -ci --exclude-standard`); the flash recipe prompts with default no, also under dash. Its P3 findings (unused zerolog require, stale `// +build` TODO, [[source-inventory]], README, roadmap row) are fixed above, except the roadmap (below).

## 4. Vault and verification

- Each commit's staged tree is exported (dndm = committed HEAD), the vault is regenerated in it with the repo's vault tool, and the regenerated files (`todo.md`, `index.md`, module tables) are staged with that commit. `make docs-check` then runs on that tree (§5).
- Script: `tmp/easyrobot/commit/split.sh` (`DRY=1` uses a temporary index and leaves the real one untouched; `DRY=0 split.sh STEP` commits one step after its gates pass). Logs: `tmp/easyrobot/commit/logs/`.

## 5. Commits (after review and Leader's go)

Made together, split into three commits. They land after the five ER-017/018 commits (shared files: `kinematics.md` gets its ELF TODO in commit 2; `build-targets.md` and ER-026 get their flash line in commit 2 and the rest in commit 3; `packages.md` goes in commit 3).

| # | Commit | Contents | Gates on its own staged tree (dry run 2026-11-09) |
|---|---|---|---|
| 1 | `chore(coding): delete wireline placeholder coders [ER-cleanup]` | `x/math/dsp/coding/{ami,b8zs,hdb3,rll,ppm}` + the five notes, [[line-coding-and-fec]], [[ER-023-coding-fixes\|ER-023]], this report | `go build ./cmd/audio`, `go vet ./x/chain/... ./cmd/audio` ok; coding tests fail only in baseline `crc`/`four5b`; `make docs-check` **0 errors** |
| 2 | `chore(manipulator): drop committed ELF and flash go:generate [ER-cleanup]` | `cmd/manipulator/manipulator`, `.gitignore`, `drivers/manipulator/{main.go,SPEC.md}`, `Makefile`, [[apps]], [[kinematics]], [[build-targets]] (flash line), [[ER-026-build-green\|ER-026]] (decision #13) | no flash `go:generate`; `make -n flash-manipulator` ok; `go build ./cmd/audio` ok; `make docs-check` **0 errors** |
| 3 | `chore(x): delete deprecated pipeline/plugin/store cluster [ER-cleanup]` | the six packages, zerolog require (`go.mod`, `go.sum`), `README.md`, `DESIGN.md`, `pipeline-step-graph`, [[multi-source-frame-sync]], [[pipeline-and-chain]], [[store-and-plugin]], [[overview]], [[packages]], [[source-inventory]], [[build-targets]], ER-026 (F5) | `go mod verify`, read-only `go list ./...`, `go build ./cmd/audio`, `go vet` ok; `go build ./...` fails only on the known darwin baseline packages; `make docs-check` **0 errors** |

After commit 3 the staged tree equals the working tree for every file of both tasks; only other peers' files (K210, research notes) stay unstaged.

## 6. Leftovers

- `x/crypto`: unused and broken (`FromSeed` returns empty keys). Fix or delete — owner's call (not in this approval).
- `x/SPEC.md` still describes `pkg/core/logger` and the pipeline (stale design doc, already listed for rewrite in [[overview]]).
- [[roadmap]] (product's note, foreign to me) still lists the deprecated cluster as open work ("held #11, delete once dndm covers transport", row at `planning/roadmap.md:209`) — for product/Leader to close.
- **dndm protobuf bump:** dndm `HEAD` (`442f088`) requires `google.golang.org/protobuf v1.36.12`, which broke EasyRobot's root `go list ./...` with `-mod=readonly`. Leader's decision (option a): EasyRobot bumps protobuf too. Committed as `db9eb8f` `chore(deps): google.golang.org/protobuf v1.36.12 to match dndm [DND-R9]` (offline from the module cache; only the protobuf lines staged, so the zerolog removal stays in the working tree for commit 3 here). Staged-tree gate with dndm HEAD: `go mod verify` ok, `go list ./...` ok, `go build ./...` fails only on the known darwin baseline packages.
- The 9.3 MB blob remains in git history; removing it needs a history rewrite and force push (not proposed).
