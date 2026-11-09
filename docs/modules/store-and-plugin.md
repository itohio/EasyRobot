---
type: module
group: "IO, camera, vision"
status: partial
tags: [easyrobot, store, plugin, options, logger, crypto, deprecated]
workstream: [W1]
packages: [x/store, x/plugin, x/options, x/logger, x/crypto, internal/concurrency]
test_result: fail
evidence: "go test -count=1 ./x/plugin → FAIL TestDefaultOptions (8 pass, 1 fail); ./x/store → build fail (vet malformed names), -vet=off → ok 2 PASS; crypto probe tmp/easyrobot/io/cryptoprobe → zero-length keys (2026-11-09)"
---
# store-and-plugin — `x/store`, `x/plugin`, `x/options`, `x/logger`, `x/crypto`, `internal/`

**Purpose:** support packages of the deprecated pipeline: a typed-key value bag (`store.Store`), a name → builder registry with option structs (`plugin`), functional options over `interface{}` (`options`), a zerolog wrapper with a no-op `logless` build (`logger`), ed25519 key helpers (`crypto`), and a goroutine submit shim (`internal/concurrency`). `x/store` and `x/plugin` carry `DEPRECATED.md` (replaced by dndm). Only `x/pipeline` imports `store`, `plugin`, `logger`, `options` and `internal/concurrency`; **nothing imports `x/crypto`**.

**Packages:**

| Package | LOC (non-test/test) | Notes |
|---|---|---|
| `x/store` | 769 / 377 | `Store` interface (`store.go:45-109`), map-backed `store` (no mutex), typed getters/setters (`store.props.go`), `FQDNType` uint16 key codes (`constants.go`), stringer output (`fqdntype_string.go`, `//go:generate stringer` at `constants.go:3`) |
| `x/plugin` | 380 / 327 | `Registry` (`registry.go`), `Options` + `With*` funcs (`options.go`), reflection struct⇄map (`marshal.go`) |
| `x/options` | 10 / 0 | `type Option func(cfg interface{})`, `ApplyOptions` |
| `x/logger` | 42 / 0 | `// +build !logless` zerolog console logger with caller; `// +build logless` no-op `EmptyLog` (old-style build tags) |
| `x/crypto` | 55 / 0 | ed25519 `New`, `FromSeed`, `FromReader`, base58 `String` |
| `internal/concurrency` | 15 / 0 | `Submit(f)` → `go f()` (both `ants` and `!ants`; ants variant is a FIXME) |

## Public API and composition

`store.New()/NewWithName()` → `Set(FQDNType, Value)`, `Get`, typed `Int64(TIMESTAMP)`, `Image(fqdn)`, `Clone(pattern)`, `Close(pattern)`, `ForEach(pattern, cb)`; values may implement `ValueCloner`/`ValueCloser`. `plugin.Global.Register(name, Builder)`, `New(name, opts...)`; `plugin.DefaultOptions()`, `WithName/WithBufferSize/WithBlocking/WithEnable/WithEOSExit/WithIgnoreErrors/WithClose/WithMapping(map)`; `plugin.MarshalOptions(struct) map[string]interface{}` (reflection, `opts` struct tags). Pipeline steps embed `plugin.Options` and receive `options.Option`s.

## Algorithms

No non-trivial algorithms; findings are recorded here rather than in algorithm notes.

<!-- vault:algorithms h=f56d515d3b10 -->

No algorithm notes belong to this module.

<!-- /vault:algorithms -->

## Build targets and dependencies

All pure Go. `logger` pulls zerolog unless built with `-tags logless` (used by `drivers/manipulator` TinyGo flash command per baseline). `plugin` and `store` rely on reflection and `interface{}` maps — unsuitable for MCUs.

## Tests and evidence

- `go test -count=1 ./x/plugin` → FAIL `TestDefaultOptions` — `options_test.go:18`: got `{false true 1 false false true}`, want BufferSize 0. Cause: `DefaultOptions` sets `BufferSize: 1` (`x/plugin/options.go:16-22`) while the test's expected struct (`options_test.go:11-14`) omits it — the test is stale, not the code [INFERENCE: the 1 is intentional, the same default breaks `x/pipeline/steps` `TestOptions`].
- `go test -count=1 ./x/store` → build failed: vet `tests` analyzer rejects `Teststore_Name` … `Teststore_Dump` (`store_test.go:54, 81, 110, 150, 189, 228, 257, 291, 320, 350`) — lower-case after `Test`. With `-vet=off`: ok, **2 PASS**: the 10 malformed functions are not recognised as tests by `go test`, so they never ran even before vet started enforcing this.
- `x/crypto` probe (`~/src/tmp/easyrobot/io/cryptoprobe/main.go`): `crypto.New()` → `len(pub)=0 len(priv)=0 pub=""`.

## Defects

- **`crypto.FromSeed` returns empty keys:** `var pub PubKey; var priv PrivKey` are nil slices (`PubKey`/`PrivKey` are slice types), so `copy(pub[:], pb)` copies 0 bytes (`x/crypto/crypto.go:26-31`); `New()` discards the error too (:17-19). `FromReader` uses single `Read` calls instead of `io.ReadFull` (:39-44).
- **FQDN code collision:** `AUDIO = 0x0200` and `ALGORITHMS = 0x0200` (`x/store/constants.go:26, 35`), so `KEY_POINTS == AUDIO+1`, etc.; also `NONE == BASIC == 0`.
- **Patterns ignored:** `store.ForEach` ignores `fqdnPattern` (`store.go:190-194`), so `Clone(pattern)`/`Close(pattern)`/`Dump(pattern)` act on every key.
- `store` map has no mutex while steps share stores across goroutines [INFERENCE: copies are made at step boundaries in some steps].
- `plugin.Registry.ForEach` ignores its `pattern` (`registry.go:59-65`); `Registry.New` calls the builder while holding the read lock (`registry.go:50-57`) — a builder that registers plugins deadlocks.

## API notes (ease of use, composability)

- `options.Option func(interface{})` is maximally generic and minimally safe: an option for the wrong struct silently does nothing (every `With*` does `if opt, ok := o.(*Options)`). Typed generic options (`type Option[T any] func(*T)`) give the same flexibility with compile-time checks.
- `store.Store` keys are global `uint16` codes in one const block — collisions (above) and no namespacing; values are `interface{}` with ~40 typed getters. dndm's typed topics/protobuf messages replace this cleanly ([[dndm-and-protobuf]]).
- `x/crypto` looks like an identity/signing helper for dndm-style peers but is unused and broken; either fix (4-line change) or delete.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Fix `FromSeed` (allocate or return the generated keys directly) and propagate the error from `New` — x/crypto/crypto.go:17-31 #p1 #backlog
- [ ] Update stale `TestDefaultOptions` expectation (BufferSize 1) — x/plugin/options_test.go:11-14 #p2
- [ ] Rename `Teststore_*` tests so they run — x/store/store_test.go:54-350 #p2
- [ ] Resolve `AUDIO`/`ALGORITHMS` code collision — x/store/constants.go:26,35 #p2
- [ ] Honour `fqdnPattern` in `ForEach` — x/store/store.go:190-194 #p2
- [ ] Decide deletion of the deprecated cluster (`x/store`, `x/plugin`, `x/pipeline`, `x/options`, `x/logger`, `internal/concurrency`) after dndm migration — x/store/DEPRECATED.md #p3

## Inferred objectives (proposals)

- **Proposal:** delete the deprecated cluster in one change once dndm covers the remaining uses (none outside `x/pipeline` today); keep `x/logger`'s `logless` idea for TinyGo builds if anything else needs it.
- **Proposal:** if signed feature streams or peer identity are wanted (K230 ↔ ESP32 ↔ PC), fix `x/crypto` and move it next to dndm.
