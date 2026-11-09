---
type: module
group: "IO, camera, vision"
status: partial
tags: [easyrobot, crypto, deleted]
workstream: [W1]
packages: [x/crypto]
test_result: none
evidence: "x/store, x/plugin, x/options, x/logger and internal/concurrency deleted in ER-cleanup (owner-approved); crypto probe tmp/easyrobot/io/cryptoprobe → zero-length keys (2026-11-09)"
---
# store-and-plugin — what remains: `x/crypto`

**History.** This note covered the support packages of the deprecated `x/pipeline`: `x/store` (a typed-key value bag), `x/plugin` (a name → builder registry), `x/options` (`func(interface{})` options), `x/logger` (a zerolog wrapper) and `internal/concurrency` (a `go f()` shim). They were deleted in [[ER-cleanup]] (owner: "easyrobot deletions - ok."). The deletion followed `x/pipeline/DEPRECATED.md` ("Pipeline, Plugin, Store will be replaced by dndm package and associated patterns"), after a grep found no importer outside the deleted cluster. Their defects (FQDN code collision, ignored patterns, a registry deadlock, stale tests) went with them.

**What remains:** `x/crypto` (55 lines, no tests, no importers): ed25519 `New`, `FromSeed`, `FromReader`, base58 `String`.

## Algorithms

<!-- vault:algorithms h=f56d515d3b10 -->

No algorithm notes belong to this module.

<!-- /vault:algorithms -->

## Defects

- **`crypto.FromSeed` returns empty keys:** `var pub PubKey; var priv PrivKey` are nil slices, so `copy(pub[:], pb)` copies 0 bytes (`x/crypto/crypto.go:26-31`); `New()` discards the error too (:17-19). `FromReader` uses single `Read` calls instead of `io.ReadFull` (:39-44). Probe: `crypto.New()` → `len(pub)=0 len(priv)=0`.

## API notes (ease of use, composability)

- `x/crypto` looks like an identity/signing helper for dndm-style peers but is unused and broken. Either fix it (a 4-line change) or delete it together with a decision on dndm's identity layer ([[dndm-and-protobuf]]).

## TODOs

Module-level items only.

- [ ] Fix `FromSeed` (allocate or return the generated keys directly) and propagate the error from `New`, or delete `x/crypto` — x/crypto/crypto.go:17-31 #p1 #backlog
