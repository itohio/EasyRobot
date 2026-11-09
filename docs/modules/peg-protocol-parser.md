---
type: module
group: "Signal, codecs, audio"
status: partial
tags: [easyrobot, protocol, parser]
workstream: [W1, W4, W8]
packages: [x/math/protocol/peg]
test_result: fail
evidence: "go test -count=1 ./x/math/protocol/peg → FAIL 58 pass / 5 fail top-level (2026-11-09); go vet clean; probes tmp/easyrobot/coding/pegprobe (pegprobe_out.txt, pegdup_out.txt, -race)"
---
# peg-protocol-parser — `x/math/protocol/peg`

**Purpose:** describe binary packet layouts with a terse PEG-like pattern (`^55AA%L%{id:u,val:uu}%*f$`) and let a byte-stream reader decide after each byte whether to keep accumulating, drop, or emit a packet, with decoded fields. Used by `cmd/monitor` (serial/stream monitor with optional CRC checking, `cmd/monitor/main.go:154-205`). Natural fit for lidar frames ([[lidar]]), CRSF ([[crsf-elrs]]), the K230→ESP32 SPI feature stream framing ([[W4-k230-feature-stream]]), and audio-modem frames.

**Package:** one package, 3 085 LOC non-test (`ast.go` 641, `parser.go` 869, `expression.go` 742, `ops.go` 407, `program.go` 175, `state.go` 158, `fields.go` 66, `decision.go` 11, `test_helpers.go` 16), 2 821 LOC tests, plus `SPEC.md` (242) and `FEATURES.md` (1 282, per-feature test map). No build tags.

## Public API

- `Compile(pattern string) (*Program, error)` (`parser.go:25`). There is no `New(pattern, opts...)`/`WithMaxLen` despite SPEC.md.
- `(*Program).Decide(State) (Decision, error)` (`program.go:166`); `Program` embeds `*graph.GenericExpressionGraph[...]` so `Compute(nil, state)` also works (single root node wrapping the AST interpreter).
- `Decision`: `DecisionContinue`, `DecisionDrop`, `DecisionEmit` (`decision.go`) — SPEC.md's `DecisionKeep` does not exist.
- `State` interface (`state.go:4-29`): packet bytes, fields, declared/max length, decision, offset, `Clone/Merge` for choice backtracking; `NewDefaultState()`.
- Field model: `FieldSpec`, `FieldType` (u8…u64 LE/BE, i8…i64, f32/f64, varint u/s, Pascal/C/fixed strings), `FieldKindLength`; `DecodedField{Name, Offset, Type, Value interface{}}`.
- Pattern features (FEATURES.md): literals, alternation `(a,b)`, `*`, `*N`, skip-until `*?hex?`, offsets `#N/#±N`, lookahead groups `(#…)`, typed fields, named fields, length fields `L/ll/LL`, arrays `%N…`, stride arrays, star arrays, structs `%{…}`, arrays of structs, expressions `%(…)`, guards `@(…)`, end anchor `$`/`$N` (max length).

## Algorithms

<!-- vault:algorithms h=5f08f0fe284e -->

Counts: 0 tested · 0 implemented · 2 partial · 0 stub · 0 unverified (2 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[peg-expression-evaluator]] | partial | fail | 2 of 5 failures (impl bugs) |
| [[peg-packet-matcher]] | partial | fail | 3 of 5 failures (1 impl bug, 2 wrong tests); races, stale AST mutation, field duplication found by… |

<!-- /vault:algorithms -->

Trivial helpers kept here: `decodeField`/varint decoders (`ops.go:332-407`), `DefaultState` length clamping (`state.go:113-138`).

## Build targets and dependencies

- Pure Go; std (`fmt`, `strconv`, `strings`, `unicode`, `bytes`, `encoding/binary`, `math`) + `x/math/graph`.
- `test_helpers.go` is **not** a `_test.go` file: it imports `testing` and `github.com/stretchr/testify/require`, so every consumer (cmd/monitor, any firmware) links them.
- Heavy for MCUs: `interface{}` field values, `float64` expressions, `fmt.Errorf` per mismatch, state `Clone` per alternation branch, whole-packet re-parse per byte (O(n²) per packet). Works in principle under TinyGo [INFERENCE: not built], but not suited to per-byte use on ESP32 at high rates without incremental matching.

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 ./x/math/protocol/peg` (2026-11-09): FAIL — 58 pass / 5 fail: `TestLengthDrivenEmit` (impl: AST mutation leak), `TestLengthDrivenEmitWildcard` (test data inconsistent + multi-`*` semantics undefined), `TestOffsetJumpEdgeCases/offset jump beyond MaxLength` (test asserts Continue; impl Drop is right and `program_test.go:313-331` asserts Drop), `TestBitwiseOperations/bitwise XOR invert` and `TestExpressionRangeCondition/within range` (impl: base-type detection). Same five as baseline-2026-11-08.
- `go vet` clean.
- Probes (`tmp/easyrobot/coding/pegprobe`): reuse of one `Program` changes results (`^AA*2BB`: Emit then Drop), streaming leak (`^55AA*ll$` byte-wise → Continue at complete packet), 29 data races with 8 goroutines, fields triple on byte-wise feeding, un-anchored patterns never search, `#(expr)` unimplemented.

## API notes (ease of use, composability)

- Pattern language is compact and expressive (structs, arrays, expressions) — good for quick protocol sniffing; FEATURES.md is a useful spec-by-test map.
- **A compiled `Program` is mutable shared state** (wildcard counts written during eval) → not reusable across packets reliably, not goroutine-safe. Must be fixed before the multi-candidate loop in SPEC.md (several `State`s against one `Program`) is trustworthy.
- `Decide` re-parses from byte 0 and appends fields each call; callers must `ResetFields()` themselves (undocumented). An incremental API (`Feed(byte) Decision` keeping a resume point) would be both faster and harder to misuse.
- Start anchoring is implicit and `^` is ignored; resynchronisation (sliding start) is left to the caller — `cmd/monitor` implements it by spawning a new `State` per byte.
- Errors are swallowed into `DecisionDrop` (`program.go:31-34`) — no way to learn *why* a packet was dropped (useful for protocol debugging).
- Length semantics: `L/ll/LL` = **total packet length** including header; 32-bit length fields are ignored. Document and/or parameterise (payload-length vs total-length, offset adjust).
- Not a `chain.Processor`: no adapter from a byte `buffer.Reader` to a stream of emitted packets; integrating with `x/chain` pipelines needs glue.
- Duplicate implementation (`ops.go` graph ops) and stale SPEC.md make it unclear which API is canonical.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Move `test_helpers.go` into a `_test.go` file — x/math/protocol/peg/test_helpers.go #p2
- [ ] Return drop reasons (error) from `Decide` instead of swallowing — x/math/protocol/peg/program.go:31-34 #p3
- [ ] Rewrite SPEC.md to the real API; remove or integrate `ops.go` `New*Op` — x/math/protocol/peg/SPEC.md, ops.go #p3

## Inferred objectives (proposals)

- **Proposal:** make PEG the declarative framer for every byte protocol in the robot (lidar, CRSF, NMEA-binary, SPI feature stream), with a `chain.Processor[[]byte, byte]` wrapper that emits packets + decoded fields.
- **Proposal:** add an incremental, allocation-free matcher (compiled to a small bytecode) for ESP32 use, keeping the current interpreter as the host-side reference.
