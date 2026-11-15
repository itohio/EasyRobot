---
type: algorithm
status: partial
tags: [easyrobot, protocol, parser]
workstream: [W1, W4, W8]
module: "[[peg-protocol-parser]]"
summary: "3 of 5 failures (1 impl bug, 2 wrong tests); races, stale AST mutation, field duplication found by…"
source: [x/math/protocol/peg/parser.go, x/math/protocol/peg/ast.go, x/math/protocol/peg/program.go, x/math/protocol/peg/state.go, x/math/protocol/peg/fields.go, x/math/protocol/peg/ops.go]
tests: [x/math/protocol/peg/parser_test.go, x/math/protocol/peg/program_test.go, x/math/protocol/peg/pattern_basic_test.go, x/math/protocol/peg/pattern_advanced_test.go, x/math/protocol/peg/pattern_features_test.go, x/math/protocol/peg/pattern_user_requirements_test.go, x/math/protocol/peg/field_test.go, x/math/protocol/peg/state_test.go]
test_result: fail
complete: no
correct: no
evidence: "go test -count=1 ./x/math/protocol/peg → FAIL, 58 PASS / 5 FAIL top-level (2026-11-09); probes (+ -race): wildcard count mutation leaks across Decide calls, 29 data races, fields duplicated per call"
---
# PEG-style binary packet matcher

**What it does:** compiles a compact pattern (e.g. `^55AA*ll$`, `^AA%{x:uu,y:u}%*f$`) into an AST and, on every call, re-matches the whole accumulated packet from offset 0, returning `Continue` (need more bytes), `Drop` (mismatch / too long) or `Emit` (complete). Intended for byte-at-a-time framing of serial/radio streams (`cmd/monitor`).

**Where:**
- Compiler: `Compile` `parser.go:25-60`, recursive descent `parseSequence` :68, `parseElement` :91, groups `(a,b)` :131, literals :167, wildcards `*`/`*N` :187, skip-until `*?hex?` :207, offsets `#N`/`#±N` :235 (`#(expr)` → error, TODO :263), conditions `@(expr)` :271, fields `%…` :283, arrays/structs :422-611, type tokens `fieldSpec` :612.
- Interpreter: `ast.go` nodes — `sequenceNode.eval` :23-99 (incl. wildcard look-ahead heuristic :32-85), `choiceNode` :105 (clones state per branch), `literalNode` :141, `lookaheadNode` :160, `wildcardNode` :174, `fieldNode` :213 (fixed ints/floats LE/BE, varints, Pascal/C/fixed strings; `L`/`ll`/`LL` set `DeclaredLength`), `offsetNode` :329, `skipUntilNode` :361, array/struct nodes :377-570, `expressionNode` :581 → [[peg-expression-evaluator]].
- Decision: `newProgram` op `program.go:24-147` (max length, declared length = **total packet length**, end-anchor coverage check :74-142); `Program.Decide` :166.
- State: `State` interface + `DefaultState` `state.go`.
- `ops.go`: the `New*Op` constructors (`NewMatchLiteralOp` :23, `NewDecodeFieldOp` :59, `NewSequenceOp` :275, `NewChoiceOp` :304, …) are a second, graph-op based implementation used **only by their own tests** (`ops_*_test.go`); `Compile` never uses them. Only `decodeField` :332 and the varint decoders :378-407 are shared with the AST.

**Inputs → outputs:** `Compile(pattern) (*Program, error)`; `Program.Decide(State) (Decision, error)` or `Program.Compute(nil, state)` (single-node `graph.GenericExpressionGraph`). Decoded values land in `state.Fields()` as `DecodedField{Name, Offset, Type, Value interface{}}`.

## Completeness review

- Complete: no — `#(expr)` unimplemented (`parser.go:263`); `^` is parsed but never used (`anchorStart` stored only, `program.go:15,158`): every pattern is start-anchored, so un-anchored search / resync is the caller's job (probe: `AA%u` on `00 AA 01` → Drop); 32/64-bit length fields do not set `DeclaredLength` (`ast.go:310-317`); SPEC.md still says "no intermediate AST" and documents `New(pattern, opts)` / `DecisionKeep`, which do not exist.
- Tested: 63 top-level tests, 5 fail. Diagnosis (probe + code):
  1. `TestLengthDrivenEmit` (`^55AA*ll$` fed byte-by-byte, expects Emit at 5 bytes) — **implementation bug.** `sequenceNode.eval` writes the look-ahead size into the shared AST node (`wc.count = minNeeded`, `ast.go:69`) and returns `statusNeedMore` before the restore at :94-96, so the next `Decide` sees a stale fixed count. Probe: same packet → Emit on a fresh program, Continue after byte-by-byte feeding.
  2. `TestLengthDrivenEmitWildcard` (`^*%ll*$`, data `FF 02 00 AA BB`, expects Emit) — **test inconsistent.** The look-ahead gives up when another unbounded `*` follows (`ast.go:52-56`), so the first `*` eats everything (Continue). Even with `*1`, declared length 2 ≠ 5-byte packet → Drop under the "declared = total length" rule that `TestLengthDrivenEmit` asserts (probe: `^*1%ll*$` with length 5 → Emit). Test data must say 0x0005, and the grammar needs a rule for multiple `*`.
  3. `TestOffsetJumpEdgeCases/offset jump beyond MaxLength` — **test wrong.** Message says "should continue or drop" but asserts Continue; implementation drops once the packet exceeds `$100` (`program.go:37-39`), which matches SPEC and the sibling test `program_test.go:313-331` that asserts Drop for the same case.
  4. `TestBitwiseOperations/bitwise XOR invert` and 5. `TestExpressionRangeCondition/within range` — base-type detection bugs, see [[peg-expression-evaluator]] (implementation wrong).
- Correct: no.
  - **Program is not reusable or concurrency-safe:** besides the leak above, the reset `wc.count = -1` at `ast.go:94-96` also fires for explicit `*N` wildcards, turning `*2` into `*` after its first success (probe: `^AA*2BB` on `AA 01 02 BB CC` → Emit, then Drop on the identical call). `go run -race` with 8 goroutines sharing one Program: 29 races (ast.go:32-95, 176, 201), 15 930/16 000 wrong decisions.
  - **Fields accumulate across calls:** `Decide` never clears fields, and every call re-decodes from offset 0 (probe: `^AA%u%u` fed byte-wise → 3 fields after the final call). `cmd/monitor` calls `Decide` per byte on one state (`cmd/monitor/main.go:175`) [INFERENCE: its printed fields contain duplicates unless it resets them].
  - Cost: whole-packet re-parse per byte = O(n²) per packet; `fmt.Errorf` on every mismatch; `Clone` per choice branch.

## TODOs

- [ ] Stop mutating AST nodes during eval (pass computed wildcard length as a local) — x/math/protocol/peg/ast.go:62-96 #p1 #ER-023
- [ ] Reset (or rebuild) `state.Fields()` at the start of each `Decide` — x/math/protocol/peg/program.go:24-30 #p1 #ER-023
- [ ] Fix `TestLengthDrivenEmitWildcard` data (length 0x0005) and define semantics for multiple unbounded `*` — pattern_advanced_test.go:701-725, ast.go:52-56 #p2
- [ ] Fix `TestOffsetJumpEdgeCases` assertion to accept Drop — pattern_advanced_test.go:503-517 #p2
- [ ] Implement `#(expr)` — parser.go:263 #p2
- [ ] Implement un-anchored mode (honour missing `^`, scan for start) — program.go:24-47 #p2
- [ ] Delete or wire up the unused `New*Op` graph-op implementation (keep `decodeField`) — x/math/protocol/peg/ops.go:23-330 #p3
- [ ] Rename `test_helpers.go` → `helpers_test.go`: it is a non-test file importing `testing` and testify into every consumer build — x/math/protocol/peg/test_helpers.go:1-16 #p2
- [ ] Update SPEC.md to the real API (`Compile`, `Decide`, `DecisionContinue`, AST) — SPEC.md #p3
- [ ] Incremental matching (resume from last offset) to avoid O(n²) — ast.go, program.go #p3

## Links

- Module: [[peg-protocol-parser]]
- Related: [[peg-expression-evaluator]], [[crsf-elrs]], [[nmea]], [[lidar]]
