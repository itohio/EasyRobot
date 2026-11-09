---
type: algorithm
status: partial
tags: [easyrobot, protocol, parser, expression]
workstream: [W1, W8]
module: "[[peg-protocol-parser]]"
summary: "2 of 5 failures (impl bugs)"
source: [x/math/protocol/peg/expression.go, x/math/protocol/peg/parser.go, x/math/protocol/peg/ast.go]
tests: [x/math/protocol/peg/pattern_advanced_test.go, x/math/protocol/peg/pattern_features_test.go]
test_result: fail
complete: no
correct: no
evidence: "go test -count=1 ./x/math/protocol/peg → TestBitwiseOperations/bitwise_XOR_invert and TestExpressionRangeCondition/within_range FAIL (2026-11-09); probe pegprobe: `(uu^0xFFFF)` Drop but `(uu ^0xFFFF)` Emit; `(50<uu && uu<1500)` Drop but `(uu>50 && uu<1500)` Emit"
---
# PEG field expressions (`%name:(expr)`, `@(cond)`)

**What it does:** precedence-climbing parser + tree-walking evaluator over `float64`/`bool` values for derived fields and guards inside [[peg-packet-matcher]] patterns: `%(uu/360.0)` decodes a `uu` and stores `uu/360`; `%(50<uu && uu<1500)` should decode `uu` and drop the packet when false; `@(cond)` checks without consuming bytes. Identifiers resolve to the current base value (`value`, the field name, or the type token) or to previously decoded fields by name (`exprEnv.lookup`, `expression.go:42-69`).

**Where:** parser `expression.go:71-511` (levels: `||` :91, `&&` :111, `== !=` :131, `< <= > >=` :159, `|` :199, `^` :223, `&` :243, `<< >>` :267, `+ -` :311, `* /` :339, unary :367, primary :389, hex/decimal numbers :418); evaluation `binaryExpr.eval` :571-710 (bitwise/shift via `int64` conversion, division-by-zero error, shift count checks); base-type detection `parser.findBaseTypeToken` `parser.go:716-797`; node `expressionNode.eval` `ast.go:581-637`.

**Inputs → outputs:** expression source string → AST; at match time: bytes of the base field (if any) + earlier fields → numeric field (`FieldFloat64`) or boolean guard (false → error → `Drop`).

## Completeness review

- Complete: no.
  - Base-type detection only inspects the **first** identifier and only accepts it when followed by one of `( ) + - * / < > = ! & |` or space (`parser.go:786-791`). Missing `^` → `(uu^0xFFFF)` consumes no bytes, `uu` resolves to the *previous* field, the 2 unconsumed bytes break the `$` anchor → Drop (failing XOR test; probe shows adding a space fixes it). A constant first (`50<uu …`) → no base field, `uu` is an unknown identifier → error → Drop (failing range test). Both are **implementation** defects; the tests express the documented grammar (FEATURES.md 4.3.4, SPEC.md "Bitwise").
  - SPEC.md's prose precedence (shift binds tighter than `+`) contradicts its own EBNF and the code (C-like: shift looser than `+`, `parseShift → parseAddSub`, `expression.go:267`).
- Tested: arithmetic, scaling, named/unnamed references, hex constants, AND/OR/shift cases pass; the two cases above fail.
- Correct: partly — arithmetic/comparison verified by passing tests; all values pass through `float64`, so 64-bit integer fields above 2^53 lose precision and bitwise ops on them are wrong [derived from `int64(float64)` conversion at `expression.go:614-640`].

## TODOs

- [ ] Detect the base type token anywhere in the expression (or require explicit `%uu:(…)`) and include `^` in the follow set — x/math/protocol/peg/parser.go:716-797 #p1 #ER-023
- [ ] Keep integers as `int64/uint64` in `exprValue` for bitwise ops — x/math/protocol/peg/expression.go:14-33 #p2
- [ ] Align SPEC.md precedence prose with the EBNF/code — x/math/protocol/peg/SPEC.md #p3

## Links

- Module: [[peg-protocol-parser]]
- Related: [[peg-packet-matcher]]
