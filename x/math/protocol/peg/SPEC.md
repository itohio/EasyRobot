# Protocol PEG Specification

## Purpose
- Describe binary packet layouts via a concise PEG-inspired grammar.
- Keep the parser runtime stateless—callers provide a `State` object that stores packet bytes, decoded fields, and derived metadata.
- After each byte arrives, the caller appends it to `State.Packet()` and calls `matcher.Compute(nil, state)` to learn whether to **keep**, **drop**, or **emit** the packet.

## Scope
- Package lives in `x/math/protocol/peg`.
- Public API:
  - `New(pattern string, opts ...Option) (*Program, error)` - compiles pattern into expression graph
  - `Program.Compute(start Node, inputs ...State) ([]Decision, error)` - evaluates pattern (inherited from GenericExpressionGraph)
  - `State` interface (DefaultState provided) with fields:
    - `Packet() []byte`, `SetPacket([]byte)`, `AppendPacket(...byte)`
    - `CurrentLength()` (len of packet), `DeclaredLength()`, `SetDeclaredLength(int)`
    - `MaxLength()`, `SetMaxLength(int)`
    - `Fields() []DecodedField`, `AddField`, `ResetFields`
    - `Decision() Decision`, `SetDecision(Decision)`
  - `Decision` enum: `DecisionKeep`, `DecisionDrop`, `DecisionEmit`.
- Grammar highlights:
  - Literals/alternation `(55AA|123A)`
  - Wildcards `*N`, `*(expr)`, future `*?pattern?`
  - Offset jumps `#N`, `#(expr)`
  - Anchors `$N` (max length), `$ (expr)`
  - Typed fields `%u`, `%uu`, `%f`, `%s`, `%S`, `%Ns`
  - Arrays `%Ntype`, `%*{struct}`
  - Structs `%{field:%uu, status:%u}`
  - Arithmetic expressions `%(uu/360.0)`

## Non-Goals
- No regex backtracking semantics; evaluation is deterministic and sequential.
- No implicit CRC or checksum verification (callers can add nodes that capture CRC bytes, but validation happens outside).
- No schema inference—the CLI pattern remains developer-authored.

## Architecture
1. **Operations (`ops.go`)** – independent, testable parsing operations:
   - `MatchLiteral(value []byte)` - matches exact byte sequence
   - `MatchWildcard(count int)` - matches N arbitrary bytes
   - `DecodeField(spec FieldSpec)` - decodes typed field, updates state
   - `SetLength(spec FieldSpec)` - decodes length field, sets state.DeclaredLength
   - `SetMaxLength(value int)` - sets state.MaxLength
   - `CheckLength()` - checks if packet length matches declared length
   - `CheckMaxLength()` - checks if packet exceeds max length
   - Each operation is an `ExpressionOp[State, Decision]` that:
     - Reads from `state.Packet()` at current offset
     - Updates state (adds fields, sets lengths, etc.)
     - Returns `DecisionKeep` (continue), `DecisionDrop` (mismatch), or `DecisionEmit` (complete)
     - Returns `false` if more bytes needed (incomplete match)

2. **Parser (`parser.go`)** – recursive-descent builder that compiles pattern directly into `GenericExpressionGraph`:
   - No intermediate AST - pattern → expression graph nodes directly
   - Each grammar element becomes one or more operation nodes
   - Nodes connected via edges to form evaluation sequence
   - Parser tests (`parser_test.go`) only verify pattern → graph compilation

3. **Program (`program.go`)** – wraps `GenericExpressionGraph[OpData, float32, State, Decision]`:
   - `New(pattern, opts...)` compiles pattern and returns Program
   - `Compute(nil, state)` evaluates graph, updates `state.Decision()` directly
   - Program tests (`program_test.go`) verify end-to-end packet matching

4. **State** – caller-owned struct containing:
   - `Packet []byte` - accumulated packet bytes
   - `Fields []DecodedField` - decoded field values
   - `DeclaredLength` (from `%L`, `%ll`, `%LL`, etc.)
   - `MaxLength` (minimum of CLI `--maxlen`, `$N`, `$expr`, etc.)
   - `Decision` - set by `Compute()` to indicate Keep/Drop/Emit

5. **Decision Loop** – caller's responsibility:
   ```go
   matcher := peg.New(pattern, peg.WithMaxLen(2048))
   for byte := range bytes {
     if len(packets) == 0 {
       state := peg.NewDefaultState()
       state.AppendPacket(byte)
       packets = append(packets, state)
     }
     for i, packet := range packets {
       packet.AppendPacket(byte)
       matcher.Compute(nil, packet)
       switch packet.Decision() {
       case peg.DecisionDrop:
         // remove packet
       case peg.DecisionEmit:
         // emit packet, then remove
       case peg.DecisionKeep:
         // continue accumulating
       }
     }
   }
   ```

### Example Timeline (`55AA*ll`)

| Bytes Seen       | Decision | Notes                                       |
|------------------|----------|---------------------------------------------|
| `BB`             | Drop     | Literal mismatch                            |
| `55`             | Keep     | Prefix matches so far                       |
| `55 AA`          | Keep     | Still matching literals                     |
| `55 AA <byte>`   | Keep     | Wildcard consumed, waiting for length field |
| `55 AA <byte> ll`| Keep     | Length field read; `DeclaredLength` updated |
| After `DeclaredLength` bytes total | Emit | Packet complete (length reached) |

## Grammar Sketch (EBNF)
```
pattern      := "^"? sequence "$"?               # anchors optional
sequence     := element*
element      := literal | wildcard | group | field | array | struct | anchor | offset | separator
separator    := ";" element             # Separator allows matching at specific offsets
group        := "(" sequence ("|" sequence)+ ")"
literal      := HEXBYTE+
wildcard     := "*" NUMBER?              # *N skips N bytes, * alone matches until MaxLength
offset       := "#" ("+" | "-")? NUMBER  # #N (absolute), #+N (forward), #-N (backward)
anchor       := "$" (NUMBER | "(" expr ")")?
field        := "%" array_prefix? (typed | string | struct | expr_field)
typed        := "u" | "uu" | "UU" | "uuuu" | "UUUU" | "f" | "F" | "c" | "cc" | "L" | "LL" | "ll"
string       := "s" | "S" | NUMBER "s"
array_prefix := NUMBER | "*"
struct       := "{" (field ("," field)*) "}"
expr_field   := "(" base typed_expr ")"
```

## Testing
- Table-driven tests covering literals, choices, wildcards, length fields, strings, anchors.
- Streaming tests verify decisions after each byte (drop/keep/emit).
- State tests validate clone/merge and length/max-length clamping.

## Implementation Plan: Advanced Pattern Features

The following grammar elements are part of the original requirements but are not yet implemented.  This plan documents the intended behaviour and incremental work packages before coding.

### 1. Offset Jumps (`#N`, `#(expr)`)
- **Syntax**
  - `#N` – jump to absolute offset `N` (0-based) within the current packet buffer.
  - `#(expr)` – evaluate arithmetic expression to produce jump offset.
- **Semantics**
  - Save current offset in a stack, jump to target, evaluate nested elements, then resume from saved offset when the sub-sequence completes.
  - Clamp jumps to `min(len(packet), maxLen)`; emit `DecisionContinue` until enough bytes arrive.
- **Plan**
  1. Extend parser to recognise `#` elements and emit `offsetJumpNode`.
  2. Extend AST/runtime to maintain an offset stack inside `State`.
  3. Add tests for simple jumps, expressions, backtracking inside groups.

### 2. Skip-Until (`*?pattern?`)
- **Syntax**: `*?FF00?` scans forward until byte sequence `FF00` is found.
- **Semantics**
  - Consumes bytes (DecisionContinue) until pattern matches; fails with `DecisionDrop` on mismatch if anchor rules violated.
- **Plan**: Parser support for `*?` tokens, runtime scanner operation, tests for found/not-found/edge cases.

### 3. Arrays
- **Fixed Count**: `%Ntype` (already partially supported by parser, needs runtime validation).
- **Stride/Struct Arrays**:
  - `%10{uu,u}` – read 10 structs, each containing fields `uu` and `u`.
  - `%10:3uu` – (legacy stride syntax) treat as 10 elements with stride 3 bytes.
- **Star Arrays**:
  - `%*type` – read elements until packet exhaustion (respecting declared/max lengths).
  - `%*{struct}` – read structs until remaining bytes are insufficient for one struct.
- **Plan**
  1. Parser: recognise array prefix before `{` or type tokens, record `ArraySpec`.
  2. AST/runtime: loop evaluation over child nodes, collecting decoded fields with indexes (e.g., `field_1[3]`).
  3. Tests: fixed arrays, stride arrays, star arrays, array-of-structs with explicit and star counts.

### 4. Structs
- **Syntax**: `%{field:%uu, status:%u}` optionally nested.
- **Semantics**
  - Acts like an inline sequence with named fields; field names become prefixes for decoded records (e.g., `struct.field`).
- **Plan**: Parser to build nested sequences, runtime to namespace field names, tests for simple/nested structs.

### 5. Arithmetic Expressions (`%(uu/360.0)`)
- **Operators**: `+`, `-`, `*`, `/`, parentheses, integer or float literals.
- **Semantics**: Evaluate expression using previously decoded field(s) or inline field; result stored as derived field.
- **Plan**: Reuse `x/math/graph/GenericExpressionGraph` nodes, add expression parser, tests covering precedence and type conversion.

### Delivery Strategy
1. **Phase 1**: Offset jumps + skip-until (enables FIFO buffer stability).
2. **Phase 2**: Structs + arrays (fixed + star + stride).
3. **Phase 3**: Arithmetic expressions and expression-based jumps.
Each phase adds parser support, AST/runtime logic, and table-driven tests that were previously skipped.

## Integration Notes
- `cmd/monitor` provides a concrete `State` implementation that also tracks reasons (e.g., CRC warnings) and the running packet buffer.
- Different protocol parsers can be precompiled and serialized using the expression graph tooling, then loaded at runtime for hot-swapping protocol definitions.
