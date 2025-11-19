# Pattern Features Status

## Test Coverage Overview

This document tracks implementation status and test coverage for each component of the PEG parser.

---

## 1. Operations (`ops.go`)

### 1.1 MatchLiteralOp (`ops_basic_test.go::TestMatchLiteralOp`)
**Tested Cases:**
- ✅ Exact match - matches complete literal sequence
- ✅ Mismatch - drops packet when bytes don't match
- ✅ Need more bytes - returns DecisionContinue with complete=false when packet too short
- ✅ Empty packet - handles zero-length packet correctly

**Missing Test Cases:**
- ❌ Literal at non-zero offset
- ❌ Very long literal (100+ bytes) for performance
- ❌ Literal matching at packet boundary
- ❌ Multiple consecutive literals

### 1.2 MatchWildcardOp (`ops_basic_test.go::TestMatchWildcardOp`)
**Tested Cases:**
- ✅ Exact bytes - matches exactly N bytes
- ✅ Need more bytes - returns DecisionContinue with complete=false when insufficient bytes
- ✅ Empty packet - handles zero-length packet
- ✅ Offset tracking - verifies offset advances correctly

**Missing Test Cases:**
- ❌ Wildcard count = 0 (no-op behavior)
- ❌ Wildcard at non-zero offset
- ❌ Very large wildcard count (1000+ bytes)
- ❌ Wildcard followed by other operations (offset state preservation)

### 1.3 DecodeFieldOp (`ops_basic_test.go::TestDecodeFieldOp`)
**Tested Cases:**
- ✅ uint8 - decodes single byte correctly
- ✅ uint16 LE/BE - little-endian and big-endian decoding
- ✅ Need more bytes - partial field data
- ✅ Length field - sets DeclaredLength correctly
- ✅ Pascal string - variable-length string with length prefix
- ✅ C string - null-terminated string
- ✅ Fixed string - fixed-size string

**Missing Test Cases:**
- ❌ All field types (int8, int16/32/64, uint32/64, float32/64, varint) - only uint8/16 tested at ops level
- ❌ Field at non-zero offset
- ❌ Invalid field data (e.g., corrupted varint)
- ❌ Field with named vs unnamed
- ❌ Length field interactions with MaxLength

### 1.4 SetMaxLengthOp (`ops_control_test.go::TestSetMaxLengthOp`)
**Tested Cases:**
- ✅ Basic setting - sets MaxLength correctly

**Missing Test Cases:**
- ❌ MaxLength clamping with existing DeclaredLength
- ❌ MaxLength = 0 (unlimited)
- ❌ MaxLength < current packet length
- ❌ Multiple SetMaxLengthOp calls (should use minimum)

### 1.5 CheckLengthOp (`ops_control_test.go::TestCheckLengthOp`)
**Tested Cases:**
- ✅ Exact match - emits when packet length equals declared length
- ✅ Need more - continues when packet shorter than declared
- ✅ Too long - drops when packet exceeds declared length
- ✅ No declared length - continues when no length set

**Missing Test Cases:**
- ❌ DeclaredLength = 0 edge case
- ❌ DeclaredLength > MaxLength interaction
- ❌ Length check at non-final position in sequence

### 1.6 CheckMaxLengthOp (`ops_control_test.go::TestCheckMaxLengthOp`)
**Tested Cases:**
- ✅ Within limit - continues when under MaxLength
- ✅ Exceeds limit - drops when over MaxLength
- ✅ No max length - continues when MaxLength = 0

**Missing Test Cases:**
- ❌ MaxLength = 1 edge case
- ❌ MaxLength exactly equal to packet length
- ❌ MaxLength check in middle of sequence

### 1.7 SequenceOp (`ops_control_test.go::TestSequenceOp`)
**Tested Cases:**
- ✅ All continue - propagates continue when all children continue
- ✅ One drop - propagates drop immediately
- ✅ All emit - propagates emit when all children emit
- ✅ Mixed continue and emit - continues when any child continues

**Missing Test Cases:**
- ❌ Empty sequence (no children)
- ❌ Sequence with 10+ children (stress test)
- ❌ Sequence where child decision changes on re-evaluation
- ❌ Error propagation through sequence

### 1.8 ChoiceOp (`ops_control_test.go::TestChoiceOp`)
**Tested Cases:**
- ✅ One continue - propagates continue when one branch continues
- ✅ One emit - propagates emit when one branch emits
- ✅ All drop - propagates drop when all branches fail

**Missing Test Cases:**
- ❌ Choice with 10+ branches (stress test)
- ❌ Choice where multiple branches could match (first match wins?)
- ❌ Choice with nested sequences
- ❌ Choice backtracking behavior verification

---

## 2. Parser (`parser.go`)

### 2.1 Parser Error Handling (`parser_test.go::TestParserErrors`)
**Tested Cases:**
- ✅ Invalid hex literal - odd number of hex digits
- ✅ Invalid hex byte - non-hex character (G)
- ✅ Missing closing paren - unmatched parenthesis
- ✅ Unexpected character - unknown token (@)
- ✅ Unknown field token - invalid field type (X)
- ✅ Invalid fixed string size - non-numeric size
- ✅ Unexpected token after end - tokens after $ anchor

**Missing Test Cases:**
- ❌ Empty pattern string
- ❌ Pattern with only whitespace
- ❌ Multiple consecutive errors
- ❌ Invalid offset jump syntax (#abc, #-abc)
- ❌ Invalid expression syntax (unmatched parens, invalid operators)
- ❌ Invalid struct syntax (unmatched braces, missing colons)
- ❌ Invalid array syntax (%abc, %*abc)

### 2.2 Parser Edge Cases (`parser_test.go::TestParserEdgeCases`)
**Tested Cases:**
- ✅ Single child sequence - pattern with one element
- ✅ Single branch group - group with no alternation
- ✅ End anchor with max length - $N syntax
- ✅ Wildcard without number - * matches all bytes
- ✅ Field with count - %3u array syntax
- ✅ Fixed string - %5s syntax

**Missing Test Cases:**
- ❌ Pattern with only anchor (^$)
- ❌ Nested groups depth (3+ levels)
- ❌ Very long pattern (1000+ characters)
- ❌ Pattern with special characters in literals
- ❌ Multiple anchors (^...$ with multiple $)

### 2.3 Field Type Recognition (`parser_test.go::TestAllFieldTypes`)
**Tested Cases:**
- ✅ int8, int16 LE/BE, int32 LE/BE, int64 LE/BE
- ✅ uint32/64 LE/BE (complementary to int types)
- ✅ float32, float64
- ✅ CRC8, CRC16
- ✅ Length fields: L, ll, LL

**Missing Test Cases:**
- ❌ Named fields for each type (%name:uu)
- ❌ Array variants for each type (%3u, %5uu, etc.)
- ❌ Varint types (v, V) parser recognition
- ❌ String types (s, S, %Ns) parser recognition
- ❌ Expression fields for each type (%(uu*2))

### 2.4 Parser Number Parsing (`parser_test.go::TestParserNumberErrors`)
**Tested Cases:**
- ✅ Invalid number after $ anchor

**Missing Test Cases:**
- ❌ Number overflow (very large numbers)
- ❌ Negative numbers where not allowed
- ❌ Number parsing in wildcards (*1000000)
- ❌ Number parsing in offset jumps (#1000000)
- ❌ Number parsing in arrays (%1000000u)

---

## 3. Expression Graph / Program (`program.go`)

### 3.1 Decision Calculation (`program_test.go`)

#### 3.1.1 Literal Flow (`program_test.go::TestDecideLiteralFlow`)
**Tested Cases:**
- ✅ Incremental matching - byte-by-byte append matching
- ✅ Final emit - DecisionEmit when pattern complete
- ✅ Field decoding - correct field extraction after match

**Missing Test Cases:**
- ❌ Decision changes: Continue → Drop (mismatch discovered)
- ❌ Decision changes: Continue → Continue → Emit (incremental)
- ❌ Multiple fields in sequence
- ❌ Decision with MaxLength constraint
- ❌ Decision with DeclaredLength constraint

#### 3.1.2 Mismatch Handling (`program_test.go::TestDecideMismatch`)
**Tested Cases:**
- ✅ Immediate drop - DecisionDrop on first byte mismatch

**Missing Test Cases:**
- ❌ Mismatch after partial match (AA matched, BB doesn't)
- ❌ Mismatch in group alternation (all branches fail)
- ❌ Mismatch in nested sequences
- ❌ Mismatch with wildcards (pattern continues but fails later)

#### 3.1.3 Length-Driven Emit (`program_test.go::TestLengthDrivenEmit`)
**Tested Cases:**
- ✅ Incremental wildcard matching
- ✅ Length field reading
- ✅ Decision transitions: Continue → Continue → Continue → Emit
- ✅ DeclaredLength setting and validation

**⚠️ KNOWN ISSUE:** Test currently fails - wildcard followed by length field not working correctly when exactly minNeeded bytes are available.

**Missing Test Cases:**
- ❌ DeclaredLength < actual packet length (should drop)
- ❌ DeclaredLength > actual packet length (should continue)
- ❌ Multiple length fields (which one wins?)
- ❌ Length field interaction with MaxLength
- ❌ Length field with offset jumps

#### 3.1.4 Program Edge Cases (`program_test.go::TestProgramEdgeCases`)
**Tested Cases:**
- ✅ AnchorEnd with wrong length - drops when packet too long
- ✅ MaxLen exceeded - drops when exceeds max length
- ✅ Declared length exceeded - drops when packet too long
- ✅ Declared length not reached - continues when packet too short

**Missing Test Cases:**
- ❌ MaxLength = DeclaredLength edge case
- ❌ MaxLength < DeclaredLength conflict resolution
- ❌ AnchorEnd with backward offset jumps
- ❌ AnchorEnd with fields spanning packet (correct coverage check)
- ❌ Nil state handling (partially tested but incomplete)

#### 3.1.5 Nil State Handling (`program_test.go::TestProgramNilState`)
**Tested Cases:**
- ✅ Nil state doesn't panic - creates default state internally

**Missing Test Cases:**
- ❌ Nil state behavior verification (should it work or error?)
- ❌ Nil state with Compute(nil, nil) vs Decide(nil)

---

## 4. Pattern Matching (End-to-End)

### 4.1 Basic Patterns (`pattern_basic_test.go`)

#### 4.1.1 Group Alternation (`pattern_basic_test.go::TestGroupAlternation`)
**Tested Cases:**
- ✅ Match first branch - (A|B) matches A
- ✅ Match second branch - (A|B) matches B
- ✅ No match - all branches fail
- ✅ Three branches - (A|B|C) matches middle branch
- ✅ Group with fields - alternation preserves field decoding

**Missing Test Cases:**
- ❌ Nested alternation ((A|B)|(C|D))
- ❌ Alternation with wildcards
- ❌ Alternation with offset jumps
- ❌ Alternation with expressions
- ❌ Backtracking in alternation (when first branch partially matches then fails)

#### 4.1.2 Skip N Bytes (`pattern_basic_test.go::TestSkipNBytes`)
**Tested Cases:**
- ✅ Skip 3 bytes - *3 skips exactly 3 bytes
- ✅ Skip 0 bytes - *0 is valid no-op
- ✅ Skip with field before and after
- ✅ Need more bytes for skip - returns Continue

**Missing Test Cases:**
- ✅ Wildcard * (no number) followed by fields - **FIXED: TestWildcardSkipWhenFollowedByFields covers this**
- ❌ Wildcard at packet boundary
- ❌ Wildcard with MaxLength constraint
- ❌ Multiple consecutive wildcards

### 4.2 Advanced Patterns (`pattern_advanced_test.go`)

#### 4.2.1 Struct Fields (`pattern_advanced_test.go::TestStructFields`)
**Tested Cases:**
- ✅ Simple struct - %{x:%uu,y:%u}
- ✅ Struct with length field - length field inside struct
- ✅ Nested struct - struct within struct

**Missing Test Cases:**
- ❌ Struct with named fields vs unnamed
- ❌ Struct with arrays (%3u within struct)
- ❌ Struct with wildcards
- ❌ Empty struct %{}
- ❌ Struct with expressions
- ❌ Array of structs %3{...}

#### 4.2.2 Expressions (`pattern_advanced_test.go::TestExpressions`)
**Tested Cases:**
- ✅ Division expression - %(uu/360.0)
- ✅ Multiplication expression - %(u*2)
- ✅ Addition expression - %(u+10)
- ✅ Subtraction expression - %(u-3)
- ✅ Parentheses expression - %((uu+10)*2)
- ✅ Expression result as float64
- ✅ Base field storage verification (DECISION: base should be stored too)

**Missing Test Cases:**
- ❌ Expression division by zero
- ❌ Expression overflow (very large numbers)
- ❌ Expression with parentheses nesting (3+ levels)
- ⚠️ Base field storage (DECISION: base should be stored too) - **TODO: Implementation stores only expression result, not base field**

#### 4.2.2.1 Expressions with Named Field References (`pattern_advanced_test.go::TestExpressionsWithNamedFields`)
**Tested Cases:**
- ✅ Reference named field in expression - %length:uu%value:(length-2000)
- ✅ Multiple field references in expression - %length:uu%offset:uu%sum:(length+offset)
- ✅ Reference non-existent field - returns DecisionDrop at runtime (DECISION: parse error - **TODO: should be parse-time validation**)

#### 4.2.2.2 Expressions with Unnamed Field References (`pattern_advanced_test.go::TestExpressionsWithUnnamedFields`)
**Tested Cases:**
- ✅ Reference unnamed field by generated name - %uu%derived:(field_1*2) (DECISION: field_1, field_2, etc.)
- ✅ Cannot reference unnamed field by type token - %uu%derived:(uu*2) decodes new bytes (DECISION: uu always decodes new bytes)
- ✅ Multiple unnamed fields - reference by order - %uu%uu%sum:(field_1+field_2)

#### 4.2.2.3 Expression Field Syntax (`pattern_advanced_test.go::TestExpressionFieldSyntax`)
**Tested Cases:**
- ✅ Expression with parentheses - %(uu/360.0)
- ✅ Expression without parentheses, ending in semicolon - %uu/360.0; (DECISION: either parentheses or `;`)
- ✅ Named expression with parentheses - %value:(uu/360.0)
- ✅ Expression without parentheses, no semicolon - parse error
- ✅ Expression decode only - %(uu) (DECISION: valid, decode and store the field)
- ✅ Empty expression - %() (DECISION: invalid)

#### 4.2.2.4 Named Expression Fields (`pattern_advanced_test.go::TestNamedExpressionField`)
**Tested Cases:**
- ✅ Named expression field - %start:(uu-2000) stores expression result with name "start"
- ⚠️ Base field storage - **TODO: DECISION says base should be stored too, but implementation only stores expression result**

**Known Implementation Gaps:**
- Base fields decoded by type token in expression are not stored separately (e.g., for %start:(uu-2000), only "start" result is stored, not the decoded uu field)

#### 4.2.3 Skip Until Pattern (`pattern_advanced_test.go::TestSkipUntilPattern`)
**Status:** ⏭️ Tests written but skipped - **NOT IMPLEMENTED**

**Tested Cases (when implemented):**
- ✅ Skip until FF00 - finds pattern and continues
- ✅ Pattern at start - no bytes skipped
- ✅ Pattern not found - returns Continue (needs more bytes)
- ✅ Skip until with field after

**Missing Test Cases:**
- ❌ Skip until with MaxLength (should drop if pattern not found within limit)
- ❌ Skip until pattern that overlaps with search start
- ❌ Skip until very long pattern (10+ bytes)
- ❌ Skip until pattern at end of packet

#### 4.2.4 Offset Jumps (`pattern_advanced_test.go::TestGoToOffset`)
**Tested Cases:**
- ✅ Jump to offset 5 - absolute offset #5
- ✅ Jump forward - relative offset #+N
- ✅ Jump backward - relative offset #-N
- ✅ Field reading at jumped offset

**Missing Test Cases:**
- ❌ Offset jump to negative offset (should error)
- ❌ Offset jump beyond packet length (should Continue)
- ❌ Offset jump beyond MaxLength (should Drop)
- ❌ Multiple consecutive offset jumps
- ❌ Offset jump with expressions #(expr)
- ❌ Offset jump to field boundary (correct field reading)

#### 4.2.5 Offset Jumps in Groups (`pattern_advanced_test.go::TestGoToOffsetInGroup`)
**Status:** ⏭️ Tests written but skipped - **PARTIALLY IMPLEMENTED** (backtracking not working)

**Tested Cases (when implemented):**
- ✅ Jump in group with backtracking - AA(#25%uu)ii should read both fields

**Missing Test Cases:**
- ❌ Nested offset jumps in groups
- ❌ Offset jump in alternation branches
- ❌ Offset jump backtracking with multiple fields

#### 4.2.6 Simple Pattern Match (`pattern_advanced_test.go::TestSimplePatternMatch`)
**Tested Cases:**
- ✅ Pattern with wildcard followed by fields - ^55AA*%uu%start:uu%end:uu
- ✅ Field offsets after wildcard skip
- ✅ Named field extraction

**Missing Test Cases:**
- ❌ Pattern with multiple wildcards
- ❌ Pattern with wildcard at start
- ❌ Pattern with wildcard at end

#### 4.2.7 Wildcard Skip When Followed By Fields (`pattern_advanced_test.go::TestWildcardSkipWhenFollowedByFields`)
**Tested Cases:**
- ✅ Wildcard followed by single uint16 - must skip at least 1 byte
- ✅ Wildcard followed by multiple uint16 fields - correct offset calculation
- ✅ Wildcard followed by mixed field types - uint8 and uint16
- ✅ Wildcard with exactly minimum bytes - edge case handling
- ✅ Wildcard with insufficient bytes - returns DecisionContinue (DECISION: wait for more data)
- ✅ Wildcard skips exactly 1 byte (DECISION: * skips EXACTLY 1 byte)
- ✅ Wildcard followed by named fields - named field extraction

**This test was added to fix the critical bug where wildcard * was not skipping bytes correctly.**

#### 4.2.7.1 Wildcard With Count (`pattern_advanced_test.go::TestWildcardWithCount`)
**Tested Cases:**
- ✅ Wildcard *5 with exactly 5 bytes - matches exactly 5 bytes (DECISION: *5 matches exactly any 5 bytes)
- ✅ Wildcard *5 needs more bytes - returns DecisionContinue (DECISION: must wait for 5 bytes)
- ✅ Wildcard *5 with insufficient bytes - returns DecisionContinue

#### 4.2.8 Loop Simulation (`pattern_advanced_test.go::TestLoopSimulation`)
**Tested Cases:**
- ✅ Byte-by-byte packet accumulation
- ✅ Skip-until pattern matching
- ✅ Field extraction after match
- ✅ Real-world scenario simulation

**Missing Test Cases:**
- ❌ Loop with multiple packets in sequence
- ❌ Loop with packet drop scenarios
- ❌ Loop with MaxLength constraints
- ❌ Loop with DeclaredLength constraints

### 4.3 Feature-Specific Patterns (`pattern_features_test.go`)

#### 4.3.1 Named Fields (`pattern_features_test.go::TestNamedFields`)
**Tested Cases:**
- ✅ Multiple named fields - %temperature:uu%voltage:uu
- ✅ Field name extraction and value association

**Missing Test Cases:**
- ❌ Named vs unnamed field mixing
- ❌ Duplicate field names (should be allowed or error?)
- ❌ Very long field names
- ❌ Field names with special characters

#### 4.3.2 Separator Literal (`pattern_features_test.go::TestSeparatorLiteral`)
**Tested Cases:**
- ✅ Separator with offset jump - AA#2;BB matches at offset 2

**Missing Test Cases:**
- ❌ Separator with field - %u#5;%uu
- ❌ Separator with wildcard - AA*5;BB
- ❌ Separator at packet boundary
- ❌ Separator beyond packet length

#### 4.3.3 Expression Scaling (`pattern_features_test.go::TestExpressionScaling`)
**Tested Cases:**
- ✅ Expression without base field - %(uu/360.0) with implicit field

**Missing Test Cases:**
- ❌ Expression with explicit base field name
- ❌ Expression result type conversions
- ❌ Expression precision (float64 vs float32)

#### 4.3.4 Expression Range Condition (`pattern_features_test.go::TestExpressionRangeCondition`)
**Tested Cases:**
- ✅ Within range - expression condition true (emits)
- ✅ Outside range - expression condition false (drops)

**Missing Test Cases:**
- ❌ Range with equals (50<=uu<=1500)
- ❌ Range with OR conditions
- ❌ Range with NOT conditions
- ❌ Complex nested conditions

#### 4.3.5 Conditional Guard (`pattern_features_test.go::TestConditionalGuard`)
**Tested Cases:**
- ✅ Condition satisfied - @(end>start) allows emit
- ✅ Condition violated - @(end>start) causes drop

**Missing Test Cases:**
- ❌ Guard with complex expression
- ❌ Guard with field references
- ❌ Guard in sequence (early drop)
- ❌ Guard in alternation (branch filtering)

#### 4.3.6 Array with Stride (`pattern_features_test.go::TestArrayWithStride`)
**Tested Cases:**
- ✅ Stride array - %3:2u reads 3 elements with stride 2

**Missing Test Cases:**
- ❌ Stride array with struct - %3:10{uu,u}
- ❌ Stride array boundary conditions
- ❌ Stride array with offset jumps

#### 4.3.7 Star Array Named (`pattern_features_test.go::TestStarArrayNamed`)
**Tested Cases:**
- ✅ Star array - %*data:uu reads until exhausted

**Missing Test Cases:**
- ❌ Star array with MaxLength
- ❌ Star array with DeclaredLength
- ❌ Star array with odd packet length (partial element)
- ❌ Star array field naming (data_0, data_1, etc.)

#### 4.3.8 Star Array of Structs (`pattern_features_test.go::TestStarArrayOfStructs`)
**Tested Cases:**
- ✅ Star array of structs - %*{uu,u} reads structs until exhausted

**Missing Test Cases:**
- ❌ Star array with partial struct (insufficient bytes)
- ❌ Star array with nested structs
- ❌ Star array with expressions

### 4.4 User Requirements (`pattern_user_requirements_test.go::TestUserRequirements`)
**Tested Cases:**
- ✅ Literal only - exact byte sequence matching
- ✅ Wildcard until maxlen - * matches up to MaxLength
- ✅ Wildcard with separator - *5;FF syntax
- ✅ Expression with condition - complex expression patterns
- ✅ Separator with checksum - separator usage patterns
- ✅ Multiple fields - field combination patterns
- ✅ Offset exceeds maxlen drops - #N beyond MaxLength drops

**Missing Test Cases:**
- ❌ All user requirement edge cases
- ❌ Performance with large packets
- ❌ Memory usage with streaming

---

## 5. State Management (`state.go`)

### 5.1 State Operations (`state_test.go`)

#### 5.1.1 Clone (`state_test.go::TestDefaultStateClone`)
**Tested Cases:**
- ✅ Deep clone - cloned state independent of original

**Missing Test Cases:**
- ❌ Clone with fields, packet, offsets, lengths
- ❌ Clone performance with large state
- ❌ Clone with nil values

#### 5.1.2 Edge Cases (`state_test.go::TestStateEdgeCases`)
**Tested Cases:**
- ✅ SetPacket with capacity reuse
- ✅ SetDeclaredLength with MaxLength clamping
- ✅ SetMaxLength with DeclaredLength clamping
- ✅ SetDeclaredLength zero or negative
- ✅ SetMaxLength zero or negative
- ✅ ResetFields
- ✅ Merge preserves offset

**Missing Test Cases:**
- ❌ SetOffset beyond packet length
- ❌ SetOffset negative
- ❌ AppendPacket performance (large packets)
- ❌ Merge with conflicting lengths (which wins?)
- ❌ Merge with duplicate field names

---

## 6. Field Decoding (`fields.go`, `ops.go`)

### 6.1 Field Types (`field_test.go`)

#### 6.1.1 Int64 Fields (`field_test.go::TestInt64Fields`)
**Tested Cases:**
- ✅ int64 little-endian
- ✅ int64 big-endian
- ✅ uint64 little-endian
- ✅ uint64 big-endian

**Missing Test Cases:**
- ❌ int64/uint64 at non-zero offsets
- ❌ int64/uint64 boundary values (max/min)
- ❌ int64/uint64 with partial data

#### 6.1.2 Varint Fields (`field_test.go::TestVarintFields`, `TestVarintIncremental`, `TestVarintEdgeCases`)
**Tested Cases:**
- ✅ Unsigned varint small/medium/large
- ✅ Signed varint positive/negative/zero
- ✅ Incremental varint reading (byte-by-byte)
- ✅ Incomplete varint with continuation bit
- ✅ Varint with many continuation bits (9+ bytes)

**Missing Test Cases:**
- ❌ Varint overflow (shift >= 64)
- ❌ Varint at non-zero offset
- ❌ Varint with MaxLength constraint

#### 6.1.3 Pascal String (`field_test.go::TestPascalStringField`)
**Tested Cases:**
- ✅ Pascal string decoding - length prefix + string

**Missing Test Cases:**
- ❌ Pascal string with length = 0
- ❌ Pascal string with length > MaxLength
- ❌ Pascal string with invalid UTF-8
- ❌ Pascal string at non-zero offset

#### 6.1.4 Field Decoding (`field_test.go::TestDecodeFieldAllTypes`, `TestDecodeFieldLengthFieldTypes`)
**Tested Cases:**
- ✅ All integer types (int8/16/32, uint8/16/32)
- ✅ Float types (float32, float64)
- ✅ Length field types (uint8/16/32)

**Missing Test Cases:**
- ❌ Field decoding at non-zero offset
- ❌ Field decoding with partial data (all types)
- ❌ Field decoding error cases (all types)
- ❌ Field decoding boundary values

---

## 7. Known Issues and Gaps

### 7.1 Critical Issues
1. ~~**Wildcard followed by fields**~~ **FIXED**
   - Issue: When wildcard * is followed by fields and exactly minNeeded bytes are available, decision calculation is incorrect
   - Test: `program_test.go::TestLengthDrivenEmit`, `pattern_advanced_test.go::TestWildcardSkipWhenFollowedByFields`
   - Status: ✅ Fixed - wildcard now correctly skips exactly 1 byte when followed by fields (DECISION: * skips EXACTLY 1 byte)

2. **Base field storage in expressions**
   - Issue: DECISION states "base should be stored too" - when %start:(uu-2000) is used, both the decoded uu field and the expression result should be stored
   - Current: Only expression result is stored
   - Status: ⚠️ TODO - Implementation needs to store base field separately (with auto-generated name if needed)
   
3. **Parse-time field reference validation**
   - Issue: DECISION states "parse error" when referencing non-existent field names
   - Current: Runtime error (converted to DecisionDrop) when field doesn't exist
   - Status: ⚠️ TODO - Parser should validate field references during compilation

### 7.2 Missing Implementation
1. **Skip Until Pattern (`*?pattern?`)**
   - Tests written but skipped
   - File: `pattern_advanced_test.go::TestSkipUntilPattern`

2. **Offset Jumps in Groups (backtracking)**
   - Tests written but skipped
   - File: `pattern_advanced_test.go::TestGoToOffsetInGroup`

### 7.3 Test Coverage Gaps

#### High Priority
1. **Wildcard behavior comprehensive testing**
   - Wildcard * (no number) in all contexts
   - Wildcard with MaxLength
   - Wildcard with offset jumps
   - Wildcard with expressions

2. **Expression evaluation edge cases**
   - Division by zero
   - Overflow handling
   - Field reference errors
   - Type conversions

3. **Offset jump edge cases**
   - Negative offsets
   - Offset beyond MaxLength
   - Multiple consecutive jumps
   - Offset with expressions

#### Medium Priority
1. **State management stress tests**
   - Large packet handling (10KB+)
   - Many fields (100+)
   - Deep nesting (10+ levels)

2. **Parser error recovery**
   - Multiple errors in one pattern
   - Error messages quality
   - Error position reporting

3. **Performance tests**
   - Pattern compilation time
   - Packet matching time
   - Memory usage

#### Low Priority
1. **Edge case combinations**
   - All features combined in complex patterns
   - Stress tests with random data
   - Fuzzing tests

---

## 8. Proposed Additional Tests

### 8.1 Operations Tests (`ops_comprehensive_test.go` - NEW FILE)

#### 8.1.1 TestMatchLiteralOpComprehensive
**Priority:** Medium
**Test Cases:**
- Literal at non-zero offset (packet: [0xFF, 0xAA, 0xBB], pattern: ^AA, offset: 1)
- Very long literal (100+ bytes) for performance testing
- Literal matching at packet boundary (exact match, no extra bytes)
- Multiple consecutive literals (^AABBCCDD with data [0xAA, 0xBB, 0xCC, 0xDD])
- Partial match then mismatch (^AABB, data: [0xAA, 0xBC] - should drop on second byte)

#### 8.1.2 TestMatchWildcardOpComprehensive
**Priority:** Medium
**Test Cases:**
- Wildcard count = 0 (*0) - should be no-op, verify offset doesn't change
- Wildcard at non-zero offset (packet: [0xFF, ...], pattern: ^FF*5, offset: 1)
- Very large wildcard count (1000+ bytes) - performance and memory
- Wildcard followed by other operations - verify offset state preservation
- Wildcard * with MaxLength constraint (should drop if exceeds MaxLength)

#### 8.1.3 TestDecodeFieldOpAllTypes
**Priority:** High
**Test Cases:**
- All integer types: int8, int16 LE/BE, int32 LE/BE, int64 LE/BE
- All unsigned types: uint32/64 LE/BE
- Float types: float32, float64
- Varint types: v (unsigned), V (signed)
- String types: s (Pascal), S (C string), %Ns (fixed)
- Length field types: L (uint8), ll (uint16 LE), LL (uint32 LE)
- CRC types: CRC8, CRC16
- Field at non-zero offset for each type
- Named vs unnamed field storage for each type
- Partial data handling for each type (should return statusNeedMore)

#### 8.1.4 TestSetMaxLengthOpInteractions
**Priority:** High
**Test Cases:**
- MaxLength clamping with existing DeclaredLength (MaxLength < DeclaredLength)
- MaxLength = 0 (unlimited) behavior
- MaxLength < current packet length (should drop)
- Multiple SetMaxLengthOp calls (should use minimum value)
- MaxLength with wildcard * (should limit wildcard consumption)
- MaxLength with offset jumps (should drop if jump beyond MaxLength)

#### 8.1.5 TestCheckLengthOpEdgeCases
**Priority:** Medium
**Test Cases:**
- DeclaredLength = 0 edge case (should continue, no check)
- DeclaredLength > MaxLength interaction (conflict resolution)
- Length check at non-final position in sequence (early validation)
- DeclaredLength exactly equals packet length (should emit)
- DeclaredLength with partial packet (should continue)

#### 8.1.6 TestSequenceOpStress
**Priority:** Low
**Test Cases:**
- Empty sequence (no children) - edge case
- Sequence with 10+ children (stress test)
- Sequence where child decision changes on re-evaluation (backtracking)
- Error propagation through sequence (first error should stop evaluation)

#### 8.1.7 TestChoiceOpStress
**Priority:** Low
**Test Cases:**
- Choice with 10+ branches (stress test)
- Choice where multiple branches could match (first match wins behavior)
- Choice with nested sequences
- Choice backtracking behavior verification

### 8.2 Parser Tests (`parser_comprehensive_test.go` - NEW FILE)

#### 8.2.1 TestParserAllFieldTypes
**Priority:** High
**Test Cases:**
- Named fields for each type (%name:uu, %name:ii, %name:f, etc.)
- Array variants for each type (%3u, %5uu, %10ii, etc.)
- Varint types (v, V) parser recognition
- String types (s, S, %Ns) parser recognition
- Expression fields for each type (%(uu*2), %(ii+10), %(f/360.0))
- Struct fields for each type (%{x:uu,y:ii})
- Combined: named array of structs (%3{x:uu,y:ii})

#### 8.2.2 TestParserErrorHandlingComprehensive
**Priority:** High
**Test Cases:**
- Empty pattern string (should error)
- Pattern with only whitespace (should error or be valid?)
- Multiple consecutive errors (parser recovery)
- Invalid offset jump syntax (#abc, #-abc, #123.45)
- Invalid expression syntax:
  - Unmatched parentheses: %(uu+10
  - Invalid operators: %(uu@10)
  - Missing operand: %(uu+)
  - Invalid number: %(uu/abc)
- Invalid struct syntax:
  - Unmatched braces: %{x:uu
  - Missing colons: %{x uu}
  - Invalid separators: %{x:uu|y:ii}
- Invalid array syntax:
  - Non-numeric count: %abc
  - Invalid star: %*abc (should be %*name:type)
  - Negative count: %-5u

#### 8.2.3 TestParserEdgeCasesComprehensive
**Priority:** Medium
**Test Cases:**
- Pattern with only anchor (^$) - empty pattern
- Nested groups depth (3+ levels): ^((((AA))))$
- Very long pattern (1000+ characters) - performance
- Pattern with special characters in literals (hex bytes 0x00-0xFF)
- Multiple anchors (^...$ with multiple $) - should error or last one wins?
- Pattern with only wildcard: ^*$
- Pattern with only field: ^%u$

#### 8.2.4 TestParserNumberParsing
**Priority:** Medium
**Test Cases:**
- Number overflow (very large numbers: *1000000, #1000000, %1000000u)
- Negative numbers where not allowed (array count, offset jumps - should error)
- Number parsing in wildcards (*1000000) - boundary testing
- Number parsing in offset jumps (#1000000) - boundary testing
- Number parsing in arrays (%1000000u) - boundary testing
- Zero values (*0, #0, %0u) - edge cases

#### 8.2.5 TestParserComplexPatterns
**Priority:** Medium
**Test Cases:**
- Deep nesting: ^((((((AA))))))$
- Mixed nesting: ^(AA|(BB|CC))%u((DD|EE)%u)$
- Complex structs: %{x:uu,y:{a:ii,b:f},z:u}
- Complex arrays: %3{x:uu,y:%5u}
- Complex expressions: %((uu+length)*2-100)/scaling
- All features combined: ^55AA*%length:uu%3{x:uu,y:(field_1*2)}@(length>100)$

### 8.3 Program/Expression Graph Tests (`program_comprehensive_test.go` - NEW FILE)

#### 8.3.1 TestDecisionTransitions
**Priority:** High
**Test Cases:**
- Continue → Drop (mismatch discovered: ^AABB, data: [0xAA, 0xBC])
- Continue → Continue → Emit (incremental: ^AA%u, data: [0xAA] → [0xAA, 0x42])
- Continue → Continue → Continue → Emit (multiple fields)
- Continue → Emit (single byte match: ^AA$, data: [0xAA])
- Drop → (should not transition, packet reset)
- Emit → (should not transition, packet complete)

#### 8.3.2 TestDecisionWithConstraints
**Priority:** High
**Test Cases:**
- MaxLength = DeclaredLength edge case (both equal, packet exactly that length)
- MaxLength < DeclaredLength conflict (which constraint wins?)
- MaxLength > DeclaredLength (both must be satisfied)
- DeclaredLength with MaxLength exceeded (should drop on MaxLength first)
- Multiple length fields (which one wins? last one?)
- Length field interaction with MaxLength (DeclaredLength can't exceed MaxLength)

#### 8.3.3 TestDecisionWithOffsetJumps
**Priority:** High
**Test Cases:**
- Offset jump to negative offset (should error or drop)
- Offset jump beyond packet length (should Continue, wait for more bytes)
- Offset jump beyond MaxLength (should Drop)
- Multiple consecutive offset jumps (#5#10%u)
- Offset jump with expressions (#(length+10)%u) - when implemented
- Offset jump to field boundary (verify correct field reading after jump)
- Backward offset jump (#-5) with anchor end (coverage check)

#### 8.3.4 TestDecisionWithExpressions
**Priority:** High
**Test Cases:**
- Expression division by zero: %(uu/0) should error (returns DecisionDrop)
- Expression overflow (very large numbers: %(uu*1000000000))
- Expression with parentheses nesting (3+ levels): %(((uu+10)*2)-5)
- Expression field reference errors (non-existent field - already tested)
- Expression type conversions (int to float, etc.)
- Expression boolean result in field (when used as condition)

#### 8.3.5 TestDecisionWithGroups
**Priority:** Medium
**Test Cases:**
- Group with all branches failing (should Drop)
- Group with one branch emitting (should Emit)
- Group with one branch continuing (should Continue)
- Nested groups decision propagation
- Group with backtracking (first branch partially matches then fails)
- Group with offset jumps (backtracking with state restoration)

#### 8.3.6 TestDecisionWithWildcards
**Priority:** Medium
**Test Cases:**
- Wildcard * with MaxLength (should drop if exceeds)
- Wildcard *N with MaxLength (should drop if exceeds)
- Wildcard at packet start (^*AA)
- Wildcard at packet end (^AA*$)
- Multiple consecutive wildcards (^*5*10%u)
- Wildcard with offset jumps (^*5#10%u)
- Wildcard with expressions (when expressions support offset calculation)

#### 8.3.7 TestMismatchScenarios
**Priority:** High
**Test Cases:**
- Mismatch after partial match (^AABB, data: [0xAA, 0xBC])
- Mismatch in group alternation (all branches fail: ^(AA|BB|CC)$, data: [0xDD])
- Mismatch in nested sequences (^AA(BB(CC))$, data: [0xAA, 0xBB, 0xCD])
- Mismatch with wildcards (^AA*BB$, data: [0xAA, 0xFF, 0xFF, 0xBC])
- Mismatch with expressions (%(uu>100) condition false)
- Mismatch with conditional guard (@(end>start) condition false)

### 8.4 Pattern Integration Tests (`pattern_integration_test.go` - NEW FILE)

#### 8.4.1 TestExpressionErrorCases
**Priority:** High
**Test Cases:**
- Expression division by zero: %(uu/0) - should return DecisionDrop with error
- Expression overflow: %(uu*1e20) - very large result
- Expression underflow: %(uu/1e20) - very small result
- Expression with NaN/Infinity results
- Expression with invalid field reference (already tested, but verify error propagation)

#### 8.4.2 TestOffsetJumpEdgeCases
**Priority:** High
**Test Cases:**
- Offset jump to negative offset (#-5) - should error or handle gracefully
- Offset jump beyond packet length (#1000) - should Continue
- Offset jump beyond MaxLength (#1000 with MaxLength=100) - should Drop
- Multiple consecutive offset jumps (#5#10#15%u) - verify state
- Offset jump to field boundary (verify correct alignment)
- Offset jump backward then forward (#-5#10%u)
- Offset jump in group with backtracking (state restoration)

#### 8.4.3 TestWildcardWithMaxLength
**Priority:** High
**Test Cases:**
- Wildcard * with MaxLength=10, packet=15 bytes (should drop at 10)
- Wildcard *5 with MaxLength=3 (should drop before matching)
- Wildcard * with MaxLength and fields after (*%u with MaxLength=5, need 2 for %u)
- Wildcard * consuming exactly MaxLength bytes
- Wildcard * with MaxLength=0 (unlimited)

#### 8.4.4 TestStructVariations
**Priority:** Medium
**Test Cases:**
- Struct with named fields vs unnamed (%{x:uu,y:u} vs %{uu,u})
- Struct with arrays (%{x:%3u,y:uu})
- Struct with wildcards (%{x:uu}*5{y:u})
- Empty struct %{} - should error or be valid?
- Struct with expressions (%{x:uu,y:(x*2)})
- Array of structs %3{x:uu,y:u}
- Nested structs %{a:{x:uu,y:u},b:ii}

#### 8.4.5 TestArrayEdgeCases
**Priority:** Medium
**Test Cases:**
- Star array with MaxLength (%*data:uu with MaxLength=100)
- Star array with DeclaredLength (%*data:uu with DeclaredLength=50)
- Star array with odd packet length (partial element: 5 bytes for %*uu)
- Star array field naming (verify data_0, data_1, etc.)
- Star array with struct (%*{uu,u} with partial struct at end)
- Stride array boundary conditions (%3:2u with insufficient bytes)
- Stride array with offset jumps (%3:2u#10)

#### 8.4.6 TestConditionalGuardVariations
**Priority:** Medium
**Test Cases:**
- Guard with complex expression (@((end-start)>100))
- Guard with field references (@(length>100 && value<200))
- Guard in sequence (early drop: ^%length:uu@(length>100)%data:uu)
- Guard in alternation (branch filtering: ^(AA@(length>100)|BB)%length:uu)
- Multiple guards in sequence
- Guard with expression field reference

#### 8.4.7 TestRealWorldProtocols
**Priority:** Low
**Test Cases:**
- Simulate Modbus RTU protocol parsing
- Simulate CAN bus frame parsing
- Simulate serial communication protocol
- Complex nested protocol structures
- Protocol with variable length fields

#### 8.4.8 TestStressTests
**Priority:** Low
**Test Cases:**
- Large packets (10KB+) with many fields
- Many fields (100+) in single pattern
- Deep nesting (10+ levels)
- Complex expressions (10+ operations)
- Many branches in alternation (50+ branches)
- Memory usage with streaming (large packet accumulation)

#### 8.4.9 TestErrorRecovery
**Priority:** Medium
**Test Cases:**
- Malformed packets (wrong length, invalid data)
- Partial data handling (byte-by-byte accumulation)
- Packet boundary detection
- Recovery after drop (new packet start)
- Invalid field data (corrupted varint, invalid UTF-8)

#### 8.4.10 TestMultiPacketScenarios
**Priority:** Medium
**Test Cases:**
- Handling multiple packets in sequence
- Packet delimiter detection
- Interleaved packets (packet boundary in middle)
- Packet drop and continue scenarios
- State reset between packets

### 8.5 Field Decoding Tests (`field_comprehensive_test.go` - NEW FILE)

#### 8.5.1 TestFieldDecodingEdgeCases
**Priority:** High
**Test Cases:**
- All field types at non-zero offset
- All field types with partial data (should return statusNeedMore)
- All field types with boundary values (max/min for integers, ±Inf for floats)
- Field decoding error cases (corrupted varint, invalid UTF-8)
- Field decoding with MaxLength constraint
- Field decoding spanning packet boundary

#### 8.5.2 TestVarintEdgeCases
**Priority:** Medium
**Test Cases:**
- Varint overflow (shift >= 64) - should error
- Varint at non-zero offset
- Varint with MaxLength constraint
- Varint with many continuation bits (10+ bytes, malformed)
- Varint with invalid continuation bit pattern

#### 8.5.3 TestStringFieldEdgeCases
**Priority:** Medium
**Test Cases:**
- Pascal string with length = 0
- Pascal string with length > MaxLength
- Pascal string with invalid UTF-8
- Pascal string at non-zero offset
- C string without null terminator (should Continue)
- Fixed string with insufficient bytes
- String field with very long content (1000+ bytes)

### 8.6 State Management Tests (`state_comprehensive_test.go` - NEW FILE)

#### 8.6.1 TestStateCloneComprehensive
**Priority:** Medium
**Test Cases:**
- Clone with fields, packet, offsets, lengths
- Clone performance with large state (10KB packet, 100 fields)
- Clone with nil values (edge case)
- Clone independence (modify clone, verify original unchanged)
- Clone with DeclaredLength and MaxLength

#### 8.6.2 TestStateSetOffsetEdgeCases
**Priority:** Medium
**Test Cases:**
- SetOffset beyond packet length (should clamp or error?)
- SetOffset negative (should error or clamp to 0)
- SetOffset with empty packet
- SetOffset with MaxLength (offset > MaxLength?)

#### 8.6.3 TestStateMergeConflicts
**Priority:** Medium
**Test Cases:**
- Merge with conflicting lengths (which wins?)
- Merge with duplicate field names (both kept or one overwrites?)
- Merge with different offsets
- Merge with different MaxLength values

### 8.7 Fuzz Tests (`fuzz_test.go` - NEW FILE)

#### 8.7.1 FuzzPatternParser
**Priority:** Low
**Test Cases:**
- Fuzz pattern parsing with random strings
- Fuzz pattern parsing with valid syntax variations
- Verify parser never panics (only returns errors)
- Verify parser handles all ASCII/UTF-8 characters

#### 8.7.2 FuzzPacketMatching
**Priority:** Low
**Test Cases:**
- Fuzz packet matching with valid patterns
- Random packet data against known patterns
- Verify Decision is always valid (Emit, Drop, or Continue)
- Verify no panics on any input

#### 8.7.3 FuzzFieldDecoding
**Priority:** Low
**Test Cases:**
- Fuzz field decoding with random data
- All field types with random bytes
- Verify graceful error handling (no panics)
- Verify partial data handling

---

## 9. Summary Statistics

### Test Files
- **ops_basic_test.go**: 3 test functions, ~235 lines
- **ops_control_test.go**: 5 test functions, ~205 lines
- **parser_test.go**: 5 test functions, ~189 lines
- **program_test.go**: 5 test functions, ~133 lines
- **pattern_basic_test.go**: 2 test functions, ~126 lines
- **pattern_advanced_test.go**: 12 test functions, ~1055 lines (updated count)
- **pattern_features_test.go**: 8 test functions, ~144 lines
- **pattern_user_requirements_test.go**: 1 test function (multiple sub-tests), ~136 lines
- **field_test.go**: 7 test functions, ~276 lines
- **state_test.go**: 3 test functions, ~95 lines

**Total:** ~2,594 lines of test code across 10 files (updated with recent additions)

### Coverage by Component
- **Operations**: ~60% coverage (basic cases, missing edge cases)
- **Parser**: ~70% coverage (good error handling, missing complex patterns)
- **Program/Expression Graph**: ~50% coverage (basic decisions, missing complex scenarios)
- **Pattern Matching**: ~65% coverage (many features tested, missing combinations)
- **State Management**: ~70% coverage (basic operations, missing stress tests)
- **Field Decoding**: ~75% coverage (most types, missing edge cases)

### Critical Missing Tests
1. ~~Wildcard * (no number) comprehensive testing~~ ✅ **FIXED** - TestWildcardSkipWhenFollowedByFields and TestWildcardWithCount
2. Expression evaluation error cases ⚠️ (partial: missing division by zero, overflow)
3. Offset jump edge cases ⚠️
4. Decision calculation correctness in complex scenarios ⚠️
5. Performance and stress tests
6. Fuzz tests

### Recently Added Tests (Based on DECISIONS in Section 10)
1. ✅ **TestExpressionsWithNamedFields** - Named field references in expressions
2. ✅ **TestExpressionsWithUnnamedFields** - Unnamed field references (field_1, field_2)
3. ✅ **TestExpressionFieldSyntax** - Expression syntax variations (parentheses, semicolons, %(uu))
4. ✅ **TestNamedExpressionField** - Named expression fields (%start:(uu-2000))
5. ✅ **TestWildcardWithCount** - Wildcard with explicit count (*5 matches exactly 5 bytes)
6. ✅ **TestWildcardSkipWhenFollowedByFields** - Updated to match DECISION: * skips EXACTLY 1 byte

---

## 10. Pattern Format Clarifications

Before implementing additional tests, the following questions about pattern format semantics need clarification:

### 10.1 Expression Field References

**Question 1: Field reference by type token vs field name**

Given pattern: `^%length:uu%start:(uu-2000)%end:(uu-2000)$` with data `[0x68, 0x01, 0x68, 0x01, 0x68, 0x01]` (three uint16 values = 360, 360, 360)

- Does `%start:(uu-2000)` decode a new `uu` field and use its value in the expression?
  - Example: Decodes bytes 2-3 (second uu), evaluates (360-2000) = -1640
  
- Or does `uu` in `(uu-2000)` refer to the previously decoded field `length`?
  - Example: Uses length=360, evaluates (360-2000) = -1640, but what bytes does it consume?
  
- If we want to reference `length` field explicitly, should the syntax be:
  - `%start:(length-2000)` - reference by field name?
  - Or is there a different syntax?

   **DECISION**: %start:(uu-2000) acts the same way as %(uu-2000) which acts the same way as %uu except that it also subtracts 2000 from decoded value and assigns name "start" to the field.

**Question 2: Type token at start of expression**

Given pattern: `^%uu%(uu/360.0)$` with data `[0x68, 0x01, 0x68, 0x01]` (two uint16 = 360, 360)

Current test expects this decodes TWO fields:
1. First `%uu` decodes bytes 0-1 = 360
2. `%(uu/360.0)` decodes bytes 2-3 = 360, then evaluates (360/360.0) = 1.0

- Is this correct? Should `%(uu/360.0)` decode its own `uu` field?
- Or should it reference the first decoded `uu` field?
- If we want to reference the first field, what's the syntax?

**DECISION**: Every % decodes bytes and advances the position by the number of bytes the type occupies. So first %uu decodes two byts and advances position two bytes. %(uu/360.0) decodes new two bytes, casts to float and devides by 360.0

   **TEST STATUS**: ✅ Tested in TestExpressions

**Question 3: Type token in middle of expression**

Given pattern: `^%length:uu%(start:(uu-2000))$` with expression `(start:(uu-2000))`
- If expression is `(start:(uu-2000))`, does the `uu` inside the nested parentheses decode a field?
- Or does `uu` always refer to a previously decoded field when it's not at the start?

**DECISION**: uu always decodes new bytes. In general, new `%` decodes new bytes always. If referring to other fields - we'd use the field name in the expression. Users cannot use field names that are any of the reserved keywords like types or checksum alias. e.g. `%scaling:f%value:((uu - 200)/scaling)` would decode float field named scaling, after that decode uint16, subtract 200, and divide by previously read scaling field and store the result in field named value. in other words - uu is not a valid field reference - it is a type, same as ii, or s, etc.

   **TEST STATUS**: ✅ Tested in TestExpressionsWithUnnamedFields (cannot_reference_unnamed_field_by_type_token)

### 10.2 Named Field References

**Question 4: Referencing named fields in expressions**

Given pattern: `^%length:uu%start:(length-2000)%end:(length-2000)$`
- Does `(length-2000)` reference the field named `length`? **DECISION** yes
- What happens if no field named `length` exists - parse error or runtime error? **DECISION** parse error
- Can expressions reference multiple fields? Example: `%(length+offset)` where both are previously decoded? **DECISION** yes. However, it should be possible to reference fields we don't have bytes yet - so need to defer the evaluation until we have bytes. We don't support this yet, but parser should not fail now. expression graph should(since it couldn't find the field that hasn't been decoded yet!).

   **TEST STATUS**: ✅ Tested in TestExpressionsWithNamedFields (reference_named_field_in_expression, multiple_field_references_in_expression, reference_non-existent_field)
   **IMPLEMENTATION STATUS**: ⚠️ Currently runtime DecisionDrop, not parse error - needs parse-time validation

### 10.3 Unnamed Field References

**Question 5: Referencing unnamed fields**

Given pattern: `^%uu%derived:(uu*2)$` where first `%uu` is unnamed
- Can `(uu*2)` reference the unnamed field by type token `uu`? **DECISION** NO
- Or must unnamed fields be referenced differently? **DECISION** field_1, field_2, etc.
- What if multiple unnamed `uu` fields exist - which one is referenced? **DECISION** each unnamed field has unique name determined during parsing in order of appearance.

   **TEST STATUS**: ✅ Tested in TestExpressionsWithUnnamedFields

### 10.4 Wildcard Behavior

**Question 6: Wildcard skip behavior edge cases**

Given pattern: `^AA*%u$` with data `[0xAA, 0x42]`
- Does `*` skip 0 bytes (no skip) or at least 1 byte? **DECISION** skips EXACTLY 1 byte
- Should pattern `^AA*%u$` match `[0xAA, 0x42]` (skip 0) or only `[0xAA, 0xFF, 0x42]` (skip 1)? **DECISION** skip 1

Given pattern: `^AA*%u$` with data `[0xAA]` (only one byte after AA)
- Should `*` wait for more bytes, or skip 0 bytes and try to read `%u` (which needs more)? **DECISION** wait for more data. Basically whildcard means any byte matches. so there must be a byte.

   **TEST STATUS**: ✅ Tested in TestWildcardSkipWhenFollowedByFields (wildcard_skips_exactly_1_byte, wildcard_with_insufficient_bytes)

**Question 7: Wildcard with exactly enough bytes**

Given pattern: `^AA*%u$` with exactly 3 bytes total: `[0xAA, 0xFF, 0x42]`
- Does `*` skip 1 byte (leaving room for `%u`), or consume all remaining bytes? **DECISION** skip exactly 1 byte. or "matches any byte". Similarly `*5` matches exactly any 5 bytes, thus must wait for 5 bytes.

   **TEST STATUS**: ✅ Tested in TestWildcardSkipWhenFollowedByFields (wildcard_skips_exactly_1_byte) and TestWildcardWithCount

### 10.5 Conditional Guards

**Question 8: Field references in conditional guards**

Given pattern: `^%start:uu%end:uu@(end>start)$`
- Does `@(end>start)` reference fields by name (`end`, `start`)? **DECISION** yes, only by field names
- What if field names don't match - parse error or runtime error? **DECISION** must be decided during parsing. field names must exist.
- Can conditions reference type tokens like `@(uu>0)` where `uu` refers to a field? **DECISION** it can

### 10.6 Expression Base Type

**Question 9: Base type in expression evaluation**

Given pattern: `^%(uu/360.0)$` (no preceding field)
- Does this decode a `uu` field first, then evaluate `(uu/360.0)`? **DECISION** decode first, evaluate later
- Or is this an error because there's no field to decode?
- What should happen if the expression doesn't start with a type token? **DECISION** parser error

   **TEST STATUS**: ✅ Tested in TestExpressions

**Question 10: Multiple expressions with same type token**

Given pattern: `^%length:uu%start:(uu-2000)%end:(uu-2000)$`
- Does each `(uu-2000)` decode its own `uu` field from the packet? **DECISION** each `%` decode their own, each `%` must contain exactly one type token unless it is a struct, then each struct element must have exactly one type token. struct fields can have name too, e.g. `%{field1:uu,float:f}`
- Or do they both reference the same `length` field? **DECISION** already explained
- What's the byte consumption for each expression? **DECISION** each % consumes the number of bytes needed to encode the type

### 10.7 Expression Result Storage

**Question 11: Expression field name and value**

Given pattern: `^%length:uu%derived:(uu/360.0)$`
- The expression `(uu/360.0)` - is the field name `derived` or auto-generated? **DECISION** derived. the syntax is `name:value`, where value has a type token and expression.
- What's the type of the result field? (Always float64?) **DECISION** float64 if dividing by a float
- Does the base field (decoded by type token in expression) also get stored, or only the expression result? **DECISION** base should be stored too

   **TEST STATUS**: ✅ Tested in TestNamedExpressionField, TestExpressions
   **IMPLEMENTATION STATUS**: ⚠️ TODO - Currently only expression result is stored, base field is not stored separately

### 10.8 Pattern Syntax Edge Cases

**Question 12: Expression without parentheses**

- Is `%uu/360.0` valid, or must expressions always be in parentheses `%(uu/360.0)`? **DECISION** either parentheses or ending in `;`
- What about `%length:uu/360.0` vs `%length:(uu/360.0)`? **DECISION** same - either parentheses or `;`. parentheses are needed to group operations and separate from the rest of the pattern.

   **TEST STATUS**: ✅ Tested in TestExpressionFieldSyntax (expression_with_parentheses, expression_without_parentheses_ending_in_semicolon)
   **IMPLEMENTATION STATUS**: ⚠️ TODO - Parser doesn't support expressions without parentheses ending in `;` yet

**Question 13: Empty expressions**

- Is `%()` valid? What should it do? **DECISION** invalid
- Is `%(uu)` valid? Does it just decode and store the field? **DECISION** valid, decode and store the field

   **TEST STATUS**: ✅ Tested in TestExpressionFieldSyntax (empty_expression, expression_decode_only)

### 10.9 Array and Struct in Expressions

**Question 14: Arrays in expressions**

- Can expressions reference array elements? Example: `%(data_0+data_1)` where `data` is an array? **DECISION** no
- Or `%(data[0]+data[1])` syntax? **DECISION** leave this for a future feature, because that also leads to iterating over elements if we want to reference in arrays which leads to complex computations... this blows up pretty quickly.

   **TEST STATUS**: ❌ Not yet implemented or tested

**Question 15: Struct fields in expressions**

- Can expressions reference struct fields? Example: `%{x:%uu,y:%uu}%(x+y)`? **DECISION** should be possible to reference struct fields, though out of scope now, add as a feature
- What's the syntax for nested structs? `%{a:%{x:%uu,y:%uu}}%(a.x+a.y)`? **DECISION** this is the way.

   **TEST STATUS**: ❌ Not yet implemented or tested

---

**Note:** These clarifications will guide test implementation and may reveal bugs in current implementation.
