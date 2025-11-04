# Tensor Rename Analysis

## Proposed Changes

1. **Rename `Tensor` struct to `EagerTensor`**
2. **Rename `TensorInterface` to `Tensor`** (the interface becomes the public API)
3. **Update all method receivers from `Tensor` to `EagerTensor`**
4. **Update all function parameters accepting `Tensor` to accept `Tensor` interface**

## Consequences Analysis

### ✅ Positive Consequences

1. **Clear Separation of Concerns**
   - `EagerTensor` clearly indicates eager evaluation semantics
   - `Tensor` interface provides abstraction for future implementations
   - Enables future lazy evaluation, views, and other tensor implementations

2. **Better API Design**
   - Interface-based design follows Go best practices
   - Allows for multiple tensor implementations without breaking consumers
   - Supports dependency injection and testing

3. **Future Extensibility**
   - Can add `LazyTensor`, `ViewTensor`, `SparseTensor`, etc. implementing `Tensor`
   - Consumers work with interface, not concrete type
   - Easier to add optimizations behind the interface

4. **Type Safety Maintained**
   - All existing operations continue to work
   - Compiler will catch breaking changes

### ⚠️ Breaking Changes

#### 1. **All Consumer Code Must Update**

**Impact: HIGH** - Every file that imports `tensor` package will need changes:

```go
// OLD
var t tensor.Tensor
t = tensor.New(tensor.DTFP32, shape)

// NEW  
var t tensor.Tensor  // Now an interface
t = tensor.NewEager(tensor.DTFP32, shape)  // or New() returns Tensor interface
```

**Affected Packages:**
- `pkg/core/math/nn/layers/*` - All layer implementations
- `pkg/core/math/nn/losses.go` - Loss functions
- `pkg/core/math/learn/*` - Training and optimization code
- All test files throughout the codebase

**Estimated Files to Update: ~50-100 files**

#### 2. **Function Return Types**

**Impact: MEDIUM**

```go
// OLD
func New(dtype DataType, shape Shape) Tensor  // Returns struct
func (t Tensor) Clone() *Tensor                // Returns pointer to struct

// NEW
func New(dtype DataType, shape Shape) Tensor           // Returns interface
func NewEager(dtype DataType, shape Shape) EagerTensor // Returns concrete type if needed
func (t *EagerTensor) Clone() Tensor                   // Returns interface
```

**Key Considerations:**
- Functions returning `*Tensor` now return `Tensor` interface
- Need to decide: return interface or concrete type?
  - **Recommendation**: Return interface for public API
  - Internal implementations can use concrete types

#### 3. **Value vs Pointer Semantics**

**Impact: HIGH**

Current codebase uses **value semantics** for parameters:
```go
func (t *Tensor) Add(other Tensor) *Tensor  // other is value
```

With interface, **must use value semantics** (interfaces are values in Go):
```go
func (t *EagerTensor) Add(other Tensor) Tensor  // other is interface value
```

**Consequences:**
- Interface values contain pointer to concrete type (one indirection)
- Slight performance overhead from interface dispatch
- Method calls become virtual calls (vtable lookup)
- Value assignments copy interface value (not the underlying data)

#### 4. **Type Assertions Required**

**Impact: MEDIUM**

When concrete type is needed (e.g., for internal operations):

```go
// OLD
func process(t tensor.Tensor) {
    // Direct access to t.data, t.shape
}

// NEW
func process(t tensor.Tensor) {
    eager, ok := t.(*tensor.EagerTensor)
    if !ok {
        panic("expected EagerTensor")
    }
    // Access eager.data, eager.shape
}
```

**Affected Areas:**
- Internal helper functions (`copyTo`, `sameShape`, etc.)
- Functions that need direct data access
- Performance-critical paths

#### 5. **Nil Handling**

**Impact: MEDIUM**

```go
// OLD
var t *tensor.Tensor  // Can be nil
if t == nil { ... }

// NEW
var t tensor.Tensor  // Interface can be nil
if t == nil { ... }  // Still works, but different semantics
```

**Key Difference:**
- Interface nil requires both type and value to be nil
- Need careful nil checks in all code

#### 6. **Helper Functions**

**Impact: LOW**

```go
// OLD
func ZerosLike(t tensor.Tensor) *tensor.Tensor

// NEW
func ZerosLike(t tensor.Tensor) tensor.Tensor  // Returns interface
```

Need to update all helper functions to return interface.

#### 7. **Test Code Impact**

**Impact: HIGH**

All test code needs updates:
- Type assertions for concrete types in tests
- Comparison logic changes
- Mock implementations become possible (positive)

```go
// OLD
t := tensor.New(...)
assert.Equal(t, expectedShape, t.Shape())

// NEW  
t := tensor.New(...)  // Returns Tensor interface
eager := t.(*tensor.EagerTensor)  // If concrete access needed
assert.Equal(t, expectedShape, t.Shape())  // Interface methods work
```

### 🔧 Implementation Challenges

#### 1. **Method Receiver Types**

**Decision Needed:**
- Keep `*EagerTensor` receivers? (mutation methods)
- Keep `EagerTensor` receivers? (non-mutating methods)

**Recommendation:**
- In-place operations: `func (t *EagerTensor) Add(other Tensor) Tensor`
- Non-mutating: `func (t EagerTensor) Clone() Tensor` or `func (t *EagerTensor) Clone() Tensor`

#### 2. **Internal vs External API**

**Challenge:** Some operations need concrete type internally

**Solution:** 
- Public API uses `Tensor` interface
- Internal functions can use `*EagerTensor` directly
- Use type assertions at boundaries

#### 3. **Backward Compatibility**

**Options:**
1. **Breaking change** (recommended) - Clean slate, force migration
2. **Type alias transition** - `type Tensor = EagerTensor` temporarily
3. **Dual support** - Support both for a version

### 📊 Migration Effort Estimate

| Category | Files | Complexity | Time Estimate |
|----------|-------|------------|---------------|
| Core tensor package | 8-10 | High | 2-3 days |
| NN layers package | 15-20 | Medium | 2-3 days |
| Loss functions | 3-5 | Low | 0.5 days |
| Learn package | 5-10 | Medium | 1-2 days |
| Test files | 20-30 | Medium | 2-3 days |
| Documentation | All | Low | 1 day |
| **Total** | **50-75** | **Mixed** | **8-13 days** |

### 🎯 Recommendations

1. **Do the Rename** - Benefits outweigh costs for long-term maintainability

2. **Phased Approach:**
   - Phase 1: Create interface, keep struct as `Tensor` (no breaking change)
   - Phase 2: Rename struct to `EagerTensor`, update package
   - Phase 3: Update all consumers
   - Phase 4: Remove old type aliases

3. **API Design Decisions:**
   - **Return interface** from public constructors: `New() Tensor`
   - **Accept interface** in all public functions
   - **Use concrete type** internally in package
   - **Pointer receivers** for all methods (consistent with current)

4. **Testing Strategy:**
   - Create mock `Tensor` implementation for testing
   - Ensure all tests pass with interface
   - Add interface conformance tests

5. **Performance Considerations:**
   - Benchmark interface vs direct calls
   - Consider keeping hot paths with concrete types
   - Profile to identify bottlenecks

### ⚡ Performance Impact

**Expected Overhead:**
- Interface method calls: ~1-2ns overhead per call (vtable lookup)
- Interface value copying: Negligible (interface is 2 words)
- Type assertions: ~1-2ns when successful

**Mitigation:**
- Most operations already use method calls (no change)
- Type assertions only at package boundaries
- Compiler optimizations (devirtualization) can help

**Conclusion:** Performance impact should be minimal (< 5% overhead in typical usage)

### 📝 Breaking Changes Summary

1. ✅ All `tensor.Tensor` variables now hold interface
2. ✅ All function signatures accept `Tensor` interface
3. ✅ Return types changed from `*Tensor` to `Tensor`
4. ✅ Type assertions needed for concrete type access
5. ✅ Helper functions return `Tensor` interface
6. ✅ Nil semantics slightly different (but compatible)

### 🔄 Migration Checklist

- [ ] Create `Tensor` interface from `TensorInterface`
- [ ] Rename `Tensor` struct to `EagerTensor`
- [ ] Update all method receivers
- [ ] Update constructors to return interface
- [ ] Update helper functions (`ZerosLike`, `OnesLike`, etc.)
- [ ] Add type assertions in internal code
- [ ] Update all layer implementations
- [ ] Update loss functions
- [ ] Update learn package
- [ ] Update all test files
- [ ] Update documentation
- [ ] Benchmark performance
- [ ] Run full test suite
- [ ] Update examples
