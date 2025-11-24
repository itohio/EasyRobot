# Chain Package

The `chain` package provides generic, type-safe mechanisms for chaining and composing data processors in sequence. It is designed for robotics and embedded systems where efficient data transformation pipelines are critical.

## Overview

This package provides two main abstractions:

1. **Chain**: Chains multiple processors where all processors have the same input and output type (Processor[T, T])
2. **Adapter**: Composes two processors with different types, allowing type conversion (SRC -> MID -> DST)

Both abstractions support:
- Lazy buffer allocation (buffers allocated on first use)
- Size calculation mode (pass `nil` destination to calculate output size)
- Zero-copy operations where possible
- Efficient memory management with configurable buffer factories

## Core Concepts

### Processor Interface

All processors must implement the `Processor[DST, SRC]` interface:

```go
type Processor[DST, SRC Sample] interface {
    Process(dst buffer.Writer[DST], src buffer.Reader[SRC]) int
    Reset()
}
```

- `Process`: Transforms data from `SRC` to `DST`. Returns number of samples written.
  - If `dst` is `nil`, calculates theoretical maximum output size
  - If `dst` is not `nil`, performs actual processing
- `Reset`: Resets processor state for reuse

### Sample Type Constraint

The `Sample` type constraint allows numeric types:

```go
type Sample interface {
    types.Number | types.Unsigned
}
```

This includes: `int`, `int8`, `int16`, `int32`, `int64`, `uint`, `uint8`, `uint16`, `uint32`, `uint64`, `float32`, `float64`, `complex64`, `complex128`, and `byte`.

### Streamer Interface

Buffers must implement the `Streamer[T]` interface:

```go
type Streamer[T any] interface {
    buffer.Writer[T]
    buffer.Reader[T]
    buffer.Initializer
}
```

This ensures buffers can both read and write, and support initialization operations.

## Chain

`Chain` sequences multiple processors where all processors have the same input and output type (Processor[T, T]).

**Important**: Chain requires all processors to have the same input and output type. If you need to convert between different types, use `Adapter` instead.

### Use Cases

- Multiple processing layers: `encryption -> reed-solomon -> manchester` (all operate on same type)
- Signal processing pipelines: `filter -> normalize -> compress` (all operate on same type)
- Data validation chains: `validate -> sanitize -> transform` (all operate on same type)

### Example

```go
// Create buffer factory
factory := func(cap int) *buffer.Ring[float32] {
    return buffer.NewRing(make([]float32, 0, cap))
}

// Create processors (all must be Processor[float32, float32])
processor1 := NewFilterProcessor[float32, float32]()
processor2 := NewNormalizeProcessor[float32, float32]()
processor3 := NewCompressProcessor[float32, float32]()

// Create chain - note: single type parameter T
chain := New[float32, *buffer.Ring[float32]](
    factory,
    processor1, processor2, processor3,
)

// Process data
src := buffer.New([]float32{1.0, 2.0, 3.0})
dst := buffer.New(make([]float32, 0, 1024))
written := chain.Process(dst, src)
```

### Processing Flow

```
src (T) -> processor1 -> buffer1 (T) -> processor2 -> buffer2 (T) -> processor3 -> dst (T)
```

All processors and buffers use the same type T, so no type conversion is needed.

## Adapter

`Adapter` composes two processors with different types, allowing type conversion in the pipeline.

**Use Adapter when**: You need to connect processors with different input/output types.
- First processor: SRC -> MID (reads SRC, writes MID)
- Second processor: MID -> DST (reads MID, writes DST)
- The intermediate buffer stores MID type data, bridging the type gap

**Use Chain when**: All processors have the same input/output type (Processor[T, T]).

### Use Cases

- Type conversion: `byte -> []byte -> float32`
- Format conversion: `raw -> encoded -> processed`
- Multi-stage processing: `source format -> intermediate format -> destination format`

### Example

```go
// Create buffer factory for intermediate type
factory := func(cap int) *buffer.Ring[[]byte] {
    return buffer.NewRing(make([][]byte, 0, cap))
}

// Create processors with different types
srcProcessor := NewByteToBytesProcessor()  // Processor[[]byte, byte]
dstProcessor := NewBytesToFloatProcessor()  // Processor[float32, []byte]

// Create adapter
adapter := NewAdapter[float32, []byte, byte, *buffer.Ring[[]byte]](
    factory,
    dstProcessor,  // MID -> DST
    srcProcessor,  // SRC -> MID
)

// Process data
src := buffer.New([]byte{0x01, 0x02, 0x03})
dst := buffer.New(make([]float32, 0, 1024))
written := adapter.Process(dst, src)
```

### Processing Flow

```
src (SRC) -> srcProcessor -> buffer (MID) -> dstProcessor -> dst (DST)
```

The intermediate buffer stores data of type MID, bridging the type gap between SRC and DST.

## Size Calculation

Both `Chain` and `Adapter` support size calculation mode by passing `nil` as the destination:

```go
// Calculate output size without actually processing
src := buffer.New([]float32{1.0, 2.0, 3.0})
outputSize := chain.Process(nil, src)
// outputSize is the theoretical maximum number of samples that would be produced
```

This is useful for:
- Pre-allocating destination buffers
- Validating buffer sizes
- Planning memory allocation

**Note**: Size calculations are estimates. Actual output may vary slightly due to processor-specific behavior.

## Buffer Management

### Lazy Allocation

Buffers are allocated on first use (when `Process` is called with a non-nil destination). This avoids unnecessary allocations when only calculating sizes.

### Buffer Sizing

Buffers are sized with a 20% overhead to handle size variations:
- Minimum buffer size: 64 samples
- Calculated size: `estimatedSize + estimatedSize/5`

### Buffer Factory Pattern

The buffer factory pattern allows:
- Custom buffer types (Ring, BitsRing, custom implementations)
- Pre-configured buffer settings
- Memory pool integration

Example with custom buffer:

```go
factory := func(cap int) *MyCustomBuffer[float32] {
    return NewMyCustomBuffer(cap)
}
chain := New[float32, *MyCustomBuffer[float32]](factory, processors...)
```

## Performance Considerations

### Hot Path Optimizations

The implementation is optimized for embedded systems:

1. **Zero allocations in hot paths**: Buffers are pre-allocated and reused
2. **Early returns**: Processing stops immediately if any stage produces no output
3. **Efficient iteration**: Uses `for range` over slices for better BCE (Boundary Check Elimination)
4. **Minimal type conversions**: Type adapters use unsafe pointers only when necessary (compile-time safe when SRC == DST)

### Memory Efficiency

- Buffers retain capacity between `Reset()` calls
- Buffer size calculations include overhead to minimize reallocations
- Intermediate buffers are reset (not reallocated) between processing calls

### Best Practices

1. **Reuse chains/adapters**: Create once, reuse many times
2. **Call Reset() between batches**: Clears state without deallocating buffers
3. **Pre-calculate sizes**: Use size calculation mode to pre-allocate destination buffers
4. **Choose appropriate buffer types**: Use `Ring` for streaming, `BitsRing` for bit-level operations

## Error Handling

The package uses panic for configuration errors (nil factories) to catch programming errors early. Runtime errors (e.g., buffer full) are handled through the standard `buffer.Reader`/`buffer.Writer` error returns.

## Thread Safety

Neither `Chain` nor `Adapter` are thread-safe. If concurrent access is needed:
- Use separate instances per goroutine
- Add external synchronization
- Use channels to serialize access

## Examples

### Complete Chain Example

```go
package main

import (
    "fmt"
    "github.com/itohio/EasyRobot/x/buffer"
    "github.com/itohio/EasyRobot/x/chain"
)

func main() {
    // Create buffer factory
    factory := func(cap int) *buffer.Ring[float32] {
        return buffer.NewRing(make([]float32, 0, cap))
    }

    // Create processors (all must be Processor[float32, float32])
    processor1 := NewFilterProcessor[float32, float32]()
    processor2 := NewNormalizeProcessor[float32, float32]()
    processor3 := NewCompressProcessor[float32, float32]()

    // Create chain - note: single type parameter T
    chain := chain.New[float32, *buffer.Ring[float32]](
        factory,
        processor1, processor2, processor3,
    )

    // Prepare source data
    srcData := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
    src := buffer.New(srcData)

    // Calculate output size
    outputSize := chain.Process(nil, src)
    fmt.Printf("Output size: %d\n", outputSize)

    // Allocate destination buffer
    dst := buffer.New(make([]float32, 0, outputSize))

    // Process data
    written := chain.Process(dst, src)
    fmt.Printf("Processed %d samples\n", written)

    // Reset for next batch
    chain.Reset()
}
```

### Complete Adapter Example

```go
package main

import (
    "fmt"
    "github.com/itohio/EasyRobot/x/buffer"
    "github.com/itohio/EasyRobot/x/chain"
)

func main() {
    // Create buffer factory for intermediate []byte type
    factory := func(cap int) *buffer.Ring[[]byte] {
        return buffer.NewRing(make([][]byte, 0, cap))
    }

    // Create processors with different types
    byteToBytes := NewByteToBytesProcessor()    // Processor[[]byte, byte]
    bytesToFloat := NewBytesToFloatProcessor()  // Processor[float32, []byte]

    // Create adapter
    adapter := chain.NewAdapter[float32, []byte, byte, *buffer.Ring[[]byte]](
        factory,
        bytesToFloat,  // MID -> DST
        byteToBytes,   // SRC -> MID
    )

    // Prepare source data
    srcData := []byte{0x01, 0x02, 0x03, 0x04}
    src := buffer.New(srcData)

    // Process data
    dst := buffer.New(make([]float32, 0, 1024))
    written := adapter.Process(dst, src)
    fmt.Printf("Processed %d samples\n", written)
}
```

## Architecture

```mermaid
graph LR
    A[Source Reader] --> B[Chain/Adapter]
    B --> C[Processor 1]
    C --> D[Buffer 1]
    D --> E[Processor 2]
    E --> F[Buffer 2]
    F --> G[Processor N]
    G --> H[Destination Writer]
    
    style B fill:#e1f5ff
    style C fill:#fff4e1
    style E fill:#fff4e1
    style G fill:#fff4e1
    style D fill:#e8f5e9
    style F fill:#e8f5e9
```

## See Also

- `x/buffer`: Buffer implementations (Ring, BitsRing, Values)
- `x/types`: Type constraints and utilities

