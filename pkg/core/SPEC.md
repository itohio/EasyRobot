# Core Framework Specification

## Overview

The `pkg/core` package provides the foundational framework components for EasyRobot, including pipeline processing, plugin management, data storage, transport, mathematics, and logging.

## Components

### 1. Pipeline (`pkg/core/pipeline`)

**Purpose**: Graph-based processing pipeline for data flow

**Core Interfaces**:
```go
type Step interface {
    In(<-chan Data)
    Out() <-chan Data
    Run(ctx context.Context)
    Reset()
}
```

**Step Types**:
- **Source**: Produces data (e.g., camera, video file)
- **Processor**: Transforms data (e.g., image processing, feature extraction)
- **Sink**: Consumes data (e.g., display, file writer)
- **FanIn**: Merges multiple inputs into one output
- **FanOut**: Splits one input into multiple outputs
- **Join**: Synchronizes multiple inputs
- **Sync**: Frame synchronization

**Data Type**: `store.Store` - Typed key-value container

**Communication**: Go channels (zero-copy for local communication)

**Configuration**:
- Blocking vs non-blocking send
- Buffer size for channels
- Name for step identification

**Questions**:
1. Should pipeline support cyclic graphs?
2. How should pipeline handle step failures?
3. Should we support dynamic pipeline modification at runtime?
4. How to handle backpressure in pipeline steps?
5. Should pipeline steps support priority-based scheduling?
6. How to handle pipeline step lifecycle (start/stop/pause/resume)?
7. Should pipeline support distributed execution across network?
8. How to handle pipeline step dependencies?
9. Should we support pipeline versioning?
10. How to handle pipeline step configuration persistence?

### 2. Plugin System (`pkg/core/plugin`)

**Purpose**: Dynamic plugin registry for extensibility

**Core Components**:
- **Registry**: Manages plugin builders
- **Builder**: Factory function for plugin creation
- **Options**: Configuration mechanism

**Plugin Lifecycle**:
1. Register plugin builder
2. Create plugin instance with options
3. Use plugin
4. Unregister plugin (optional)

**Configuration**:
- Options-based configuration
- Marshaling support for persistence

**Questions**:
1. Should plugins support hot-reloading?
2. How to handle plugin versioning?
3. Should plugins support dependency injection?
4. How to handle plugin conflicts?
5. Should plugins support lazy loading?
6. How to handle plugin lifecycle events?
7. Should plugins support resource cleanup?
8. How to validate plugin configuration?
9. Should plugins support metrics/observability?
10. How to handle plugin errors?

### 3. Store (`pkg/core/store`)

**Purpose**: Typed key-value storage for pipeline data

**Type System**: FQDN-based (Fully Qualified Domain Name)
- `FQDNType`: uint16 identifier
- Pattern matching support via `ForEach`

**Value Types**:
- Primitive types: int, float, string, bytes
- Composite types: slices, maps
- Custom types: images, matrices, vectors
- Nested stores: StoreList, StoreMap

**Value Lifecycle**:
- **Set**: Store value
- **Get**: Retrieve value
- **Close**: Cleanup resources
- **Clone**: Deep copy value

**Standard FQDN Types**:
- `INDEX`: Frame index
- `TIMESTAMP`: Timestamp
- `FPS`: Frames per second
- `DROPPED_FRAMES`: Dropped frame count

**Questions**:
1. Should Store support value versioning?
2. How to handle concurrent access to Store?
3. Should Store support transactions?
4. How to handle Store memory limits?
5. Should Store support value expiration?
6. How to handle value serialization for persistence?
7. Should Store support value validation?
8. How to handle nested Store updates?
9. Should Store support value change notifications?
10. How to optimize Store for embedded systems (memory usage)?

### 4. Transport (`pkg/core/transport`)

**Purpose**: Communication layer for distributed systems

**Current Backends**:
- **NATS**: Message broker for distributed communication
- **Protobuf**: Message serialization

**Message Types**:
```protobuf
message StreamMsg {
    StreamType Type = 1;
    int64 Index = 2;
    int64 Timestamp = 3;
    bytes Data = 7;  // Protobuf-encoded payload
}
```

**Stream Types**:
- Video: RAW, PNG, JPG, H264, H265
- Audio: PCM, MP3, OGG
- Features: DNN, CV
- Sensors: RAW

**Questions**:
1. Should transport support multiple backends (NATS, DNDM, TCP, UDP)?
2. How to handle transport connection failures?
3. Should transport support message compression?
4. How to handle message ordering in distributed scenarios?
5. Should transport support message acknowledgments?
6. How to handle transport backpressure?
7. Should transport support message filtering?
8. How to optimize transport for embedded systems?
9. Should transport support encryption?
10. How to handle transport protocol versioning?

### 5. Math (`pkg/core/math`)

**Purpose**: Mathematical primitives for robotics

**Components**:
- **Vector**: 2D, 3D, 4D, generic
- **Matrix**: 2x2, 3x3, 4x4, generic, sparse
- **Interpolation**: Linear, Bezier, Cosine, Spline
- **Filters**: PID, AHRS (Madgwick/Mahony)
- **Tensor**: Dense, sparse

**Characteristics**:
- In-place operations (minimize allocations)
- `float32` precision (embedded-friendly)
- Zero-copy where possible

**Questions**:
1. Should math operations support SIMD optimization?
2. How to handle math errors (overflow, NaN, Inf)?
3. Should math support fixed-point arithmetic for embedded?
4. How to optimize math operations for different architectures (ARM, Xtensa)?
5. Should math support GPU acceleration?
6. How to handle math precision issues?
7. Should math support complex numbers?
8. How to optimize matrix operations (block algorithms, parallel)?
9. Should math support symbolic computation?
10. How to handle math operations on embedded systems (memory constraints)?

### 6. Logger (`pkg/core/logger`)

**Purpose**: Logging abstraction with backend support

**Backends**:
- **ZeroLog**: Full-featured logging
- **Empty**: Optimized away via build tags (`logless`)

**Characteristics**:
- Structured logging
- Build-tag based optimization for embedded

**Questions**:
1. Should logger support multiple log levels at runtime?
2. How to handle log filtering?
3. Should logger support log rotation?
4. How to optimize logger for embedded systems?
5. Should logger support distributed tracing?
6. How to handle logger performance impact?
7. Should logger support structured logging with context?
8. How to handle sensitive data in logs?
9. Should logger support metrics integration?
10. How to handle logger configuration?

## Common Patterns

### Options Pattern

All components use options-based configuration:
```go
type Option func(cfg interface{})
```

**Benefits**:
- Flexible configuration
- Optional parameters
- Backward compatibility

**Questions**:
1. Should options support validation?
2. How to handle option conflicts?
3. Should options support merging?
4. How to document options?
5. Should options support defaults?

### Context Propagation

All components support context for cancellation and timeouts:
```go
func Run(ctx context.Context) error
```

**Questions**:
1. How to handle context propagation through pipeline?
2. Should context support value propagation?
3. How to handle context cancellation in long-running operations?
4. Should context support deadline propagation?

## Integration Points

### Pipeline → Store
- Pipeline steps use Store as message type
- Store provides type-safe data access

### Pipeline → Plugin
- Pipeline steps can be registered as plugins
- Plugin registry used for step discovery

### Pipeline → Transport
- Transport steps can bridge local and distributed communication
- NATS transport encodes/decodes Store data

### Math → Store
- Math types (vectors, matrices) stored in Store
- Store provides type-safe access to math types

## Design Questions

### Architecture

1. **Modularity**:
   - Should core components be more loosely coupled?
   - How to handle component dependencies?
   - Should we support component lifecycle management?

2. **Extensibility**:
   - How to extend core components?
   - Should we support component composition?
   - How to handle component versioning?

3. **Performance**:
   - How to optimize for embedded systems?
   - Should we support memory pooling?
   - How to handle memory constraints?

### Compatibility

4. **Go vs TinyGo**:
   - How to handle TinyGo limitations?
   - Should we provide TinyGo-specific implementations?
   - How to test TinyGo compatibility?

5. **Platform Support**:
   - How to handle platform-specific features?
   - Should we provide platform abstraction layer?
   - How to handle missing features on embedded platforms?

6. **Backward Compatibility**:
   - How to handle breaking changes?
   - Should we support migration tools?
   - How to handle API deprecation?

## Known Issues

### Current Limitations

1. **Type Safety**:
   - Runtime type checking in Store
   - No compile-time verification of pipeline connections
   - Type conversion errors only caught at runtime

2. **Error Handling**:
   - Limited error propagation
   - No structured error types
   - Error recovery not well-defined

3. **Testing**:
   - Missing comprehensive tests
   - Limited integration tests
   - No performance benchmarks

4. **Documentation**:
   - Incomplete API documentation
   - Missing usage examples
   - No architecture diagrams

### Potential Improvements

1. **Generics**:
   - Use Go generics for type-safe Store operations
   - Compile-time type checking
   - Better API ergonomics

2. **Metrics**:
   - Add observability (metrics, tracing)
   - Performance monitoring
   - Resource usage tracking

3. **Testing**:
   - Comprehensive test suite
   - Integration tests
   - Performance benchmarks

4. **Documentation**:
   - Complete API documentation
   - Usage examples
   - Architecture diagrams

5. **Performance**:
   - Memory pooling
   - Zero-copy optimization
   - SIMD support where applicable

