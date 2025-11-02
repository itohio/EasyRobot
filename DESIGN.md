# EasyRobot Library - High Level Design

## Overview

EasyRobot is a cross-platform robotics framework designed to enable algorithm development on desktop/cloud environments and deployment on embedded systems (Raspberry Pi, Nano Pi, ARM microcontrollers). The framework emphasizes:

- **Cross-platform compatibility**: Intel/AMD, ARM, Xtensa, RP2040, WASM
- **Multi-compiler support**: Go and TinyGo
- **Minimal copying**: Zero-copy message passing where possible using typed protobuf messages
- **Modular architecture**: ROS-like component system with typed communication
- **Performance**: Optimized for embedded systems with in-place operations

## Architecture

### Core Philosophy

1. **Pipeline-based processing**: Data flows through typed pipelines with steps (sources, processors, sinks)
2. **Protobuf as communication layer**: All inter-component communication uses protobuf for type safety and serialization
3. **Plugin system**: Extensible plugin registry for dynamic component loading
4. **Backend abstraction**: Multiple backends (GoCV, TF, TFLite, native) for algorithm implementations
5. **Zero-copy where possible**: Channels and reference passing minimize memory copies

### Component Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                       │
│  (Vision pipelines, Robot control, Navigation, etc.)       │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Domain Packages                           │
│  pkg/vision    pkg/robot    pkg/navigation    pkg/ai       │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Core Framework                           │
│  Pipeline │ Plugin │ Store │ Transport │ Math │ Logger      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                  Backend Implementations                    │
│  GoCV │ TF │ TFLite │ Native (float32) │ Device Drivers    │
└─────────────────────────────────────────────────────────────┘
```

## Package Structure

### Core Packages (`pkg/core`)

**pipeline**: Graph-based processing pipeline
- Steps: Source, Processor, Sink, FanIn, FanOut, Join, Sync
- Data flow: Channels with typed Store messages
- Configuration: JSON serializable pipeline definitions

**plugin**: Dynamic plugin system
- Registry-based plugin loading
- Options-based configuration
- Marshaling support for persistence

**store**: Typed key-value store
- FQDN-based type system
- Value lifecycle management (Close, Clone)
- Type-safe getters/setters

**transport**: Communication layer
- NATS backend implementation
- Protobuf message encoding/decoding
- Stream-based transport (video, audio, sensors)

**math**: Mathematical primitives
- Vector operations (2D, 3D, 4D, generic)
- Matrix operations (2x2, 3x3, 4x4, generic)
- Interpolation (linear, bezier, cosine, spline)
- Filters (PID, AHRS - Madgwick/Mahony)
- Tensor support (dense/sparse)

**logger**: Logging abstraction
- ZeroLog backend
- Build-tag based optimization (`logless` for embedded)

### Robot Packages (`pkg/robot`)

**kinematics**: Forward and inverse kinematics
- Denavit-Hartenberg parameterization
- Planar kinematics (2DOF, 3DOF)
- Wheel kinematics (differential, mechanum)
- Configurable via protobuf

**actuator**: Actuator control
- Motor control (PWM, encoder)
- Servo control
- Protocol-based configuration

**transport**: Robot-specific transport
- Packet-based protocol
- Header with magic, ID, CRC
- Type-safe packet routing

### Vision Packages (`pkg/vision`)

**reader**: Image/video input
- Device capture (camera)
- Video file reading
- Image file reading
- Backend: GoCV

**transform**: Image transformations
- Color space conversion
- Format conversion
- Stereo processing

**extract**: Feature extraction
- DNN inference (OpenCV DNN backend)
- Feature detection (ORB, SIFT)
- Backend: GoCV

**display**: Visualization
- Image display
- Backend: GoCV

**writer**: Image/video output
- Image file writing
- Video file writing
- Null sink (for testing)

## Communication Architecture

### Current Implementation

1. **Pipeline Steps**: Use Go channels for zero-copy message passing
2. **NATS Transport**: Distributed communication via NATS with protobuf encoding
3. **Robot Transport**: Packet-based protocol for embedded devices (I2C/SPI/Serial/CAN)

### Potential Integration with DNDM

DNDM (Decentralized Named Data Mesh) could replace or complement current transport:

**Benefits**:
- Intent/Interest pattern matches pipeline source/sink model
- Type-safe routes (`Type@path` format)
- Zero-copy local communication
- Distributed communication via endpoints
- Mesh networking support

**Migration Considerations**:
- Pipeline `Step` interface could use DNDM routes instead of channels
- NATS transport could be implemented as DNDM endpoint
- Robot transport packets could be wrapped in DNDM messages

## Data Flow

### Pipeline Processing

```
Source → [Processor1] → [FanOut] → [Processor2] → [FanIn] → [Sink]
            ↓                           ↓
         [Sync]                      [Join]
```

**Data Type**: `store.Store` - Typed key-value container
**Transport**: Go channels (local), NATS/DNDM (distributed)

### Message Types

1. **Store**: Generic container with FQDN-typed values
2. **Protobuf Messages**: Generated from `.proto` files
3. **Transport Messages**: Wrapped protobuf with headers/metadata

## Target Platforms

### Compiler Support

**Go (standard)**:
- Full feature set
- All backends available
- Development and testing

**TinyGo**:
- Subset of features (no reflection, limited stdlib)
- Embedded-friendly code generation
- Direct hardware access

### Architecture Support

**Intel/AMD (x86_64)**:
- Full backend support
- Optimized math operations
- Desktop/server deployment

**ARM (AArch64, ARMv7)**:
- Full backend support
- Raspberry Pi, Nano Pi
- Optimized for embedded Linux

**Xtensa**:
- Limited backend support (native math only)
- ESP32, ESP8266
- No GoCV, minimal dependencies

**RP2040**:
- Minimal backend support
- Raspberry Pi Pico
- Native math, basic I/O

**WASM**:
- Subset of features
- Web deployment
- Browser-based visualization

## Design Questions

### Architecture Questions

1. **Communication Strategy**:
   - Should DNDM completely replace NATS transport?
   - How should pipeline steps communicate across network boundaries?
   - Should we support both channel-based (local) and DNDM-based (distributed) communication?
   - How to handle hybrid scenarios (some steps local, some remote)?

2. **Type System**:
   - Should we maintain FQDN-based type system or migrate to protobuf message types?
   - How to handle type conversion between different backends (GoCV Mat vs image.Image vs TF tensor)?
   - Should type information be available at runtime for dynamic pipelines?

3. **Backend Strategy**:
   - How to handle missing backends on embedded platforms?
   - Should we provide fallback implementations?
   - How to configure backend selection at compile-time vs runtime?

4. **Pipeline Configuration**:
   - Should pipeline definitions be declarative (JSON/YAML)?
   - How to handle dynamic pipeline modification?
   - Should we support pipeline versioning?

### Performance Questions

5. **Memory Management**:
   - Should we use memory pools for frequently allocated types (Store, matrices)?
   - How to handle zero-copy constraints with serialization?
   - Should we support explicit memory management for embedded systems?

6. **Concurrency**:
   - How to handle goroutine lifecycle in pipeline steps?
   - Should we support worker pools for parallel processing?
   - How to prevent goroutine leaks in long-running pipelines?

7. **Real-time Constraints**:
   - How to handle real-time requirements on embedded systems?
   - Should we support priority-based scheduling?
   - How to handle deadline misses?

### Integration Questions

8. **DNDM Integration**:
   - Should DNDM routes map directly to pipeline step names?
   - How to handle pipeline step discovery across network?
   - Should pipeline connections be established automatically via DNDM?

9. **Hardware Abstraction**:
   - How to abstract hardware-specific features (GPIO, I2C, SPI)?
   - Should we provide HAL (Hardware Abstraction Layer)?
   - How to handle different hardware capabilities across platforms?

10. **Deployment**:
    - How to package and deploy pipelines to embedded devices?
    - Should we support over-the-air updates?
    - How to handle configuration management?

## Known Issues and Areas for Improvement

### Current Limitations

1. **Type Safety**:
   - Runtime type checking in Store (could be compile-time with generics)
   - No compile-time verification of pipeline connections
   - Type conversion errors only caught at runtime

2. **Error Handling**:
   - Error propagation through pipeline could be improved
   - No structured error types
   - Limited error recovery mechanisms

3. **Testing**:
   - Missing comprehensive test coverage
   - Limited integration tests
   - No performance benchmarks

4. **Documentation**:
   - Incomplete API documentation
   - Missing usage examples
   - No architecture diagrams

5. **Memory**:
   - No memory pooling for frequent allocations
   - Potential memory leaks in long-running pipelines
   - No memory usage monitoring

### Potential Improvements

1. **Generics**: Use Go generics (1.18+) for type-safe Store operations
2. **Context Propagation**: Better context handling throughout pipeline
3. **Metrics**: Add observability (metrics, tracing)
4. **Backpressure**: Implement proper backpressure handling
5. **Serialization**: Optimize protobuf encoding/decoding
6. **Code Generation**: Better code generation for platform-specific optimizations
7. **Testing**: Comprehensive test suite with embedded device simulation
8. **Examples**: Complete examples for common use cases

## Future Considerations

### Planned Features

- [ ] Metrics and observability
- [ ] Distributed tracing
- [ ] Configuration management
- [ ] Over-the-air updates
- [ ] Hardware abstraction layer
- [ ] Simulator integration
- [ ] ROS bridge/compatibility
- [ ] Web-based visualization
- [ ] Machine learning training pipelines

### Research Areas

- Zero-copy serialization across network boundaries
- Real-time scheduling for embedded systems
- Distributed pipeline execution
- Hardware acceleration integration
- WASM performance optimization

