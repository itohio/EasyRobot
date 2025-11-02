# Transport Layer Specification

## Overview

The transport layer provides communication mechanisms for distributed EasyRobot systems. It supports multiple backends including NATS and protobuf message encoding.

## Current Implementation

### NATS Backend (`pkg/core/transport/nats`)

**Purpose**: Distributed communication via NATS message broker

**Message Format**:
```protobuf
message StreamMsg {
    StreamType Type = 1;
    int64 Index = 2;
    int64 Timestamp = 3;
    bytes Data = 7;  // Protobuf-encoded payload
}

message Robot {
    int64 Timestamp = 1;
    StatusMsg Status = 2;
    RobotMsg Robot = 3;
    repeated EventMsg Events = 4;
    repeated StreamMsg Stream = 5;
    bytes Signature = 6;
}
```

**Stream Types**:
- Video: `VIDEO_RAW`, `VIDEO_PNG`, `VIDEO_JPG`, `VIDEO_H264`, `VIDEO_H265`
- Features: `FEATURES_DNN`, `FEATURES_CV`
- Audio: `AUDIO_PCM`, `AUDIO_MP3`, `AUDIO_OGG`
- Sensors: `SENSORS_RAW`

**Integration**:
- Pipeline step implementation
- GoCV backend only (limited)
- Manual encoding/decoding

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

### Protobuf Schema (`pkg/core/transport/schema.proto`)

**Purpose**: Core message definitions for transport layer

**Message Types**:
- `StreamMsg`: Stream data with type, index, timestamp
- `StatusMsg`: Robot status (OWD, RTT, Jitter, Battery, Health)
- `ChatMsg`: Chat messages with timestamp, username
- `EventMsg`: Event messages with ID, flags, data
- `RobotMsg`: Robot capability advertisement
- `Robot`: Complete robot state message

**Questions**:
1. Should schema support message versioning?
2. How to handle schema evolution?
3. Should schema support optional fields?
4. How to optimize schema for embedded systems?
5. Should schema support message compression hints?

## Pipeline Integration

### NATS Step

NATS transport is implemented as pipeline step:
```go
natsStep, _ := nats.NewGoCV(nats.WithTopicPub("robot.video"), ...)
pipeline.ConnectSteps(source, natsStep)
```

**Characteristics**:
- Publishes to NATS topic
- Subscribes from NATS topic
- Encodes/decodes using protobuf
- GoCV-specific (limited)

**Questions**:
1. Should transport step support generic types?
2. How to handle transport step errors?
3. Should transport step support automatic reconnection?
4. How to optimize transport step for embedded systems?
5. Should transport step support batching?

## Encoding/Decoding

### GoCV Encoding (`pkg/core/transport/nats/encode.gocv.go`)

**Purpose**: Encode GoCV Mat to protobuf

**Current Implementation**:
- GoCV Mat → protobuf bytes
- Manual encoding

**Questions**:
1. Should encoding support multiple formats (PNG, JPG, RAW)?
2. How to handle encoding errors?
3. Should encoding support compression?
4. How to optimize encoding for embedded systems?
5. Should encoding support progressive encoding?

### GoCV Decoding (`pkg/core/transport/nats/decode.gocv.go`)

**Purpose**: Decode protobuf to GoCV Mat

**Current Implementation**:
- Protobuf bytes → GoCV Mat
- Manual decoding

**Questions**:
1. Should decoding support multiple formats?
2. How to handle decoding errors?
3. Should decoding support validation?
4. How to optimize decoding for embedded systems?
5. Should decoding support streaming?

## Blob Support (`pkg/core/transport/blob.go`)

**Purpose**: Custom protobuf type for binary data

**Characteristics**:
- Zero-copy where possible
- Efficient serialization

**Questions**:
1. Should blob support compression?
2. How to handle blob size limits?
3. Should blob support streaming?
4. How to optimize blob for embedded systems?

## DNDM Integration Potential

### Current Model

- Pipeline steps communicate via channels (local)
- NATS transport for distributed communication
- Manual serialization/deserialization

### DNDM Model

- Intent/Interest pattern for publish/subscribe
- Routes: `Type@path` format
- Type-safe via protobuf message types
- Zero-copy local communication
- Distributed via endpoints

**Migration Strategy**:
- Option 1: Replace NATS with DNDM
- Option 2: DNDM as transport backend
- Option 3: Hybrid (channels local, DNDM distributed)

**Questions**:
1. Should DNDM replace NATS completely?
2. How to migrate existing NATS-based code?
3. Should we support both NATS and DNDM?
4. How to handle DNDM performance for local communication?
5. Should DNDM support be optional or required?

## Design Questions

### Architecture

1. **Backend Abstraction**:
   - Should transport support pluggable backends?
   - How to handle backend-specific features?
   - Should we provide backend abstraction layer?

2. **Protocol Design**:
   - Should we support multiple protocols?
   - How to handle protocol versioning?
   - Should we support protocol negotiation?

3. **Message Format**:
   - Should all messages use protobuf?
   - How to handle non-protobuf types?
   - Should we support alternative encodings (JSON, MessagePack)?

### Performance

4. **Serialization**:
   - How to minimize serialization overhead?
   - Should we support zero-copy serialization?
   - How to optimize for embedded systems?

5. **Network Efficiency**:
   - Should we support message batching?
   - How to handle network congestion?
   - Should we support message prioritization?

6. **Embedded Systems**:
   - How to optimize for memory-constrained devices?
   - Should we support message fragmentation?
   - How to handle limited network bandwidth?

### Reliability

7. **Error Handling**:
   - How to handle connection failures?
   - Should we support automatic retry?
   - How to distinguish transient vs permanent errors?

8. **Message Delivery**:
   - Should we support message acknowledgments?
   - How to handle message retransmission?
   - Should we support at-least-once vs exactly-once delivery?

9. **Ordering**:
   - Should we maintain message ordering?
   - How to handle out-of-order messages?
   - Should we support ordered vs unordered delivery?

## Known Issues

1. **Backend Limitation**: Only GoCV backend supported
2. **Manual Encoding**: Encoding/decoding requires manual implementation
3. **No Abstraction**: Transport tied to NATS implementation
4. **Limited Testing**: Missing comprehensive tests
5. **No Compression**: Messages not compressed
6. **No Encryption**: Messages not encrypted

## Potential Improvements

1. **Backend Abstraction**: Generic transport interface
2. **DNDM Integration**: Support for DNDM backend
3. **Automatic Encoding**: Generic encoding/decoding
4. **Compression**: Message compression support
5. **Encryption**: Message encryption support
6. **Testing**: Comprehensive test suite
7. **Documentation**: Complete API documentation
8. **Performance**: Zero-copy optimization
9. **Reliability**: Better error handling and recovery

