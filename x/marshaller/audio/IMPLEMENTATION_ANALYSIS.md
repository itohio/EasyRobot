# Audio Marshaller Implementation Analysis

## 1. SPEC.md Implementation Completeness

### ✅ Implemented Features

#### Phase 2: WAV File Support (Unmarshaller)
- ✅ **2.1 Single WAV File Reading** - COMPLETE
  - WAV file reader using `github.com/go-audio/wav`
  - Format detection (sample rate, channels, bit depth)
  - Tensor creation with matching dtype (INT16, INT32, FP32, UINT8)
  - Shape tensor as `[frames, channels]`
  - Format info storage (`SampleRate()`, `Channels()`, `DataType()`)

#### Phase 3: WAV File Support (Marshaller)
- ✅ **3.1 Single WAV File Writing** - COMPLETE
  - WAV file writer using `github.com/go-audio/wav`
  - Writes tensor as-is when format matches
  - Format conversion when needed (INT16, INT32, FP32, UINT8)
  - Configurable sample rate and channels
  - Defaults to 16-bit PCM if conversion needed

### ❌ Missing Features

#### Phase 2: WAV File Support (Unmarshaller)
- ❌ **2.2 Multiple WAV File Concatenation** - NOT IMPLEMENTED
  - Error: "multiple WAV files not yet implemented"
  - Required: Support multiple WAV paths, verify same sample rate, combine channels

- ❌ **2.3 Buffer Pooling for WAV** - NOT IMPLEMENTED
  - No buffer pooling implemented
  - Should use separate pools for PCM (byte) and typed buffers

#### Phase 3: WAV File Support (Marshaller)
- ❌ **3.2 Multiple WAV File Writing** - NOT IMPLEMENTED
  - Error: "multiple WAV files not yet implemented"
  - Required: Distribute tensor channels across multiple WAV files

- ❌ **3.3 Buffer Pooling for WAV Writing** - NOT IMPLEMENTED
  - No buffer pooling implemented

#### Phase 4: Audio Device Support (Unmarshaller)
- ❌ **4.1 Device Enumeration** - NOT IMPLEMENTED
  - Error: "device enumeration not yet implemented"
  - Required: Use malgo (via gocoustics) to enumerate devices
  - Return `[]AudioDeviceInfo` with device capabilities

- ❌ **4.2 Single Device Capture** - NOT IMPLEMENTED
  - Error: "device capture not yet implemented"
  - Required: Open device, async I/O, create FrameStream

- ❌ **4.3 Multiple Device Capture** - NOT IMPLEMENTED
  - Required: Multiple devices, synchronization, channel combination

- ❌ **4.4 Mixed Sources (WAV + Devices)** - NOT IMPLEMENTED
  - Required: Support WAV files + devices in same unmarshaller

#### Phase 5: Audio Device Support (Marshaller)
- ❌ **5.1 Single Device Playback** - NOT IMPLEMENTED
  - Error: "device playback not yet implemented"
  - Required: Open device, async I/O, write tensor/stream

- ❌ **5.2 Multiple Device Playback** - NOT IMPLEMENTED
  - Required: Distribute channels across devices, synchronization

- ❌ **5.3 Mixed Outputs (WAV + Devices)** - NOT IMPLEMENTED
  - Required: Write to both WAV and devices simultaneously

#### Phase 6: Buffer Pooling and Optimization
- ❌ **6.1 PCM Buffer Pool** - NOT IMPLEMENTED
- ❌ **6.2 Typed Buffer Pool** - NOT IMPLEMENTED
- ❌ **6.3 Ring Buffer for Async I/O** - NOT IMPLEMENTED
- ❌ **6.4 Statistics and Health Monitoring** - NOT IMPLEMENTED
  - `Health()` methods return empty maps (TODO comments)

#### Phase 7: Format Conversion
- ⚠️ **7.1 PCM to Tensor Conversion** - PARTIALLY IMPLEMENTED
  - Basic conversion exists in `fillTensorFromPCM()`
  - Missing: Optimized primitives, format-aware zero-copy

- ⚠️ **7.2 Tensor to PCM Conversion** - PARTIALLY IMPLEMENTED
  - Basic conversion exists in write methods
  - Missing: Optimized primitives, format-aware zero-copy

#### Phase 8: Unit Tests
- ⚠️ **8.1 WAV File Tests** - PARTIALLY IMPLEMENTED
  - `unmarshaller_test.go` exists but needs review
  - Missing: Multiple file tests, format preservation tests, round-trip tests

- ❌ **8.2 Device Tests** - NOT IMPLEMENTED
- ❌ **8.3 Integration Tests** - NOT IMPLEMENTED

### Summary

**Implementation Status:**
- ✅ Single WAV file read/write: **COMPLETE**
- ⚠️ Format conversion: **PARTIAL** (basic implementation, missing optimizations)
- ❌ Multiple WAV files: **NOT IMPLEMENTED**
- ❌ Audio device support: **NOT IMPLEMENTED**
- ❌ Buffer pooling: **NOT IMPLEMENTED**
- ❌ Health monitoring: **NOT IMPLEMENTED**
- ❌ Multi-source/multi-output: **NOT IMPLEMENTED**

**Overall Completion: ~25%** (Core single-file WAV I/O works, but most advanced features missing)

---

## 2. protocol.go Location Analysis

### Current State

**File:** `x/marshaller/audio/protocol.go`

**Purpose:** Defines a simple packet protocol for wrapping audio data:
- Magic bytes: "AUDO"
- Protocol version: 1
- Header: magic (4 bytes) + version (1 byte) + data length (4 bytes)
- CRC32 checksum (4 bytes)
- Stateful unmarshaller for stream parsing

**Current Usage:**
- ❌ **NOT USED** anywhere in the audio marshaller codebase
- ❌ **NOT MENTIONED** in SPEC.md
- ❌ **NOT MENTIONED** in PLAN.md
- ❌ **NO REFERENCES** in marshaller.go, unmarshaller.go, or any other files

### Analysis: Should it be in the marshaller?

**Arguments AGAINST keeping it in `x/marshaller/audio`:**
1. **Not part of SPEC.md** - The spec defines WAV file and audio device I/O, not packet protocols
2. **Not used** - No code references it, suggesting it's either:
   - Dead code
   - Planned for future use but not yet integrated
   - Belongs to a different layer/concern
3. **Different abstraction level** - The marshaller handles audio format conversion (WAV ↔ Tensor), not communication protocols
4. **Separation of concerns** - Packet framing/protocol is a communication layer concern, not a marshalling concern

**Arguments FOR keeping it in `x/marshaller/audio`:**
1. **Audio-specific** - The magic bytes are "AUDO" (audio-specific)
2. **Could be for audio streaming** - Might be intended for streaming audio over network/serial
3. **Convenience** - If it's only used for audio, keeping it nearby makes sense

### Analysis: Should it be in `x/math/protocol`?

**Current `x/math/protocol/peg` purpose:**
- PEG (Parsing Expression Grammar) parser for describing binary packet layouts
- General-purpose protocol description system
- Uses pattern matching, field extraction, state management
- Much more sophisticated than the simple protocol in `protocol.go`

**Arguments AGAINST moving to `x/math/protocol`:**
1. **Different purpose** - `x/math/protocol/peg` is a protocol description language, not a protocol implementation
2. **Different abstraction** - PEG is declarative (describe what you want), protocol.go is imperative (how to do it)
3. **Package structure** - `x/math/protocol/peg` is specifically for PEG parsing, not general protocols

**Arguments FOR moving to `x/math/protocol`:**
1. **Higher-level abstraction** - Protocol framing is a general concern, not audio-specific
2. **Reusability** - Other marshallers (video, image) might need similar packet protocols
3. **Logical grouping** - Protocol-related code belongs together
4. **Could create `x/math/protocol/packet`** - A new subpackage for simple packet protocols

### Recommendation

**Option 1: Move to `x/math/protocol/packet` (RECOMMENDED)**

Create a new subpackage `x/math/protocol/packet` for simple packet protocol implementations:
- `x/math/protocol/packet` - Simple packet protocols (header + CRC)
- `x/math/protocol/peg` - PEG-based protocol description

**Rationale:**
- Protocol framing is a general concern, not audio-specific
- Makes the code reusable for other marshallers
- Separates concerns: marshaller handles format conversion, protocol handles framing
- Follows the pattern: `x/math` contains general-purpose utilities

**Option 2: Keep in `x/marshaller/audio` but document it**

If this protocol is specifically for audio streaming over network/serial:
- Add documentation explaining its purpose
- Add it to SPEC.md if it's part of the design
- Integrate it into the marshaller/unmarshaller if needed
- Remove it if it's dead code

**Option 3: Remove if unused**

If this is dead code:
- Remove `protocol.go`
- If needed later, it can be recreated or moved to appropriate location

### Decision Matrix

| Criterion | Keep in audio | Move to x/math/protocol | Remove |
|-----------|---------------|-------------------------|--------|
| Used in codebase | ❌ No | ❌ No | N/A |
| Part of SPEC.md | ❌ No | ❌ No | N/A |
| Reusable | ❌ No | ✅ Yes | N/A |
| Separation of concerns | ❌ Poor | ✅ Good | N/A |
| Future-proof | ⚠️ Maybe | ✅ Yes | ❌ No |

**Final Recommendation: Move to `x/math/protocol/packet`**

This provides:
- Better code organization
- Reusability for other marshallers
- Clear separation: marshaller = format conversion, protocol = framing
- Future-proof if protocol is needed

---

## 3. Action Items

### Immediate Actions

1. **Document protocol.go purpose** - Determine if it's needed or dead code
2. **If needed:** Move to `x/math/protocol/packet` subpackage
3. **If not needed:** Remove `protocol.go`
4. **Update SPEC.md** - Add protocol section if protocol is part of design

### Implementation Priorities

Based on SPEC.md analysis, prioritize:
1. **Multiple WAV file support** (Phase 2.2, 3.2) - Core feature
2. **Device enumeration** (Phase 4.1) - Foundation for device support
3. **Single device capture/playback** (Phase 4.2, 5.1) - Core device feature
4. **Health monitoring** (Phase 6.4) - Important for debugging
5. **Buffer pooling** (Phase 6.1-6.3) - Performance optimization

