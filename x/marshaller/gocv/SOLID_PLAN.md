# GoCV Marshaller SOLID Alignment Plan

## Context
- Current GoCV marshaller/unmarshaller implementation uses lossy image encoding (PNG/JPEG) instead of the envelope-based serialization described in the original `DESIGN.md`. Frame streams are consumed with side effects (file writing, display) rather than being serialized for storage/transmission. DNN support is limited to ONNX unmarshaling only.
  - This is expected and DESIGN should be updated.
  - display serialization is a special case that also allows reading user input.
- Configuration and option handling mixes concerns for codec, streaming, display, and DNNs inside a single `config` struct, violating Single Responsibility and Interface Segregation principles.
- Source loaders directly depend on GoCV types (`cv.VideoCapture`, `cv.Mat`) instead of abstractions, making testing difficult and violating Dependency Inversion.
  - this is expected. although they should also support tensor gocv wrapper too.
- Before refactoring runtime code, we need an agreed plan that restores intended usability features and aligns the module with SOLID principles.

## Current State Analysis

### Actual Implementation (as of analysis)

**Serialization Formats:**
- Mat/Image/Tensor: Raw PNG/JPEG/BMP bytes (lossy, no metadata)
- Custom .mat files: Binary format with magic number (used for disk I/O only, not marshaller)
- DNN: Raw bytes pass-through (ONNX only for unmarshaling)
- FrameStream: Not serialized (consumed with side effects)

**Configuration Structure:**
- Single `config` struct with 17 fields mixing:
  - Codec settings (imageEncoding, tensorOpts)
  - Stream settings (sources, sequential, allowBestEffort, sorter)
  - Display settings (displayEnabled, displayTitle, displayWidth, displayHeight, onKey, onMouse, eventLoop)
  - DNN settings (dnnFormat, netBackend, netTarget)
  - Context (ctx)

**Source Loaders:**
- `imageLoader`, `fileListLoader`, `videoFileLoader`, `videoDeviceLoader`
- All depend directly on `cv.*` types
- Return internal `frameItem` struct (not interface)

**FrameStream Marshalling:**
- `Marshaller.writeFrameStream` performs:
  1. Output directory resolution
  2. File writer creation
  3. Display writer creation
  4. Event loop execution
  5. Frame consumption and writing
  6. Summary generation
- This is a side-effect operation, not serialization

## Findings Summary
- **Missing usability features**
  1. **Lossy serialization:** Mat/image/tensor marshaling converts to PNG/JPEG bytes, losing:
     - Original data type (float32/float64 → uint8)
     - Channel count (inferred from image)
     - Mat type, continuous flag, and other metadata
     - Precision for non-uint8 types
     - can be made non-lossy by specifying proper file extension
  2. **FrameStream not serializable:** `Marshaller.Marshal` consumes the stream immediately with side effects (file writing, display). No way to persist or transmit a `FrameStream` for later consumption.
    - this is expected. streaming frames purpose is to video files, images, or display.
  3. **DNN limitations:**
     - Only ONNX format supported for unmarshaling
     - No marshaling support for `cv.Net` (cannot serialize loaded networks)
     - No support for TensorFlow/Caffe/Darknet formats
       - this is fine. if gocv does not support these formats - then we don't support these formats.
     - Requires temporary file during unmarshaling
- **SOLID violations**
  1. **Single Responsibility Principle (SRP):**
     - `Marshaller.writeFrameStream` manages option resolution, writer creation, event-loop lifecycle, and I/O side effects
     - `config` struct mixes codec, streaming, display, and DNN concerns
     - Source loaders handle both I/O and frame construction
  2. **Open/Closed Principle (OCP):**
     - Adding new source types requires modifying `buildStreams` switch statement
     - Adding new DNN formats requires modifying `loadNetFromBytes` switch
     - Display/stream/file writing tightly coupled in `writeFrameStream`
  3. **Liskov Substitution Principle (LSP):**
     - Source loaders return concrete `frameItem` instead of interface
     - No abstraction for video capture or file I/O
  4. **Interface Segregation Principle (ISP):**
     - Single `config` struct forces all components to depend on unrelated settings
     - `types.Option` interface used for all concerns (codec, stream, display, DNN)
     - No way to construct components with minimal dependencies
  5. **Dependency Inversion Principle (DIP):**
     - Source loaders depend on concrete `cv.*` types instead of interfaces
     - `Marshaller`/`Unmarshaller` depend on concrete `config` struct
     - No abstractions for codec operations (MatEncoder, MatDecoder)

## Objectives
1. Keep GoCV in charge of image/video/DNN bytes while introducing structured (protobuf) metadata for everything GoCV cannot serialize natively (manifests, options, sidecars).
2. Introduce a transport-friendly representation for frame streams (directory manifests or chunked protobuf stream) and separate stream consumption from serialization.
3. Restructure configuration/option handling into composable units that can be injected/mocked independently.

## Proposed Refactor Steps

### 1. Transport Encoding Strategy (GoCV-first, Protobuf-as-needed)
**Goal:** Keep GoCV in charge of the media types it already understands (images, video frames, DNN models) while introducing protobuf wrappers only for data that GoCV cannot serialize natively (manifests, metadata, control structures).

#### 1.1 GoCV-Native Assets
- **Mat/Image/Tensor payloads:** Continue to encode/decode pixels via GoCV (`cv.IMEncode`, `cv.IMDecode`, `tensor/gocv`). Persist them as standard image formats (PNG/JPEG/BMP) so existing GoCV tooling can consume them without extra adapters.
- **Video streams:** Keep using GoCV video capture/writer APIs; serialization remains a sequence of encoded frames plus metadata (handled by GoCV and existing file/display writers).
- **DNN weights:** Saving/loading stays delegated to GoCV (`cv.ReadNet`, future byte loaders). We only surface format/backend hints via options, not via protobuf payloads.

#### 1.2 Protobuf Wrappers for Non-GoCV Types
Use protobuf only for structural data that GoCV does not serialize:
- Frame stream manifests, source specifications, synchronization hints.
- High-level metadata about stored frames (e.g., names, tags, timestamps) when transporting between services.
- Error/status responses and progress information.

Define schemas in `proto/types/marshaller/gocv.proto` (alongside `graph_metadata.proto`). Example:
```protobuf
syntax = "proto3";

package types.marshaller;

option go_package = "github.com/itohio/EasyRobot/types/marshaller";

import "types/core/frame.proto";

message FrameStreamManifest {
  repeated SourceSpec sources = 1;
  string sync_mode = 2;   // "parallel", "sequential"
  bool best_effort = 3;
  map<string, string> metadata = 4;
}

message SourceSpec {
  SourceKind kind = 1;
  string path = 2;
  DeviceSpec device = 3;
  repeated string files = 4;
}

enum SourceKind {
  SOURCE_KIND_UNKNOWN = 0;
  SOURCE_KIND_SINGLE = 1;
  SOURCE_KIND_VIDEO_FILE = 2;
  SOURCE_KIND_VIDEO_DEVICE = 3;
  SOURCE_KIND_FILE_LIST = 4;
}

message DeviceSpec {
  int32 id = 1;
  int32 width = 2;
  int32 height = 3;
}

// Optional metadata-only records (no raw pixels)
message MatMetadata {
  int32 rows = 1;
  int32 cols = 2;
  int32 channels = 3;
  int32 mat_type = 4;
  map<string, string> annotations = 5;
}
```

#### 1.3 Implementation Steps
1. **Document the policy:** Update DESIGN docs to state that image/video/DNN bytes remain in GoCV-managed formats; protobuf is used only for metadata and non-GoCV types.
2. **Add protobuf manifests:** Generate code for `FrameStreamManifest`, `SourceSpec`, `MatMetadata`, etc., to describe how to rebuild sources and streams.
3. **Wire manifests into marshaller:** When serializing a `FrameStream`, emit the protobuf manifest first, then stream GoCV-encoded frames via sinks (Section 2). When deserializing, read the manifest to configure loaders; actual frame bytes are still PNG/JPEG handled by GoCV.
4. **Metadata sidecars (optional):** If consumers need Mat metadata, write/read a protobuf `MatMetadata` sidecar next to the raw image bytes instead of embedding pixels into protobuf.
5. **Codec interfaces:** Keep `MatEncoder`/`MatDecoder` abstractions to allow swapping PNG/JPEG encoders, but they remain GoCV-based. Only metadata wrappers depend on protobuf.

**Benefits:**
- Respects user requirement: GoCV continues to own image/video/DNN serialization.
- Protobuf is introduced only where GoCV offers no native format (manifests, metadata).
- Maintains compatibility with existing GoCV tooling while enabling structured metadata exchange.

### 2. FrameStream Transport & Sink Abstractions
**Goal:** Separate stream serialization from consumption.

**Steps:**
- Define `StreamSink` interface:
  ```go
  type StreamSink interface {
      WriteFrame(frame types.Frame) error
      Close() error
  }
  ```
- Implement sinks:
  - `ProtobufSink`: Serializes frames to protobuf stream (using `types.core.Frame`)
  - `GobSink`: Serializes frames to gob stream (alternative, if not using protobuf)
  - `DirectorySink`: Writes frames to disk (existing `fileWriter` logic)
  - `DisplaySink`: Displays frames (existing `displayWriter` logic)
  - `MultiSink`: Composes multiple sinks
- Define stream manifest format (protobuf):
  - Use `FrameStreamManifest` from `proto/types/marshaller/gocv.proto` (see Option A above)
  - Or define as Go struct if using gob (Option B)
- Update `Marshaller.Marshal` for `FrameStream`:
  - Write manifest to `io.Writer` (protobuf-encoded)
  - Serialize frames to provided sinks (via options)
  - Support both serialization (ProtobufSink/GobSink) and side effects (DirectorySink, DisplaySink)
  - Use `types.core.Frame` proto for frame serialization (reuse existing schema)
  - Frame pixel payloads continue to be PNG/JPEG bytes emitted by GoCV; protobuf only wraps metadata/tensors.
- Update `Unmarshaller.Unmarshal` for `FrameStream`:
  - Read manifest from `io.Reader` (protobuf-decoded)
  - Rebuild stream from manifest instead of mutating `config`
  - Support reading from protobuf stream or from configured sources
  - Deserialize frames using `types.core.Frame` proto
- Move file/display writers behind `StreamSink` interface.

**Benefits:**
- FrameStream can be serialized for storage/transmission
- Clear separation between serialization and consumption
- Composability (multiple sinks can be used)
- Reuse of existing `types.core.Frame` protobuf schema
- Language-agnostic frame serialization (protobuf option)

### 3. DNN Artifact Support
**Goal:** Keep GoCV responsible for model bytes while providing structured metadata and loader abstractions for additional formats.

**Steps:**
- Create `NetLoader` interface:
  ```go
  type NetLoader interface {
      Load(data []byte, format string, cfg dnnConfig) (cv.Net, error)
  }
  ```
- Implement `LoaderRegistry`:
  ```go
  type LoaderRegistry struct {
      loaders map[string]NetLoader
  }
  
  func (r *LoaderRegistry) Register(format string, loader NetLoader)
  func (r *LoaderRegistry) Load(data []byte, format string, cfg dnnConfig) (cv.Net, error)
  ```
- Register loaders:
  - ONNX loader (existing logic)
  - TensorFlow loader (when GoCV supports it)
  - Caffe loader (prototxt + caffemodel handling)
  - Darknet loader (config + weights)
- Implement `NetExtractor` interface for marshaling:
  ```go
  type NetExtractor interface {
      ExtractWeights(net cv.Net) ([]byte, string, error) // data, format, error
  }
  ```
- Update `Marshaller.Marshal` for `cv.Net`:
  - Prefer pass-through of raw bytes supplied by caller (GoCV already knows how to consume them).
  - If metadata must be transported, wrap only the metadata (format/backend/target) in protobuf while keeping weights as-is or referenced by path.
- Update `Unmarshaller.Unmarshal` for `cv.Net`:
  - Feed raw bytes directly to GoCV loaders.
  - Apply backend/target preferences after load.

**Benefits:**
- Extensible format support
- Marshaling support for `cv.Net`
- Testable via mock loaders/extractors
- GoCV remains responsible for actual model serialization; protobuf only carries metadata when needed.

### 4. Configuration & Dependency Injection
**Goal:** Split monolithic config and enable dependency injection.

**Steps:**
- Split `config` into focused structs:
  ```go
  type codecConfig struct {
      imageEncoding string
      tensorOpts    []tensorgocv.Option
  }
  
  type streamConfig struct {
      sources         []sourceSpec
      sequential      bool
      allowBestEffort bool
      sorter          fileSorter
  }
  
  type displayConfig struct {
      enabled  bool
      title    string
      width    int
      height   int
      onKey    func(int) bool
      onMouse  func(int, int, int, int) bool
      eventLoop func(context.Context, func() bool)
  }
  
  type dnnConfig struct {
      format  string
      backend cv.NetBackendType
      target  cv.NetTargetType
  }
  ```
- Create constructor functions:
  ```go
  func NewCodecConfig(opts ...CodecOption) codecConfig
  func NewStreamConfig(opts ...StreamOption) streamConfig
  func NewDisplayConfig(opts ...DisplayOption) displayConfig
  func NewDNNConfig(opts ...DNNOption) dnnConfig
  ```
- Update option constructors to only touch relevant config section.
- Update `Marshaller`/`Unmarshaller` constructors:
  ```go
  func NewMarshaller(
      codecCfg codecConfig,
      matEncoder MatEncoder,
      opts ...types.Option,
  ) types.Marshaller
  ```
- Accept interfaces in constructors, return concrete structs.

**Benefits:**
- Components only depend on what they need
- Easier to test (mock dependencies)
- Clear separation of concerns

### 5. Source Loader Interfaces
**Goal:** Abstract GoCV dependencies for testability.

**Steps:**
- Define source interfaces:
  ```go
  type VideoCapture interface {
      Read(mat *cv.Mat) bool
      Close() error
  }
  
  type FileSystem interface {
      Glob(pattern string) ([]string, error)
      ReadFile(path string) ([]byte, error)
      Stat(path string) (os.FileInfo, error)
  }
  
  type SourceLoader interface {
      Next(ctx context.Context) (frameItem, bool, error)
      Close() error
  }
  ```
- Implement GoCV-backed adapters:
  ```go
  type gocvVideoCapture struct {
      cap *cv.VideoCapture
  }
  
  type osFileSystem struct{}
  ```
- Refactor loaders to depend on interfaces:
  ```go
  func newImageLoader(
      path string,
      fs FileSystem,
      cfg codecConfig,
  ) SourceLoader
  ```
- Create factory for production use:
  ```go
  func NewSourceLoaderFactory(
      fs FileSystem,
      videoCaptureFactory func(int) (VideoCapture, error),
  ) SourceLoaderFactory
  ```

**Benefits:**
- Testable without GoCV dependencies
- Can swap implementations (e.g., mock file system)
- Honors Dependency Inversion Principle

## Testing Strategy

### Unit Tests
- **Envelope serialization:**
  - Table-driven tests for Mat round-trips (uint8, float32, float64)
  - Image encode/decode round-trips
  - Tensor envelope serialization
  - Legacy format backward compatibility
- **Stream sinks:**
  - Mock `StreamSink` implementations
  - Test `GobSink` serialization/deserialization
  - Test `MultiSink` composition
- **DNN loaders:**
  - Mock `NetLoader` implementations
  - Test format registry
  - Test backend/target application
- **Source loaders:**
  - Mock `FileSystem` and `VideoCapture`
  - Test loader implementations without GoCV
  - Test error handling paths

### Integration Tests
- Keep existing e2e tests:
  - `streams_test.go` - FrameStream from directory/video
  - `stream_display_manual_test.go` - Display functionality
- Add new tests:
  - Envelope round-trip with various Mat types
  - FrameStream gob serialization/deserialization
  - DNN format loading (ONNX, TensorFlow when available)
  - Mixed source synchronization

### Test Utilities
- Fake `FileSystem` implementation for testing
- Fake `VideoCapture` implementation for testing
- Test fixtures (small images, videos, ONNX models)

## Migration Strategy

### Phase 1: Add Envelope Support (Backward Compatible)
1. ✅ **Define protobuf schemas:** - **COMPLETED**
   - ✅ Created `proto/types/marshaller/gocv.proto` with FrameStreamManifest, SourceSpec, DeviceSpec, MatMetadata
   - ✅ Reused `types.core.Frame` from existing schemas
   - ✅ Generated Go code from protobuf
2. ✅ **Implement protobuf codec:** - **COMPLETED**
   - ✅ Protobuf manifest serialization/deserialization implemented
   - ✅ Format detection implemented (peek at first 4 bytes, fallback to legacy text)
   - ✅ Legacy raw image bytes still supported (GoCV-first policy)
   - ✅ Legacy text path format still supported
3. ✅ **Update tests:** - **COMPLETED**
   - ✅ Added tests for protobuf manifest round-trips (`TestManifestSerializationRoundTrip`)
   - ✅ Test format detection and fallback (`TestManifestFormatDetection`)
   - ✅ Test backward compatibility with legacy formats (`TestLegacyTextFormatBackwardCompatibility`, `TestLegacyTextFormatWithEmptyLines`)
   - ✅ Test edge cases for format detection (`TestFormatDetectionEdgeCases`)
   - ✅ Test multiple sources (`TestManifestWithMultipleSources`)
   - ✅ Test sync mode preservation (`TestManifestSyncMode`)
   - ✅ Test best-effort flag preservation (`TestManifestBestEffort`)

### Phase 2: Refactor Configuration
1. ✅ Split `config` into focused structs - **COMPLETED**
2. ✅ Update option constructors - **COMPLETED**
3. ✅ Update `Marshaller`/`Unmarshaller` to accept new configs - **COMPLETED**
4. ✅ Maintain backward compatibility via adapter - **COMPLETED** (legacy fields removed, focused configs used throughout)

### Phase 3: Abstract Dependencies
1. ❌ Introduce interfaces (`MatEncoder`, `NetLoader`, `FileSystem`, etc.) - **NOT STARTED**
2. ❌ Refactor loaders to use interfaces - **NOT STARTED**
3. ❌ Create production implementations - **NOT STARTED**
4. ❌ Update tests to use mocks - **NOT STARTED**

**Note:** This phase improves testability but is not critical for functionality. Current implementation works but is harder to test without GoCV dependencies.

### Phase 4: FrameStream Serialization
1. ✅ Implement `StreamSink` interface and implementations - **COMPLETED**
2. ✅ Add stream manifest format - **COMPLETED** (protobuf FrameStreamManifest)
3. ✅ Update `Marshaller.Marshal` for `FrameStream` - **COMPLETED** (writes manifest, uses StreamSink)
4. ✅ Update `Unmarshaller.Unmarshal` for `FrameStream` - **COMPLETED** (reads manifest, rebuilds stream)
5. ✅ Deprecate side-effect behavior (or make it opt-in) - **COMPLETED** (separated via StreamSink, side effects still supported)

## Open Questions

1. **Serialization Format Choice:**
   - **Protobuf vs. Gob:** Should we use protobuf (recommended) or gob for envelopes?
   - **Recommendation:** Use protobuf for consistency with codebase (`proto/types/marshaller/`), language-agnostic support, and reuse of existing `types.math.Tensor` and `types.core.Frame` schemas.
   - Protobuf schemas should be defined in `proto/types/marshaller/gocv.proto` following the pattern of `graph_metadata.proto`.
   **DECISION:** Protobuf

2. **Backward Compatibility:**
   - Do we need to support existing raw PNG/JPEG payloads indefinitely?
   - Should we add a format version header to distinguish envelope vs. legacy?
   - **Recommendation:** Add format detection (magic number or protobuf wire format detection) with fallback to legacy decode. Protobuf messages start with varint field tags, which can be used for detection.
   **DECISION:** we do not envelope RPN/JPEG and other image/video formats that are supported by gocv.

3. **FrameStream Chunking:**
   - How should very long streams be handled? (protobuf streaming vs. NDJSON)
   - Should manifests be separate files or embedded in stream?
   - **Recommendation:** Use protobuf streaming with embedded manifest at start. For very large streams, consider chunked format with multiple protobuf messages or length-prefixed records.
   **DECISION:** out of scope

4. **DNN Marshaling:**
   - Should `cv.Net` marshaling require original file path, or extract weights from loaded network?
   - GoCV doesn't provide API to extract weights from loaded `cv.Net`. Options:
     - Require original file path in options
     - Store original bytes alongside `cv.Net` (memory overhead)
     - Only support unmarshaling (current behavior)
   - **Recommendation:** For now, only support unmarshaling. If marshaling is needed, require original file path via option.
  **DECISION:** unmarshaling only

5. **Performance:**
   - Envelope format will be larger than raw image bytes (metadata overhead)
   - Gob encoding/decoding adds CPU overhead
   - **Recommendation:** Profile and optimize. Consider compression for large payloads. Protobuf encoding is generally more efficient than gob for cross-language scenarios.
**DECISION:** protobuf. but don't envelope images! GOCV should handle image formats it knows about.

6. **API Surface:**
   - Should envelope format be opt-in or default?
   - How to expose format selection to users?
   - **Recommendation:** Make protobuf envelope format default, with opt-out for legacy compatibility. Add `WithEnvelopeFormat(string)` option accepting `"protobuf"`, `"gob"`, or `"legacy"` (raw image bytes).
**DECISION:** envelope format is opinionated default. no format selection to users. (image format is selected via file extension and let gocv decide file format. NOTE: Video encoding option could be useful though)

7. **Protobuf Schema Location:**
   - Where should gocv-specific protobuf schemas be defined?
   - **Recommendation:** Follow existing pattern: `proto/types/marshaller/gocv.proto` (alongside `graph_metadata.proto`). This keeps marshaller-specific schemas together and follows the codebase convention.
**DECISION:** recommendation is correct.

## Success Criteria

- [ ] Mat/Image/Tensor round-trip preserves all metadata and data types
  - **Status:** By design, uses GoCV image encoding (can be lossy). Optional metadata sidecars can preserve Mat properties (not yet implemented).
- [x] FrameStream can be serialized and deserialized
  - **Status:** ✅ **COMPLETED** - Protobuf manifest serialization implemented
- [ ] DNN support for at least ONNX + TensorFlow formats
  - **Status:** Only ONNX supported (limited by GoCV). TensorFlow support depends on GoCV.
- [x] Configuration split into focused units
  - **Status:** ✅ **COMPLETED** - Split into `codecConfig`, `streamConfig`, `displayConfig`, `dnnConfig`
- [ ] Source loaders use interfaces (testable without GoCV)
  - **Status:** ❌ **NOT STARTED** - Phase 3 pending
- [x] All existing tests pass
  - **Status:** ✅ **VERIFIED** - All tests pass
- [x] New tests cover envelope format and stream serialization
  - **Status:** ✅ **COMPLETED** - Tests added for protobuf manifest serialization/deserialization, format detection, and backward compatibility
- [x] Backward compatibility maintained (legacy format support)
  - **Status:** ✅ **COMPLETED** - Legacy text path format still supported, format detection implemented

## Implementation Status Summary

### ✅ Completed Phases

**Phase 1: Add Envelope Support (Backward Compatible)**
- ✅ Protobuf schemas defined and generated
- ✅ Protobuf manifest serialization/deserialization implemented
- ✅ Format detection with fallback to legacy formats
- ❌ Tests for protobuf manifest round-trips (pending)

**Phase 2: Refactor Configuration**
- ✅ Config split into focused structs (`codecConfig`, `streamConfig`, `displayConfig`, `dnnConfig`)
- ✅ All option constructors updated to use focused configs
- ✅ All code migrated from legacy fields to focused configs
- ✅ Legacy fields completely removed

**Phase 4: FrameStream Serialization**
- ✅ `StreamSink` interface and implementations (`ProtobufSink`, `DirectorySink`, `DisplaySink`, `MultiSink`)
- ✅ Protobuf `FrameStreamManifest` format implemented
- ✅ `Marshaller.Marshal` writes manifest and uses StreamSink
- ✅ `Unmarshaller.Unmarshal` reads manifest and rebuilds stream
- ✅ Side-effect behavior separated from serialization

### ❌ Remaining Work

**Phase 1: Testing**
- ✅ Add tests for protobuf manifest serialization/deserialization round-trips
- ✅ Test format detection (protobuf vs legacy text)
- ✅ Test backward compatibility with legacy text path format
- ✅ Test edge cases and multiple scenarios

**Phase 3: Abstract Dependencies** (Optional - improves testability)
- ❌ Introduce interfaces (`MatEncoder`, `NetLoader`, `FileSystem`, `VideoCapture`, `SourceLoader`)
- ❌ Refactor loaders to use interfaces
- ❌ Create production implementations
- ❌ Update tests to use mocks

**Optional Enhancements:**
- ❌ Mat metadata sidecar support (preserve Mat properties alongside images)
- ❌ Video encoding options (currently format determined by file extension)
- ❌ DNN format registry for extensible format support (when GoCV adds more formats)

### Next Steps

1. ✅ **High Priority:** Add tests for protobuf manifest serialization/deserialization - **COMPLETED**
2. **Medium Priority:** Consider Phase 3 (abstract dependencies) if testability becomes a concern (optional)
3. **Low Priority:** Optional enhancements (metadata sidecars, video encoding options)

