# GoCV Marshaller Design

## Goals

- Provide marshaller/unmarshaller pair for GoCV backed data.
- Support domain objects:
  - `gocv.Mat` and EasyRobot GoCV tensor wrapper.
  - `image.Image`.
  - DNN models via `gocv.Net`.
  - Streaming sources (images, video files, capture devices, mixed paths) exposed as `types.FrameStream`.
- Expose options for context cancellation, backend/target selection, image/tensor conversion.

## High-Level Architecture

### Configuration Structure

The marshaller uses a **focused configuration architecture** with separate config structs for different concerns:

- **`codecConfig`** – Image encoding format, tensor options
- **`streamConfig`** – Frame stream sources, sync mode, file sorter
- **`displayConfig`** – Display window settings, event handlers
- **`dnnConfig`** – DNN format, backend, target preferences

These focused configs are embedded in the main `config` struct for backward compatibility. Components can gradually migrate to use focused configs directly.

### Options

- Shared `types.Option` values (`WithContext`, `WithHint`, etc.) flow through constructors.
- GoCV-specific options (implemented):
  - **Codec options:**
    - `WithImageEncoding(format string)` – image encoding format for marshaling (png, jpeg, bmp, etc.)
    - `WithTensorOptions(...tensorgocv.Option)` – propagate tensor creation preferences
  - **Stream options:**
    - `WithPath(path string)` – configure input sources (files, directories, devices)
    - `WithVideoDevice(id int, width, height int)` – configure video capture device
    - `WithBestEffortDevices(enable bool)` – best-effort synchronization for video devices
    - `WithSequential(enable bool)` – sequential vs parallel source consumption
    - `WithSorter(sorter func([]string) []string)` – filename sorting function
  - **Display options:**
    - `WithDisplay(ctx context.Context)` – enable display window
    - `WithTitle(title string)` – display window title
    - `WithWindowSize(width, height int)` – display window size
    - `WithOnKey(handler func(int) bool)` – key event handler
    - `WithOnMouse(handler func(int, int, int, int) bool)` – mouse event handler
    - `WithEventLoop(loop func(context.Context, func() bool))` – custom event loop
  - **DNN options:**
    - `WithNetBackend(backend cv.NetBackendType)` – DNN backend selection
    - `WithNetTarget(target cv.NetTargetType)` – DNN target selection
    - `WithDNNFormat(format string)` – DNN format hint (currently only "onnx" supported)
- Options update both focused configs and legacy fields for backward compatibility.

### Data Serialization Format

**GoCV-First Policy:**
The marshaller delegates all image, video, and DNN serialization to GoCV itself. GoCV handles:
- Image formats (PNG, JPEG, BMP, TIFF, etc.) via `cv.IMEncode`/`cv.IMDecode`
- Video formats (AVI, MP4, MOV, MKV, etc.) via GoCV video capture/writer APIs
- DNN model formats (ONNX, TensorFlow, Caffe, Darknet - as supported by GoCV)

**Protobuf for Metadata:**
Protobuf is used only for structural data that GoCV cannot serialize natively:
- Frame stream manifests (source specifications, sync mode)
- Optional Mat metadata sidecars (preserve Mat properties alongside images)
- Control structures and synchronization hints

**Mat/Image/Tensor Format:**
- Encoded as raw image bytes (PNG/JPEG/BMP) using GoCV's `IMEncode`.
- Format determined by file extension or `WithImageEncoding()` option.
- No embedded metadata (dimensions inferred from image, type/channels lost).
- Precision limited to 8-bit per channel for standard formats (can be non-lossy with appropriate format selection).
- Optional metadata sidecar (protobuf `MatMetadata`) can preserve Mat properties.
- See `FILE_FORMAT.md` for detailed format specification.

**Custom .mat File Format:**
- Binary format with magic number (`0xabcdef0012345678`), dimensions, type, and raw data bytes.
- Used for reading custom mat files from disk.
- Not used for marshaller/unmarshaller serialization.
- See `FILE_FORMAT.md` for detailed format specification.

**DNN Format:**
- Raw bytes written directly to `io.Writer` (GoCV-managed format).
- Currently only ONNX format supported for unmarshaling.
- Format hint provided via `WithDNNFormat("onnx")`.
- Temporary file created during unmarshaling (cleaned up automatically).
- Backend/target preferences applied post-load (not serialized with model).

**FrameStream Serialization:**
- **Manifest (protobuf):** `FrameStreamManifest` written to `io.Writer` when serializing.
  - Contains source specifications, sync mode, best-effort flag, metadata.
  - Actual frame pixel data remains in GoCV-managed formats (not embedded).
- **Consumption:** When `io.Writer` is provided, manifest is written first, then frames are consumed.
- **Side effects:** Frames can be written to disk and/or displayed via `StreamSink` implementations.
- **Summary:** Optional summary of written file paths written after consumption.
- See `FILE_FORMAT.md` for detailed format specification.


### Marshaller Responsibilities

- Type switch:
  - `gocv.Mat` / `*gocv.Mat` → encode as image bytes (PNG/JPEG/BMP) via GoCV.
  - EasyRobot GoCV tensor (`tensor/gocv.Tensor`) → extract Mat via `Accessor.MatClone()` → encode as image bytes via GoCV.
  - `image.Image` / `*image.Image` → convert to Mat (RGBA→BGR) → encode as image bytes via GoCV.
  - `types.FrameStream` / `*types.FrameStream` → write protobuf manifest, consume stream via `StreamSink`, optionally write summary.
  - `[]byte` → pass-through (for DNN weights - GoCV format).
  - Fallback returns `types.NewError("marshal", "gocv", ...)`.
- Conversions:
  - Mat → image bytes via `cv.IMEncode` (format from `config.codec.imageEncoding`).
  - Tensor → Mat via `tensorgocv.Accessor.MatClone()`.
  - Image → Mat via `cv.ImageToMatRGBA` + `cv.CvtColor` (RGBA→BGR).
- Write raw bytes directly to `io.Writer` (GoCV-managed formats, no envelope wrapper).
- FrameStream serialization:
  - Write protobuf `FrameStreamManifest` to `io.Writer` (if provided).
  - Create `StreamSink` implementations based on configuration (DirectorySink, DisplaySink, ProtobufSink).
  - Consume frames from stream and write to sinks.
  - Write summary of written files after consumption (if file writer used).

### Unmarshaller Responsibilities

- Accept destination pointers:
  - `*gocv.Mat`, `**gocv.Mat` → decode image bytes via `cv.IMDecode` (GoCV format).
  - `*tensorgocv.Tensor` or `*types.Tensor` → decode Mat → convert to tensor via `tensorgocv.FromMat`.
  - `*image.Image`, `**image.Image` → decode image bytes → convert to `image.Image` via `mat.ToImage()`.
  - `*types.FrameStream` → read protobuf manifest from `io.Reader` (or legacy text paths) → rebuild stream from manifest.
  - `*gocv.Net`, `**gocv.Net` → read raw bytes → load DNN model (ONNX only, via temp file - GoCV format).
- `FrameStream` handling:
  - **Format detection:** Try to read protobuf `FrameStreamManifest` from `io.Reader`.
    - If manifest found: use it to configure stream sources and sync mode.
    - If no manifest: fall back to legacy newline-separated text paths.
    - If `config.stream.sources` already configured: use configured sources (manifest/reader ignored).
  - **Source resolution:** Classify paths (image, video, directory/glob), open video devices.
  - **Stream creation:** Spawn worker goroutine reading from configured sources (files, directories, video capture, devices).
  - **Frame metadata:** Each frame becomes `types.Frame` with metadata map:
    - `index`, `timestamp`, `path`, `source`, `device`, `filename`, `name`, `set`, `set_index`, etc.
    - `error` key if frame acquisition fails.
  - **Shutdown:** Graceful shutdown using `FrameStream.Close()` plus context cancellation.
- Source discovery:
  - `WithPath` supports files, directories (glob patterns), and video files.
  - Directories/globs enumerated and sorted (lexicographically by default, customizable via `WithSorter`).
  - Video files read frame-by-frame using `gocv.VideoCaptureFile`.
  - Devices opened using `gocv.OpenVideoCapture` with optional width/height configuration.
  - Mixed sources combined by round-robin index alignment (parallel) or sequential consumption.
  - Synchronization: attempt to read same index from each source; stop when any reaches EOS (unless `allowBestEffort` enabled for devices).
- Tensor conversion:
  - Use `tensor/gocv.FromMat` with configured options from `config.codec.tensorOpts`.
  - Destination type conversion via `types.Options.DestinationType` (currently reserved for future use).

### DNN Loader

- **Current Implementation:**
  - Only ONNX format supported for unmarshaling.
  - Raw bytes read from `io.Reader` → written to temporary file → loaded via `cv.ReadNet`.
  - Temporary directory cleaned up after load.
  - Backend/target preferences applied post-load via `net.SetPreferableBackend` / `net.SetPreferableTarget`.
- **Planned (not implemented):**
  - Support for TensorFlow, Caffe, Darknet formats.
  - Use GoCV byte loaders when available (e.g., `gocv.ReadNetFromONNXBytes`).
  - For formats requiring multiple files (e.g., Caffe prototxt + caffemodel), write to temporary directory ensuring cleanup.

### FrameStream Marshalling Behavior

**Serialization and Consumption:**
`Marshaller.Marshal` for `FrameStream` supports both serialization and consumption:

1. **Manifest Serialization (if `io.Writer` provided):**
   - Writes protobuf `FrameStreamManifest` to `io.Writer`.
   - Manifest contains source specifications, sync mode, and metadata.
   - Frame pixel data is NOT embedded (remains in GoCV-managed formats).

2. **Frame Consumption:**
   - Consumes the stream immediately (reads all frames).
   - Uses `StreamSink` implementations to handle frames:
     - `DirectorySink`: Writes frames to disk (if output directories configured).
     - `DisplaySink`: Displays frames (if display enabled).
     - `ProtobufSink`: Serializes frame metadata to protobuf stream (optional).
     - `MultiSink`: Composes multiple sinks.

3. **Summary:**
   - Optionally writes a summary of written file paths to `io.Writer` after consumption (if file writer used).

**StreamSink Interface:**
The `StreamSink` interface separates serialization from consumption:
- `WriteFrame(frame types.Frame) error` – write a frame to the sink
- `Close() error` – close the sink

This allows frames to be written to multiple destinations simultaneously (e.g., file + display + protobuf stream).

### Context & Cancellation

- Honor `types.Options.Context`; default to `context.Background()`.
- Worker goroutines select on context done channel + internal close channel.
- `FrameStream.Close()` triggers early stop via internal cancel function.
- Event loop respects context cancellation (default loop or custom via `WithEventLoop`).

### Error Handling

- Wrap failures with `types.NewError(op, "gocv", message, err)`.
- For streaming, surface fatal errors through frame metadata (`error` key) and closing stream.
- Frame writers (file/display) can return `errStopLoop` to signal early termination.
- Errors during frame acquisition result in error frames sent to stream before closing.

### Testing Strategy

- Table-driven tests covering:
  - Mat marshal/unmarshal round-trip (UINT8, lossy for FP32 due to image encoding).
  - Image encode/decode.
  - Tensor output path.
  - FrameStream from directory of images (use small fixtures).
  - Video capture simulation (use sample video in testdata or synthetic generator).
  - Mixed source alignment (dual image paths).
  - Context cancellation and `FrameStream.Close()`.
  - DNN loader using small ONNX fixture.
  - Error paths (unsupported types, missing files, device unavailable).
- Manual tests for display functionality (see `stream_display_manual_test.go`).

### StreamSink Architecture

**Interface:**
```go
type StreamSink interface {
    WriteFrame(frame types.Frame) error
    Close() error
}
```

**Implementations:**
- **`ProtobufSink`** – Serializes frame metadata to protobuf stream (pixel data not embedded per GoCV-first policy).
- **`DirectorySink`** – Writes frames to disk as GoCV-encoded images.
- **`DisplaySink`** – Displays frames in GoCV window.
- **`MultiSink`** – Composes multiple sinks for simultaneous writing.

**Benefits:**
- Clear separation between serialization and consumption.
- Composability (multiple sinks can be used together).
- Testability (sinks can be mocked).
- Extensibility (new sink types can be added).

### Known Limitations

1. **Lossy Serialization (by design):** Mat/Image/Tensor marshaling uses GoCV image encoding (PNG/JPEG), which may lose:
   - Original data type (float32/float64 → uint8) for standard formats.
   - Channel count information (inferred from image).
   - Continuous flag and other Mat metadata.
   - Precision for non-uint8 types.
   - **Mitigation:** Can be non-lossy with appropriate format selection (e.g., `.exr` for float32). Optional metadata sidecars can preserve Mat properties.

2. **DNN Support Limited:** Only formats supported by GoCV are available. Currently only ONNX format supported for unmarshaling; no marshaling support for `cv.Net` (unmarshaling only).

3. **Configuration Migration:** Focused configs are available but legacy fields remain for backward compatibility. Gradual migration to focused configs is ongoing.

### Recent Improvements

1. **✅ FrameStream Serialization:** Protobuf manifest serialization implemented. FrameStream can now be serialized for storage/transmission.

2. **✅ StreamSink Abstraction:** Stream consumption separated from serialization via `StreamSink` interface.

3. **✅ Focused Configuration:** Config split into focused units (codec, stream, display, DNN) with backward compatibility.

4. **✅ Protobuf Metadata:** Protobuf used for manifests and metadata (per GoCV-first policy).

### Open Questions / TODO

- Add Mat metadata sidecar support (optional feature).
- Gradually migrate components to use focused configs directly (remove legacy fields over time).
- Add tests for protobuf manifest serialization/deserialization.
- Consider adding video encoding options (currently format determined by file extension).


