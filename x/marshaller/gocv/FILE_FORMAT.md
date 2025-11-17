# GoCV Marshaller File Format Specification

This document describes the file formats used by the GoCV marshaller/unmarshaller for serializing and deserializing data.

## Design Philosophy

**GoCV-First Approach:** The GoCV marshaller delegates all image, video, and DNN serialization to GoCV itself. GoCV handles:
- Image formats (PNG, JPEG, BMP, TIFF, etc.)
- Video formats (AVI, MP4, MOV, MKV, etc.)
- DNN model formats (ONNX, TensorFlow, Caffe, Darknet - as supported by GoCV)

**Protobuf for Metadata:** Protobuf is used only for structural data that GoCV cannot serialize natively:
- Frame stream manifests and source specifications
- Metadata sidecars (optional annotations about Mat/Image properties)
- Control structures and synchronization hints

This approach ensures compatibility with existing GoCV tooling while enabling structured metadata exchange.

## Overview

The GoCV marshaller supports several data types, each with its own serialization format:

1. **Mat/Image/Tensor** - GoCV-managed image formats (PNG/JPEG/BMP)
2. **Custom .mat files** - Binary format with magic number (for disk I/O only)
3. **DNN models** - GoCV-managed model formats (ONNX, etc.)
4. **FrameStream manifests** - Protobuf-encoded metadata
5. **Mat metadata sidecars** - Optional protobuf metadata (alongside images)

## Format Details

### 1. Mat/Image/Tensor Format

**Format:** GoCV-managed image bytes (PNG, JPEG, BMP, TIFF, etc.)

**Policy:** GoCV handles all image encoding/decoding. The marshaller delegates to GoCV's native image format support.

**Encoding:**
- `gocv.Mat` → encoded via `cv.IMEncode(cv.FileExt(ext), mat)`
- `image.Image` → converted to Mat (RGBA→BGR) → encoded via `cv.IMEncode`
- `types.Tensor` → extracted Mat via `Accessor.MatClone()` → encoded via `cv.IMEncode`

**Format Selection:**
- Format determined by file extension (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, etc.)
- Default: PNG (`.png`) if no extension specified
- Configurable via `WithImageEncoding(format string)` for marshalling
- GoCV automatically detects format during unmarshalling based on file extension or content

**Byte Layout:**
```
[Image Bytes - GoCV format]
```

The image bytes are written directly to `io.Writer` without any wrapper, header, or metadata. This ensures compatibility with standard image viewers and GoCV tooling.

**Limitations:**
- **Lossy for non-uint8 types:** Original data type is lost (float32/float64 → uint8) when using standard image formats
- **Can be non-lossy:** For lossless storage, use formats that support the data type (e.g., `.exr` for float32, `.tif` with appropriate compression)
- **No embedded metadata:** Dimensions inferred from image, channel count inferred, Mat type/continuous flag lost
- **Precision:** Limited to 8-bit per channel for standard formats (PNG/JPEG)
- **Color space:** Images converted to BGR (OpenCV standard)

**Optional Metadata Sidecar:**
If Mat metadata needs to be preserved, a protobuf `MatMetadata` sidecar can be written alongside the image file (see Section 6).

**Example:**
```go
mat := gocv.IMRead("input.jpg", gocv.IMReadColor)
marshaller.Marshal(writer, mat) // Writes PNG bytes directly (GoCV format)

// With metadata sidecar (optional)
marshaller.Marshal(writer, mat, gocv.WithMetadataSidecar(true))
// Writes: image.png + image.png.meta (protobuf)
```

### 2. Custom .mat File Format

**Format:** Binary format with magic number and structured header

**Purpose:** Used for reading custom mat files from disk (not used for marshaller/unmarshaller serialization)

**Byte Layout:**
```
Offset  Size    Type      Description
------  ------  --------  -----------------------------------------
0       8        uint64    Magic number: 0xabcdef0012345678
8       1        uint8     Number of dimensions (lenSizes)
9       4*N      int32[]   Dimension sizes (N = lenSizes)
9+4*N   4        MatType   OpenCV Mat type (cv.MatType)
13+4*N  8        uint64    Data length in bytes (lenBytes)
21+4*N  lenBytes byte[]    Raw Mat data bytes
```

**Endianness:** Little-endian

**Example:**
```go
// Reading a custom .mat file
mat, err := readCustomMat("custom.mat")
// File contains:
// - Magic: 0xabcdef0012345678
// - Dimensions: [480, 640, 3] (height, width, channels)
// - Type: cv.MatTypeCV8UC3
// - Data: 480*640*3 bytes
```

### 3. DNN Model Format

**Format:** GoCV-managed model bytes (ONNX, TensorFlow, Caffe, Darknet - as supported by GoCV)

**Policy:** GoCV handles all DNN model loading/saving. The marshaller delegates to GoCV's native DNN format support.

**Encoding:**
- Raw bytes written directly to `io.Writer` (pass-through for `[]byte`)
- No wrapper or metadata added
- Format determined by file extension or `WithDNNFormat()` option

**Supported Formats:**
- **ONNX** (`.onnx`) - Currently supported for unmarshaling
- **TensorFlow, Caffe, Darknet** - Supported as GoCV adds support
- Format hint provided via `WithDNNFormat("onnx")` if not inferrable from extension

**Unmarshaling Process:**
1. Read all bytes from `io.Reader`
2. Write bytes to temporary file (with appropriate extension)
3. Load model via `cv.ReadNet(modelPath, configPath)` or GoCV byte loaders when available
4. Apply backend/target preferences if configured (`WithNetBackend`, `WithNetTarget`)
5. Clean up temporary file

**Byte Layout:**
```
[Model Bytes - GoCV format]
```

**Example:**
```go
// Marshaling (pass-through)
data, _ := os.ReadFile("model.onnx")
marshaller.Marshal(writer, data) // Writes raw ONNX bytes (GoCV format)

// Unmarshaling
var net gocv.Net
unmarshaller.Unmarshal(reader, &net, gocv.WithDNNFormat("onnx"))
// Reads bytes, writes to temp file, loads via cv.ReadNet
```

**Limitations:**
- Only formats supported by GoCV are available
- No marshaling support for `cv.Net` (network cannot be serialized - unmarshaling only)
- Temporary file required during unmarshaling (cleaned up automatically)
- Backend/target preferences applied post-load (not serialized with model)

### 4. FrameStream Manifest Format

**Format:** Length-prefixed protobuf-encoded `FrameStreamManifest` message

**Purpose:** Used to serialize FrameStream configuration (sources, sync mode, metadata) for transport/storage. Actual frame pixel data remains in GoCV-managed formats.

**Schema:** Defined in `proto/types/marshaller/gocv.proto`

**Byte Layout:**
```
[4 bytes: uint32 length (little-endian)]
[length bytes: Protobuf Wire Format - FrameStreamManifest]
```

**Serialization Process:**
1. Convert internal `config` to protobuf `FrameStreamManifest` via `configToManifest()`.
2. Marshal protobuf message to bytes using `proto.Marshal()`.
3. Write 4-byte little-endian length prefix.
4. Write protobuf message bytes.

**Structure:**
- `sources`: Array of `SourceSpec` messages describing frame sources
- `sync_mode`: String ("parallel" or "sequential")
- `best_effort`: Boolean (best-effort synchronization for devices)
- `metadata`: Map of string key-value pairs

**SourceSpec Fields:**
- `kind`: Enum (`SOURCE_KIND_SINGLE`, `SOURCE_KIND_VIDEO_FILE`, `SOURCE_KIND_VIDEO_DEVICE`, `SOURCE_KIND_FILE_LIST`)
- `path`: String (file path, directory, or glob pattern)
- `device`: `DeviceSpec` (if kind is `SOURCE_KIND_VIDEO_DEVICE`)
  - `id`: int32 (device ID)
  - `width`: int32 (optional width)
  - `height`: int32 (optional height)
- `files`: Array of strings (pre-resolved file list if kind is `SOURCE_KIND_FILE_LIST`)

**Deserialization Process:**
1. Peek at first 4 bytes to detect format (protobuf vs legacy text).
2. If length prefix detected and reasonable (< 1MB), read length.
3. Read protobuf message bytes.
4. Unmarshal to `FrameStreamManifest`.
5. Convert manifest to internal `config` via `manifestToConfig()`.
6. Rebuild stream from sources using GoCV.

**Format Detection:**
The unmarshaller uses heuristics to detect format:
- **Protobuf manifest:** First 4 bytes are a reasonable little-endian uint32 length (< 1MB).
- **Legacy text:** First bytes are printable characters, newlines, or zero-length.

**Legacy Format (Newline-separated paths):**
For backward compatibility, unmarshaller also supports newline-separated text paths when no protobuf manifest is detected:

```
[Path 1]\n
[Path 2]\n
[Path 3]\n
...
```

**Path Types:**
- Image files: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff` (GoCV formats)
- Video files: `.avi`, `.mp4`, `.mov`, `.mkv`, `.wmv` (GoCV formats)
- Directories: Will be enumerated (glob patterns supported)
- Custom .mat files: `.mat`

**Example:**
```go
// Serializing FrameStream with protobuf manifest
var stream types.FrameStream
var buf bytes.Buffer
marshaller.Marshal(&buf, stream)
// buf contains:
//   [4 bytes: length]
//   [protobuf FrameStreamManifest bytes]
//   [optional: summary of written files]

// Deserializing FrameStream
var restored types.FrameStream
unmarshaller.Unmarshal(&buf, &restored)
// Reads manifest, rebuilds stream from sources
```

**Note:** Frame pixel data is not embedded in the manifest. Frames are read from the specified sources using GoCV when the stream is consumed. The manifest only contains configuration/metadata.

### 5. Mat Metadata Sidecar Format (Optional)

**Format:** Protobuf-encoded `MatMetadata` message

**Purpose:** Optional metadata sidecar file written alongside GoCV-encoded images to preserve Mat properties (rows, cols, channels, type, continuous flag) that are lost in standard image formats.

**Schema:** Defined in `proto/types/marshaller/gocv.proto`

**File Naming Convention:**
- Image file: `image.png`
- Metadata sidecar: `image.png.meta` (or `image.meta`)

**Byte Layout:**
```
[Protobuf Wire Format - MatMetadata]
```

**Structure:**
- `rows`: Number of rows
- `cols`: Number of columns
- `channels`: Number of channels
- `mat_type`: OpenCV MatType (cv.MatType)
- `continuous`: Continuous flag
- `annotations`: Map of string key-value pairs for additional metadata

**Usage:**
```go
// Writing with metadata sidecar
mat := gocv.IMRead("input.jpg", gocv.IMReadColor)
marshaller.Marshal(writer, mat, gocv.WithMetadataSidecar(true))
// Writes: image.png + image.png.meta

// Reading with metadata sidecar
var restored gocv.Mat
unmarshaller.Unmarshal(reader, &restored, gocv.WithMetadataSidecar(true))
// Reads image.png and image.png.meta, restores Mat properties
```

**Note:** This is optional. If metadata sidecar is not present, Mat properties are inferred from the image (may lose precision/type information).

### 6. FrameStream Summary Format

**Format:** Newline-separated file paths

**Purpose:** Written to `io.Writer` after `FrameStream` consumption (if file writer was used)

**Byte Layout:**
```
[Output Path 1]\n
[Output Path 2]\n
[Output Path 3]\n
...
```

**Encoding:** UTF-8 text, newline-separated (`\n`)

**Content:** Full paths to files written during stream consumption (GoCV-encoded images)

**Example:**
```
/output/dir/image_000000.png
/output/dir/image_000001.png
/output/dir/image_000002.png
```

**Usage:**
```go
var stream types.FrameStream
// ... configure stream ...
var buf bytes.Buffer
marshaller.Marshal(&buf, stream, gocv.WithPath("/output/dir"))
// buf now contains newline-separated paths of written files
```

### 7. StreamSink Architecture

**Purpose:** The `StreamSink` interface separates frame serialization from consumption, allowing frames to be written to multiple destinations simultaneously.

**Interface:**
```go
type StreamSink interface {
    WriteFrame(frame types.Frame) error
    Close() error
}
```

**Implementations:**

1. **`ProtobufSink`**
   - **Format:** Length-prefixed protobuf `types.core.Frame` messages
   - **Purpose:** Serialize frame metadata to protobuf stream
   - **Byte Layout:** `[4 bytes: length][protobuf Frame bytes]` (per frame)
   - **Note:** Per GoCV-first policy, pixel data is NOT embedded in protobuf. Only metadata is serialized.

2. **`DirectorySink`**
   - **Format:** GoCV-encoded images (PNG/JPEG/BMP) written to disk
   - **Purpose:** Write frames to filesystem directories
   - **File Naming:** `image_%06d.{ext}` (e.g., `image_000000.png`)
   - **Encoding:** Determined by file extension or `WithImageEncoding()` option

3. **`DisplaySink`**
   - **Format:** GoCV window display (no serialization)
   - **Purpose:** Display frames in real-time using GoCV window
   - **Configuration:** Window title, size, event handlers via display options

4. **`MultiSink`**
   - **Format:** Composes multiple sinks
   - **Purpose:** Write frames to multiple destinations simultaneously
   - **Usage:** `NewMultiSink(sink1, sink2, sink3)`

**Frame Serialization Flow:**
```
FrameStream → StreamSink.WriteFrame() → [DirectorySink, DisplaySink, ProtobufSink, ...]
```

**Benefits:**
- Clear separation of concerns (serialization vs consumption)
- Composability (multiple sinks can be used together)
- Testability (sinks can be mocked)
- Extensibility (new sink types can be added)

## Format Comparison

| Type | Format | Metadata | Lossy | Wrapper | Handler |
|------|--------|----------|-------|---------|---------|
| Mat/Image/Tensor | PNG/JPEG/BMP (GoCV) | Optional sidecar | Yes* | No | GoCV |
| Custom .mat | Binary | Yes | No | Yes (magic + header) | Custom |
| DNN | Raw bytes (GoCV) | No | No | No | GoCV |
| FrameStream manifest | Protobuf | Yes | No | Yes (protobuf) | Protobuf |
| Mat metadata sidecar | Protobuf | Yes | No | Yes (protobuf) | Protobuf |
| FrameStream summary | Text (newline) | No | No | No | Text |

*Can be non-lossy with appropriate format selection (e.g., `.exr` for float32)

## Design Decisions

**GoCV-First Policy:**
- Images, video, and DNN models are handled by GoCV in their native formats
- No protobuf envelopes for pixel/model data
- Protobuf is used only for metadata and control structures

**Rationale:**
1. **Compatibility:** Standard image/model formats work with existing GoCV tooling
2. **Performance:** No unnecessary encoding/decoding overhead
3. **Simplicity:** Leverage GoCV's proven format support
4. **Flexibility:** Optional metadata sidecars when needed

**Protobuf Usage:**
- Frame stream manifests (source specifications, sync mode)
- Optional Mat metadata sidecars (preserve Mat properties)
- Future: Error responses, progress information, control structures

## Backward Compatibility

**Current State:**
- Raw PNG/JPEG/BMP bytes are supported (GoCV format)
- Newline-separated path lists are supported for FrameStream (legacy)
- Custom .mat binary format is supported (disk I/O only)

**Future:**
- Protobuf FrameStream manifests will be the default
- Legacy newline-separated paths will be supported via format detection
- Mat metadata sidecars are optional (backward compatible when absent)

## References

- GoCV documentation: https://gocv.io/
- OpenCV Mat documentation: https://docs.opencv.org/
- PNG specification: https://www.w3.org/TR/PNG/
- JPEG specification: ISO/IEC 10918

