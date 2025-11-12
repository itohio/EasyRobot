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

### Options

- Shared `types.Option` values (`WithContext`, `WithHint`, etc.) flow through constructors.
- GoCV-specific options (planned):
  - `WithPaths(paths ...string)` – configure input sources (files, directories, devices).
  - `WithVideoDevice(id int, width, height int)`.
  - `WithImageEncoding(format string)` for image marshaling.
  - `WithTensorOptions(...tensorgocv.Option)` to propagate tensor creation preferences.
  - DNN tweaks: `WithNetBackend`, `WithNetTarget`, `WithDNNFormat`.
- Options build internal `config` struct consumed by marshaller/unmarshaller.

### Data Envelopes

- `matEnvelope`: rows, cols, channels, type (OpenCV int), data bytes, continuous flag, metadata map.
- `imageEnvelope`: format string (`"png"` default) + byte payload.
- `dnnEnvelope`: format (`onnx`, `tensorflow`, etc.), weight bytes, optional config bytes, backend/target overrides.
- Envelopes serialized via `encoding/gob`.

### Marshaller Responsibilities

- Type switch:
  - `gocv.Mat` / `*gocv.Mat`.
  - EasyRobot GoCV tensor (`tensor/gocv.Tensor`) via accessor interface.
  - `image.Image`.
  - `gocv.Net` (not supported, error) – GoCV networks are loaded-only.
  - Fallback returns `types.NewError("marshal", "gocv", ...)`.
- Conversions:
  - Mat → envelope by cloning/ensuring continuity.
  - Tensor → Mat path.
  - Image → encode (PNG default).
- Write envelope using `gob.NewEncoder`.

### Unmarshaller Responsibilities

- Accept destination pointers:
  - `*gocv.Mat`, `**gocv.Mat`.
  - `*tensorgocv.Tensor` or `*types.Tensor`.
  - `*image.Image`.
  - `*types.FrameStream` to receive streaming outputs.
- `FrameStream` handling:
  - Spawn worker goroutine reading from configured sources (files, directories, video capture, devices).
  - Each frame becomes `types.Frame` with metadata map:
    - `index`, `timestamp`, `path`, `source`, `device`, etc.
    - `error` key if frame acquisition fails.
  - Provide graceful shutdown using `FrameStream.Close()` plus context cancellation.
- Source discovery:
  - `WithPaths` supports files + directories; directories enumerated lexicographically; multi path zipped by index for stereo pairs.
  - Video files read frame-by-frame using `gocv.VideoCaptureFile`.
  - Devices opened using `gocv.OpenVideoCapture`.
  - Mixed sources combined by round-robin index alignment.
  - Synchronization: attempt to read same index from each source; stop when any reaches EOS unless repeat is enabled (future option).
- Tensor conversion:
  - Use `tensor/gocv.FromMat` with configured options.
  - Destination type conversion via `types.Options.DestinationType`.

### DNN Loader

- Support in-memory ONNX (preferred), TensorFlow, Caffe, Darknet.
- Use GoCV byte loaders when available:
  - `gocv.ReadNetFromONNXBytes`.
  - `gocv.ReadNetFromTensorflowBytes`.
- For formats requiring files (e.g., Caffe prototxt + caffemodel), write to temporary directory ensuring cleanup.
- Apply backend/target preferences post-load.
- Return `*gocv.Net` or wrap as custom type if necessary.

### Context & Cancellation

- Honor `types.Options.Context`; default to `context.Background()`.
- Worker goroutines select on context done channel + internal close channel.
- `FrameStream.Close()` triggers early stop via internal cancel function.

### Error Handling

- Wrap failures with `types.NewError(op, "gocv", message, err)`.
- For streaming, surface fatal errors through frame metadata and closing stream with stored error accessible via option or separate method (TBD).

### Testing Strategy

- Table-driven tests covering:
  - Mat marshal/unmarshal round-trip (UINT8, FP32).
  - Image encode/decode.
  - Tensor output path.
  - FrameStream from directory of images (use small fixtures).
  - Video capture simulation (use sample video in testdata or synthetic generator).
  - Mixed source alignment (dual image paths).
  - Context cancellation and `FrameStream.Close()`.
  - DNN loader using small ONNX fixture.
  - Error paths (unsupported types, missing files, device unavailable).

### Open Questions / TODO

- Confirm availability of GoCV byte-based DNN loaders in v0.42.0; implement fallbacks if missing.
- Decide representation for multi-source synchronization errors (drop frame vs propagate).
- Explore pluggable source abstraction to simplify tests.


