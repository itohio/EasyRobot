# Audio Marshaller Specification

## Overview

The Audio marshaller provides a unified interface for audio I/O operations using the malgo library (via gocoustics). It supports marshalling and unmarshalling of audio data as tensors, reading from and writing to WAV files, and real-time audio device capture/playback. The marshaller follows the same patterns as the GoCV marshaller, providing a consistent API across different media types.

Key features:
- **WAV File Support**: Read/write WAV files as tensors with automatic format detection
- **Multi-file Concatenation**: Read multiple WAV files and combine into multichannel tensors (e.g., two mono files → stereo tensor)
- **Audio Device Support**: Real-time capture and playback using malgo
- **Device Enumeration**: List available audio input/output devices
- **Tensor Integration**: Seamless conversion between audio samples and EasyRobot tensors
- **Multichannel Support**: Read/write mono, stereo, and multichannel audio

## Core Concepts

### Data Types

The marshaller handles several core data types:

1. **Tensor**: Audio samples represented as tensors with shape `[frames, channels]`
2. **WAV Files**: Standard WAV format files (PCM, 16-bit, various sample rates)
3. **Audio Devices**: Input (capture) and output (playback) devices via malgo
4. **FrameStream**: Real-time audio stream from devices (reuses `types.FrameStream` from marshaller types)

### Audio Tensor Format

Audio data is represented as tensors with the following conventions:
- **Shape**: `[frames, channels]` for row-major layout (frames × channels)
- **Data Type**: Matches source format to minimize conversion overhead:
  - **16-bit PCM WAV/device**: `INT16` (int16) tensors
  - **24-bit PCM WAV/device**: `INT32` (int32) tensors  
  - **32-bit PCM WAV/device**: `INT32` (int32) tensors
  - **32-bit float WAV/device**: `FP32` (float32) tensors
  - **8-bit PCM WAV/device**: `INT8` (int8) tensors
- **Frame**: One sample per channel at a specific time point
- **Channel**: One audio channel (mono=1, stereo=2, etc.)

**Important**: Tensors preserve the native format from the source. Users can convert to FP32 or other formats as needed using tensor operations. This avoids unnecessary conversion overhead.

Example shapes:
- Mono audio, 48000 samples: `[48000, 1]`
- Stereo audio, 48000 samples: `[48000, 2]`
- 5.1 surround, 48000 samples: `[48000, 6]`

### Marshalling vs Unmarshalling

**Single Responsibility Principle:**
- **Unmarshaller**: Reads from WAV files and/or audio devices, combines them into tensors
  - No resampling - preserves original sample rates
  - Synchronizes multiple sources (devices/files) when combining
  - Combines channels from multiple sources into single tensor
- **Marshaller**: Writes tensors to WAV files and/or audio devices
  - No resampling - uses tensor's sample rate or configured rate
  - Splits multichannel tensors across multiple devices if needed
  - Synchronizes writes to multiple outputs

### Configuration Architecture

The marshaller uses focused configuration structs:

```go
type config struct {
    ctx          context.Context
    wav          wavConfig      // WAV file format settings
    device       deviceConfig   // Audio device settings
    stream       streamConfig   // Stream synchronization
    tensor       tensorConfig   // Tensor conversion options
}
```

## Use Cases and Examples

### 1. Reading a Single WAV File

**Unmarshalling a WAV file to tensor:**
```go
unmarshaller := audio.NewUnmarshaller()
var tensor types.Tensor
err := unmarshaller.Unmarshal(strings.NewReader("audio.wav"), &tensor)
// tensor shape: [frames, channels]
// tensor dtype: matches WAV format (INT16 for 16-bit PCM, FP32 for float, etc.)

// Get format information
sampleRate := unmarshaller.SampleRate()
channels := unmarshaller.Channels()
dataType := tensor.DataType() // INT16, FP32, etc.
```

**With explicit path option:**
```go
unmarshaller := audio.NewUnmarshaller(audio.WithWAVPath("audio.wav"))
var tensor types.Tensor
err := unmarshaller.Unmarshal(nil, &tensor)
```

### 2. Writing a Tensor to WAV File

**Marshalling a tensor to WAV:**
```go
marshaller := audio.NewMarshaller()
var buf bytes.Buffer
err := marshaller.Marshal(&buf, tensor, 
    audio.WithWAVPath("output.wav"),
    audio.WithSampleRate(48000),
    audio.WithChannels(2))
// Tensor is written as-is (INT16 → 16-bit PCM, FP32 → 32-bit float)
// If WAV format doesn't support tensor dtype, automatic conversion occurs
```

**Or write directly to file:**
```go
marshaller := audio.NewMarshaller(
    audio.WithWAVPath("output.wav"),
    audio.WithSampleRate(48000),
    audio.WithChannels(2))
err := marshaller.Marshal(nil, tensor)
```

### 3. Reading Multiple WAV Files as Multichannel Tensor

**Combine two mono files into stereo tensor:**
```go
unmarshaller := audio.NewUnmarshaller(
    audio.WithWAVPath("left.wav"),
    audio.WithWAVPath("right.wav"))
var tensor types.Tensor
err := unmarshaller.Unmarshal(nil, &tensor)
// tensor shape: [frames, 2] (stereo)
// Channel 0: left.wav, Channel 1: right.wav
// Note: Files must have same sample rate (no resampling)
```

**Combine multiple files (must have same sample rate):**
```go
unmarshaller := audio.NewUnmarshaller(
    audio.WithWAVPath("file1.wav"),
    audio.WithWAVPath("file2.wav"),
    audio.WithWAVPath("file3.wav"))
var tensor types.Tensor
err := unmarshaller.Unmarshal(nil, &tensor)
// tensor shape: [frames, 3]
// All files must have same sample rate - unmarshaller does NOT resample
```

### 4. Writing Multichannel Tensor to WAV

**Write stereo tensor to WAV:**
```go
// tensor shape: [frames, 2]
marshaller := audio.NewMarshaller(
    audio.WithWAVPath("stereo.wav"),
    audio.WithSampleRate(48000))
err := marshaller.Marshal(nil, tensor)
// Creates 2-channel WAV file
```

**Write 5.1 surround tensor:**
```go
// tensor shape: [frames, 6]
marshaller := audio.NewMarshaller(
    audio.WithWAVPath("surround.wav"),
    audio.WithSampleRate(48000))
err := marshaller.Marshal(nil, tensor)
// Creates 6-channel WAV file
```

### 5. Audio Device Capture (Input)

**Capture from single device:**
```go
// Get device ID from enumeration first
unmarshaller := audio.NewUnmarshaller()
var devices []types.AudioDeviceInfo
err := unmarshaller.Unmarshal(strings.NewReader("list"), &devices)
deviceID := devices[0].ID // Use first capture device

// Capture from device
unmarshaller = audio.NewUnmarshaller(
    audio.WithAudioDevice(deviceID),
    audio.WithSampleRate(48000),
    audio.WithChannels(2))
var stream types.FrameStream
err = unmarshaller.Unmarshal(nil, &stream)
defer stream.Close()

for frame := range stream.C {
    tensor := frame.Tensors[0]
    // Process audio frame
    // tensor shape: [frames, channels]
    tensor.Release()
}
```

**Capture from multiple devices (combined channels):**
```go
// Two sound cards as input - channels combined into single tensor
unmarshaller := audio.NewUnmarshaller(
    audio.WithAudioDevice("device-id-0"), // First device (e.g., 2 channels)
    audio.WithAudioDevice("device-id-1"), // Second device (e.g., 2 channels)
    audio.WithSampleRate(48000)) // All devices must use same sample rate
var stream types.FrameStream
err := unmarshaller.Unmarshal(nil, &stream)
defer stream.Close()

for frame := range stream.C {
    tensor := frame.Tensors[0]
    // tensor shape: [frames, 4] (2 channels from device 0 + 2 channels from device 1)
    // Channels synchronized across devices
    tensor.Release()
}
```

**Capture from multiple devices to tensor (non-streaming):**
```go
// Capture from 3 devices and combine into single tensor
unmarshaller := audio.NewUnmarshaller(
    audio.WithAudioDevice("device-0"), // 2 channels
    audio.WithAudioDevice("device-1"), // 2 channels
    audio.WithAudioDevice("device-2"), // 1 channel (mono)
    audio.WithSampleRate(48000),
    audio.WithChannels(2)) // Per-device channel count
var tensor types.Tensor
err := unmarshaller.Unmarshal(nil, &tensor)
// tensor shape: [frames, 5] (2+2+1 channels combined)
// All devices synchronized, shorter captures padded with zeros
```

**Capture with specific configuration:**
```go
unmarshaller := audio.NewUnmarshaller(
    audio.WithAudioDevice("device-id-0"),
    audio.WithSampleRate(48000),
    audio.WithChannels(2),
    audio.WithFormat("S16"))
var stream types.FrameStream
err := unmarshaller.Unmarshal(nil, &stream)
defer stream.Close()

for frame := range stream.C {
    tensor := frame.Tensors[0]
    // Process audio frame
    tensor.Release()
}
```

**Mixed sources (WAV files + devices):**
```go
unmarshaller := audio.NewUnmarshaller(
    audio.WithWAVPath("file1.wav"),      // WAV file (e.g., 2 channels)
    audio.WithAudioDevice("device-id-0"), // Audio device (e.g., 2 channels)
    audio.WithSampleRate(48000))         // All sources must have same sample rate
var stream types.FrameStream
err := unmarshaller.Unmarshal(nil, &stream)
defer stream.Close()

for frame := range stream.C {
    tensor := frame.Tensors[0]
    // tensor shape: [frames, 4] (2 channels from WAV + 2 channels from device)
    // All sources synchronized
    tensor.Release()
}
```

**Mixed sources to tensor (non-streaming):**
```go
// Combine WAV file and multiple devices into single tensor
unmarshaller := audio.NewUnmarshaller(
    audio.WithWAVPath("background.wav"), // 2 channels
    audio.WithAudioDevice("mic-0"),       // 1 channel
    audio.WithAudioDevice("mic-1"),       // 1 channel
    audio.WithSampleRate(48000))
var tensor types.Tensor
err := unmarshaller.Unmarshal(nil, &tensor)
// tensor shape: [frames, 4] (2+1+1 channels)
// WAV file read in chunks, devices captured in real-time, all synchronized
```

### 6. Audio Device Playback (Output)

**Play tensor to single device:**
```go
// Get device ID from enumeration first
unmarshaller := audio.NewUnmarshaller()
var devices []types.AudioDeviceInfo
err := unmarshaller.Unmarshal(strings.NewReader("list"), &devices)
deviceID := devices[0].ID // Use first playback device

// Play to device
marshaller := audio.NewMarshaller(
    audio.WithAudioDevice(deviceID),
    audio.WithSampleRate(48000))
err = marshaller.Marshal(nil, tensor)
// Blocks until playback completes
```

**Play multichannel tensor to multiple devices:**
```go
// 4-channel tensor split across two stereo devices
// tensor shape: [frames, 4]
marshaller := audio.NewMarshaller(
    audio.WithAudioDevice("device-id-0"), // First stereo device (channels 0-1)
    audio.WithAudioDevice("device-id-1"), // Second stereo device (channels 2-3)
    audio.WithSampleRate(48000))
err := marshaller.Marshal(nil, tensor)
// Channels 0-1 → device 0, Channels 2-3 → device 1
// Writes synchronized across devices
```

**Play 6-channel tensor to multiple devices:**
```go
// 6-channel tensor (5.1 surround) distributed across devices
// tensor shape: [frames, 6]
marshaller := audio.NewMarshaller(
    audio.WithAudioDevice("device-0"), // Stereo device (channels 0-1: L, R)
    audio.WithAudioDevice("device-1"), // Stereo device (channels 2-3: C, LFE)
    audio.WithAudioDevice("device-2"), // Stereo device (channels 4-5: SL, SR)
    audio.WithSampleRate(48000))
err := marshaller.Marshal(nil, tensor)
// Channels distributed: 0-1 → device 0, 2-3 → device 1, 4-5 → device 2
// All devices play synchronized
```

**Stream playback to multiple devices:**
```go
marshaller := audio.NewMarshaller(
    audio.WithAudioDevice("device-id-0"),
    audio.WithAudioDevice("device-id-1"),
    audio.WithSampleRate(48000))
err := marshaller.Marshal(nil, stream)
// Streams audio frames to multiple devices
// stream is types.FrameStream with audio tensors
// Channels distributed across devices, synchronized writes
```

**Mixed outputs (WAV file + device):**
```go
marshaller := audio.NewMarshaller(
    audio.WithWAVPath("output.wav"),     // Write to file
    audio.WithAudioDevice("device-id-0"), // Also play to device
    audio.WithSampleRate(48000))
err := marshaller.Marshal(nil, tensor)
// Writes to both WAV file and device simultaneously
// Same tensor data written to both outputs
```

**Mixed outputs (multiple WAV files + multiple devices):**
```go
// 4-channel tensor written to 2 WAV files and 2 devices
// tensor shape: [frames, 4]
marshaller := audio.NewMarshaller(
    audio.WithWAVPath("left.wav"),        // Channels 0-1
    audio.WithWAVPath("right.wav"),      // Channels 2-3
    audio.WithAudioDevice("device-0"),   // Channels 0-1
    audio.WithAudioDevice("device-1"),   // Channels 2-3
    audio.WithSampleRate(48000))
err := marshaller.Marshal(nil, tensor)
// Channels 0-1 → left.wav + device-0
// Channels 2-3 → right.wav + device-1
// All outputs synchronized
```

### 7. Listing Audio Devices

**Enumerate available devices:**
```go
unmarshaller := audio.NewUnmarshaller()
var devices []types.AudioDeviceInfo
err := unmarshaller.Unmarshal(strings.NewReader("list"), &devices)
for _, dev := range devices {
    fmt.Printf("Device %d: %s (channels: %d-%d, sample rate: %d-%d)\n",
        dev.ID, dev.Name, dev.MinChannels, dev.MaxChannels,
        dev.MinSampleRate, dev.MaxSampleRate)
}
```

## API Design

### Unmarshaller Interface

```go
type Unmarshaller struct {
    opts types.Options
    cfg  config
}

func NewUnmarshaller(opts ...types.Option) *Unmarshaller

func (u *Unmarshaller) Format() string // Returns "audio"

func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error

// SampleRate returns the sample rate of the last unmarshalled source
func (u *Unmarshaller) SampleRate() int

// Channels returns the number of channels of the last unmarshalled source
func (u *Unmarshaller) Channels() int

// DataType returns the data type of the last unmarshalled tensor
// Returns DT_UNKNOWN if no tensor has been unmarshalled yet
func (u *Unmarshaller) DataType() types.DataType

// Health returns statistics about the unmarshaller state
// For device streams: frames, queue lengths, dropped buffers, byte counts
// For WAV files: frames, bytes read
func (u *Unmarshaller) Health() map[string]int64
```

**Supported destination types:**
- `*types.Tensor`: Single tensor from WAV file(s) and/or device(s)
  - Multiple sources combined: channels concatenated, synchronized
  - All sources must have same sample rate (no resampling)
- `*types.FrameStream`: Streaming audio from device(s) (reuses FrameStream from types package)
  - Multiple devices: channels combined, synchronized in real-time
  - Mixed sources: WAV files + devices supported
- `*[]types.AudioDeviceInfo`: List of available devices (when r contains "list")
  
  Note: `types.AudioDeviceInfo` needs to be added to `x/marshaller/types` package, similar to `types.CameraInfo`

**Multi-source Behavior:**
- When multiple WAV files specified: channels concatenated into single tensor
- When multiple devices specified: channels combined, synchronized capture
- When mixed (WAV + devices): all channels combined, synchronized
- Sample rate must match across all sources (unmarshaller does NOT resample)

### Marshaller Interface

```go
type Marshaller struct {
    opts types.Options
    cfg  config
}

func NewMarshaller(opts ...types.Option) *Marshaller

func (m *Marshaller) Format() string // Returns "audio"

func (m *Marshaller) Marshal(w io.Writer, value any, opts ...types.Option) error

// SetSampleRate sets the target sample rate for marshalling
func (m *Marshaller) SetSampleRate(rate int)

// SampleRate returns the configured sample rate
func (m *Marshaller) SampleRate() int

// Health returns statistics about the marshaller state
// For device streams: frames, queue lengths, dropped buffers, byte counts
// For WAV files: frames, bytes written
func (m *Marshaller) Health() map[string]int64
```

**Supported source types:**
- `types.Tensor`: Audio tensor to write to WAV file(s) and/or device(s)
  - Multiple outputs: channels distributed across outputs
  - Example: 4-channel tensor → 2 stereo devices (ch 0-1 → dev 0, ch 2-3 → dev 1)
- `types.FrameStream`: Stream audio frames to device(s) (reuses FrameStream from types package)
  - Multiple devices: channels distributed, synchronized writes
  - Mixed outputs: WAV file + devices supported

**Format Selection:**
- Marshaller automatically selects best format based on tensor dtype and target capabilities
- If tensor is INT16 and WAV/device supports 16-bit → direct write (zero-copy)
- If tensor is FP32 and WAV/device supports float → direct write (zero-copy)
- Conversion only occurs when format mismatch requires it

**Multi-output Behavior:**
- When multiple WAV files specified: channels distributed across files
- When multiple devices specified: channels distributed, synchronized playback
- When mixed (WAV + devices): channels distributed across all outputs
- Sample rate from tensor or configuration (marshaller does NOT resample)

### Configuration Options

```go
// WAV file options
func WithWAVPath(path string) types.Option
func WithWAVPaths(paths ...string) types.Option // Multiple files for multichannel

// Audio format options
func WithSampleRate(rate int) types.Option
func WithChannels(channels int) types.Option
func WithFormat(format string) types.Option // "F32", "S32", "S24", "S16", "U8"

// Device options (can be called multiple times for multiple devices)
func WithAudioDevice(deviceID int) types.Option // Add device (input for unmarshaller, output for marshaller)
func WithAudioDeviceName(name string) types.Option
func WithCaptureDevice(deviceID int) types.Option // Explicit input device (unmarshaller only)
func WithPlaybackDevice(deviceID int) types.Option // Explicit output device (marshaller only)

// Multiple devices can be specified:
// unmarshaller := audio.NewUnmarshaller(
//     audio.WithAudioDevice(0),
//     audio.WithAudioDevice(1)) // Combines channels from both devices

// Stream options
func WithFrameSize(frames int) types.Option // Frames per read/write
func WithBufferSize(size int) types.Option // Buffer size in frames
```

## Implementation Details

### Performance Optimizations (from gocoustics)

The implementation should closely replicate the highly optimized patterns from the gocoustics codebase:

#### 1. Buffer Pooling (`gocoustics/internal/audio/buffer.pcm.go`)

**Reference**: `pcmBufferPool` struct and `buffer.MemoryPool` interface

- **Separate pools for different data types**:
  - PCM buffer pool (`pcmBufferPool`) for raw byte buffers
  - Real buffer pool (`MemoryPool[T]`) for typed samples (float32, etc.)
- **Pool initialization**: `initBuffer(channels, sampleRate, sampleSize)` calculates buffer sizes
- **Buffer allocation**: `MakeRaw()` with `WithFrames()` option for frame-based allocation
- **Buffer release**: `Release()` returns buffers to pool for reuse
- **Second-size calculation**: `secondSize()` returns bytes needed for one second of audio

**Implementation pattern**:
```go
type pcmBufferPool struct {
    buffer.MemoryPool[byte]
    sampleSize int
}
// Channels calculated as: MemoryPool.Channels() / sampleSize
```

#### 2. Asynchronous I/O with Ring Buffers (`gocoustics/internal/audio/reader.go`)

**Reference**: `chanReader` struct with `ringbuffer.RingBuffer`

- **Channel-based async I/O**: Uses buffered channels (`chan []byte`) for producer/consumer pattern
- **Ring buffer for smoothing**: `ringbuffer.RingBuffer` with 3x second-size capacity
- **Timeout-based reading**: Context with timeout prevents blocking indefinitely
- **Automatic buffer release**: Released buffers returned to pool after use

**Key optimizations**:
- Input channel: 32-buffer capacity for capture
- Output channel: 32-buffer capacity for playback  
- Ring buffer size: `pool.secondSize() * 3` (3 seconds of buffering)
- Timeout: 20ms for output, 1s for input

#### 3. Statistics and Health Monitoring (`gocoustics/internal/audio/dev.wrapper.malgo.go`)

**Reference**: `Health()` method with atomic counters

- **Atomic counters** for lock-free statistics:
  - `frames`: Total frames processed
  - `capturedBuffers` / `playedBuffers`: Buffer counts
  - `droppedBuffers`: Overflow/underflow detection
  - `inBytesIn` / `inBytesOut`: Input byte statistics
  - `outBytesIn` / `outBytesOut`: Output byte statistics
- **Queue monitoring**: Tracks channel lengths and capacities
- **Frame tracking**: `LastFrame()` and `LastPhase()` for position tracking

**Implementation pattern**:
```go
func (r *audioDevice[T]) Health() types.HealthMap {
    return map[string]int64{
        "frames": atomic.LoadInt64(&r.frame),
        "inQueueLen": int64(len(r.pcmIn)),
        "droppedBuffers": atomic.LoadInt64(&r.dropped),
        // ... more stats
    }
}
```

#### 4. Device Callback Optimization (`gocoustics/internal/audio/dev.wrapper.malgo.go:92-114`)

**Reference**: `malgo.DeviceCallbacks.Data` function

- **Non-blocking channel operations**: Uses `select` with `default` to prevent blocking
- **Immediate buffer copy**: Copies input data to pooled buffer before queuing
- **Dropped buffer tracking**: Counts buffers dropped due to queue full
- **Separate readers**: `outputReader` and `inputReader` with different timeouts

**Key pattern**:
```go
callbacks := malgo.DeviceCallbacks{
    Data: func(output, input []byte, framecount uint32) {
        // Get pooled buffer
        buf := types.Must(r.pool.MakeRaw(...))
        copy(buf, input)
        
        // Non-blocking send
        select {
        case r.pcmIn <- buf:
            atomic.AddInt64(&r.captured, 1)
        default:
            atomic.AddInt64(&r.dropped, 1)
        }
    },
}
```

#### 5. Format-Aware Conversion (`gocoustics/pkg/dsp/ops/convert.go`)

**Reference**: `PCM2Real[T]` and `Real2PCM[T]` functions

- **Generic type parameters**: Uses Go generics for type-safe conversion
- **Format-specific conversion**: Handles S16, S24, S32, F32, U8 formats
- **Channel-aware processing**: Processes interleaved PCM correctly
- **Minimal overhead**: Direct type conversions when possible

**Replication strategy**:
- Use tensor's native dtype when matching source format
- Only convert when format mismatch requires it
- Use optimized conversion primitives from tensor package

### WAV File Handling

**Reading:**
- Uses `github.com/go-audio/wav` for WAV decoding
- Automatically detects sample rate, channels, and bit depth
- **Preserves native format**: Creates tensor with dtype matching WAV format
  - 16-bit PCM → INT16 tensor (no conversion)
  - 24-bit PCM → INT32 tensor (24-bit stored in int32)
  - 32-bit PCM → INT32 tensor (no conversion)
  - 32-bit float → FP32 tensor (no conversion)
- Uses buffer pooling for efficient memory management
- Supports multiple file concatenation for multichannel tensors
- **No resampling**: All files must have same sample rate when combining
- **Synchronization**: Reads from multiple files in lockstep, combines channels

**Writing:**
- Uses `github.com/go-audio/wav` for WAV encoding
- **Writes tensor as-is** if WAV format supports tensor dtype:
  - INT16 tensor → 16-bit PCM (direct write)
  - INT32 tensor → 32-bit PCM or 24-bit PCM (direct write or conversion)
  - FP32 tensor → 32-bit float (direct write if supported)
- **Automatic format selection**: Chooses best WAV format based on tensor dtype
- Default: 16-bit PCM (if tensor conversion required)
- Supports multichannel output

### Audio Device Handling

**Device Enumeration:**
- Uses malgo context to enumerate devices
- Returns device info with capabilities (channels, sample rates, formats)
- Supports both capture (input) and playback (output) devices

**Device I/O:**
- Uses malgo device callbacks for real-time I/O
- **Preserves native format**: Creates tensors matching device format
  - Device S16 → INT16 tensor (no conversion)
  - Device S32 → INT32 tensor (no conversion)
  - Device F32 → FP32 tensor (no conversion)
- **Asynchronous I/O pattern** (replicate `gocoustics/internal/audio/dev.wrapper.malgo.go`):
  - Buffered channels (32 capacity) for async producer/consumer
  - Ring buffers for smoothing (3x second-size)
  - Non-blocking channel operations with timeout
  - Atomic statistics tracking
- **Buffer pooling**: Separate pools for PCM (byte) and real (typed) buffers
- **Format-aware conversion**: Only converts when tensor dtype doesn't match device format
- Supports configurable buffer sizes and frame sizes
- **Health monitoring**: Tracks frames, bytes, dropped buffers, queue lengths

**Multiple Device Support:**
- **Input (Unmarshaller)**: Multiple devices combined into single tensor
  - Each device has its own ring buffer and channel reader
  - Synchronized capture: read from all devices in lockstep
  - Channels concatenated: device 0 channels → tensor ch 0..N, device 1 → ch N+1..M
  - All devices must use same sample rate (no resampling)
  - Timing differences handled with per-device buffering
- **Output (Marshaller)**: Multichannel tensor distributed across devices
  - Channels split: tensor ch 0-1 → device 0, ch 2-3 → device 1, etc.
  - Synchronized writes: write to all devices in lockstep
  - Per-device buffer management and statistics

### Tensor Conversion

**PCM to Tensor (Unmarshalling):**
- Reads PCM samples from WAV or device
- **Preserves native format** - no automatic conversion:
  - 16-bit PCM → `INT16` tensor (direct copy, no conversion)
  - 24-bit PCM → `INT32` tensor (24-bit samples stored in int32)
  - 32-bit PCM → `INT32` tensor (direct copy)
  - 32-bit float → `FP32` tensor (direct copy)
  - 8-bit PCM → `INT8` tensor (direct copy)
- Reshapes to `[frames, channels]` format
- **Zero-copy when possible**: If tensor dtype matches source format, direct memory copy

**Tensor to PCM (Marshalling):**
- Takes tensor with shape `[frames, channels]`
- **Writes as-is** if target format supports tensor dtype:
  - `INT16` tensor → 16-bit PCM (direct copy if WAV/device supports 16-bit)
  - `INT32` tensor → 32-bit PCM or 24-bit PCM (direct copy or conversion)
  - `FP32` tensor → 32-bit float (direct copy if WAV/device supports float)
- **Automatic conversion only when necessary**:
  - If WAV only supports 16-bit but tensor is FP32 → convert FP32 to 16-bit PCM
  - If device only supports S16 but tensor is INT32 → convert INT32 to S16
  - Conversion uses optimized primitives to minimize overhead
- Clamps values during conversion to prevent overflow

### Multi-source Concatenation

**Multiple WAV Files:**
When multiple WAV files are specified:
1. Open all files and read metadata
2. **Verify same sample rate** - all files must have identical sample rate (no resampling)
3. Determine total frames (use minimum length, pad shorter files with zeros)
4. Concatenate channels: file 0 → channel 0, file 1 → channel 1, etc.
5. Create tensor with shape `[frames, num_files]`

**Multiple Audio Devices:**
When multiple devices are specified:
1. Open all devices with same sample rate configuration
2. **Synchronize capture** - read frames from all devices in lockstep
3. Combine channels: device 0 channels → tensor channels 0..N, device 1 → N+1..M, etc.
4. Create tensor with shape `[frames, total_channels]`
5. Handle timing differences with buffering (ring buffers per device)

**Mixed Sources (WAV + Devices):**
1. Open all sources (files and devices)
2. **Verify same sample rate** - all sources must match (no resampling)
3. Synchronize reads: WAV files read in chunks, devices read in real-time
4. Combine channels maintaining synchronization
5. Create tensor with shape `[frames, total_channels]`

**Synchronization Strategy:**
- Use ring buffers per source to handle timing variations
- Read from fastest source first, buffer others
- Drop frames if synchronization cannot be maintained (track in health stats)
- Best-effort synchronization with configurable tolerance

### Error Handling

- Invalid WAV files: return error with file path
- Device unavailable: return error with device ID
- Format mismatch: return error with details
- Buffer underrun/overrun: log warning, continue with best effort
- **Sample rate mismatch**: When combining multiple sources, return error if sample rates differ
  - Unmarshaller does NOT resample - all sources must have same rate
  - Error message lists all source sample rates for debugging
- **Synchronization failure**: When multiple devices cannot be synchronized
  - Track dropped frames in health statistics
  - Continue with best-effort synchronization if configured
  - Return error if synchronization completely fails

## Testing Requirements

### Unit Tests

**WAV File Tests:**
- Read single mono WAV file
- Read single stereo WAV file
- Read multichannel WAV file (5.1, 7.1)
- Write tensor to WAV and verify
- Read multiple mono files as stereo tensor (same sample rate)
- Read files with different sample rates (should error - no resampling)
- Read files with different lengths (pad shorter files with zeros)
- Write multichannel tensor to multiple WAV files (channel distribution)

**Tensor Format Tests:**
- Verify tensor shape `[frames, channels]`
- Verify tensor dtype matches source format:
  - 16-bit PCM WAV → INT16 tensor
  - 32-bit float WAV → FP32 tensor
  - 24-bit PCM WAV → INT32 tensor
- Verify no unnecessary conversion (zero-copy when formats match)
- Verify round-trip: WAV → tensor → WAV (format preservation)
- Verify conversion when format mismatch occurs

**Device Tests:**
- Enumerate devices (mock if no devices available)
- Open device with specific config
- Read/write frames from/to device
- Handle device errors gracefully
- **Multiple device input**: Read from 2 devices, verify channels combined
- **Multiple device output**: Write 4-channel tensor to 2 stereo devices
- **Synchronization**: Verify frames synchronized across multiple devices
- **Mixed sources**: WAV file + device input, verify channels combined
- **Mixed outputs**: WAV file + device output, verify channels distributed

**Integration Tests:**
- Full pipeline: device → tensor → WAV
- Full pipeline: WAV → tensor → device
- Multi-file concatenation with real files
- **Multi-device pipeline**: 2 devices → tensor → 2 devices (channel routing)
- **Mixed pipeline**: WAV + device → tensor → WAV + device
- **Synchronization**: Verify timing accuracy across multiple sources/sinks

### Test Data

User will provide testdata directory with:
- Mono WAV files (various sample rates)
- Stereo WAV files
- Multichannel WAV files
- Files with different lengths
- Files with different sample rates

## Implementation Status

✅ **Fully Implemented:**
- WAV file reading (single and multi-file concatenation)
- WAV file writing (single file)
- Audio device enumeration
- Single device capture (unmarshalling to tensor and FrameStream)
- Multi-device capture with channel combination
- Single device playback (marshalling from tensor)
- Multi-device playback with channel distribution
- Mixed sources (WAV + devices)
- Mixed outputs (WAV + devices)
- Buffer pooling for PCM and tensor data
- Statistics tracking with atomic counters
- FrameStream support for device capture and playback
- PCM ↔ Tensor conversion for all supported formats

**Note:** Multi-output for WAV files is intentionally not implemented per requirements (only multi-source needed).

## Dependencies

- `github.com/gen2brain/malgo` - Audio I/O (cross-platform, used directly)
- `github.com/go-audio/wav` - WAV file format
- `github.com/itohio/EasyRobot/x/math/tensor` - Tensor operations
- `github.com/itohio/EasyRobot/x/marshaller/types` - Marshaller interfaces
- `github.com/itohio/EasyRobot/x/math/primitive/generics/helpers` - Buffer pooling
- `github.com/smallnest/ringbuffer` - Ring buffer for async I/O smoothing

## Type Definitions Needed

The following types need to be added to `x/marshaller/types` package:

```go
// AudioDeviceInfo represents information about an audio device
type AudioDeviceInfo struct {
    ID            int      // Device identifier
    Name          string   // Device name
    IsDefault     bool     // Whether this is the default device
    MinChannels   int      // Minimum supported channels
    MaxChannels   int      // Maximum supported channels
    MinSampleRate int      // Minimum supported sample rate
    MaxSampleRate int      // Maximum supported sample rate
    Formats       []string // Supported formats: "F32", "S32", "S24", "S16", "U8"
    DeviceType    string   // "capture" or "playback"
}
```

This follows the same pattern as `CameraInfo` in the types package.

## Performance Considerations

### Memory Management
- **Buffer pooling**: Reuse buffers to minimize allocations (reference: `gocoustics/pkg/audio/buffer`)
- **Zero-copy when possible**: Direct memory operations when formats match
- **Separate pools**: PCM buffers (byte) vs typed buffers (int16, float32, etc.)

### I/O Optimization
- **Asynchronous channels**: Non-blocking I/O prevents audio glitches
- **Ring buffers**: Smooth out timing variations (3 seconds capacity)
- **Timeout handling**: Prevents indefinite blocking (20ms output, 1s input)

### Statistics and Monitoring
- **Atomic counters**: Lock-free statistics collection
- **Health metrics**: Track dropped buffers, queue lengths, throughput
- **Frame tracking**: Position and phase information for synchronization

### Format Conversion
- **Minimize conversions**: Preserve native formats, convert only when necessary
- **Optimized primitives**: Use tensor package conversion functions
- **Type-aware**: Leverage Go generics for efficient type-specific operations

## Future Enhancements

- Support for other audio formats (FLAC, OGG, MP3)
- Audio resampling library integration
- Audio effects pipeline (filtering, mixing)
- Real-time audio processing callbacks
- Audio format conversion utilities
- Stream synchronization for multi-device capture
- Health monitoring API for runtime statistics

