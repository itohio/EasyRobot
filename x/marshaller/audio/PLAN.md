# Audio Marshaller Implementation Plan

## Overview

This plan outlines the implementation of the audio marshaller package (`x/marshaller/audio`) and the command-line tool (`cmd/audio`) with subcommands for listing devices, recording, and playing audio.

**Success Criteria:**
- `cmd/audio list` - enumerates audio devices (input/output)
- `cmd/audio record` - records from audio device to WAV file
- `cmd/audio play` - plays WAV file to audio device
- All commands work similar to `gocoustics/cmd/enum`, `gocoustics/cmd/record`, `gocoustics/cmd/play`

## Implementation Phases

### Phase 1: Foundation and Types
**Status: TODO**

#### 1.1 Add AudioDeviceInfo Type
- [X] **DONE** Add `AudioDeviceInfo` to `x/marshaller/types` package
  - Similar to `CameraInfo` structure
  - Fields: ID, Name, IsDefault, MinChannels, MaxChannels, MinSampleRate, MaxSampleRate, Formats, DeviceType
  - Location: `x/marshaller/types/audio.go` (new file)

#### 1.2 Review Tensor Package Support
- [X] **DONE** Verify tensor package supports INT16, INT32, INT8, FP32 data types
  - All types supported via `x/math/tensor/types/dtype.go`
  - INT16, INT32, INT8, FP32 all available
- [X] **DONE** Verify tensor manipulation operations (reshape, slice, concatenate)
  - Reshape: ✅ Available (`Reshape(dst Tensor, newShape Shape) Tensor`)
  - Slice: ✅ Available (`Slice(dst Tensor, dim int, start int, length int) Tensor`)
  - Concat: ⚠️ No explicit function, but can combine manually using Reshape/Slice
- [X] **DONE** Check tensor pooling/memory management capabilities
  - Generic `Pool[T]` type available: `x/math/primitive/generics/helpers/pool.go`
  - Type-specific pools: `fp32.Pool`, `qi16.Pool`, `qi32.Pool`, `qi8.Pool`
  - Functions: `MakeTensorData()`, `ReleaseTensorData()`, `CloneTensorDataTo()`
  - See `TENSOR_VERIFICATION.md` for details

#### 1.3 Create Package Structure
- [X] **DONE** Create `x/marshaller/audio` package directory
- [X] **DONE** Create initial files:
  - `marshaller.go` - Marshaller implementation (skeleton with Format, Marshal, SampleRate, Health methods)
  - `unmarshaller.go` - Unmarshaller implementation (skeleton with Format, Unmarshal, SampleRate, Channels, DataType, Health methods)
  - `options.go` - Configuration options (WithWAVPath, WithSampleRate, WithChannels, WithFormat, WithAudioDevice, etc.)
  - `config.go` - Internal configuration structs (wavConfig, deviceConfig, streamConfig, tensorConfig, config)
  - `types.go` - Internal types (sourceKind, sourceSpec, deviceSpec)

---

### Phase 2: WAV File Support (Unmarshaller)
**Status: IN PROGRESS**

#### 2.1 Single WAV File Reading
- [X] **DONE** Implement WAV file reader using `github.com/go-audio/wav`
- [X] **DONE** Detect WAV format (sample rate, channels, bit depth)
- [X] **DONE** Create tensor with matching dtype:
  - 16-bit PCM → INT16 tensor
  - 24-bit PCM → INT32 tensor
  - 32-bit PCM → INT32 tensor
  - 32-bit float → FP32 tensor
- [X] **DONE** Shape tensor as `[frames, channels]`
- [X] **DONE** Store format info (sample rate, channels, dtype) in unmarshaller
- [X] **DONE** Implement `SampleRate()`, `Channels()`, `DataType()` methods

#### 2.2 Multiple WAV File Concatenation
- [ ] **TODO** Support multiple WAV paths in options
- [ ] **TODO** Open all files and verify same sample rate (error if different)
- [ ] **TODO** Read frames from all files in lockstep
- [ ] **TODO** Combine channels: file 0 → ch 0, file 1 → ch 1, etc.
- [ ] **TODO** Handle different file lengths (pad shorter files with zeros)
- [ ] **TODO** Create single tensor with shape `[frames, total_channels]`

#### 2.3 Buffer Pooling for WAV
- [ ] **TODO** Implement buffer pool for WAV reading (reference: `gocoustics/internal/audio/wav.wrapper.go`) Verify that tensors support pooling from the getgo. There is generics Pool type that helps with this. Verify that it has sufficient functionalitites.
- [ ] **TODO** Use separate pools for PCM (byte) and typed buffers
- [ ] **TODO** Implement buffer release after use

---

### Phase 3: WAV File Support (Marshaller)
**Status: TODO**

#### 3.1 Single WAV File Writing
- [ ] **TODO** Implement WAV file writer using `github.com/go-audio/wav`
- [ ] **TODO** Write tensor as-is if WAV format supports tensor dtype
- [ ] **TODO** Convert tensor dtype only when format mismatch requires it
- [ ] **TODO** Support configurable sample rate and channels
- [ ] **TODO** Default to 16-bit PCM if conversion needed

#### 3.2 Multiple WAV File Writing
- [ ] **TODO** Support multiple WAV paths in options
- [ ] **TODO** Distribute tensor channels across files
- [ ] **TODO** Example: 4-channel tensor → 2 stereo WAV files

#### 3.3 Buffer Pooling for WAV Writing
- [ ] **TODO** Implement buffer pool for WAV writing
- [ ] **TODO** Reuse buffers for efficient memory management

---

### Phase 4: Audio Device Support (Unmarshaller)
**Status: TODO**

#### 4.1 Device Enumeration
- [ ] **TODO** Implement device enumeration using malgo (via gocoustics)
- [ ] **TODO** List capture devices (input)
- [ ] **TODO** List playback devices (output)
- [ ] **TODO** Return `[]AudioDeviceInfo` with device capabilities
- [ ] **TODO** Reference: `gocoustics/internal/audio/dev.system.malgo.go:enumerate()`

#### 4.2 Single Device Capture
- [ ] **TODO** Open audio device using malgo
- [ ] **TODO** Configure device (sample rate, channels, format)
- [ ] **TODO** Implement async I/O pattern (reference: `gocoustics/internal/audio/dev.wrapper.malgo.go`)
  - Buffered channels (32 capacity)
  - Ring buffers for smoothing (3x second-size)
  - Non-blocking channel operations
- [ ] **TODO** Create tensor with dtype matching device format
- [ ] **TODO** Return `FrameStream` for streaming capture
- [ ] **TODO** Implement health statistics tracking

#### 4.3 Multiple Device Capture
- [ ] **TODO** Support multiple devices in options
- [ ] **TODO** Open all devices with same sample rate
- [ ] **TODO** Implement per-device ring buffers
- [ ] **TODO** Synchronize capture across devices (lockstep reading)
- [ ] **TODO** Combine channels: device 0 → ch 0..N, device 1 → ch N+1..M
- [ ] **TODO** Handle timing differences with buffering
- [ ] **TODO** Track dropped frames per device

#### 4.4 Mixed Sources (WAV + Devices)
- [ ] **TODO** Support WAV files + devices in same unmarshaller
- [ ] **TODO** Verify all sources have same sample rate
- [ ] **TODO** Synchronize reads: WAV files (chunks) + devices (real-time)
- [ ] **TODO** Combine all channels into single tensor

---

### Phase 5: Audio Device Support (Marshaller)
**Status: TODO**

#### 5.1 Single Device Playback
- [ ] **TODO** Open audio device for playback using malgo
- [ ] **TODO** Configure device (sample rate, channels, format)
- [ ] **TODO** Implement async I/O pattern for playback
- [ ] **TODO** Write tensor as-is if device format matches
- [ ] **TODO** Convert tensor dtype only when necessary
- [ ] **TODO** Implement health statistics tracking

#### 5.2 Multiple Device Playback
- [ ] **TODO** Support multiple devices in options
- [ ] **TODO** Distribute tensor channels across devices
- [ ] **TODO** Example: 4-channel tensor → 2 stereo devices
- [ ] **TODO** Synchronize writes across devices (lockstep)
- [ ] **TODO** Per-device buffer management

#### 5.3 Mixed Outputs (WAV + Devices)
- [ ] **TODO** Support WAV file + devices in same marshaller
- [ ] **TODO** Write to both WAV and devices simultaneously
- [ ] **TODO** Distribute channels appropriately

---

### Phase 6: Buffer Pooling and Optimization
**Status: TODO**

#### 6.1 PCM Buffer Pool
- [ ] **TODO** Implement `pcmBufferPool` (reference: `gocoustics/internal/audio/buffer.pcm.go`)
- [ ] **TODO** Separate pool for raw byte buffers
- [ ] **TODO** Initialize with channels, sample rate, sample size
- [ ] **TODO** Implement `newSamples()` and `secondSize()` methods

#### 6.2 Typed Buffer Pool
- [ ] **TODO** Implement typed buffer pool for int16, int32, float32
- [ ] **TODO** Use tensor package pooling if available
- [ ] **TODO** Reference: `gocoustics/pkg/audio/buffer/pool.go`

#### 6.3 Ring Buffer for Async I/O
- [ ] **TODO** Implement `chanReader` with ring buffer (reference: `gocoustics/internal/audio/reader.go`)
- [ ] **TODO** Use `github.com/smallnest/ringbuffer`
- [ ] **TODO** Size: 3x second-size for smoothing
- [ ] **TODO** Timeout-based reading (20ms output, 1s input)

#### 6.4 Statistics and Health Monitoring
- [ ] **TODO** Implement atomic counters for statistics
- [ ] **TODO** Track: frames, bytes, dropped buffers, queue lengths
- [ ] **TODO** Implement `Health()` method returning `map[string]int64`
- [ ] **TODO** Reference: `gocoustics/internal/audio/dev.wrapper.malgo.go:Health()`

---

### Phase 7: Format Conversion
**Status: TODO**

#### 7.1 PCM to Tensor Conversion
- [ ] **TODO** Implement format-aware conversion
- [ ] **TODO** Preserve native format (no conversion when possible)
- [ ] **TODO** Reference: `gocoustics/pkg/dsp/ops/convert.go:PCM2Real`
- [ ] **TODO** Support: S16, S24, S32, F32, U8 formats

#### 7.2 Tensor to PCM Conversion
- [ ] **TODO** Implement format-aware conversion
- [ ] **TODO** Write as-is when format matches
- [ ] **TODO** Convert only when necessary
- [ ] **TODO** Reference: `gocoustics/pkg/dsp/ops/convert.go:Real2PCM`

---

### Phase 8: Unit Tests
**Status: TODO**

#### 8.1 WAV File Tests
- [ ] **TODO** Test reading single mono WAV file
- [ ] **TODO** Test reading single stereo WAV file
- [ ] **TODO** Test reading multichannel WAV (5.1, 7.1)
- [ ] **TODO** Test writing tensor to WAV
- [ ] **TODO** Test multiple mono files → stereo tensor
- [ ] **TODO** Test error on different sample rates (no resampling)
- [ ] **TODO** Test different file lengths (padding)
- [ ] **TODO** Test format preservation (INT16 → 16-bit PCM)
- [ ] **TODO** Test round-trip: WAV → tensor → WAV

#### 8.2 Device Tests
- [ ] **TODO** Test device enumeration (mock if no devices)
- [ ] **TODO** Test single device capture
- [ ] **TODO** Test single device playback
- [ ] **TODO** Test multiple device input (channel combination)
- [ ] **TODO** Test multiple device output (channel distribution)
- [ ] **TODO** Test synchronization across devices
- [ ] **TODO** Test mixed sources (WAV + device)
- [ ] **TODO** Test mixed outputs (WAV + device)
- [ ] **TODO** Test error handling (device unavailable, etc.)

#### 8.3 Integration Tests
- [ ] **TODO** Test full pipeline: device → tensor → WAV
- [ ] **TODO** Test full pipeline: WAV → tensor → device
- [ ] **TODO** Test multi-device pipeline: 2 devices → tensor → 2 devices
- [ ] **TODO** Test mixed pipeline: WAV + device → tensor → WAV + device
- [ ] **TODO** Test synchronization accuracy

#### 8.4 Test Data
- [ ] **TODO** Create testdata directory structure
- [ ] **TODO** Add mono WAV files (various sample rates)
- [ ] **TODO** Add stereo WAV files
- [ ] **TODO** Add multichannel WAV files
- [ ] **TODO** Add files with different lengths
- [ ] **TODO** Add files with different sample rates (for error testing)

---

### Phase 9: Command-Line Tool
**Status: TODO**

#### 9.1 Create cmd/audio Structure
- [ ] **TODO** Create `cmd/audio` directory
- [ ] **TODO** Create `main.go` with cobra CLI structure
- [ ] **TODO** Implement subcommands: `list`, `record`, `play`, `gen-sine`, `gen-noise`, `gen-chirp`, `gen-iq`, `gen-psk`, `gen-fsk`, `gen-qam`

#### 9.2 List Subcommand
- [ ] **TODO** Implement `audio list` command
- [ ] **TODO** Enumerate input devices (capture)
- [ ] **TODO** Enumerate output devices (playback)
- [ ] **TODO** Display device info: ID, name, default flag, channels, sample rates, formats
- [ ] **TODO** Match output format similar to `gocoustics/cmd/enum`
- [ ] **TODO** Example output:
  ```
  Readers:
    0 [default] Device Name [1-2, 44100-48000] id=abc123
    1 [] Another Device [2, 48000] id=def456
  Writers:
    0 [default] Output Device [2, 48000] id=xyz789
  ```

#### 9.3 Record Subcommand
- [ ] **TODO** Implement `audio record` command
- [ ] **TODO** Flags:
  - `-o, --output`: Output WAV file path (required)
  - `-id, --device-id`: Capture device ID (optional, uses default if not specified)
  - `-ch, --channels`: Number of channels (default: 1)
  - `-sr, --sample-rate`: Sample rate (default: 48000)
  - `-fmt, --format`: Format (S16, S32, F32, etc.) (optional)
- [ ] **TODO** List available capture devices on startup
- [ ] **TODO** Open capture device
- [ ] **TODO** Create WAV file writer
- [ ] **TODO** Stream capture → WAV file
- [ ] **TODO** Handle Ctrl+C to stop and save
- [ ] **TODO** Display format info (channels, sample rate)
- [ ] **TODO** Match behavior similar to `gocoustics/cmd/record`

#### 9.4 Play Subcommand
- [ ] **TODO** Implement `audio play` command
- [ ] **TODO** Flags:
  - `-i, --input`: Input WAV file path (required)
  - `-id, --device-id`: Playback device ID (optional, uses default if not specified)
- [ ] **TODO** List available playback devices on startup
- [ ] **TODO** Open WAV file reader
- [ ] **TODO** Open playback device (match WAV format)
- [ ] **TODO** Stream WAV file → device
- [ ] **TODO** Handle Ctrl+C to stop
- [ ] **TODO** Display format info (channels, sample rate)
- [ ] **TODO** Display health statistics on exit
- [ ] **TODO** Match behavior similar to `gocoustics/cmd/play`

#### 9.5 Multi-Device Support in Commands
- [ ] **TODO** Add `--device-id` flag support for multiple devices (comma-separated)
- [ ] **TODO** Record: combine channels from multiple devices
- [ ] **TODO** Play: distribute channels across multiple devices

#### 9.6 Gen-Sine Subcommand
- [ ] **TODO** Implement `audio gen-sine` command
- [ ] **TODO** Flags:
  - `-o, --output`: Output WAV file path (required)
  - `-a, --amplitude`: Amplitude in dB (default: -6dB)
  - `-f, --frequency`: Frequency in Hz per channel (comma-separated, default: 1000)
  - `-p, --phase`: Phase in degrees per channel (comma-separated, default: 0)
  - `-ch, --channels`: Number of channels (default: 1)
  - `-sr, --sample-rate`: Sample rate (default: 48000)
  - `-d, --duration`: Duration in seconds (default: 5)
  - `-dt, --data-type`: Data type (INT16, INT32, FP32, UINT8, default: INT16)
- [ ] **TODO** Generate sine wave(s) using `x/math/dsp` SignalGenerator1D
- [ ] **TODO** Convert dB to linear amplitude
- [ ] **TODO** Support per-channel frequency and phase
- [ ] **TODO** Write to WAV file using audio marshaller

#### 9.7 Gen-Noise Subcommand
- [ ] **TODO** Implement `audio gen-noise` command
- [ ] **TODO** Flags:
  - `-o, --output`: Output WAV file path (required)
  - `-a, --amplitude`: Amplitude in dB per channel (comma-separated, default: -6dB)
  - `-t, --type`: Noise type (white, pink, brown, default: white)
  - `-ch, --channels`: Number of channels (default: 1)
  - `-sr, --sample-rate`: Sample rate (default: 48000)
  - `-d, --duration`: Duration in seconds (default: 5)
  - `-dt, --data-type`: Data type (INT16, INT32, FP32, UINT8, default: INT16)
- [ ] **TODO** Generate noise using `x/math/dsp` SignalGenerator1D (NoiseWhite, NoisePink, NoiseBrown)
- [ ] **TODO** Support per-channel amplitude
- [ ] **TODO** Write to WAV file

#### 9.8 Gen-Chirp Subcommand
- [ ] **TODO** Implement `audio gen-chirp` command
- [ ] **TODO** Flags:
  - `-o, --output`: Output WAV file path (required)
  - `-a, --amplitude`: Amplitude in dB (default: -6dB)
  - `-fmin, --freq-min`: Minimum frequency in Hz (default: 100)
  - `-fmax, --freq-max`: Maximum frequency in Hz (default: 10000)
  - `-ch, --channels`: Number of channels (default: 1)
  - `-sr, --sample-rate`: Sample rate (default: 48000)
  - `-d, --duration`: Duration in seconds (default: 5)
  - `-dt, --data-type`: Data type (INT16, INT32, FP32, UINT8, default: INT16)
- [ ] **TODO** Generate chirp using `x/math/dsp` SignalGenerator1D.Chirp
- [ ] **TODO** Write to WAV file

#### 9.9 Gen-IQ Subcommand
- [ ] **TODO** Implement `audio gen-iq` command
- [ ] **TODO** Flags:
  - `-o, --output`: Output WAV file path (required)
  - `-a, --amplitude`: Amplitude in dB (default: -6dB)
  - `-f, --frequency`: Carrier frequency in Hz (default: 1000)
  - `-ch, --channels`: Number of channels (default: 1)
  - `-sr, --sample-rate`: Sample rate (default: 48000)
  - `-d, --duration`: Duration in seconds (default: 5)
  - `-dt, --data-type`: Data type (INT16, INT32, FP32, UINT8, default: INT16)
- [ ] **TODO** Generate I and Q signals (sine/cosine)
- [ ] **TODO** Use `x/math/dsp` IQEncoder.GenerateIQSignal
- [ ] **TODO** Write to WAV file

#### 9.10 Gen-PSK Subcommand
- [ ] **TODO** Implement `audio gen-psk` command
- [ ] **TODO** Flags:
  - `-o, --output`: Output WAV file path (required)
  - `-a, --amplitude`: Amplitude in dB (default: -6dB)
  - `-f, --frequency`: Carrier frequency in Hz (default: 1000)
  - `-data, --data`: Data as string (default: "Hello")
  - `-hex, --hex`: Data as hex string (alternative to -data)
  - `-sr, --sample-rate`: Sample rate (default: 48000)
  - `-sym, --symbol-rate`: Symbol rate in baud (default: 1000)
  - `-order, --modulation-order`: Modulation order (2=BPSK, 4=QPSK, default: 2)
  - `-dt, --data-type`: Data type (INT16, INT32, FP32, UINT8, default: INT16)
- [ ] **TODO** Parse data (string or hex)
- [ ] **TODO** Use `x/math/dsp` PSKEncoder.GenerateSignal
- [ ] **TODO** Write to WAV file

#### 9.11 Gen-FSK Subcommand
- [ ] **TODO** Implement `audio gen-fsk` command
- [ ] **TODO** Flags:
  - `-o, --output`: Output WAV file path (required)
  - `-a, --amplitude`: Amplitude in dB (default: -6dB)
  - `-f1, --freq-1`: Frequency for bit 0 in Hz (default: 1000)
  - `-f2, --freq-2`: Frequency for bit 1 in Hz (default: 2000)
  - `-data, --data`: Data as string (default: "Hello")
  - `-hex, --hex`: Data as hex string (alternative to -data)
  - `-sr, --sample-rate`: Sample rate (default: 48000)
  - `-sym, --symbol-rate`: Symbol rate in baud (default: 1000)
  - `-dt, --data-type`: Data type (INT16, INT32, FP32, UINT8, default: INT16)
- [ ] **TODO** Parse data (string or hex)
- [ ] **TODO** Use `x/math/dsp` FSKEncoder.GenerateSignal
- [ ] **TODO** Write to WAV file

#### 9.12 Gen-QAM Subcommand
- [ ] **TODO** Implement `audio gen-qam` command
- [ ] **TODO** Flags:
  - `-o, --output`: Output WAV file path (required)
  - `-a, --amplitude`: Amplitude in dB (default: -6dB)
  - `-f, --frequency`: Carrier frequency in Hz (default: 1000)
  - `-n, --poles`: Number of constellation points (4, 16, 64, 256, default: 16)
  - `-data, --data`: Data as string (default: "Hello")
  - `-hex, --hex`: Data as hex string (alternative to -data)
  - `-sr, --sample-rate`: Sample rate (default: 48000)
  - `-sym, --symbol-rate`: Symbol rate in baud (default: 1000)
  - `-dt, --data-type`: Data type (INT16, INT32, FP32, UINT8, default: INT16)
- [ ] **TODO** Parse data (string or hex)
- [ ] **TODO** Use `x/math/dsp` QAM16Encoder.GenerateSignal (or create generic QAM encoder)
- [ ] **TODO** Write to WAV file

---

### Phase 10: Documentation and Examples
**Status: TODO**

#### 10.1 Package Documentation
- [ ] **TODO** Add package-level documentation
- [ ] **TODO** Document all exported types and functions
- [ ] **TODO** Add usage examples in doc comments

#### 10.2 Command Documentation
- [ ] **TODO** Add README.md for `cmd/audio`
- [ ] **TODO** Document all subcommands and flags
- [ ] **TODO** Add usage examples

#### 10.3 Example Code
- [ ] **TODO** Create example_test.go with basic usage examples
- [ ] **TODO** Examples: single WAV read/write, device capture/playback, multi-device

---

## Implementation Order

### Recommended Sequence

1. **Phase 1** - Foundation (types, structure)
2. **Phase 2** - WAV reading (single file first, then multiple)
3. **Phase 3** - WAV writing (single file first, then multiple)
4. **Phase 4** - Device capture (single device first, then multiple)
5. **Phase 5** - Device playback (single device first, then multiple)
6. **Phase 6** - Optimization (buffer pooling, statistics)
7. **Phase 7** - Format conversion (integrate with existing code)
8. **Phase 8** - Unit tests (write tests as features are implemented)
9. **Phase 9** - Command-line tool (after core functionality works)
10. **Phase 10** - Documentation (ongoing, finalize at end)

---

## Key Dependencies

- `github.com/itohio/gocoustics/pkg/audio` - Audio I/O via malgo (malgo backend only)
- `github.com/go-audio/wav` - WAV file format support
- `github.com/itohio/EasyRobot/x/math/tensor` - Tensor operations
- `github.com/itohio/EasyRobot/x/marshaller/types` - Marshaller interfaces
- `github.com/smallnest/ringbuffer` - Ring buffer for async I/O
- `github.com/spf13/cobra` - CLI framework (for cmd/audio)

---

## Code References for Optimization

### Buffer Pooling
- `gocoustics/internal/audio/buffer.pcm.go` - PCM buffer pool implementation
- `gocoustics/pkg/audio/buffer/pool.go` - Typed buffer pool

### Async I/O Pattern
- `gocoustics/internal/audio/dev.wrapper.malgo.go` - Device wrapper with async I/O
- `gocoustics/internal/audio/reader.go` - Channel reader with ring buffer

### Statistics and Health
- `gocoustics/internal/audio/dev.wrapper.malgo.go:Health()` - Health statistics
- `gocoustics/internal/audio/wav.wrapper.go:Health()` - WAV statistics

### Format Conversion
- `gocoustics/pkg/dsp/ops/convert.go` - PCM ↔ Real conversion functions

### Device Enumeration
- `gocoustics/internal/audio/dev.system.malgo.go:enumerate()` - Device enumeration

---

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock audio devices when hardware unavailable
- Use testdata WAV files for file I/O tests

### Integration Tests
- Test full pipelines (device → tensor → WAV, etc.)
- Test multi-device scenarios (if hardware available)
- Test error conditions (format mismatches, device unavailable)

### Manual Testing
- Test `cmd/audio` commands on real hardware
- Verify behavior matches `gocoustics/cmd` commands
- Test with various audio devices and WAV files

---

## Success Criteria Checklist

- [ ] `cmd/audio list` enumerates devices correctly
- [ ] `cmd/audio record` records from device to WAV file
- [ ] `cmd/audio play` plays WAV file to device
- [ ] All commands handle Ctrl+C gracefully
- [ ] Format information displayed correctly
- [ ] Health statistics work (for play command)
- [ ] Multi-device input/output works
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Documentation complete

---

## Notes

- **No Resampling**: Unmarshaller and marshaller do NOT resample. All sources must have same sample rate.
- **Format Preservation**: Tensors preserve native format (INT16, INT32, FP32, etc.) to minimize conversion overhead.
- **Synchronization**: Multi-device operations require careful synchronization using ring buffers and lockstep I/O.
- **Error Handling**: Clear error messages for format mismatches, device unavailable, etc.
- **Performance**: Replicate optimizations from gocoustics (buffer pooling, async I/O, statistics).

---

## Status Legend

- **TODO** - Not started
- **IN PROGRESS** - Currently being worked on
- **DONE** - Completed and tested

