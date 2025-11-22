package audio

import (
	"fmt"
	"io"
	"os"
	"sync/atomic"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// stats tracks statistics for the marshaller using atomic counters.
type marshallerStats struct {
	frames int64 // Total frames written
	bytes  int64 // Total bytes written
}

// Marshaller encodes audio tensors to WAV files or audio devices.
type Marshaller struct {
	opts types.Options
	cfg  config

	// Statistics tracking
	stats marshallerStats

	// Buffer pools
	intPool *helpers.Pool[int] // Pool for PCM int buffers
}

// NewMarshaller constructs an audio marshaller instance.
func NewMarshaller(opts ...types.Option) *Marshaller {
	baseOpts, baseCfg := applyOptions(types.Options{}, defaultConfig(), opts)
	return &Marshaller{
		opts:    baseOpts,
		cfg:     baseCfg,
		intPool: &helpers.Pool[int]{},
	}
}

// Format returns the format identifier.
func (m *Marshaller) Format() string {
	return "audio"
}

// Marshal serializes audio tensors to WAV files or writes to audio devices.
func (m *Marshaller) Marshal(w io.Writer, value any, opts ...types.Option) error {
	if value == nil {
		return types.NewError("marshal", "audio", "nil value", nil)
	}

	localOpts, localCfg := applyOptions(m.opts, m.cfg, opts)
	_ = localOpts // reserved for future use

	switch v := value.(type) {
	case types.Tensor:
		return m.writeTensor(w, v, localCfg)
	case types.FrameStream:
		return m.writeFrameStream(w, v, localCfg)
	case *types.FrameStream:
		if v == nil {
			return types.NewError("marshal", "audio", "nil *FrameStream", nil)
		}
		return m.writeFrameStream(w, *v, localCfg)
	default:
		return types.NewError("marshal", "audio", fmt.Sprintf("unsupported value %T", value), nil)
	}
}

// SetSampleRate sets the target sample rate for marshalling.
func (m *Marshaller) SetSampleRate(rate int) {
	if rate > 0 {
		m.cfg.wav.sampleRate = rate
		m.cfg.device.sampleRate = rate
	}
}

// SampleRate returns the configured sample rate.
func (m *Marshaller) SampleRate() int {
	if m.cfg.wav.sampleRate > 0 {
		return m.cfg.wav.sampleRate
	}
	if m.cfg.device.sampleRate > 0 {
		return m.cfg.device.sampleRate
	}
	return 48000 // default
}

// Health returns statistics about the marshaller state.
// For device streams: frames, queue lengths, dropped buffers, byte counts.
// For WAV files: frames, bytes written.
func (m *Marshaller) Health() map[string]int64 {
	return map[string]int64{
		"frames": atomic.LoadInt64(&m.stats.frames),
		"bytes":  atomic.LoadInt64(&m.stats.bytes),
	}
}

// writeTensor writes a tensor to WAV file(s) and/or audio device(s).
func (m *Marshaller) writeTensor(w io.Writer, t types.Tensor, cfg config) error {
	if t.Empty() {
		return types.NewError("marshal", "audio", "empty tensor", nil)
	}

	shape := t.Shape()
	if len(shape) != 2 {
		return types.NewError("marshal", "audio", "tensor must be 2D [frames, channels] for audio", nil)
	}

	numFrames := shape[0]
	numChannels := shape[1]

	// Get sample rate and channels from config or tensor metadata
	sampleRate := cfg.wav.sampleRate
	if sampleRate == 0 {
		sampleRate = 48000 // default
	}

	// Check if we have devices configured
	if len(cfg.device.deviceIDs) > 0 {
		if len(cfg.wav.paths) > 0 {
			// Mixed outputs: WAV + devices
			return m.writeMixedOutputs(t, cfg)
		}
		// Device-only playback
		if len(cfg.device.deviceIDs) == 1 {
			return m.playbackSingleDevice(cfg.device.deviceIDs[0], t, cfg)
		}
		return m.playbackMultipleDevices(cfg.device.deviceIDs, t, cfg)
	}

	// Check if we have WAV paths configured
	if len(cfg.wav.paths) == 0 {
		return types.NewError("marshal", "audio", "no WAV paths or devices specified", nil)
	}

	// Single WAV file writing (Phase 3.1)
	// Note: Multi-output for WAV files not needed per requirements
	if len(cfg.wav.paths) == 1 {
		return m.writeSingleWAV(cfg.wav.paths[0], t, numFrames, numChannels, sampleRate, cfg)
	}

	// Multiple WAV files - not supported per requirements (only multi-source needed)
	return types.NewError("marshal", "audio", "multiple WAV file output not supported (only multi-source for WAV files)", nil)
}

// writeSingleWAV writes a tensor to a single WAV file.
func (m *Marshaller) writeSingleWAV(path string, t types.Tensor, numFrames, numChannels, sampleRate int, cfg config) error {
	// Open/create WAV file
	file, err := os.Create(path)
	if err != nil {
		return types.NewError("marshal", "audio", fmt.Sprintf("failed to create WAV file: %s", path), err)
	}
	defer file.Close()

	// Determine bit depth from tensor dtype
	bitDepth := 16 // default
	dtype := t.DataType()
	switch dtype {
	case tensortypes.UINT8, tensortypes.INT8:
		bitDepth = 8
	case tensortypes.INT16:
		bitDepth = 16
	case tensortypes.INT32:
		bitDepth = 32
	case tensortypes.FP32:
		bitDepth = 32 // Use 32-bit PCM for float
	default:
		bitDepth = 16 // Default to 16-bit
	}

	// Override with config if specified
	if cfg.wav.bitDepth > 0 {
		bitDepth = cfg.wav.bitDepth
	}

	// Create WAV encoder
	encoder := wav.NewEncoder(file, sampleRate, bitDepth, numChannels, 1)
	defer encoder.Close()

	// Get tensor data
	tensorData := t.Data()
	if tensorData == nil {
		return types.NewError("marshal", "audio", "tensor data is nil", nil)
	}

	// Convert tensor data to PCM and write
	switch dtype {
	case tensortypes.INT16:
		data := tensorData.([]int16)
		return m.writeInt16PCM(encoder, data, numFrames, numChannels)
	case tensortypes.INT32:
		data := tensorData.([]int32)
		return m.writeInt32PCM(encoder, data, numFrames, numChannels, bitDepth)
	case tensortypes.FP32:
		data := tensorData.([]float32)
		return m.writeFloat32PCM(encoder, data, numFrames, numChannels, bitDepth)
	case tensortypes.UINT8:
		data := tensorData.([]uint8)
		return m.writeUint8PCM(encoder, data, numFrames, numChannels)
	default:
		return types.NewError("marshal", "audio", fmt.Sprintf("unsupported tensor dtype: %v", dtype), nil)
	}
}

// writeInt16PCM writes int16 tensor data to WAV.
func (m *Marshaller) writeInt16PCM(encoder *wav.Encoder, data []int16, numFrames, numChannels int) error {
	// Get pooled buffer for frame data
	frameData := m.intPool.Get(numChannels)
	defer m.intPool.Put(frameData)

	for frame := 0; frame < numFrames; frame++ {
		for ch := 0; ch < numChannels; ch++ {
			frameData[ch] = int(data[frame*numChannels+ch])
		}
		buf := &audio.IntBuffer{
			Format: &audio.Format{
				NumChannels: numChannels,
				SampleRate:  encoder.SampleRate,
			},
			Data: frameData,
		}
		if err := encoder.Write(buf); err != nil {
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to write frame %d", frame), err)
		}
	}

	// Update statistics
	atomic.AddInt64(&m.stats.frames, int64(numFrames))
	atomic.AddInt64(&m.stats.bytes, int64(numFrames*numChannels*2)) // int16 = 2 bytes

	return nil
}

// writeInt32PCM writes int32 tensor data to WAV.
func (m *Marshaller) writeInt32PCM(encoder *wav.Encoder, data []int32, numFrames, numChannels, bitDepth int) error {
	// Get pooled buffer for frame data
	frameData := m.intPool.Get(numChannels)
	defer m.intPool.Put(frameData)

	for frame := 0; frame < numFrames; frame++ {
		for ch := 0; ch < numChannels; ch++ {
			val := int(data[frame*numChannels+ch])
			// Clamp to bit depth range
			if bitDepth == 16 {
				if val > 32767 {
					val = 32767
				} else if val < -32768 {
					val = -32768
				}
			}
			frameData[ch] = val
		}
		buf := &audio.IntBuffer{
			Format: &audio.Format{
				NumChannels: numChannels,
				SampleRate:  encoder.SampleRate,
			},
			Data: frameData,
		}
		if err := encoder.Write(buf); err != nil {
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to write frame %d", frame), err)
		}
	}

	// Update statistics
	atomic.AddInt64(&m.stats.frames, int64(numFrames))
	bytesPerSample := 4
	if bitDepth == 16 {
		bytesPerSample = 2
	}
	atomic.AddInt64(&m.stats.bytes, int64(numFrames*numChannels*bytesPerSample))

	return nil
}

// writeFloat32PCM writes float32 tensor data to WAV.
func (m *Marshaller) writeFloat32PCM(encoder *wav.Encoder, data []float32, numFrames, numChannels, bitDepth int) error {
	// Get pooled buffer for frame data
	frameData := m.intPool.Get(numChannels)
	defer m.intPool.Put(frameData)

	for frame := 0; frame < numFrames; frame++ {
		for ch := 0; ch < numChannels; ch++ {
			val := data[frame*numChannels+ch]
			// Convert float to int based on bit depth
			if bitDepth == 16 {
				intVal := int(val * 32767.0)
				if intVal > 32767 {
					intVal = 32767
				} else if intVal < -32768 {
					intVal = -32768
				}
				frameData[ch] = intVal
			} else if bitDepth == 32 {
				intVal := int(val * 2147483647.0)
				if intVal > 2147483647 {
					intVal = 2147483647
				} else if intVal < -2147483648 {
					intVal = -2147483648
				}
				frameData[ch] = intVal
			} else {
				// Default to 16-bit
				intVal := int(val * 32767.0)
				if intVal > 32767 {
					intVal = 32767
				} else if intVal < -32768 {
					intVal = -32768
				}
				frameData[ch] = intVal
			}
		}
		buf := &audio.IntBuffer{
			Format: &audio.Format{
				NumChannels: numChannels,
				SampleRate:  encoder.SampleRate,
			},
			Data: frameData,
		}
		if err := encoder.Write(buf); err != nil {
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to write frame %d", frame), err)
		}
	}

	// Update statistics
	atomic.AddInt64(&m.stats.frames, int64(numFrames))
	bytesPerSample := 4
	if bitDepth == 16 {
		bytesPerSample = 2
	}
	atomic.AddInt64(&m.stats.bytes, int64(numFrames*numChannels*bytesPerSample))

	return nil
}

// writeUint8PCM writes uint8 tensor data to WAV.
func (m *Marshaller) writeUint8PCM(encoder *wav.Encoder, data []uint8, numFrames, numChannels int) error {
	// Get pooled buffer for frame data
	frameData := m.intPool.Get(numChannels)
	defer m.intPool.Put(frameData)

	for frame := 0; frame < numFrames; frame++ {
		for ch := 0; ch < numChannels; ch++ {
			// Convert from [0, 255] to signed [-128, 127] for WAV
			frameData[ch] = int(data[frame*numChannels+ch]) - 128
		}
		buf := &audio.IntBuffer{
			Format: &audio.Format{
				NumChannels: numChannels,
				SampleRate:  encoder.SampleRate,
			},
			Data: frameData,
		}
		if err := encoder.Write(buf); err != nil {
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to write frame %d", frame), err)
		}
	}

	// Update statistics
	atomic.AddInt64(&m.stats.frames, int64(numFrames))
	atomic.AddInt64(&m.stats.bytes, int64(numFrames*numChannels)) // uint8 = 1 byte

	return nil
}

// playbackSingleDevice plays tensor to a single audio device.
func (m *Marshaller) playbackSingleDevice(deviceID string, t types.Tensor, cfg config) error {
	if t.Empty() {
		return types.NewError("marshal", "audio", "empty tensor", nil)
	}

	shape := t.Shape()
	if len(shape) != 2 {
		return types.NewError("marshal", "audio", "tensor must be 2D [frames, channels] for audio", nil)
	}

	numFrames := shape[0]
	numChannels := shape[1]

	// Create device
	dev, err := newAudioDevice(deviceID, "playback", cfg.device, cfg.stream, cfg.ctx)
	if err != nil {
		return types.NewError("marshal", "audio", fmt.Sprintf("failed to create device: %v", err), err)
	}

	// Start device
	if err := dev.start(); err != nil {
		return types.NewError("marshal", "audio", fmt.Sprintf("failed to start device: %v", err), err)
	}
	defer dev.stop()

	// Get device format
	format := getDeviceFormat(dev)
	deviceChannels := getDeviceChannels(dev)

	// Verify channel count
	if numChannels != deviceChannels {
		return types.NewError("marshal", "audio",
			fmt.Sprintf("channel mismatch: tensor has %d channels, device expects %d", numChannels, deviceChannels), nil)
	}

	// Get tensor data
	tensorData := t.Data()
	if tensorData == nil {
		return types.NewError("marshal", "audio", "tensor data is nil", nil)
	}

	// Write frames to device
	frameSize := cfg.stream.frameSize
	if frameSize == 0 {
		frameSize = 1024
	}

	for frameOffset := 0; frameOffset < numFrames; frameOffset += frameSize {
		select {
		case <-cfg.ctx.Done():
			return cfg.ctx.Err()
		default:
		}

		// Calculate actual frame count for this chunk
		chunkFrames := frameSize
		if frameOffset+chunkFrames > numFrames {
			chunkFrames = numFrames - frameOffset
		}

		// Extract chunk from tensor
		chunkData, err := extractTensorChunk(tensorData, t.DataType(), frameOffset, chunkFrames, numChannels)
		if err != nil {
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to extract chunk: %v", err), err)
		}

		// Convert tensor to PCM
		pcmData, err := tensorToPCM(chunkData, format, numChannels, chunkFrames)
		if err != nil {
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to convert to PCM: %v", err), err)
		}

		// Write to device
		if err := dev.writeFrame(pcmData); err != nil {
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to write frame: %v", err), err)
		}

		// Update statistics
		atomic.AddInt64(&m.stats.frames, int64(chunkFrames))
		atomic.AddInt64(&m.stats.bytes, int64(len(pcmData)))
	}

	return nil
}

// playbackMultipleDevices distributes tensor channels across multiple devices.
func (m *Marshaller) playbackMultipleDevices(deviceIDs []string, t types.Tensor, cfg config) error {
	if t.Empty() {
		return types.NewError("marshal", "audio", "empty tensor", nil)
	}

	shape := t.Shape()
	if len(shape) != 2 {
		return types.NewError("marshal", "audio", "tensor must be 2D [frames, channels] for audio", nil)
	}

	numFrames := shape[0]
	numChannels := shape[1]

	if len(deviceIDs) == 0 {
		return types.NewError("marshal", "audio", "no device IDs provided", nil)
	}

	// Create all devices
	devices := make([]deviceInterface, 0, len(deviceIDs))
	deviceChannels := make([]int, 0, len(deviceIDs))
	totalDeviceChannels := 0

	for _, id := range deviceIDs {
		dev, err := newAudioDevice(id, "playback", cfg.device, cfg.stream, cfg.ctx)
		if err != nil {
			// Cleanup
			for _, d := range devices {
				d.stop()
			}
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to create device %s: %v", id, err), err)
		}
		devices = append(devices, dev)
		channels := getDeviceChannels(dev)
		deviceChannels = append(deviceChannels, channels)
		totalDeviceChannels += channels
	}

	// Verify channel count
	if numChannels != totalDeviceChannels {
		// Cleanup
		for _, dev := range devices {
			dev.stop()
		}
		return types.NewError("marshal", "audio",
			fmt.Sprintf("channel mismatch: tensor has %d channels, devices total %d", numChannels, totalDeviceChannels), nil)
	}

	// Start all devices
	for _, dev := range devices {
		if err := dev.start(); err != nil {
			// Cleanup
			for _, d := range devices {
				d.stop()
			}
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to start device: %v", err), err)
		}
	}
	defer func() {
		for _, dev := range devices {
			dev.stop()
		}
	}()

	// Verify all devices have same sample rate
	commonSampleRate := getDeviceSampleRate(devices[0])
	for i, dev := range devices {
		if getDeviceSampleRate(dev) != commonSampleRate {
			return types.NewError("marshal", "audio",
				fmt.Sprintf("sample rate mismatch: device %d has %d Hz, expected %d Hz", i, getDeviceSampleRate(dev), commonSampleRate), nil)
		}
	}

	// Get device format
	format := getDeviceFormat(devices[0])

	// Get tensor data
	tensorData := t.Data()
	if tensorData == nil {
		return types.NewError("marshal", "audio", "tensor data is nil", nil)
	}

	// Write frames to all devices in lockstep
	frameSize := cfg.stream.frameSize
	if frameSize == 0 {
		frameSize = 1024
	}

	for frameOffset := 0; frameOffset < numFrames; frameOffset += frameSize {
		select {
		case <-cfg.ctx.Done():
			return cfg.ctx.Err()
		default:
		}

		// Calculate actual frame count for this chunk
		chunkFrames := frameSize
		if frameOffset+chunkFrames > numFrames {
			chunkFrames = numFrames - frameOffset
		}

		// Distribute channels across devices
		channelOffset := 0
		for devIdx, dev := range devices {
			devChannels := deviceChannels[devIdx]

			// Extract channels for this device
			chunkData, err := extractDeviceChannels(tensorData, t.DataType(), frameOffset, chunkFrames, channelOffset, devChannels, numChannels)
			if err != nil {
				return types.NewError("marshal", "audio", fmt.Sprintf("failed to extract channels: %v", err), err)
			}

			// Convert tensor to PCM
			pcmData, err := tensorToPCM(chunkData, format, devChannels, chunkFrames)
			if err != nil {
				return types.NewError("marshal", "audio", fmt.Sprintf("failed to convert to PCM: %v", err), err)
			}

			// Write to device
			if err := dev.writeFrame(pcmData); err != nil {
				return types.NewError("marshal", "audio", fmt.Sprintf("failed to write to device %d: %v", devIdx, err), err)
			}

			channelOffset += devChannels
		}

		// Update statistics
		atomic.AddInt64(&m.stats.frames, int64(chunkFrames))
		atomic.AddInt64(&m.stats.bytes, int64(chunkFrames*totalDeviceChannels*sampleSizeFromFormat(format)))
	}

	return nil
}

// writeMixedOutputs writes to both WAV file(s) and device(s).
func (m *Marshaller) writeMixedOutputs(t types.Tensor, cfg config) error {
	shape := t.Shape()
	if len(shape) != 2 {
		return types.NewError("marshal", "audio", "tensor must be 2D [frames, channels] for audio", nil)
	}

	// Write to WAV files
	if len(cfg.wav.paths) > 0 {
		if err := m.writeTensor(nil, t, cfg); err != nil {
			return err
		}
	}

	// Play to devices
	if len(cfg.device.deviceIDs) > 0 {
		if len(cfg.device.deviceIDs) == 1 {
			if err := m.playbackSingleDevice(cfg.device.deviceIDs[0], t, cfg); err != nil {
				return err
			}
		} else {
			if err := m.playbackMultipleDevices(cfg.device.deviceIDs, t, cfg); err != nil {
				return err
			}
		}
	}

	return nil
}

// writeFrameStream writes a frame stream to WAV file(s) and/or audio device(s).
func (m *Marshaller) writeFrameStream(w io.Writer, stream types.FrameStream, cfg config) error {
	// Check if we have devices configured
	if len(cfg.device.deviceIDs) > 0 {
		// Create devices for streaming
		devices := make([]deviceInterface, 0, len(cfg.device.deviceIDs))
		for _, id := range cfg.device.deviceIDs {
			dev, err := newAudioDevice(id, "playback", cfg.device, cfg.stream, cfg.ctx)
			if err != nil {
				for _, d := range devices {
					d.stop()
				}
				return types.NewError("marshal", "audio", fmt.Sprintf("failed to create device: %v", err), err)
			}
			devices = append(devices, dev)
		}

		// Start all devices
		for _, dev := range devices {
			if err := dev.start(); err != nil {
				for _, d := range devices {
					d.stop()
				}
				return types.NewError("marshal", "audio", fmt.Sprintf("failed to start device: %v", err), err)
			}
		}
		defer func() {
			for _, dev := range devices {
				dev.stop()
			}
		}()

		format := getDeviceFormat(devices[0])
		deviceChannels := make([]int, len(devices))
		totalDeviceChannels := 0
		for i, dev := range devices {
			channels := getDeviceChannels(dev)
			deviceChannels[i] = channels
			totalDeviceChannels += channels
		}

		// Stream frames to devices
		for frame := range stream.C {
			if len(frame.Tensors) == 0 {
				continue
			}

			t := frame.Tensors[0]
			if t.Empty() {
				continue
			}

			shape := t.Shape()
			if len(shape) != 2 {
				continue
			}

			tensorData := t.Data()
			if tensorData == nil {
				continue
			}

			chunkFrames := shape[0]
			tensorChannels := shape[1]

			// Distribute channels across devices
			channelOffset := 0
			for devIdx, dev := range devices {
				devChannels := deviceChannels[devIdx]

				// Extract channels for this device
				chunkData, err := extractDeviceChannels(tensorData, t.DataType(), 0, chunkFrames, channelOffset, devChannels, tensorChannels)
				if err != nil {
					continue
				}

				// Convert tensor to PCM
				pcmData, err := tensorToPCM(chunkData, format, devChannels, chunkFrames)
				if err != nil {
					continue
				}

				// Write to device
				dev.writeFrame(pcmData)

				channelOffset += devChannels
			}

			// Update statistics
			atomic.AddInt64(&m.stats.frames, int64(chunkFrames))
			atomic.AddInt64(&m.stats.bytes, int64(chunkFrames*totalDeviceChannels*sampleSizeFromFormat(format)))

			// Release frame if needed
			if cfg.ReleaseAfterProcessing {
				t.Release()
			}
		}

		return nil
	}

	// If only WAV files, accumulate frames and write to WAV
	if len(cfg.wav.paths) > 0 {
		// Accumulate all frames from stream
		var frameTensors []types.Tensor
		var totalFrames int
		var numChannels int
		var commonDtype tensortypes.DataType
		var sampleRate int

		// First pass: collect frames and determine format
		for frame := range stream.C {
			if len(frame.Tensors) == 0 {
				continue
			}

			t := frame.Tensors[0]
			if t.Empty() {
				continue
			}

			shape := t.Shape()
			if len(shape) != 2 {
				continue
			}

			frameTensors = append(frameTensors, t)
			totalFrames += shape[0]
			numChannels = shape[1]
			commonDtype = t.DataType()
			// Get sample rate from config
			sampleRate = cfg.wav.sampleRate
			if sampleRate == 0 {
				sampleRate = 48000 // default
			}
		}

		if len(frameTensors) == 0 {
			return types.NewError("marshal", "audio", "no frames in stream", nil)
		}

		// Combine all frames into single tensor
		combinedShape := tensor.NewShape(totalFrames, numChannels)
		combined := tensor.New(commonDtype, combinedShape)
		if combined.Empty() {
			return types.NewError("marshal", "audio", "failed to create combined tensor", nil)
		}

		combinedData := combined.Data()
		if combinedData == nil {
			return types.NewError("marshal", "audio", "combined tensor data is nil", nil)
		}

		// Copy frames into combined tensor
		frameOffset := 0
		for _, t := range frameTensors {
			tShape := t.Shape()
			frameCount := tShape[0]
			tData := t.Data()
			if tData == nil {
				continue
			}

			// Copy data based on dtype
			switch commonDtype {
			case tensortypes.INT16:
				dst := combinedData.([]int16)
				src := tData.([]int16)
				copy(dst[frameOffset*numChannels:(frameOffset+frameCount)*numChannels], src)
			case tensortypes.INT32:
				dst := combinedData.([]int32)
				src := tData.([]int32)
				copy(dst[frameOffset*numChannels:(frameOffset+frameCount)*numChannels], src)
			case tensortypes.FP32:
				dst := combinedData.([]float32)
				src := tData.([]float32)
				copy(dst[frameOffset*numChannels:(frameOffset+frameCount)*numChannels], src)
			case tensortypes.UINT8:
				dst := combinedData.([]uint8)
				src := tData.([]uint8)
				copy(dst[frameOffset*numChannels:(frameOffset+frameCount)*numChannels], src)
			default:
				return types.NewError("marshal", "audio", fmt.Sprintf("unsupported dtype: %v", commonDtype), nil)
			}

			frameOffset += frameCount

			// Release frame if needed
			if cfg.ReleaseAfterProcessing {
				t.Release()
			}
		}

		// Write combined tensor to WAV file
		return m.writeSingleWAV(cfg.wav.paths[0], combined, totalFrames, numChannels, sampleRate, cfg)
	}

	return types.NewError("marshal", "audio", "no outputs specified", nil)
}
