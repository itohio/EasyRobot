package audio

import (
	"fmt"
	"io"
	"os"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// Marshaller encodes audio tensors to WAV files or audio devices.
type Marshaller struct {
	opts types.Options
	cfg  config
}

// NewMarshaller constructs an audio marshaller instance.
func NewMarshaller(opts ...types.Option) *Marshaller {
	baseOpts, baseCfg := applyOptions(types.Options{}, defaultConfig(), opts)
	return &Marshaller{
		opts: baseOpts,
		cfg:  baseCfg,
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
	// TODO: Implement health statistics tracking
	return make(map[string]int64)
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

	// Check if we have WAV paths configured
	if len(cfg.wav.paths) == 0 {
		// No WAV files, check for devices
		if len(cfg.device.deviceIDs) > 0 {
			return types.NewError("marshal", "audio", "device playback not yet implemented", nil)
		}
		return types.NewError("marshal", "audio", "no WAV paths or devices specified", nil)
	}

	// Single WAV file writing (Phase 3.1)
	if len(cfg.wav.paths) == 1 {
		return m.writeSingleWAV(cfg.wav.paths[0], t, numFrames, numChannels, sampleRate, cfg)
	}

	// Multiple WAV files (Phase 3.2) - TODO
	return types.NewError("marshal", "audio", "multiple WAV files not yet implemented", nil)
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
	for frame := 0; frame < numFrames; frame++ {
		intData := make([]int, numChannels)
		for ch := 0; ch < numChannels; ch++ {
			intData[ch] = int(data[frame*numChannels+ch])
		}
		buf := &audio.IntBuffer{
			Format: &audio.Format{
				NumChannels: numChannels,
				SampleRate:  encoder.SampleRate,
			},
			Data: intData,
		}
		if err := encoder.Write(buf); err != nil {
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to write frame %d", frame), err)
		}
	}
	return nil
}

// writeInt32PCM writes int32 tensor data to WAV.
func (m *Marshaller) writeInt32PCM(encoder *wav.Encoder, data []int32, numFrames, numChannels, bitDepth int) error {
	for frame := 0; frame < numFrames; frame++ {
		intData := make([]int, numChannels)
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
			intData[ch] = val
		}
		buf := &audio.IntBuffer{
			Format: &audio.Format{
				NumChannels: numChannels,
				SampleRate:  encoder.SampleRate,
			},
			Data: intData,
		}
		if err := encoder.Write(buf); err != nil {
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to write frame %d", frame), err)
		}
	}
	return nil
}

// writeFloat32PCM writes float32 tensor data to WAV.
func (m *Marshaller) writeFloat32PCM(encoder *wav.Encoder, data []float32, numFrames, numChannels, bitDepth int) error {
	for frame := 0; frame < numFrames; frame++ {
		intData := make([]int, numChannels)
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
				intData[ch] = intVal
			} else if bitDepth == 32 {
				intVal := int(val * 2147483647.0)
				if intVal > 2147483647 {
					intVal = 2147483647
				} else if intVal < -2147483648 {
					intVal = -2147483648
				}
				intData[ch] = intVal
			} else {
				// Default to 16-bit
				intVal := int(val * 32767.0)
				if intVal > 32767 {
					intVal = 32767
				} else if intVal < -32768 {
					intVal = -32768
				}
				intData[ch] = intVal
			}
		}
		buf := &audio.IntBuffer{
			Format: &audio.Format{
				NumChannels: numChannels,
				SampleRate:  encoder.SampleRate,
			},
			Data: intData,
		}
		if err := encoder.Write(buf); err != nil {
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to write frame %d", frame), err)
		}
	}
	return nil
}

// writeUint8PCM writes uint8 tensor data to WAV.
func (m *Marshaller) writeUint8PCM(encoder *wav.Encoder, data []uint8, numFrames, numChannels int) error {
	for frame := 0; frame < numFrames; frame++ {
		intData := make([]int, numChannels)
		for ch := 0; ch < numChannels; ch++ {
			// Convert from [0, 255] to signed [-128, 127] for WAV
			val := int(data[frame*numChannels+ch]) - 128
			intData[ch] = val
		}
		buf := &audio.IntBuffer{
			Format: &audio.Format{
				NumChannels: numChannels,
				SampleRate:  encoder.SampleRate,
			},
			Data: intData,
		}
		if err := encoder.Write(buf); err != nil {
			return types.NewError("marshal", "audio", fmt.Sprintf("failed to write frame %d", frame), err)
		}
	}
	return nil
}

// writeFrameStream writes a frame stream to WAV file(s) and/or audio device(s).
func (m *Marshaller) writeFrameStream(w io.Writer, stream types.FrameStream, cfg config) error {
	// TODO: Implement frame stream writing
	return types.NewError("marshal", "audio", "writeFrameStream not yet implemented", nil)
}
