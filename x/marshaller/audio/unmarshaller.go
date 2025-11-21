package audio

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// Unmarshaller decodes audio from WAV files or audio devices into tensors.
type Unmarshaller struct {
	opts types.Options
	cfg  config

	// Format information from last unmarshal operation
	lastSampleRate int
	lastChannels   int
	lastDataType   tensortypes.DataType
}

// NewUnmarshaller constructs an audio unmarshaller instance.
func NewUnmarshaller(opts ...types.Option) *Unmarshaller {
	baseOpts, baseCfg := applyOptions(types.Options{}, defaultConfig(), opts)
	return &Unmarshaller{
		opts:         baseOpts,
		cfg:          baseCfg,
		lastDataType: tensortypes.DT_UNKNOWN,
	}
}

// Format identifies the unmarshaller.
func (u *Unmarshaller) Format() string {
	return "audio"
}

// Unmarshal decodes audio from WAV files or devices into the provided destination.
func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
	if dst == nil {
		return types.NewError("unmarshal", "audio", "nil destination", nil)
	}

	localOpts, localCfg := applyOptions(u.opts, u.cfg, opts)

	switch out := dst.(type) {
	case *types.Tensor:
		tensor, err := u.unmarshalTensor(r, localCfg, tensortypes.DataType(localOpts.DestinationType), localOpts.TensorFactory)
		if err != nil {
			return err
		}
		*out = tensor
		return nil
	case *types.FrameStream:
		stream, err := u.unmarshalStream(r, localCfg)
		if err != nil {
			return err
		}
		*out = stream
		return nil
	case *[]types.AudioDeviceInfo:
		devices, err := u.unmarshalDeviceList(r)
		if err != nil {
			return err
		}
		*out = devices
		return nil
	default:
		return types.NewError("unmarshal", "audio", fmt.Sprintf("unsupported destination %T", dst), nil)
	}
}

// SampleRate returns the sample rate of the last unmarshalled source.
func (u *Unmarshaller) SampleRate() int {
	return u.lastSampleRate
}

// Channels returns the number of channels of the last unmarshalled source.
func (u *Unmarshaller) Channels() int {
	return u.lastChannels
}

// DataType returns the data type of the last unmarshalled tensor.
// Returns DT_UNKNOWN if no tensor has been unmarshalled yet.
func (u *Unmarshaller) DataType() types.DataType {
	return types.DataType(u.lastDataType)
}

// Health returns statistics about the unmarshaller state.
// For device streams: frames, queue lengths, dropped buffers, byte counts.
// For WAV files: frames, bytes read.
func (u *Unmarshaller) Health() map[string]int64 {
	// TODO: Implement health statistics tracking
	return make(map[string]int64)
}

// unmarshalTensor creates a tensor from WAV file(s) or device(s).
func (u *Unmarshaller) unmarshalTensor(r io.Reader, cfg config, dstType tensortypes.DataType, tensorFactory func(tensortypes.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	// Check if we have WAV paths configured
	if len(cfg.wav.paths) == 0 {
		// No WAV files, check for devices
		if len(cfg.device.deviceIDs) > 0 {
			return nil, types.NewError("unmarshal", "audio", "device capture not yet implemented", nil)
		}
		return nil, types.NewError("unmarshal", "audio", "no WAV paths or devices specified", nil)
	}

	// Single WAV file reading (Phase 2.1)
	if len(cfg.wav.paths) == 1 {
		return u.readSingleWAV(cfg.wav.paths[0], cfg, dstType, tensorFactory)
	}

	// Multiple WAV files (Phase 2.2) - TODO
	return nil, types.NewError("unmarshal", "audio", "multiple WAV files not yet implemented", nil)
}

// readSingleWAV reads a single WAV file and creates a tensor.
func (u *Unmarshaller) readSingleWAV(path string, cfg config, dstType tensortypes.DataType, tensorFactory func(tensortypes.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	// Open WAV file
	file, err := os.Open(path)
	if err != nil {
		return nil, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to open WAV file: %s", path), err)
	}
	defer file.Close()

	// Create WAV decoder
	decoder := wav.NewDecoder(file)
	if decoder == nil {
		return nil, types.NewError("unmarshal", "audio", "failed to create WAV decoder", nil)
	}

	// Read format info
	decoder.ReadInfo()
	if decoder.Err() != nil {
		return nil, types.NewError("unmarshal", "audio", "failed to read WAV format info", decoder.Err())
	}

	format := decoder.Format()
	if format == nil {
		return nil, types.NewError("unmarshal", "audio", "WAV file has no format info", nil)
	}

	sampleRate := int(format.SampleRate)
	numChannels := int(format.NumChannels)

	// Store format info
	u.lastSampleRate = sampleRate
	u.lastChannels = numChannels

	// Detect bit depth from WAV decoder
	bitDepth := int(decoder.BitDepth)
	if bitDepth == 0 {
		// Fallback: try SampleBitDepth() method
		bitDepth = int(decoder.SampleBitDepth())
	}
	if bitDepth == 0 {
		// Default to 16-bit PCM if not detected
		bitDepth = 16
	}

	// Detect if float format (WAV format 3 = IEEE float)
	isFloat := decoder.WavAudioFormat == 3

	// Map WAV format to tensor dtype
	var tensorDtype tensortypes.DataType
	if dstType != tensortypes.DT_UNKNOWN {
		tensorDtype = dstType
	} else {
		tensorDtype = wavFormatToDtype(bitDepth, isFloat)
		u.lastDataType = tensorDtype
	}

	// Validate sample rate if specified
	if cfg.wav.sampleRate > 0 && cfg.wav.sampleRate != sampleRate {
		return nil, types.NewError("unmarshal", "audio",
			fmt.Sprintf("sample rate mismatch: file has %d Hz, expected %d Hz (resampling not supported)",
				sampleRate, cfg.wav.sampleRate), nil)
	}

	// Validate channels if specified
	if cfg.wav.channels > 0 && cfg.wav.channels != numChannels {
		return nil, types.NewError("unmarshal", "audio",
			fmt.Sprintf("channel count mismatch: file has %d channels, expected %d",
				numChannels, cfg.wav.channels), nil)
	}

	// Read all PCM data
	// Use IntBuffer for PCM data (works for all integer formats)
	intBuf := &audio.IntBuffer{
		Format: format,
		Data:   make([]int, 0),
	}

	// Read all frames
	for {
		buf := &audio.IntBuffer{
			Format: format,
			Data:   make([]int, 1024*numChannels), // Read in chunks
		}
		n, err := decoder.PCMBuffer(buf)
		if err != nil && err != io.EOF {
			return nil, types.NewError("unmarshal", "audio", "failed to read PCM data", err)
		}
		if n == 0 {
			break
		}
		// Append read data
		intBuf.Data = append(intBuf.Data, buf.Data[:n]...)
		if err == io.EOF {
			break
		}
	}

	numFrames := len(intBuf.Data) / numChannels
	if numFrames == 0 {
		return nil, types.NewError("unmarshal", "audio", "WAV file contains no audio data", nil)
	}

	// Create tensor with shape [frames, channels]
	shape := tensor.NewShape(numFrames, numChannels)

	// Use provided tensor factory or default
	if tensorFactory == nil {
		tensorFactory = func(dtype tensortypes.DataType, shape types.Shape) types.Tensor {
			return tensor.New(dtype, shape)
		}
	}

	// Create tensor
	t := tensorFactory(tensorDtype, shape)
	if t == nil {
		return nil, types.NewError("unmarshal", "audio", "failed to create tensor", nil)
	}

	// Convert PCM data to tensor data based on dtype
	if err := u.fillTensorFromPCM(t, intBuf.Data, numFrames, numChannels, tensorDtype, bitDepth, isFloat); err != nil {
		return nil, err
	}

	return t, nil
}

// wavFormatToDtype maps WAV file format to tensor data type.
func wavFormatToDtype(bitDepth int, isFloat bool) tensortypes.DataType {
	if isFloat {
		return tensortypes.FP32
	}
	switch bitDepth {
	case 8:
		return tensortypes.UINT8
	case 16:
		return tensortypes.INT16
	case 24, 32:
		return tensortypes.INT32
	default:
		// Default to INT16 for unknown formats
		return tensortypes.INT16
	}
}

// fillTensorFromPCM fills a tensor with PCM data, converting based on target dtype.
func (u *Unmarshaller) fillTensorFromPCM(t types.Tensor, pcmData []int, numFrames, numChannels int, dtype tensortypes.DataType, bitDepth int, isFloat bool) error {
	data := t.Data()
	if data == nil {
		return types.NewError("unmarshal", "audio", "tensor data is nil", nil)
	}

	switch dtype {
	case tensortypes.INT16:
		dataSlice, ok := data.([]int16)
		if !ok {
			return types.NewError("unmarshal", "audio", "tensor data is not []int16", nil)
		}
		for i, val := range pcmData {
			// Convert int to int16 (clamp if needed)
			if val > 32767 {
				val = 32767
			} else if val < -32768 {
				val = -32768
			}
			dataSlice[i] = int16(val)
		}
	case tensortypes.INT32:
		dataSlice, ok := data.([]int32)
		if !ok {
			return types.NewError("unmarshal", "audio", "tensor data is not []int32", nil)
		}
		for i, val := range pcmData {
			dataSlice[i] = int32(val)
		}
	case tensortypes.FP32:
		dataSlice, ok := data.([]float32)
		if !ok {
			return types.NewError("unmarshal", "audio", "tensor data is not []float32", nil)
		}
		if isFloat {
			// Already float format, convert int buffer to float
			// This shouldn't happen with go-audio/wav, but handle it
			for i, val := range pcmData {
				dataSlice[i] = float32(val) / 2147483647.0 // Normalize to [-1.0, 1.0]
			}
		} else {
			// Convert integer PCM to float
			// For 16-bit: max value is 32768 (2^15)
			// For 24-bit: max value is 8388608 (2^23)
			// For 32-bit: max value is 2147483648 (2^31)
			var maxVal float32
			switch bitDepth {
			case 8:
				maxVal = 128.0
			case 16:
				maxVal = 32768.0
			case 24:
				maxVal = 8388608.0
			case 32:
				maxVal = 2147483648.0
			default:
				maxVal = 32768.0 // Default to 16-bit
			}
			for i, val := range pcmData {
				dataSlice[i] = float32(val) / maxVal
			}
		}
	case tensortypes.UINT8:
		dataSlice, ok := data.([]uint8)
		if !ok {
			return types.NewError("unmarshal", "audio", "tensor data is not []uint8", nil)
		}
		for i, val := range pcmData {
			// Convert to unsigned 8-bit (offset by 128)
			u8Val := val + 128
			if u8Val > 255 {
				u8Val = 255
			} else if u8Val < 0 {
				u8Val = 0
			}
			dataSlice[i] = uint8(u8Val)
		}
	default:
		return types.NewError("unmarshal", "audio", fmt.Sprintf("unsupported tensor dtype: %v", dtype), nil)
	}

	return nil
}

// unmarshalStream creates a FrameStream from audio device(s).
func (u *Unmarshaller) unmarshalStream(r io.Reader, cfg config) (types.FrameStream, error) {
	// TODO: Implement stream unmarshalling
	return types.FrameStream{}, types.NewError("unmarshal", "audio", "unmarshalStream not yet implemented", nil)
}

// unmarshalDeviceList enumerates available audio devices.
func (u *Unmarshaller) unmarshalDeviceList(r io.Reader) ([]types.AudioDeviceInfo, error) {
	// Check if this is a "list" command
	if r != nil {
		buf := make([]byte, 4)
		n, err := r.Read(buf)
		if err != nil && err != io.EOF {
			return nil, types.NewError("unmarshal", "audio", "read list command", err)
		}
		command := strings.TrimSpace(string(buf[:n]))
		if command != "list" {
			return nil, types.NewError("unmarshal", "audio", "expected 'list' command, got: "+command, nil)
		}
	}

	// TODO: Enumerate audio devices using malgo (via gocoustics)
	return nil, types.NewError("unmarshal", "audio", "device enumeration not yet implemented", nil)
}
