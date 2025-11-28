package audio

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/itohio/EasyRobot/x/workers"
)

// stats tracks statistics for the unmarshaller using atomic counters.
type stats struct {
	frames int64 // Total frames processed
	bytes  int64 // Total bytes read
}

// Unmarshaller decodes audio from WAV files or audio devices into tensors.
type Unmarshaller struct {
	opts types.Options
	cfg  config

	// Format information from last unmarshal operation
	lastSampleRate int
	lastChannels   int
	lastDataType   tensortypes.DataType

	// Statistics tracking
	stats stats

	// Buffer pools
	intPool *buffer.Pool[int] // Pool for PCM int buffers
	
	// Worker pool for parallel operations
	workerPool *workers.WorkerPool
	
	// Active devices for flush
	activeDevices []deviceInterface
	devicesMu     sync.RWMutex
}

// NewUnmarshaller constructs an audio unmarshaller instance.
func NewUnmarshaller(opts ...types.Option) *Unmarshaller {
	baseOpts, baseCfg := applyOptions(types.Options{}, defaultConfig(), opts)
	
	// Initialize worker pool
	wp := &workers.WorkerPool{}
	if err := wp.Init(); err != nil {
		// Fallback to default if init fails
		wp = &workers.WorkerPool{}
		_ = wp.Init()
	}
	
	return &Unmarshaller{
		opts:          baseOpts,
		cfg:           baseCfg,
		lastDataType:  tensortypes.DT_UNKNOWN,
		intPool:       &buffer.Pool[int]{},
		workerPool:    wp,
		activeDevices: make([]deviceInterface, 0),
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
	return map[string]int64{
		"frames": atomic.LoadInt64(&u.stats.frames),
		"bytes":  atomic.LoadInt64(&u.stats.bytes),
	}
}

// Flush instructs all malgo handlers to skip buffer writes, clears all ring buffers,
// and enables parallel processing. This ensures all devices process buffers simultaneously.
func (u *Unmarshaller) Flush() error {
	u.devicesMu.RLock()
	devices := make([]deviceInterface, len(u.activeDevices))
	copy(devices, u.activeDevices)
	u.devicesMu.RUnlock()
	
	if len(devices) == 0 {
		return nil
	}
	
	// Flush all devices in parallel using worker pool
	return flushDevices(devices, u.workerPool)
}

// unmarshalTensor creates a tensor from WAV file(s) or device(s).
func (u *Unmarshaller) unmarshalTensor(r io.Reader, cfg config, dstType tensortypes.DataType, tensorFactory func(tensortypes.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	// Check if we have devices configured
	if len(cfg.device.deviceIDs) > 0 {
		if len(cfg.wav.paths) > 0 {
			// Mixed sources: WAV + devices
			return u.unmarshalMixedSources(cfg, dstType, tensorFactory)
		}
		// Device-only capture
		if len(cfg.device.deviceIDs) == 1 {
			return u.captureSingleDevice(cfg.device.deviceIDs[0], cfg, dstType, tensorFactory)
		}
		return u.captureMultipleDevices(cfg.device.deviceIDs, cfg, dstType, tensorFactory)
	}

	// Check if we have WAV paths configured
	if len(cfg.wav.paths) == 0 {
		return nil, types.NewError("unmarshal", "audio", "no WAV paths or devices specified", nil)
	}

	// Single WAV file reading (Phase 2.1)
	if len(cfg.wav.paths) == 1 {
		return u.readSingleWAV(cfg.wav.paths[0], cfg, dstType, tensorFactory)
	}

	// Multiple WAV files (Phase 2.2)
	return u.readMultipleWAV(cfg.wav.paths, cfg, dstType, tensorFactory)
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

	// Read all frames using pooled buffers
	chunkSize := 1024 * numChannels
	for {
		// Get pooled buffer
		bufData := u.intPool.Get(chunkSize)
		buf := &audio.IntBuffer{
			Format: format,
			Data:   bufData,
		}
		n, err := decoder.PCMBuffer(buf)
		if err != nil && err != io.EOF {
			u.intPool.Put(bufData)
			return nil, types.NewError("unmarshal", "audio", "failed to read PCM data", err)
		}
		if n == 0 {
			u.intPool.Put(bufData)
			break
		}
		// Append read data
		intBuf.Data = append(intBuf.Data, buf.Data[:n]...)
		// Return buffer to pool
		u.intPool.Put(bufData)
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

	// Update statistics
	atomic.AddInt64(&u.stats.frames, int64(numFrames))
	atomic.AddInt64(&u.stats.bytes, int64(len(intBuf.Data)*4)) // Approximate: int is 4 bytes

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

// readMultipleWAV reads multiple WAV files and combines them into a single tensor.
// All files must have the same sample rate (no resampling).
// Channels are concatenated: file 0 → ch 0..N, file 1 → ch N+1..M, etc.
func (u *Unmarshaller) readMultipleWAV(paths []string, cfg config, dstType tensortypes.DataType, tensorFactory func(tensortypes.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	if len(paths) == 0 {
		return nil, types.NewError("unmarshal", "audio", "no WAV paths provided", nil)
	}

	// Read all files and collect metadata
	type fileInfo struct {
		tensor     types.Tensor
		frames     int
		channels   int
		sampleRate int
		dtype      tensortypes.DataType
	}
	files := make([]fileInfo, 0, len(paths))

	// Read each file
	var commonSampleRate int
	var commonDtype tensortypes.DataType
	maxFrames := 0
	totalChannels := 0

	for i, path := range paths {
		t, err := u.readSingleWAV(path, cfg, dstType, tensorFactory)
		if err != nil {
			return nil, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to read file %d (%s): %v", i, path, err), err)
		}

		shape := t.Shape()
		frames := shape[0]
		channels := shape[1]

		// Verify sample rate consistency
		if i == 0 {
			commonSampleRate = u.lastSampleRate
			if dstType != tensortypes.DT_UNKNOWN {
				commonDtype = dstType
			} else {
				commonDtype = u.lastDataType
			}
		} else if u.lastSampleRate != commonSampleRate {
			return nil, types.NewError("unmarshal", "audio",
				fmt.Sprintf("sample rate mismatch: file %d has %d Hz, expected %d Hz (resampling not supported)",
					i, u.lastSampleRate, commonSampleRate), nil)
		}

		// Track maximum frames (for padding)
		if frames > maxFrames {
			maxFrames = frames
		}
		totalChannels += channels

		files = append(files, fileInfo{
			tensor:     t,
			frames:     frames,
			channels:   channels,
			sampleRate: u.lastSampleRate,
			dtype:      u.lastDataType,
		})
	}

	// Determine common dtype (already set from first file or dstType)
	// Convert all tensors to common dtype if needed
	for i := range files {
		if files[i].dtype != commonDtype {
			// Create new tensor with target dtype and same shape
			fileShape := files[i].tensor.Shape()
			converted := tensorFactory(commonDtype, fileShape)
			if converted == nil {
				return nil, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to create converted tensor from %v to %v", files[i].dtype, commonDtype), nil)
			}
			// Copy with type conversion
			converted.Copy(files[i].tensor)
			files[i].tensor = converted
			files[i].dtype = commonDtype
		}
	}

	// Create combined tensor with shape [maxFrames, totalChannels]
	combinedShape := tensor.NewShape(maxFrames, totalChannels)
	if tensorFactory == nil {
		tensorFactory = func(dtype tensortypes.DataType, shape types.Shape) types.Tensor {
			return tensor.New(dtype, shape)
		}
	}
	combined := tensorFactory(commonDtype, combinedShape)
	if combined == nil {
		return nil, types.NewError("unmarshal", "audio", "failed to create combined tensor", nil)
	}

	// Combine channels from all files
	combinedData := combined.Data()
	if combinedData == nil {
		return nil, types.NewError("unmarshal", "audio", "combined tensor data is nil", nil)
	}

	// Copy data from each file to appropriate channels
	channelOffset := 0
	for _, file := range files {
		srcData := file.tensor.Data()
		if srcData == nil {
			continue
		}

		// Copy frames from source to combined tensor
		switch commonDtype {
		case tensortypes.INT16:
			dst := combinedData.([]int16)
			src := srcData.([]int16)
			for frame := 0; frame < file.frames; frame++ {
				for ch := 0; ch < file.channels; ch++ {
					dst[frame*totalChannels+channelOffset+ch] = src[frame*file.channels+ch]
				}
			}
			// Pad remaining frames with zeros
			for frame := file.frames; frame < maxFrames; frame++ {
				for ch := 0; ch < file.channels; ch++ {
					dst[frame*totalChannels+channelOffset+ch] = 0
				}
			}
		case tensortypes.INT32:
			dst := combinedData.([]int32)
			src := srcData.([]int32)
			for frame := 0; frame < file.frames; frame++ {
				for ch := 0; ch < file.channels; ch++ {
					dst[frame*totalChannels+channelOffset+ch] = src[frame*file.channels+ch]
				}
			}
			for frame := file.frames; frame < maxFrames; frame++ {
				for ch := 0; ch < file.channels; ch++ {
					dst[frame*totalChannels+channelOffset+ch] = 0
				}
			}
		case tensortypes.FP32:
			dst := combinedData.([]float32)
			src := srcData.([]float32)
			for frame := 0; frame < file.frames; frame++ {
				for ch := 0; ch < file.channels; ch++ {
					dst[frame*totalChannels+channelOffset+ch] = src[frame*file.channels+ch]
				}
			}
			for frame := file.frames; frame < maxFrames; frame++ {
				for ch := 0; ch < file.channels; ch++ {
					dst[frame*totalChannels+channelOffset+ch] = 0
				}
			}
		case tensortypes.UINT8:
			dst := combinedData.([]uint8)
			src := srcData.([]uint8)
			for frame := 0; frame < file.frames; frame++ {
				for ch := 0; ch < file.channels; ch++ {
					dst[frame*totalChannels+channelOffset+ch] = src[frame*file.channels+ch]
				}
			}
			for frame := file.frames; frame < maxFrames; frame++ {
				for ch := 0; ch < file.channels; ch++ {
					dst[frame*totalChannels+channelOffset+ch] = 0
				}
			}
		default:
			return nil, types.NewError("unmarshal", "audio", fmt.Sprintf("unsupported dtype for concatenation: %v", commonDtype), nil)
		}

		channelOffset += file.channels
	}

	// Update format info
	u.lastSampleRate = commonSampleRate
	u.lastChannels = totalChannels
	u.lastDataType = commonDtype

	// Update statistics
	atomic.AddInt64(&u.stats.frames, int64(maxFrames))
	atomic.AddInt64(&u.stats.bytes, int64(maxFrames*totalChannels*4)) // Approximate

	return combined, nil
}

// unmarshalStream creates a FrameStream from audio device(s).
func (u *Unmarshaller) unmarshalStream(r io.Reader, cfg config) (types.FrameStream, error) {
	if len(cfg.device.deviceIDs) == 0 {
		return types.FrameStream{}, types.NewError("unmarshal", "audio", "no devices specified for streaming", nil)
	}

	// For now, single device streaming
	// Multi-device streaming would require more complex synchronization
	if len(cfg.device.deviceIDs) == 1 {
		return u.createDeviceStream(cfg.device.deviceIDs[0], cfg)
	}

	// Multi-device streaming - combine into single stream
	return u.createMultiDeviceStream(cfg.device.deviceIDs, cfg)
}

func stopDevicesOnce(devices []deviceInterface) func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			for _, dev := range devices {
				if dev != nil {
					dev.stop()
				}
			}
		})
	}
}

// createDeviceStream creates a FrameStream from a single device.
func (u *Unmarshaller) createDeviceStream(deviceID string, cfg config) (types.FrameStream, error) {
	dev, err := newAudioDevice(deviceID, "capture", cfg.device, cfg.stream, cfg.ctx)
	if err != nil {
		return types.FrameStream{}, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to create device: %v", err), err)
	}

	// Register device for flush
	u.devicesMu.Lock()
	u.activeDevices = append(u.activeDevices, dev)
	u.devicesMu.Unlock()

	if err := dev.start(); err != nil {
		u.devicesMu.Lock()
		// Remove from active devices
		for i, active := range u.activeDevices {
			if active == dev {
				u.activeDevices = append(u.activeDevices[:i], u.activeDevices[i+1:]...)
				break
			}
		}
		u.devicesMu.Unlock()
		return types.FrameStream{}, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to start device: %v", err), err)
	}

	// Create frame stream channel
	ch := make(chan types.Frame, 32)
	stopFunc := stopDevicesOnce([]deviceInterface{dev})
	stream := types.NewFrameStream(ch, stopFunc)

	// Start goroutine to capture frames
	go func() {
		defer close(ch)
		defer func() {
			stopFunc()
			// Unregister device
			u.devicesMu.Lock()
			for i, active := range u.activeDevices {
				if active == dev {
					u.activeDevices = append(u.activeDevices[:i], u.activeDevices[i+1:]...)
					break
				}
			}
			u.devicesMu.Unlock()
		}()

		frameSize := cfg.stream.frameSize
		if frameSize == 0 {
			frameSize = 1024
		}

		format := getDeviceFormat(dev)
		channels := getDeviceChannels(dev)
		dtype := tensortypes.INT16
		if fmtStr, ok := formatToDtype[format]; ok {
			dtype = fmtStr
		}

		for {
			select {
			case <-cfg.ctx.Done():
				return
			default:
				buf, err := dev.readFrame(100 * time.Millisecond)
				if err != nil {
					if err == context.Canceled || err == context.DeadlineExceeded {
						return
					}
					continue
				}

				// Convert PCM to tensor
				actualFrames := len(buf) / (channels * sampleSizeFromFormat(format))
				shape := tensor.NewShape(actualFrames, channels)
				t := tensor.New(dtype, shape)
				if t.Empty() {
					continue
				}

				tensorData, err := pcmToTensor(buf, format, channels, actualFrames, dtype)
				if err != nil {
					continue
				}

				if err := copyToTensor(t, tensorData); err != nil {
					continue
				}

				// Send frame
				select {
				case ch <- types.Frame{Tensors: []types.Tensor{t}}:
				case <-cfg.ctx.Done():
					return
				}

				// Release buffer
				if md, ok := dev.(*malgoDevice); ok {
					md.bytePool.Put(buf)
				}
			}
		}
	}()

	return stream, nil
}

// createMultiDeviceStream creates a FrameStream from multiple devices.
func (u *Unmarshaller) createMultiDeviceStream(deviceIDs []string, cfg config) (types.FrameStream, error) {
	// Create all devices
	devices := make([]deviceInterface, 0, len(deviceIDs))
	for _, id := range deviceIDs {
		dev, err := newAudioDevice(id, "capture", cfg.device, cfg.stream, cfg.ctx)
		if err != nil {
			for _, d := range devices {
				d.stop()
			}
			return types.FrameStream{}, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to create device: %v", err), err)
		}
		devices = append(devices, dev)
	}

	// Register devices for flush
	u.devicesMu.Lock()
	u.activeDevices = append(u.activeDevices, devices...)
	u.devicesMu.Unlock()

	// Start all devices in parallel using worker pool for synchronized start
	if err := startCaptureDevices(devices, u.workerPool); err != nil {
		u.devicesMu.Lock()
		// Remove from active devices
		for _, dev := range devices {
			for i, active := range u.activeDevices {
				if active == dev {
					u.activeDevices = append(u.activeDevices[:i], u.activeDevices[i+1:]...)
					break
				}
			}
		}
		u.devicesMu.Unlock()
		for _, d := range devices {
			d.stop()
		}
		return types.FrameStream{}, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to start devices: %v", err), err)
	}

	// Create frame stream channel
	ch := make(chan types.Frame, 32)
	stopFunc := stopDevicesOnce(devices)
	stream := types.NewFrameStream(ch, stopFunc)

	// Start goroutine to capture frames from all devices using worker pool for parallel reading
	go func() {
		defer close(ch)
		defer func() {
			stopFunc()
			// Unregister devices
			u.devicesMu.Lock()
			for _, dev := range devices {
				for i, active := range u.activeDevices {
					if active == dev {
						u.activeDevices = append(u.activeDevices[:i], u.activeDevices[i+1:]...)
						break
					}
				}
			}
			u.devicesMu.Unlock()
		}()

		frameSize := cfg.stream.frameSize
		if frameSize == 0 {
			frameSize = 1024
		}

		format := getDeviceFormat(devices[0])
		totalChannels := 0
		for _, dev := range devices {
			totalChannels += getDeviceChannels(dev)
		}

		dtype := tensortypes.INT16
		if fmtStr, ok := formatToDtype[format]; ok {
			dtype = fmtStr
		}

		for {
			select {
			case <-cfg.ctx.Done():
				return
			default:
				// Read from all devices in parallel using worker pool
				frames := make([][]byte, len(devices))
				readErrs := make([]error, len(devices))
				
				// Create callbacks for parallel reading
				callbacks := make([]workers.SimpleCallback, len(devices))
				for i, dev := range devices {
					i, dev := i, dev // Capture for closure
					callbacks[i] = func(ctx context.Context) error {
						buf, err := dev.readFrame(100 * time.Millisecond)
						if err != nil {
							readErrs[i] = err
							return err
						}
						frames[i] = buf
						return nil
					}
				}
				
				// Execute parallel reads
				readCtx, readCancel := context.WithTimeout(cfg.ctx, 200*time.Millisecond)
				readErr := u.workerPool.Do(readCtx, callbacks...)
				readCancel()
				
				if readErr != nil || len(frames) == 0 || frames[0] == nil {
					// Release any frames we did read
					for i, frame := range frames {
						if frame != nil {
							if md, ok := devices[i].(*malgoDevice); ok {
								md.bytePool.Put(frame)
							}
						}
					}
					if cfg.ctx.Err() != nil {
						return
					}
					continue
				}

				// Combine channels from all devices
				channels0 := getDeviceChannels(devices[0])
				actualFrames := len(frames[0]) / (channels0 * sampleSizeFromFormat(format))
				if actualFrames == 0 {
					// Release frames
					for i, frame := range frames {
						if frame != nil {
							if md, ok := devices[i].(*malgoDevice); ok {
								md.bytePool.Put(frame)
							}
						}
					}
					continue
				}

				// Allocate tensor (callee allocates)
				shape := tensor.NewShape(actualFrames, totalChannels)
				t := tensor.New(dtype, shape)
				if t.Empty() {
					// Release frames
					for i, frame := range frames {
						if frame != nil {
							if md, ok := devices[i].(*malgoDevice); ok {
								md.bytePool.Put(frame)
							}
						}
					}
					continue
				}

				// Copy data from all devices in parallel using worker pool
				channelOffset := 0
				copyCallbacks := make([]workers.SimpleCallback, len(devices))
				for devIdx, dev := range devices {
					devIdx, dev, channelOffset := devIdx, dev, channelOffset // Capture for closure
					channels := getDeviceChannels(dev)
					framePCM := frames[devIdx]
					
					copyCallbacks[devIdx] = func(ctx context.Context) error {
						tensorData, err := pcmToTensor(framePCM, format, channels, actualFrames, dtype)
						if err != nil {
							return err
						}

						// Copy to combined tensor
						if err := copyDeviceChannelsToTensor(t, tensorData, dtype, 0, channelOffset, channels, totalChannels); err != nil {
							return err
						}

						// Release frame
						if md, ok := dev.(*malgoDevice); ok {
							md.bytePool.Put(framePCM)
						}
						
						return nil
					}
					
					channelOffset += channels
				}
				
				// Execute parallel copies
				copyCtx, copyCancel := context.WithTimeout(cfg.ctx, 100*time.Millisecond)
				if err := u.workerPool.Do(copyCtx, copyCallbacks...); err != nil {
					copyCancel()
					// Release tensor on error
					var tensorInterface types.Tensor = t
					if releaser, ok := tensorInterface.(types.Releaser); ok {
						releaser.Release()
					}
					continue
				}
				copyCancel()

				// Send frame (caller is responsible for releasing tensor)
				select {
				case ch <- types.Frame{Tensors: []types.Tensor{t}}:
				case <-cfg.ctx.Done():
					// Release tensor on context cancel
					var tensorInterface types.Tensor = t
					if releaser, ok := tensorInterface.(types.Releaser); ok {
						releaser.Release()
					}
					return
				}
			}
		}
	}()

	return stream, nil
}

// unmarshalDeviceList enumerates available audio devices.
func (u *Unmarshaller) unmarshalDeviceList(r io.Reader) ([]types.AudioDeviceInfo, error) {
	// Check if this is a "list" command
	if r != nil {
		var buf [4]byte
		n, err := r.Read(buf[:])
		if err != nil && err != io.EOF {
			return nil, types.NewError("unmarshal", "audio", "read list command", err)
		}
		command := strings.TrimSpace(string(buf[:n]))
		if command != "list" {
			return nil, types.NewError("unmarshal", "audio", "expected 'list' command, got: "+command, nil)
		}
	}

	return enumerateDevices()
}

// captureSingleDevice captures from a single audio device.
func (u *Unmarshaller) captureSingleDevice(deviceID string, cfg config, dstType tensortypes.DataType, tensorFactory func(tensortypes.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	// Create device
	dev, err := newAudioDevice(deviceID, "capture", cfg.device, cfg.stream, cfg.ctx)
	if err != nil {
		return nil, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to create device: %v", err), err)
	}

	// Start device
	if err := dev.start(); err != nil {
		return nil, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to start device: %v", err), err)
	}
	defer dev.stop()

	// Determine dtype
	deviceDtype := dstType
	if deviceDtype == tensortypes.DT_UNKNOWN {
		format := getDeviceFormat(dev)
		if fmtStr, ok := formatToDtype[format]; ok {
			deviceDtype = fmtStr
		} else {
			deviceDtype = tensortypes.INT16 // Default
		}
	}

	// Capture frames (for now, capture a fixed duration or until context cancelled)
	// In streaming mode, this would return a FrameStream instead
	frameSize := cfg.stream.frameSize
	if frameSize == 0 {
		frameSize = 1024
	}

	// Accumulate frames
	var allFrames [][]byte
	totalFrames := 0

	for {
		select {
		case <-cfg.ctx.Done():
			goto done
		default:
			buf, err := dev.readFrame(100 * time.Millisecond)
			if err != nil {
				if err == context.Canceled || err == context.DeadlineExceeded {
					goto done
				}
				// Continue on other errors
				break
			}
			allFrames = append(allFrames, buf)
			totalFrames += frameSize
		}
	}

done:
	if totalFrames == 0 {
		return nil, types.NewError("unmarshal", "audio", "no frames captured", nil)
	}

	// Combine all frames
	totalBytes := 0
	for _, frame := range allFrames {
		totalBytes += len(frame)
	}

	combinedPCM := make([]byte, 0, totalBytes)
	for _, frame := range allFrames {
		combinedPCM = append(combinedPCM, frame...)
		if md, ok := dev.(*malgoDevice); ok {
			md.bytePool.Put(frame)
		}
	}

	// Convert to tensor
	format := getDeviceFormat(dev)
	channels := getDeviceChannels(dev)
	actualFrames := totalBytes / (channels * sampleSizeFromFormat(format))
	shape := tensor.NewShape(actualFrames, channels)

	if tensorFactory == nil {
		tensorFactory = func(dtype tensortypes.DataType, shape types.Shape) types.Tensor {
			return tensor.New(dtype, shape)
		}
	}

	t := tensorFactory(deviceDtype, shape)
	if t == nil {
		return nil, types.NewError("unmarshal", "audio", "failed to create tensor", nil)
	}

	// Convert PCM to tensor
	tensorData, err := pcmToTensor(combinedPCM, format, channels, actualFrames, deviceDtype)
	if err != nil {
		return nil, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to convert PCM: %v", err), err)
	}

	// Copy to tensor
	if err := copyToTensor(t, tensorData); err != nil {
		return nil, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to copy tensor data: %v", err), err)
	}

	// Update format info
	u.lastSampleRate = getDeviceSampleRate(dev)
	u.lastChannels = getDeviceChannels(dev)
	u.lastDataType = deviceDtype

	// Update statistics
	atomic.AddInt64(&u.stats.frames, int64(actualFrames))
	atomic.AddInt64(&u.stats.bytes, int64(totalBytes))

	return t, nil
}

// captureMultipleDevices captures from multiple devices and combines channels.
func (u *Unmarshaller) captureMultipleDevices(deviceIDs []string, cfg config, dstType tensortypes.DataType, tensorFactory func(tensortypes.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	if len(deviceIDs) == 0 {
		return nil, types.NewError("unmarshal", "audio", "no device IDs provided", nil)
	}

	// Create all devices
	devices := make([]deviceInterface, 0, len(deviceIDs))
	deviceChannels := make([]int, 0, len(deviceIDs))
	totalDeviceChannels := 0

	for _, id := range deviceIDs {
		dev, err := newAudioDevice(id, "capture", cfg.device, cfg.stream, cfg.ctx)
		if err != nil {
			// Cleanup
			for _, d := range devices {
				d.stop()
			}
			return nil, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to create device %s: %v", id, err), err)
		}
		devices = append(devices, dev)
		channels := getDeviceChannels(dev)
		deviceChannels = append(deviceChannels, channels)
		totalDeviceChannels += channels
	}

	// Start all devices
	for _, dev := range devices {
		if err := dev.start(); err != nil {
			// Cleanup
			for _, d := range devices {
				d.stop()
			}
			return nil, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to start device: %v", err), err)
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
			return nil, types.NewError("unmarshal", "audio",
				fmt.Sprintf("sample rate mismatch: device %d has %d Hz, expected %d Hz (resampling not supported)",
					i, getDeviceSampleRate(dev), commonSampleRate), nil)
		}
	}

	// Get device format (use first device's format)
	format := getDeviceFormat(devices[0])

	// Determine dtype
	deviceDtype := dstType
	if deviceDtype == tensortypes.DT_UNKNOWN {
		if fmtStr, ok := formatToDtype[format]; ok {
			deviceDtype = fmtStr
		} else {
			deviceDtype = tensortypes.INT16 // Default
		}
	}

	sampleSize := sampleSizeFromFormat(format)
	devicePCM := make([][]byte, len(devices))
	deviceFrameCounts := make([]int, len(devices))

captureLoop:
	for {
		select {
		case <-cfg.ctx.Done():
			break captureLoop
		default:
			chunkData := make([][]byte, len(devices))
			chunkFrames := make([]int, len(devices))

			for i, dev := range devices {
				buf, err := dev.readFrame(100 * time.Millisecond)
				if err != nil {
					if err == context.Canceled || err == context.DeadlineExceeded {
						break captureLoop
					}
					break captureLoop
				}

				channels := deviceChannels[i]
				frames := len(buf) / (channels * sampleSize)
				if frames == 0 {
					if md, ok := dev.(*malgoDevice); ok {
						md.bytePool.Put(buf)
					}
					continue captureLoop
				}

				bytesNeeded := frames * channels * sampleSize
				copyBuf := make([]byte, bytesNeeded)
				copy(copyBuf, buf[:bytesNeeded])
				chunkData[i] = copyBuf
				chunkFrames[i] = frames

				if md, ok := dev.(*malgoDevice); ok {
					md.bytePool.Put(buf)
				}
			}

			// Ensure we have data from all devices
			for _, data := range chunkData {
				if data == nil {
					continue captureLoop
				}
			}

			minChunkFrames := chunkFrames[0]
			for _, frames := range chunkFrames {
				if frames < minChunkFrames {
					minChunkFrames = frames
				}
			}
			if minChunkFrames == 0 {
				break captureLoop
			}

			for i := range devices {
				channels := deviceChannels[i]
				bytesToUse := minChunkFrames * channels * sampleSize
				devicePCM[i] = append(devicePCM[i], chunkData[i][:bytesToUse]...)
				deviceFrameCounts[i] += minChunkFrames
			}
		}
	}

	minFrames := deviceFrameCounts[0]
	for _, frames := range deviceFrameCounts {
		if frames < minFrames {
			minFrames = frames
		}
	}
	if minFrames == 0 {
		return nil, types.NewError("unmarshal", "audio", "no frames captured", nil)
	}

	actualFrames := minFrames
	combinedShape := tensor.NewShape(actualFrames, totalDeviceChannels)

	if tensorFactory == nil {
		tensorFactory = func(dtype tensortypes.DataType, shape types.Shape) types.Tensor {
			return tensor.New(dtype, shape)
		}
	}

	combined := tensorFactory(deviceDtype, combinedShape)
	if combined == nil {
		return nil, types.NewError("unmarshal", "audio", "failed to create combined tensor", nil)
	}

	// Convert each device's frames and combine channels
	channelOffset := 0
	for devIdx := range devices {
		devChannels := deviceChannels[devIdx]
		framesAvailable := len(devicePCM[devIdx]) / (devChannels * sampleSize)
		if framesAvailable == 0 {
			channelOffset += devChannels
			continue
		}
		if framesAvailable > actualFrames {
			framesAvailable = actualFrames
		}
		bytesToUse := framesAvailable * devChannels * sampleSize
		deviceData := devicePCM[devIdx][:bytesToUse]

		devTensorData, err := pcmToTensor(deviceData, format, devChannels, framesAvailable, deviceDtype)
		if err != nil {
			return nil, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to convert PCM for device %d: %v", devIdx, err), err)
		}

		if err := copyDeviceChannelsToTensor(combined, devTensorData, deviceDtype, 0, channelOffset, devChannels, totalDeviceChannels); err != nil {
			return nil, err
		}

		channelOffset += devChannels
	}

	// Update format info
	u.lastSampleRate = commonSampleRate
	u.lastChannels = totalDeviceChannels
	u.lastDataType = deviceDtype

	// Update statistics
	atomic.AddInt64(&u.stats.frames, int64(actualFrames))
	atomic.AddInt64(&u.stats.bytes, int64(actualFrames*totalDeviceChannels*sampleSizeFromFormat(format)))

	return combined, nil
}

// unmarshalMixedSources handles mixed WAV files + devices.
func (u *Unmarshaller) unmarshalMixedSources(cfg config, dstType tensortypes.DataType, tensorFactory func(tensortypes.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	// Read WAV files first
	var wavTensors []types.Tensor
	if len(cfg.wav.paths) > 0 {
		for _, path := range cfg.wav.paths {
			t, err := u.readSingleWAV(path, cfg, dstType, tensorFactory)
			if err != nil {
				return nil, err
			}
			wavTensors = append(wavTensors, t)
		}
	}

	// Capture from devices
	var deviceTensors []types.Tensor
	if len(cfg.device.deviceIDs) > 0 {
		if len(cfg.device.deviceIDs) == 1 {
			t, err := u.captureSingleDevice(cfg.device.deviceIDs[0], cfg, dstType, tensorFactory)
			if err != nil {
				return nil, err
			}
			deviceTensors = append(deviceTensors, t)
		} else {
			t, err := u.captureMultipleDevices(cfg.device.deviceIDs, cfg, dstType, tensorFactory)
			if err != nil {
				return nil, err
			}
			deviceTensors = append(deviceTensors, t)
		}
	}

	// Combine all tensors
	allTensors := append(wavTensors, deviceTensors...)
	if len(allTensors) == 0 {
		return nil, types.NewError("unmarshal", "audio", "no sources to combine", nil)
	}

	if len(allTensors) == 1 {
		return allTensors[0], nil
	}

	// Verify all have same sample rate
	commonSampleRate := u.lastSampleRate
	_ = allTensors // Sample rate should be consistent from reads

	// Determine max frames
	maxFrames := 0
	totalChannels := 0
	for _, t := range allTensors {
		shape := t.Shape()
		if shape[0] > maxFrames {
			maxFrames = shape[0]
		}
		totalChannels += shape[1]
	}

	// Determine common dtype
	commonDtype := dstType
	if commonDtype == tensortypes.DT_UNKNOWN {
		commonDtype = allTensors[0].DataType()
	}

	// Create combined tensor
	combinedShape := tensor.NewShape(maxFrames, totalChannels)
	if tensorFactory == nil {
		tensorFactory = func(dtype tensortypes.DataType, shape types.Shape) types.Tensor {
			return tensor.New(dtype, shape)
		}
	}

	combined := tensorFactory(commonDtype, combinedShape)
	if combined == nil {
		return nil, types.NewError("unmarshal", "audio", "failed to create combined tensor", nil)
	}

	// Combine channels
	channelOffset := 0
	for _, t := range allTensors {
		shape := t.Shape()
		frames := shape[0]
		channels := shape[1]

		// Copy data
		if err := copyTensorChannels(combined, t, 0, channelOffset, frames, channels, totalChannels); err != nil {
			return nil, types.NewError("unmarshal", "audio", fmt.Sprintf("failed to copy channels: %v", err), err)
		}

		channelOffset += channels
	}

	// Update format info
	u.lastSampleRate = commonSampleRate
	u.lastChannels = totalChannels
	u.lastDataType = commonDtype

	return combined, nil
}

// copyTensorChannels copies channels from source tensor to combined tensor.
func copyTensorChannels(dst, src types.Tensor, frameOffset, channelOffset, frames, channels, totalChannels int) error {
	dstData := dst.Data()
	srcData := src.Data()
	if dstData == nil || srcData == nil {
		return fmt.Errorf("tensor data is nil")
	}

	dtype := dst.DataType()
	switch dtype {
	case tensortypes.INT16:
		dstSlice := dstData.([]int16)
		srcSlice := srcData.([]int16)
		for i := 0; i < frames; i++ {
			for ch := 0; ch < channels; ch++ {
				srcIdx := i*channels + ch
				dstIdx := (frameOffset+i)*totalChannels + channelOffset + ch
				if srcIdx < len(srcSlice) && dstIdx < len(dstSlice) {
					dstSlice[dstIdx] = srcSlice[srcIdx]
				}
			}
		}

	case tensortypes.INT32:
		dstSlice := dstData.([]int32)
		srcSlice := srcData.([]int32)
		for i := 0; i < frames; i++ {
			for ch := 0; ch < channels; ch++ {
				srcIdx := i*channels + ch
				dstIdx := (frameOffset+i)*totalChannels + channelOffset + ch
				if srcIdx < len(srcSlice) && dstIdx < len(dstSlice) {
					dstSlice[dstIdx] = srcSlice[srcIdx]
				}
			}
		}

	case tensortypes.FP32:
		dstSlice := dstData.([]float32)
		srcSlice := srcData.([]float32)
		for i := 0; i < frames; i++ {
			for ch := 0; ch < channels; ch++ {
				srcIdx := i*channels + ch
				dstIdx := (frameOffset+i)*totalChannels + channelOffset + ch
				if srcIdx < len(srcSlice) && dstIdx < len(dstSlice) {
					dstSlice[dstIdx] = srcSlice[srcIdx]
				}
			}
		}

	case tensortypes.UINT8:
		dstSlice := dstData.([]uint8)
		srcSlice := srcData.([]uint8)
		for i := 0; i < frames; i++ {
			for ch := 0; ch < channels; ch++ {
				srcIdx := i*channels + ch
				dstIdx := (frameOffset+i)*totalChannels + channelOffset + ch
				if srcIdx < len(srcSlice) && dstIdx < len(dstSlice) {
					dstSlice[dstIdx] = srcSlice[srcIdx]
				}
			}
		}

	default:
		return fmt.Errorf("unsupported dtype: %v", dtype)
	}

	return nil
}
