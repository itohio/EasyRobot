//go:build !wasm
// +build !wasm

package audio

import (
	"context"
	"crypto/sha1"
	"encoding/base64"
	"fmt"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/gen2brain/malgo"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/itohio/EasyRobot/x/workers"
)

var malgoFormats = map[malgo.FormatType]string{
	malgo.FormatUnknown: "?",
	malgo.FormatU8:      "U8",
	malgo.FormatS16:     "S16",
	malgo.FormatS24:     "S24",
	malgo.FormatS32:     "S32",
	malgo.FormatF32:     "F32",
}

var formatToMalgo = map[string]malgo.FormatType{
	"U8":  malgo.FormatU8,
	"S16": malgo.FormatS16,
	"S24": malgo.FormatS24,
	"S32": malgo.FormatS32,
	"F32": malgo.FormatF32,
}

var formatToDtype = map[string]tensortypes.DataType{
	"U8":  tensortypes.UINT8,
	"S16": tensortypes.INT16,
	"S24": tensortypes.INT32,
	"S32": tensortypes.INT32,
	"F32": tensortypes.FP32,
}

var dtypeToFormat = map[tensortypes.DataType]string{
	tensortypes.UINT8: "U8",
	tensortypes.INT16: "S16",
	tensortypes.INT32: "S32",
	tensortypes.FP32:  "F32",
}

// malgoContext wraps malgo context for device operations.
type malgoContext struct {
	ctx *malgo.AllocatedContext
}

var globalMalgoCtx *malgoContext

func initMalgoContext() (*malgoContext, error) {
	if globalMalgoCtx != nil {
		return globalMalgoCtx, nil
	}

	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, func(message string) {
		slog.Debug("malgo", "message", message)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize malgo context: %w", err)
	}

	globalMalgoCtx = &malgoContext{ctx: ctx}
	return globalMalgoCtx, nil
}

// enumerateDevices lists all available audio devices.
func enumerateDevices() ([]types.AudioDeviceInfo, error) {
	mctx, err := initMalgoContext()
	if err != nil {
		return nil, err
	}

	var devices []types.AudioDeviceInfo

	// Enumerate capture devices
	captureInfos, err := mctx.ctx.Devices(malgo.Capture)
	if err == nil {
		for _, info := range captureInfos {
			full, err := mctx.ctx.DeviceInfo(malgo.Capture, info.ID, malgo.Shared)
			if err != nil {
				continue
			}

			id := deviceIDFromMalgoID(info.ID)
			formats := make([]string, 0, len(full.Formats))

			// Extract formats and compute min/max from Formats array
			var minChannels, maxChannels uint32
			var minSampleRate, maxSampleRate uint32

			for i, df := range full.Formats {
				if fmtStr, ok := malgoFormats[df.Format]; ok {
					formats = append(formats, fmtStr)
				}

				if i == 0 {
					minChannels = df.Channels
					maxChannels = df.Channels
					minSampleRate = df.SampleRate
					maxSampleRate = df.SampleRate
				} else {
					if df.Channels < minChannels {
						minChannels = df.Channels
					}
					if df.Channels > maxChannels {
						maxChannels = df.Channels
					}
					if df.SampleRate < minSampleRate {
						minSampleRate = df.SampleRate
					}
					if df.SampleRate > maxSampleRate {
						maxSampleRate = df.SampleRate
					}
				}
			}

			devices = append(devices, types.AudioDeviceInfo{
				ID:            id,
				Name:          info.Name(),
				IsDefault:     full.IsDefault != 0,
				MinChannels:   int(minChannels),
				MaxChannels:   int(maxChannels),
				MinSampleRate: int(minSampleRate),
				MaxSampleRate: int(maxSampleRate),
				Formats:       formats,
				DeviceType:    "capture",
			})
		}
	}

	// Enumerate playback devices
	playbackInfos, err := mctx.ctx.Devices(malgo.Playback)
	if err == nil {
		for _, info := range playbackInfos {
			full, err := mctx.ctx.DeviceInfo(malgo.Playback, info.ID, malgo.Shared)
			if err != nil {
				continue
			}

			id := deviceIDFromMalgoID(info.ID)
			formats := make([]string, 0, len(full.Formats))

			// Extract formats and compute min/max from Formats array
			var minChannels, maxChannels uint32
			var minSampleRate, maxSampleRate uint32

			for i, df := range full.Formats {
				if fmtStr, ok := malgoFormats[df.Format]; ok {
					formats = append(formats, fmtStr)
				}

				if i == 0 {
					minChannels = df.Channels
					maxChannels = df.Channels
					minSampleRate = df.SampleRate
					maxSampleRate = df.SampleRate
				} else {
					if df.Channels < minChannels {
						minChannels = df.Channels
					}
					if df.Channels > maxChannels {
						maxChannels = df.Channels
					}
					if df.SampleRate < minSampleRate {
						minSampleRate = df.SampleRate
					}
					if df.SampleRate > maxSampleRate {
						maxSampleRate = df.SampleRate
					}
				}
			}

			devices = append(devices, types.AudioDeviceInfo{
				ID:            id,
				Name:          info.Name(),
				IsDefault:     full.IsDefault != 0,
				MinChannels:   int(minChannels),
				MaxChannels:   int(maxChannels),
				MinSampleRate: int(minSampleRate),
				MaxSampleRate: int(maxSampleRate),
				Formats:       formats,
				DeviceType:    "playback",
			})
		}
	}

	return devices, nil
}

// deviceIDFromMalgoID converts malgo device ID to string ID.
func deviceIDFromMalgoID(id malgo.DeviceID) string {
	idStr := id.String()
	hash := sha1.Sum([]byte(idStr))
	return base64.RawURLEncoding.EncodeToString(hash[:])
}

// findDeviceByID finds a malgo device by string ID.
func findDeviceByID(deviceID string, devType malgo.DeviceType) (*malgo.DeviceInfo, error) {
	mctx, err := initMalgoContext()
	if err != nil {
		return nil, err
	}

	infos, err := mctx.ctx.Devices(devType)
	if err != nil {
		return nil, fmt.Errorf("failed to enumerate devices: %w", err)
	}

	for _, info := range infos {
		id := deviceIDFromMalgoID(info.ID)
		if id == deviceID {
			return &info, nil
		}
	}

	return nil, fmt.Errorf("device not found: %s", deviceID)
}

// pcmBufferPool manages PCM byte buffers.
type pcmBufferPool struct {
	buffer.Pool[byte]
	sampleSize int
	channels   int
	sampleRate int
}

func newPCMBufferPool(channels, sampleRate, sampleSize int) *pcmBufferPool {
	pool := &pcmBufferPool{
		Pool:       buffer.Pool[byte]{},
		sampleSize: sampleSize,
		channels:   channels,
		sampleRate: sampleRate,
	}
	return pool
}

func (p *pcmBufferPool) secondSize() int {
	return p.channels * p.sampleSize * p.sampleRate
}

func (p *pcmBufferPool) makeRaw(frames int) []byte {
	size := frames * p.channels * p.sampleSize
	return p.Pool.Get(size)
}

// chanReader reads from ring buffer (for output/playback).
// For input/capture, we write directly to the ring buffer from malgo callback.
type chanReader struct {
	buf *buffer.Ring[byte]
	to  time.Duration
}

func newChanReader(pool *pcmBufferPool, timeout time.Duration) *chanReader {
	// Create ring buffer with 3 seconds capacity
	capacity := pool.secondSize() * 3
	backing := make([]byte, 0, capacity)
	ringBuf := buffer.NewRing(backing)

	return &chanReader{
		buf: ringBuf,
		to:  timeout,
	}
}

func (r *chanReader) read(dst []byte) (n int, err error) {
	// Direct read from ring buffer - no channel needed
	// If buffer doesn't have enough data, return what's available
	return r.buf.Read(dst)
}

// deviceStats tracks statistics for a single audio device.
type deviceStats struct {
	frames         int64 // Total frames processed
	bytes          int64 // Total bytes processed
	droppedBuffers int64 // Buffers dropped due to queue full
	skippedFrames  int64 // Frames skipped due to channel full (input) or buffer full (output)
	queueLen       int64 // Current queue length
	bufferFillRate int64 // Current buffer fill percentage (0-100)
}

// malgoDevice wraps a malgo device for capture/playback.
type malgoDevice struct {
	id         string
	deviceType string
	cfg        deviceConfig
	streamCfg  streamConfig
	stats      *deviceStats

	mctx     *malgoContext
	malgoCfg malgo.DeviceConfig
	dev      *malgo.Device

	pcmOut   chan []byte // Only used for playback (output)
	bytePool *pcmBufferPool

	// Input path: channel -> converter goroutine -> sampleRing
	inputBytesChan chan []byte // Channel for input bytes (non-blocking, drops if full)
	inputConverter *converter  // Converter goroutine for input

	// Sample ring buffers matching device format
	sampleRing *sampleRing // Ring buffer for samples (for input/capture devices)
	byteRing   *byteRing   // Ring buffer for bytes (for output/playback devices)

	// Flush control - atomic boolean to stop writing to sample buffers
	flushMode atomic.Bool // When true, skip writing to sample buffers and exit immediately

	// Preallocated scratch buffers for direct sample->byte conversion (zero allocations)
	scratchInt16   []int16
	scratchInt32   []int32
	scratchFloat32 []float32
	scratchUint8   []uint8
	scratchBytes   []byte

	ctx    context.Context
	cancel context.CancelFunc

	format     string
	sampleRate int
	channels   int
	frameSize  int
	dtype      tensortypes.DataType // Device sample data type
}

func newMalgoDevice(id string, deviceType string, cfg deviceConfig, streamCfg streamConfig, ctx context.Context) (*malgoDevice, error) {
	mctx, err := initMalgoContext()
	if err != nil {
		return nil, err
	}

	deviceCtx, cancel := context.WithCancel(ctx)

	// Find device info
	devType := malgo.Capture
	if deviceType == "playback" {
		devType = malgo.Playback
	}

	info, err := findDeviceByID(id, devType)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("device not found: %w", err)
	}

	// Build malgo config
	malgoCfg := malgo.DefaultDeviceConfig(devType)
	if devType == malgo.Capture {
		malgoCfg.Capture.DeviceID = info.ID.Pointer()
	} else {
		malgoCfg.Playback.DeviceID = info.ID.Pointer()
	}

	// Set format
	format := cfg.format
	if format == "" {
		format = "S16" // Default
	}
	malgoFormat, ok := formatToMalgo[format]
	if !ok {
		malgoFormat = malgo.FormatS16
		format = "S16"
	}

	// Set channels and sample rate
	channels := cfg.channels
	if channels == 0 {
		channels = 2 // Default
	}
	sampleRate := cfg.sampleRate
	if sampleRate == 0 {
		sampleRate = 48000 // Default
	}

	if devType == malgo.Capture {
		malgoCfg.Capture.Format = malgoFormat
		malgoCfg.Capture.Channels = uint32(channels)
	} else {
		malgoCfg.Playback.Format = malgoFormat
		malgoCfg.Playback.Channels = uint32(channels)
	}
	malgoCfg.SampleRate = uint32(sampleRate)
	malgoCfg.Alsa.NoMMap = 1

	// Determine frame size
	frameSize := streamCfg.frameSize
	if frameSize == 0 {
		frameSize = 1024
	}

	// Calculate sample size
	sampleSize := malgo.SampleSizeInBytes(malgoFormat)

	// Determine device dtype from format
	deviceDtype, ok := formatToDtype[format]
	if !ok {
		deviceDtype = tensortypes.INT16 // Default
	}

	dev := &malgoDevice{
		id:         id,
		deviceType: deviceType,
		cfg:        cfg,
		streamCfg:  streamCfg,
		stats:      &deviceStats{},
		mctx:       mctx,
		malgoCfg:   malgoCfg,
		ctx:        deviceCtx,
		cancel:     cancel,
		pcmOut:     make(chan []byte, 32), // Only for playback
		bytePool:   newPCMBufferPool(channels, sampleRate, sampleSize),
		sampleRate: sampleRate,
		channels:   channels,
		format:     format,
		frameSize:  frameSize,
		dtype:      deviceDtype,
	}

	// Calculate buffer size (default 1 second if not specified)
	bufferSeconds := 1
	if streamCfg.bufferSize > 0 {
		// bufferSize is in frames, convert to seconds
		bufferSeconds = streamCfg.bufferSize / sampleRate
		if bufferSeconds < 1 {
			bufferSeconds = 1
		}
	}

	// Create ring buffer based on device type
	if devType == malgo.Playback {
		// For output devices, use byteRing (stores bytes directly)
		bytesPerSecond := channels * sampleRate * sampleSize
		ringCapacityBytes := bytesPerSecond * bufferSeconds
		byteRing, err := newByteRing(format, channels, ringCapacityBytes, deviceCtx)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to create byte ring: %w", err)
		}
		dev.byteRing = byteRing
	} else {
		// For input devices, use sampleRing (stores samples)
		// Also create input channel and converter goroutine
		samplesPerSecond := channels * sampleRate
		ringCapacity := samplesPerSecond * bufferSeconds
		sampleRing, err := newSampleRing(deviceDtype, channels, ringCapacity, deviceCtx)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to create sample ring: %w", err)
		}
		dev.sampleRing = sampleRing

		// Create input channel (non-blocking, drops if full)
		// Channel size: enough for ~100ms of audio at max frame size
		channelCapacity := (sampleRate / 10) * channels * sampleSize / frameSize
		if channelCapacity < 32 {
			channelCapacity = 32
		}
		dev.inputBytesChan = make(chan []byte, channelCapacity)

		// Create converter goroutine (uses device's inputBytesChan)
		dev.inputConverter = newConverter(format, channels, frameSize, sampleRing, dev.inputBytesChan, deviceCtx)
	}

	return dev, nil
}

func (d *malgoDevice) start() error {
	if d.dev != nil {
		return fmt.Errorf("device already started")
	}

	// Reset stats
	atomic.StoreInt64(&d.stats.frames, 0)
	atomic.StoreInt64(&d.stats.bytes, 0)
	atomic.StoreInt64(&d.stats.droppedBuffers, 0)
	atomic.StoreInt64(&d.stats.skippedFrames, 0)
	atomic.StoreInt64(&d.stats.queueLen, 0)
	atomic.StoreInt64(&d.stats.bufferFillRate, 0)

	// Start input converter goroutine if this is a capture device
	if d.inputConverter != nil {
		d.inputConverter.start()
	}

	// Preallocate scratch buffers for direct sample->byte conversion (zero allocations)
	// Size for maximum frame size to avoid allocations
	maxFrameSize := d.frameSize
	if maxFrameSize == 0 {
		maxFrameSize = 1024
	}
	maxSamples := maxFrameSize * d.channels
	maxBytes := maxFrameSize * d.channels * 4 // Worst case: 4 bytes per sample

	d.scratchBytes = bytePool.Get(maxBytes)
	switch d.dtype {
	case tensortypes.INT16:
		d.scratchInt16 = int16Pool.Get(maxSamples)
	case tensortypes.INT32:
		d.scratchInt32 = int32Pool.Get(maxSamples)
	case tensortypes.FP32:
		d.scratchFloat32 = float32Pool.Get(maxSamples)
	case tensortypes.UINT8:
		d.scratchUint8 = uint8Pool.Get(maxSamples)
	}

	// Setup callbacks
	callbacks := malgo.DeviceCallbacks{
		Data: func(output, input []byte, framecount uint32) {
			// Playback (output) - read directly from byteRing (non-blocking)
			if len(output) > 0 {
				// Read bytes directly from byteRing (non-blocking)
				n, err := d.byteRing.Read(output)
				if err == nil && n > 0 {
					atomic.AddInt64(&d.stats.bytes, int64(n))
					atomic.AddInt64(&d.stats.frames, int64(framecount))
					// Fill remainder with silence if needed
					if n < len(output) {
						for i := n; i < len(output); i++ {
							output[i] = 0
						}
					}
				} else {
					// No data available - fill with silence
					for i := range output {
						output[i] = 0
					}
				}
			}

			// Capture (input) - write to channel (non-blocking, drops if full)
			if len(input) > 0 {
				// Check flush mode - if enabled, skip writing and exit immediately
				if d.flushMode.Load() {
					return
				}

				// Get buffer from pool and copy data
				pcmData := bytePool.Get(len(input))
				copy(pcmData, input)

				// Try to send to channel (non-blocking)
				select {
				case d.inputBytesChan <- pcmData:
					// Successfully sent to converter goroutine
					atomic.AddInt64(&d.stats.frames, int64(framecount))
					atomic.AddInt64(&d.stats.bytes, int64(len(input)))
					atomic.AddInt64(&d.stats.queueLen, 1)
				default:
					// Channel full - drop frame and return buffer to pool
					bytePool.Put(pcmData)
					atomic.AddInt64(&d.stats.skippedFrames, int64(framecount))
					atomic.AddInt64(&d.stats.droppedBuffers, 1)
				}
			}
		},
	}

	// Initialize device
	dev, err := malgo.InitDevice(d.mctx.ctx.Context, d.malgoCfg, callbacks)
	if err != nil {
		return fmt.Errorf("failed to init device: %w", err)
	}
	d.dev = dev

	// Start device
	if d.malgoCfg.DeviceType == malgo.Playback {
		// Device callback reads directly from byteRing - no converter goroutine needed
		go dev.Start()
		// Give device a moment to initialize
		time.Sleep(10 * time.Millisecond)
	} else {
		// Input converter goroutine already started in start()
		err = dev.Start()
		if err != nil {
			// Stop converter if device start fails
			if d.inputConverter != nil {
				d.inputConverter.stop()
			}
			return fmt.Errorf("failed to start device: %w", err)
		}
	}

	return nil
}

func (d *malgoDevice) stop() error {
	if d.dev == nil {
		return nil
	}

	// Cancel context first to unblock any blocking operations
	d.cancel()

	// Stop input converter goroutine
	if d.inputConverter != nil {
		d.inputConverter.stop()
		close(d.inputBytesChan)
	}

	// Unblock rings to wake up any waiting goroutines
	if d.sampleRing != nil {
		d.sampleRing.unblock()
	}
	if d.byteRing != nil {
		d.byteRing.unblock()
	}

	d.dev.Stop()
	d.dev.Uninit()
	d.dev = nil

	close(d.pcmOut)

	// Return scratch buffers to pools
	if d.scratchBytes != nil {
		bytePool.Put(d.scratchBytes)
		d.scratchBytes = nil
	}
	if d.scratchInt16 != nil {
		int16Pool.Put(d.scratchInt16)
		d.scratchInt16 = nil
	}
	if d.scratchInt32 != nil {
		int32Pool.Put(d.scratchInt32)
		d.scratchInt32 = nil
	}
	if d.scratchFloat32 != nil {
		float32Pool.Put(d.scratchFloat32)
		d.scratchFloat32 = nil
	}
	if d.scratchUint8 != nil {
		uint8Pool.Put(d.scratchUint8)
		d.scratchUint8 = nil
	}

	return nil
}

// flush clears all buffers and enables flush mode.
func (d *malgoDevice) flush() {
	// Enable flush mode - stops writing to sample buffers
	d.flushMode.Store(true)

	// Clear all ring buffers
	if d.sampleRing != nil {
		d.sampleRing.Reset()
	}
	if d.byteRing != nil {
		d.byteRing.Reset()
	}

	// Drain input channel
	for {
		select {
		case pcmData := <-d.inputBytesChan:
			bytePool.Put(pcmData)
		default:
			goto drainDone
		}
	}
drainDone:

	// Empty channels
	for {
		select {
		case <-d.pcmOut:
		default:
			goto emptyDone
		}
	}
emptyDone:
	// Disable flush mode after clearing
	d.flushMode.Store(false)
}

// startCapture synchronizes start of all devices for simultaneous capture.
// Uses worker pool to start all devices in parallel.
func startCaptureDevices(devices []deviceInterface, workerPool *workers.WorkerPool) error {
	if len(devices) == 0 {
		return nil
	}

	// Create callbacks for parallel start
	callbacks := make([]workers.SimpleCallback, len(devices))
	for i, dev := range devices {
		dev := dev // Capture for closure
		callbacks[i] = func(ctx context.Context) error {
			return dev.start()
		}
	}

	// Start all devices in parallel using worker pool
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := workerPool.Do(ctx, callbacks...); err != nil {
		return fmt.Errorf("failed to start devices in parallel: %w", err)
	}

	// Small delay to ensure all devices are ready
	time.Sleep(50 * time.Millisecond)

	return nil
}

// flushDevices flushes all devices in parallel using worker pool.
func flushDevices(devices []deviceInterface, workerPool *workers.WorkerPool) error {
	if len(devices) == 0 {
		return nil
	}

	// Create callbacks for parallel flush
	callbacks := make([]workers.SimpleCallback, len(devices))
	for i, dev := range devices {
		dev := dev // Capture for closure
		callbacks[i] = func(ctx context.Context) error {
			if md, ok := dev.(*malgoDevice); ok {
				md.flush()
			}
			return nil
		}
	}

	// Flush all devices in parallel using worker pool
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := workerPool.Do(ctx, callbacks...); err != nil {
		return fmt.Errorf("failed to flush devices in parallel: %w", err)
	}

	return nil
}

func (d *malgoDevice) health() map[string]int64 {
	// Calculate buffer fill rate
	var fillRate int64
	if d.sampleRing != nil {
		len := d.sampleRing.Len()
		cap := d.sampleRing.Cap()
		if cap > 0 {
			fillRate = int64((len * 100) / cap)
		}
	} else if d.byteRing != nil {
		len := d.byteRing.Len()
		cap := d.byteRing.Cap()
		if cap > 0 {
			fillRate = int64((len * 100) / cap)
		}
	}
	atomic.StoreInt64(&d.stats.bufferFillRate, fillRate)

	return map[string]int64{
		"frames":         atomic.LoadInt64(&d.stats.frames),
		"bytes":          atomic.LoadInt64(&d.stats.bytes),
		"droppedBuffers": atomic.LoadInt64(&d.stats.droppedBuffers),
		"skippedFrames":  atomic.LoadInt64(&d.stats.skippedFrames),
		"queueLen":       atomic.LoadInt64(&d.stats.queueLen),
		"bufferFillRate": atomic.LoadInt64(&d.stats.bufferFillRate),
	}
}

// readFrame reads a frame from the device.
// Flow: malgo -> channel -> converter goroutine -> sampleRing -> return samples
func (d *malgoDevice) readFrame(timeout time.Duration) ([]byte, error) {
	// Converter goroutine handles conversion from channel to sampleRing
	// Just read from sampleRing
	frameSize := d.frameSize
	if frameSize == 0 {
		frameSize = 1024
	}

	ctx, cancel := context.WithTimeout(d.ctx, timeout)
	defer cancel()

	for {
		// Allocate tensor for reading (callee allocates)
		shape := tensor.NewShape(frameSize, d.channels)
		t := tensor.New(d.dtype, shape)
		if t.Empty() {
			return nil, fmt.Errorf("failed to allocate tensor")
		}

		// Read directly into tensor
		framesRead, err := d.sampleRing.ReadFrames(t, frameSize)
		if err == nil && framesRead > 0 {
			atomic.AddInt64(&d.stats.queueLen, -1)

			// Convert tensor data back to PCM bytes for backward compatibility
			tensorData := t.Data()
			if tensorData == nil {
				t.Release()
				return nil, fmt.Errorf("tensor data is nil")
			}

			pcmData, convErr := tensorToPCM(tensorData, d.format, d.channels, framesRead)
			if convErr != nil {
				t.Release()
				return nil, fmt.Errorf("failed to convert samples to PCM: %w", convErr)
			}

			// Release tensor (device is done with it)
			// Convert to types.Tensor interface for Releaser check
			var tensorInterface types.Tensor = t
			if releaser, ok := tensorInterface.(types.Releaser); ok {
				releaser.Release()
			}

			return pcmData, nil
		}

		// Release tensor if read failed
		var tensorInterface types.Tensor = t
		if releaser, ok := tensorInterface.(types.Releaser); ok {
			releaser.Release()
		}

		// Check if we should continue waiting
		select {
		case <-ctx.Done():
			if ctx.Err() == context.DeadlineExceeded {
				return nil, fmt.Errorf("timeout reading from device %s", d.id)
			}
			return nil, ctx.Err()
		default:
			// If buffer is empty, wait a short time before retrying
			if d.sampleRing.Empty() {
				select {
				case <-time.After(1 * time.Millisecond):
					continue
				case <-ctx.Done():
					return nil, fmt.Errorf("timeout reading from device %s", d.id)
				}
			}
		}
	}
}

// writeFrame writes a frame to the device (legacy API - converts bytes to samples).
// Flow: bytes -> samples -> sampleRing -> [goroutine converts] -> bytes channel -> malgo
func (d *malgoDevice) writeFrame(buf []byte) error {
	// Convert PCM bytes to samples and write to sampleRing
	sampleSize := sampleSizeFromFormat(d.format)
	frames := len(buf) / (sampleSize * d.channels)
	if frames == 0 {
		return fmt.Errorf("insufficient data for a frame")
	}

	// Convert PCM to samples
	samples, err := pcmToTensor(buf, d.format, d.channels, frames, d.dtype)
	if err != nil {
		return fmt.Errorf("failed to convert PCM to samples: %w", err)
	}

	// Write to sampleRing - converter goroutine will handle conversion to bytes
	_, err = d.sampleRing.Write(samples)
	if err != nil {
		return fmt.Errorf("failed to write to sampleRing: %w", err)
	}

	return nil
}

// readSamples reads samples directly from sampleRing (new API).
// Returns samples in device format (int16, int32, float32, uint8).
// Converter goroutine handles conversion from channel to sampleRing.
func (d *malgoDevice) readSamples(timeout time.Duration, maxFrames int) (any, int, error) {
	// Converter goroutine handles conversion from channel to sampleRing
	// Just read from sampleRing
	if maxFrames <= 0 {
		maxFrames = d.frameSize
		if maxFrames == 0 {
			maxFrames = 1024
		}
	}

	ctx, cancel := context.WithTimeout(d.ctx, timeout)
	defer cancel()

	for {
		// Allocate tensor for reading (callee allocates)
		shape := tensor.NewShape(maxFrames, d.channels)
		t := tensor.New(d.dtype, shape)
		if t.Empty() {
			return nil, 0, fmt.Errorf("failed to allocate tensor")
		}

		// Read directly into tensor
		framesRead, err := d.sampleRing.ReadFrames(t, maxFrames)
		if err == nil && framesRead > 0 {
			atomic.AddInt64(&d.stats.queueLen, -1)

			// Return tensor data (caller is responsible for releasing tensor)
			tensorData := t.Data()
			if tensorData == nil {
				var tensorInterface types.Tensor = t
				if releaser, ok := tensorInterface.(types.Releaser); ok {
					releaser.Release()
				}
				return nil, 0, fmt.Errorf("tensor data is nil")
			}

			// Note: Caller must release the tensor
			return tensorData, framesRead, nil
		}

		// Release tensor if read failed
		var tensorInterface types.Tensor = t
		if releaser, ok := tensorInterface.(types.Releaser); ok {
			releaser.Release()
		}

		// Check if we should continue waiting
		select {
		case <-ctx.Done():
			if ctx.Err() == context.DeadlineExceeded {
				return nil, 0, fmt.Errorf("timeout reading from device %s", d.id)
			}
			return nil, 0, ctx.Err()
		default:
			if d.sampleRing.Empty() {
				select {
				case <-time.After(1 * time.Millisecond):
					continue
				case <-ctx.Done():
					return nil, 0, fmt.Errorf("timeout reading from device %s", d.id)
				}
			}
		}
	}
}

// writeSamples writes samples from a tensor directly to byteRing (new API).
// Conversion from tensor to bytes happens in WriteFrames.
// Malgo callback reads bytes directly from byteRing (non-blocking).
func (d *malgoDevice) writeSamples(t types.Tensor) error {
	if d.byteRing == nil {
		return fmt.Errorf("byteRing not initialized for output device")
	}
	// Write directly from tensor to byteRing (converts to bytes internally)
	_, err := d.byteRing.WriteFrames(t)
	if err != nil {
		return fmt.Errorf("failed to write to byteRing: %w", err)
	}

	return nil
}
