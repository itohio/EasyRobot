//go:build !wasm
// +build !wasm

package audio

import (
	"context"
	"crypto/sha1"
	"encoding/base64"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/gen2brain/malgo"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/smallnest/ringbuffer"
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
		// Log callback - can be configured later
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
	*buffer.Pool[byte]
	sampleSize int
	channels   int
	sampleRate int
}

func newPCMBufferPool(channels, sampleRate, sampleSize int) *pcmBufferPool {
	pool := &pcmBufferPool{
		Pool:       &buffer.Pool[byte]{},
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

// chanReader reads from channel with ring buffer smoothing.
type chanReader struct {
	ch   chan []byte
	to   time.Duration
	buf  *ringbuffer.RingBuffer
	pool *pcmBufferPool
}

func newChanReader(pool *pcmBufferPool, ch chan []byte, timeout time.Duration) *chanReader {
	return &chanReader{
		ch:   ch,
		buf:  ringbuffer.New(pool.secondSize() * 3), // 3 seconds
		to:   timeout,
		pool: pool,
	}
}

func (r *chanReader) read(dst []byte) (n int, err error) {
	ctx, cancel := context.WithTimeout(context.Background(), r.to)
	defer cancel()

	for r.buf.Length() < len(dst) {
		select {
		case data, ok := <-r.ch:
			if !ok {
				return r.buf.Read(dst)
			}
			r.buf.Write(data)
			r.pool.Put(data)
		case <-ctx.Done():
			return r.buf.Read(dst)
		}
	}
	return r.buf.Read(dst)
}

// deviceStats tracks statistics for a single audio device.
type deviceStats struct {
	frames         int64 // Total frames processed
	bytes          int64 // Total bytes processed
	droppedBuffers int64 // Buffers dropped due to queue full
	queueLen       int64 // Current queue length
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

	pcmIn        chan []byte
	pcmOut       chan []byte
	bytePool     *pcmBufferPool
	outputReader *chanReader
	inputReader  *chanReader

	ctx    context.Context
	cancel context.CancelFunc

	format     string
	sampleRate int
	channels   int
	frameSize  int
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
		pcmIn:      make(chan []byte, 32),
		pcmOut:     make(chan []byte, 32),
		bytePool:   newPCMBufferPool(channels, sampleRate, sampleSize),
		sampleRate: sampleRate,
		channels:   channels,
		format:     format,
		frameSize:  frameSize,
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

	// Create readers
	d.outputReader = newChanReader(d.bytePool, d.pcmOut, 20*time.Millisecond)
	d.inputReader = newChanReader(d.bytePool, d.pcmIn, time.Second)

	// Setup callbacks
	callbacks := malgo.DeviceCallbacks{
		Data: func(output, input []byte, framecount uint32) {
			// Playback (output)
			if output != nil && len(output) > 0 {
				n, err := d.outputReader.read(output)
				if err == nil && n > 0 {
					atomic.AddInt64(&d.stats.bytes, int64(n))
					atomic.AddInt64(&d.stats.frames, int64(framecount))
				}
			}

			// Capture (input)
			if input != nil && len(input) > 0 {
				buf := d.bytePool.makeRaw(int(framecount))
				copy(buf, input)

				select {
				case d.pcmIn <- buf:
					atomic.AddInt64(&d.stats.frames, int64(framecount))
					atomic.AddInt64(&d.stats.bytes, int64(len(buf)))
					atomic.AddInt64(&d.stats.queueLen, 1)
				default:
					atomic.AddInt64(&d.stats.droppedBuffers, 1)
					d.bytePool.Put(buf)
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
		go dev.Start()
	} else {
		err = dev.Start()
		if err != nil {
			return fmt.Errorf("failed to start device: %w", err)
		}
	}

	return nil
}

func (d *malgoDevice) stop() error {
	if d.dev == nil {
		return nil
	}

	d.cancel()
	d.dev.Stop()
	d.dev.Uninit()
	d.dev = nil

	close(d.pcmIn)
	close(d.pcmOut)

	return nil
}

func (d *malgoDevice) health() map[string]int64 {
	return map[string]int64{
		"frames":         atomic.LoadInt64(&d.stats.frames),
		"bytes":          atomic.LoadInt64(&d.stats.bytes),
		"droppedBuffers": atomic.LoadInt64(&d.stats.droppedBuffers),
		"queueLen":       atomic.LoadInt64(&d.stats.queueLen),
	}
}

func (d *malgoDevice) readFrame(timeout time.Duration) ([]byte, error) {
	select {
	case buf := <-d.pcmIn:
		atomic.AddInt64(&d.stats.queueLen, -1)
		return buf, nil
	case <-time.After(timeout):
		return nil, fmt.Errorf("timeout reading from device %s", d.id)
	case <-d.ctx.Done():
		return nil, d.ctx.Err()
	}
}

func (d *malgoDevice) writeFrame(buf []byte) error {
	select {
	case d.pcmOut <- buf:
		return nil
	default:
		atomic.AddInt64(&d.stats.droppedBuffers, 1)
		return fmt.Errorf("queue full for device %s", d.id)
	case <-d.ctx.Done():
		return d.ctx.Err()
	}
}
