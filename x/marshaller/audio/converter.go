package audio

import (
	"context"
	"fmt"
	"sync"
)

// converter handles byte-to-sample conversion with preallocated scratch buffers.
// Uses goroutine-based pipeline: bytes channel -> converter -> sampleRing
type converter struct {
	format     string
	channels   int
	sampleRing *sampleRing

	// Preallocated scratch buffers (fixed size, reused)
	scratchBytes   []byte
	scratchInt16   []int16
	scratchInt32   []int32
	scratchFloat32 []float32
	scratchUint8   []uint8

	// Channel for receiving bytes from malgo (provided by device)
	bytesChan chan []byte

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// newConverter creates a new converter with preallocated buffers.
// bytesChan is the channel provided by the device for input bytes.
func newConverter(format string, channels, frameSize int, sampleRing *sampleRing, bytesChan chan []byte, ctx context.Context) *converter {
	// Preallocate scratch buffers for maximum frame size
	maxBytesPerFrame := frameSize * channels * 4 // Worst case: 4 bytes per sample
	maxSamplesPerFrame := frameSize * channels

	convCtx, cancel := context.WithCancel(ctx)

	return &converter{
		format:         format,
		channels:       channels,
		sampleRing:     sampleRing,
		scratchBytes:   make([]byte, maxBytesPerFrame),
		scratchInt16:   make([]int16, maxSamplesPerFrame),
		scratchInt32:   make([]int32, maxSamplesPerFrame),
		scratchFloat32: make([]float32, maxSamplesPerFrame),
		scratchUint8:   make([]uint8, maxSamplesPerFrame),
		bytesChan:      bytesChan,
		ctx:            convCtx,
		cancel:         cancel,
	}
}

// start begins the converter goroutine.
func (c *converter) start() {
	c.wg.Add(1)
	go c.run()
}

// stop stops the converter goroutine.
// Note: does not close bytesChan as it's owned by the device.
func (c *converter) stop() {
	c.cancel()
	c.wg.Wait()
}

// run is the converter goroutine that processes bytes and converts to samples.
// WriteSamples is non-blocking - writes as much as possible, drops remainder if buffer full.
func (c *converter) run() {
	defer c.wg.Done()

	for {
		select {
		case <-c.ctx.Done():
			return
		case pcmData, ok := <-c.bytesChan:
			if !ok {
				return
			}
			if len(pcmData) == 0 {
				bytePool.Put(pcmData)
				continue
			}

			// Convert bytes to samples and write to sampleRing (non-blocking)
			// WriteSamples writes as much as possible, drops remainder if buffer full
			_, err := c.sampleRing.WriteSamples(pcmData, c.format, c.scratchInt16, c.scratchInt32, c.scratchFloat32, c.scratchUint8)
			if err != nil {
				// Log error but continue
			}

			// Return buffer to pool
			bytePool.Put(pcmData)
		}
	}
}

// writeBytes sends bytes to the converter (non-blocking, returns buffer to pool if full).
func (c *converter) writeBytes(pcmData []byte) error {
	select {
	case c.bytesChan <- pcmData:
		return nil
	case <-c.ctx.Done():
		bytePool.Put(pcmData)
		return c.ctx.Err()
	default:
		// Channel full - return buffer to pool and drop
		bytePool.Put(pcmData)
		return fmt.Errorf("converter channel full")
	}
}

// reverseConverter handles sample-to-byte conversion with preallocated scratch buffers.
// Uses goroutine-based pipeline: sampleRing -> converter -> bytes channel
type reverseConverter struct {
	format     string
	channels   int
	sampleRing *sampleRing

	// Preallocated scratch buffers (fixed size, reused)
	scratchBytes   []byte
	scratchInt16   []int16
	scratchInt32   []int32
	scratchFloat32 []float32
	scratchUint8   []uint8

	// Channel for sending bytes to malgo
	bytesChan chan []byte

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// newReverseConverter creates a new reverse converter with preallocated buffers.
func newReverseConverter(format string, channels, frameSize int, sampleRing *sampleRing, ctx context.Context) *reverseConverter {
	// Preallocate scratch buffers for maximum frame size
	maxBytesPerFrame := frameSize * channels * 4 // Worst case: 4 bytes per sample
	maxSamplesPerFrame := frameSize * channels

	convCtx, cancel := context.WithCancel(ctx)

	return &reverseConverter{
		format:         format,
		channels:       channels,
		sampleRing:     sampleRing,
		scratchBytes:   make([]byte, maxBytesPerFrame),
		scratchInt16:   make([]int16, maxSamplesPerFrame),
		scratchInt32:   make([]int32, maxSamplesPerFrame),
		scratchFloat32: make([]float32, maxSamplesPerFrame),
		scratchUint8:   make([]uint8, maxSamplesPerFrame),
		bytesChan:      make(chan []byte, 32),
		ctx:            convCtx,
		cancel:         cancel,
	}
}

// start begins the reverse converter goroutine.
func (rc *reverseConverter) start() {
	rc.wg.Add(1)
	go rc.run()
}

// stop stops the reverse converter goroutine.
func (rc *reverseConverter) stop() {
	rc.cancel()
	close(rc.bytesChan)
	rc.wg.Wait()
}

// run is the reverse converter goroutine that reads samples and converts to bytes.
func (rc *reverseConverter) run() {
	defer rc.wg.Done()

	frameSize := len(rc.scratchBytes) / (rc.channels * sampleSizeFromFormat(rc.format))
	if frameSize == 0 {
		frameSize = 1024
	}

	for {
		select {
		case <-rc.ctx.Done():
			return
		default:
			// Read samples and convert to bytes using preallocated scratch buffers
			framesRead, err := rc.sampleRing.ReadSamples(rc.scratchBytes, rc.format, frameSize, rc.scratchInt16, rc.scratchInt32, rc.scratchFloat32, rc.scratchUint8)
			if err != nil || framesRead == 0 {
				// No data available or error - wait a bit
				select {
				case <-rc.ctx.Done():
					return
				default:
					continue
				}
			}

			// Calculate actual bytes written
			sampleSize := sampleSizeFromFormat(rc.format)
			bytesWritten := framesRead * rc.channels * sampleSize

			// Get buffer from pool and copy (minimal copy - only what we need)
			pcmData := bytePool.Get(bytesWritten)
			copy(pcmData, rc.scratchBytes[:bytesWritten])

			// Send to channel (non-blocking)
			select {
			case rc.bytesChan <- pcmData:
			case <-rc.ctx.Done():
				bytePool.Put(pcmData)
				return
			default:
				// Channel full - return buffer to pool and drop
				bytePool.Put(pcmData)
			}
		}
	}
}

// readBytes reads bytes from the converter (non-blocking).
func (rc *reverseConverter) readBytes() ([]byte, error) {
	select {
	case pcmData, ok := <-rc.bytesChan:
		if !ok {
			return nil, fmt.Errorf("converter channel closed")
		}
		return pcmData, nil
	case <-rc.ctx.Done():
		return nil, rc.ctx.Err()
	default:
		return nil, nil // No data available
	}
}
