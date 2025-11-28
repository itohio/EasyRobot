package audio

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"sync"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	convtypes "github.com/itohio/EasyRobot/x/types"
)

// ringInterface defines methods that are not type-dependent.
type ringInterface interface {
	Len() int
	Cap() int
	Available() int
	Reset()
}

// sampleRing is a type-agnostic ring buffer for audio samples.
// Uses a single any field with type dispatch pattern for optimized access.
type sampleRing struct {
	ring     any // *buffer.Ring[int16], *buffer.Ring[int32], *buffer.Ring[float32], or *buffer.Ring[uint8]
	dtype    tensortypes.DataType
	channels int
	mu       sync.Mutex    // Protects ring buffer
	notEmpty chan struct{} // Signaled when buffer becomes non-empty (buffered, size 1)
	notFull  chan struct{} // Signaled when buffer becomes non-full (buffered, size 1)
	ctx      context.Context
	cancel   context.CancelFunc
}

// newSampleRing creates a new sample ring buffer for the given data type and capacity.
// The context is used to cancel blocking operations when the marshaller/unmarshaller is closed.
func newSampleRing(dtype tensortypes.DataType, channels, capacity int, ctx context.Context) (*sampleRing, error) {
	var ring any

	switch dtype {
	case tensortypes.INT16:
		backing := make([]int16, 0, capacity)
		ring = buffer.NewRing(backing)
	case tensortypes.INT32:
		backing := make([]int32, 0, capacity)
		ring = buffer.NewRing(backing)
	case tensortypes.FP32:
		backing := make([]float32, 0, capacity)
		ring = buffer.NewRing(backing)
	case tensortypes.UINT8:
		backing := make([]uint8, 0, capacity)
		ring = buffer.NewRing(backing)
	default:
		return nil, fmt.Errorf("unsupported sample type: %v", dtype)
	}

	if ctx == nil {
		ctx = context.Background()
	}

	sr := &sampleRing{
		ring:     ring,
		dtype:    dtype,
		channels: channels,
		ctx:      ctx,
		cancel:   func() {},              // No-op since we use the provided context
		notEmpty: make(chan struct{}, 1), // Buffered channel, size 1
		notFull:  make(chan struct{}, 1), // Buffered channel, size 1
	}

	return sr, nil
}

// signalNotEmpty signals that the buffer is no longer empty (non-blocking).
func (sr *sampleRing) signalNotEmpty() {
	select {
	case sr.notEmpty <- struct{}{}:
	default:
		// Channel already has a signal pending, no need to send again
	}
}

// signalNotFull signals that the buffer is no longer full (non-blocking).
func (sr *sampleRing) signalNotFull() {
	select {
	case sr.notFull <- struct{}{}:
	default:
		// Channel already has a signal pending, no need to send again
	}
}

// unblock wakes up all waiting goroutines. Should be called when context is cancelled.
func (sr *sampleRing) unblock() {
	sr.signalNotEmpty()
	sr.signalNotFull()
}

// Write writes samples to the ring buffer using type dispatch.
func (sr *sampleRing) Write(samples any) (int, error) {
	switch r := sr.ring.(type) {
	case *buffer.Ring[int16]:
		data, ok := samples.([]int16)
		if !ok {
			return 0, fmt.Errorf("type mismatch: expected []int16, got %T", samples)
		}
		return r.Write(data)
	case *buffer.Ring[int32]:
		data, ok := samples.([]int32)
		if !ok {
			return 0, fmt.Errorf("type mismatch: expected []int32, got %T", samples)
		}
		return r.Write(data)
	case *buffer.Ring[float32]:
		data, ok := samples.([]float32)
		if !ok {
			return 0, fmt.Errorf("type mismatch: expected []float32, got %T", samples)
		}
		return r.Write(data)
	case *buffer.Ring[uint8]:
		data, ok := samples.([]uint8)
		if !ok {
			return 0, fmt.Errorf("type mismatch: expected []uint8, got %T", samples)
		}
		return r.Write(data)
	default:
		return 0, fmt.Errorf("unsupported sample type: %v", sr.dtype)
	}
}

// Read reads samples from the ring buffer using type dispatch.
func (sr *sampleRing) Read(samples any) (int, error) {
	switch r := sr.ring.(type) {
	case *buffer.Ring[int16]:
		data, ok := samples.([]int16)
		if !ok {
			return 0, fmt.Errorf("type mismatch: expected []int16, got %T", samples)
		}
		return r.Read(data)
	case *buffer.Ring[int32]:
		data, ok := samples.([]int32)
		if !ok {
			return 0, fmt.Errorf("type mismatch: expected []int32, got %T", samples)
		}
		return r.Read(data)
	case *buffer.Ring[float32]:
		data, ok := samples.([]float32)
		if !ok {
			return 0, fmt.Errorf("type mismatch: expected []float32, got %T", samples)
		}
		return r.Read(data)
	case *buffer.Ring[uint8]:
		data, ok := samples.([]uint8)
		if !ok {
			return 0, fmt.Errorf("type mismatch: expected []uint8, got %T", samples)
		}
		return r.Read(data)
	default:
		return 0, fmt.Errorf("unsupported sample type: %v", sr.dtype)
	}
}

// Len returns the number of samples currently in the buffer.
func (sr *sampleRing) Len() int {
	if r, ok := sr.ring.(ringInterface); ok {
		return r.Len()
	}
	return 0
}

// Cap returns the capacity of the ring buffer.
func (sr *sampleRing) Cap() int {
	if r, ok := sr.ring.(ringInterface); ok {
		return r.Cap()
	}
	return 0
}

// Available returns the number of samples that can be written.
func (sr *sampleRing) Available() int {
	if r, ok := sr.ring.(ringInterface); ok {
		return r.Available()
	}
	return 0
}

// Empty returns true if the buffer is empty.
func (sr *sampleRing) Empty() bool {
	return sr.Len() == 0
}

// Full returns true if the buffer is full.
func (sr *sampleRing) Full() bool {
	return sr.Available() == 0
}

// Reset clears the buffer.
func (sr *sampleRing) Reset() {
	if r, ok := sr.ring.(ringInterface); ok {
		r.Reset()
	}
}

// Dtype returns the data type of samples in this ring buffer.
func (sr *sampleRing) Dtype() tensortypes.DataType {
	return sr.dtype
}

// Channels returns the number of channels.
func (sr *sampleRing) Channels() int {
	return sr.channels
}

// ReadFrames reads frames directly into a preallocated tensor.
// Validates that tensor dtype matches sample type.
// Blocks until all requested frames are read (waits if buffer is empty).
// Returns the number of frames read.
func (sr *sampleRing) ReadFrames(t types.Tensor, numFrames int) (int, error) {
	if t == nil || t.Empty() {
		return 0, fmt.Errorf("tensor is nil or empty")
	}

	// Validate tensor shape
	shape := t.Shape()
	if len(shape) != 2 {
		return 0, fmt.Errorf("tensor must be 2D [frames, channels] for audio")
	}

	tensorChannels := shape[1]
	if tensorChannels != sr.channels {
		return 0, fmt.Errorf("channel count mismatch: tensor has %d channels, sampleRing expects %d", tensorChannels, sr.channels)
	}

	// Ensure tensor has enough capacity
	tensorFrames := shape[0]
	if tensorFrames < numFrames {
		numFrames = tensorFrames
	}

	if numFrames == 0 {
		return 0, nil
	}

	samplesNeeded := numFrames * sr.channels
	tensorData := t.Data()
	if tensorData == nil {
		return 0, fmt.Errorf("tensor data is nil")
	}

	// Read all samples, blocking if buffer is empty
	offset := 0
	for offset < samplesNeeded {
		// Check context before acquiring lock
		select {
		case <-sr.ctx.Done():
			return offset / sr.channels, sr.ctx.Err()
		default:
		}

		sr.mu.Lock()
		// Check if buffer has data
		available := sr.Len()

		if available == 0 {
			// Buffer is empty, wait for data
			sr.mu.Unlock()
			select {
			case <-sr.notEmpty:
				// Buffer has data, continue loop
			case <-sr.ctx.Done():
				return offset / sr.channels, sr.ctx.Err()
			}
			continue
		}

		// Calculate how much we can read
		toRead := samplesNeeded - offset
		if toRead > available {
			toRead = available
		}

		// Read data
		var n int
		var err error
		if t.DataType() == sr.dtype {
			// Fast path: types match
			switch r := sr.ring.(type) {
			case *buffer.Ring[int16]:
				data := tensorData.([]int16)
				n, err = r.Read(data[offset : offset+toRead])
			case *buffer.Ring[int32]:
				data := tensorData.([]int32)
				n, err = r.Read(data[offset : offset+toRead])
			case *buffer.Ring[float32]:
				data := tensorData.([]float32)
				n, err = r.Read(data[offset : offset+toRead])
			case *buffer.Ring[uint8]:
				data := tensorData.([]uint8)
				n, err = r.Read(data[offset : offset+toRead])
			}
		} else {
			// Slow path: convert types - read into temp buffer then convert
			sr.mu.Unlock()
			return sr.readFramesWithConversion(t, offset, samplesNeeded, numFrames)
		}

		if err != nil && err != io.EOF {
			sr.mu.Unlock()
			return offset / sr.channels, err
		}

		offset += n
		// Signal that buffer is no longer full
		if n > 0 {
			sr.signalNotFull()
		}
		sr.mu.Unlock()
	}

	return numFrames, nil
}

// readFramesWithConversion handles reading when types don't match
func (sr *sampleRing) readFramesWithConversion(t types.Tensor, offset, samplesNeeded, numFrames int) (int, error) {
	// Read in chunks with blocking
	tensorData := t.Data()
	readOffset := 0
	for readOffset < samplesNeeded {
		select {
		case <-sr.ctx.Done():
			return readOffset / sr.channels, sr.ctx.Err()
		default:
		}

		sr.mu.Lock()
		available := sr.Len()

		if available == 0 {
			sr.mu.Unlock()
			select {
			case <-sr.notEmpty:
				// Buffer has data, continue loop
			case <-sr.ctx.Done():
				return readOffset / sr.channels, sr.ctx.Err()
			}
			continue
		}

		toRead := samplesNeeded - readOffset
		if toRead > available {
			toRead = available
		}

		// Read into temp buffer and convert
		var n int
		var err error
		switch r := sr.ring.(type) {
		case *buffer.Ring[int16]:
			tempBuf := int16Pool.Get(toRead)
			n, err = r.Read(tempBuf[:toRead])
			if err == nil || err == io.EOF {
				convtypes.ConvertAny(tensorData, tempBuf[:n])
			}
			int16Pool.Put(tempBuf)
		case *buffer.Ring[int32]:
			tempBuf := int32Pool.Get(toRead)
			n, err = r.Read(tempBuf[:toRead])
			if err == nil || err == io.EOF {
				convtypes.ConvertAny(tensorData, tempBuf[:n])
			}
			int32Pool.Put(tempBuf)
		case *buffer.Ring[float32]:
			tempBuf := float32Pool.Get(toRead)
			n, err = r.Read(tempBuf[:toRead])
			if err == nil || err == io.EOF {
				convtypes.ConvertAny(tensorData, tempBuf[:n])
			}
			float32Pool.Put(tempBuf)
		case *buffer.Ring[uint8]:
			tempBuf := uint8Pool.Get(toRead)
			n, err = r.Read(tempBuf[:toRead])
			if err == nil || err == io.EOF {
				convtypes.ConvertAny(tensorData, tempBuf[:n])
			}
			uint8Pool.Put(tempBuf)
		}

		if err != nil && err != io.EOF {
			sr.mu.Unlock()
			return readOffset / sr.channels, err
		}

		readOffset += n
		if n > 0 {
			sr.signalNotFull()
		}
		sr.mu.Unlock()
	}

	return numFrames, nil
}

// WriteFrames writes frames from a tensor directly to the ring buffer.
// Converts from tensor type to sampleRing type automatically.
// Blocks until all data is written (waits if buffer is full).
// Returns the number of frames written.
func (sr *sampleRing) WriteFrames(t types.Tensor) (int, error) {
	if t == nil || t.Empty() {
		return 0, fmt.Errorf("tensor is nil or empty")
	}

	// Validate tensor shape
	shape := t.Shape()
	if len(shape) != 2 {
		return 0, fmt.Errorf("tensor must be 2D [frames, channels] for audio")
	}

	tensorChannels := shape[1]
	if tensorChannels != sr.channels {
		return 0, fmt.Errorf("channel count mismatch: tensor has %d channels, sampleRing expects %d", tensorChannels, sr.channels)
	}

	frames := shape[0]
	if frames == 0 {
		return 0, nil
	}

	samplesNeeded := frames * sr.channels
	tensorData := t.Data()
	if tensorData == nil {
		return 0, fmt.Errorf("tensor data is nil")
	}

	// Write all samples, blocking if buffer is full
	offset := 0
	for offset < samplesNeeded {
		select {
		case <-sr.ctx.Done():
			return offset / sr.channels, sr.ctx.Err()
		default:
		}

		// Check context before acquiring lock
		select {
		case <-sr.ctx.Done():
			return offset / sr.channels, sr.ctx.Err()
		default:
		}

		sr.mu.Lock()
		// Check if buffer has space
		available := sr.Available()

		if available == 0 {
			// Buffer is full, wait for space
			sr.mu.Unlock()
			select {
			case <-sr.notFull:
				// Buffer has space, continue loop
			case <-sr.ctx.Done():
				return offset / sr.channels, sr.ctx.Err()
			}
			continue
		}

		// Calculate how much we can write
		toWrite := samplesNeeded - offset
		if toWrite > available {
			toWrite = available
		}

		// Write data
		var n int
		var err error
		if t.DataType() == sr.dtype {
			// Fast path: types match
			switch r := sr.ring.(type) {
			case *buffer.Ring[int16]:
				data := tensorData.([]int16)
				n, err = r.Write(data[offset : offset+toWrite])
			case *buffer.Ring[int32]:
				data := tensorData.([]int32)
				n, err = r.Write(data[offset : offset+toWrite])
			case *buffer.Ring[float32]:
				data := tensorData.([]float32)
				n, err = r.Write(data[offset : offset+toWrite])
			case *buffer.Ring[uint8]:
				data := tensorData.([]uint8)
				n, err = r.Write(data[offset : offset+toWrite])
			}
		} else {
			// Slow path: convert types
			// This is more complex, so we'll write in chunks
			// For now, let's use the existing conversion logic
			// TODO: optimize this path
			sr.mu.Unlock()
			return sr.writeFramesWithConversion(t, offset, samplesNeeded)
		}

		if err != nil && err != io.ErrShortWrite {
			sr.mu.Unlock()
			return offset / sr.channels, err
		}

		offset += n
		// Signal that buffer is no longer empty
		if n > 0 {
			sr.signalNotEmpty()
		}
		sr.mu.Unlock()
	}

	return frames, nil
}

// writeFramesWithConversion handles writing when types don't match
func (sr *sampleRing) writeFramesWithConversion(t types.Tensor, offset, samplesNeeded int) (int, error) {
	tensorData := t.Data()
	writeOffset := 0
	for writeOffset < samplesNeeded {
		select {
		case <-sr.ctx.Done():
			return writeOffset / sr.channels, sr.ctx.Err()
		default:
		}

		sr.mu.Lock()
		available := sr.Available()

		if available == 0 {
			sr.mu.Unlock()
			select {
			case <-sr.notFull:
				// Buffer has space, continue loop
			case <-sr.ctx.Done():
				return writeOffset / sr.channels, sr.ctx.Err()
			}
			continue
		}

		toWrite := samplesNeeded - writeOffset
		if toWrite > available {
			toWrite = available
		}

		// Convert and write
		var n int
		var err error
		switch r := sr.ring.(type) {
		case *buffer.Ring[int16]:
			tempBuf := int16Pool.Get(toWrite)
			convtypes.ConvertAny(tempBuf[:toWrite], tensorData)
			n, err = r.Write(tempBuf[:toWrite])
			int16Pool.Put(tempBuf)
		case *buffer.Ring[int32]:
			tempBuf := int32Pool.Get(toWrite)
			convtypes.ConvertAny(tempBuf[:toWrite], tensorData)
			n, err = r.Write(tempBuf[:toWrite])
			int32Pool.Put(tempBuf)
		case *buffer.Ring[float32]:
			tempBuf := float32Pool.Get(toWrite)
			convtypes.ConvertAny(tempBuf[:toWrite], tensorData)
			n, err = r.Write(tempBuf[:toWrite])
			float32Pool.Put(tempBuf)
		case *buffer.Ring[uint8]:
			tempBuf := uint8Pool.Get(toWrite)
			convtypes.ConvertAny(tempBuf[:toWrite], tensorData)
			n, err = r.Write(tempBuf[:toWrite])
			uint8Pool.Put(tempBuf)
		}

		if err != nil && err != io.ErrShortWrite {
			sr.mu.Unlock()
			return writeOffset / sr.channels, err
		}

		writeOffset += n
		if n > 0 {
			sr.signalNotEmpty()
		}
		sr.mu.Unlock()
	}

	return (samplesNeeded) / sr.channels, nil
}

// WriteSamples converts bytes to samples and writes to the ring buffer (non-blocking).
// Writes as much as possible, drops remainder if buffer is full.
// Uses type dispatch pattern - no switches in hot loops.
// scratch buffers are preallocated to avoid allocations.
func (sr *sampleRing) WriteSamples(pcmData []byte, format string, scratchInt16 []int16, scratchInt32 []int32, scratchFloat32 []float32, scratchUint8 []uint8) (int, error) {
	switch r := sr.ring.(type) {
	case *buffer.Ring[int16]:
		return sr.writeSamplesInt16(r, pcmData, format, scratchInt16)
	case *buffer.Ring[int32]:
		return sr.writeSamplesInt32(r, pcmData, format, scratchInt32)
	case *buffer.Ring[float32]:
		return sr.writeSamplesFloat32(r, pcmData, format, scratchFloat32)
	case *buffer.Ring[uint8]:
		return sr.writeSamplesUint8(r, pcmData, format, scratchUint8)
	default:
		return 0, fmt.Errorf("unsupported sample type: %v", sr.dtype)
	}
}

// writeSamplesInt16 converts bytes to int16 samples and writes (no switches in loop).
// scratch is a preallocated buffer to avoid allocations.
func (sr *sampleRing) writeSamplesInt16(r *buffer.Ring[int16], pcmData []byte, format string, scratch []int16) (int, error) {
	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < sampleSize*sr.channels {
		return 0, fmt.Errorf("insufficient PCM data")
	}

	frames := len(pcmData) / (sampleSize * sr.channels)
	if frames == 0 {
		return 0, nil
	}

	samplesNeeded := frames * sr.channels
	if len(scratch) < samplesNeeded {
		// Fallback to pool if scratch too small
		samples := int16Pool.Get(samplesNeeded)
		defer int16Pool.Put(samples)
		return sr.writeSamplesInt16WithBuffer(r, pcmData, format, samples[:samplesNeeded], frames)
	}

	return sr.writeSamplesInt16WithBuffer(r, pcmData, format, scratch[:samplesNeeded], frames)
}

// writeSamplesInt16WithBuffer does the actual conversion (no switches in loop).
// Non-blocking: writes as much as possible, drops remainder if buffer full.
func (sr *sampleRing) writeSamplesInt16WithBuffer(r *buffer.Ring[int16], pcmData []byte, format string, samples []int16, frames int) (int, error) {
	if format == "S16" {
		for i := 0; i < frames*sr.channels; i++ {
			samples[i] = int16(binary.LittleEndian.Uint16(pcmData[i*2 : (i+1)*2]))
		}
	} else {
		return 0, fmt.Errorf("unsupported format conversion: %s to INT16", format)
	}

	// Write non-blocking - write as much as possible, drop remainder if buffer full
	samplesNeeded := frames * sr.channels
	sr.mu.Lock()
	available := r.Available()
	if available == 0 {
		// Buffer full - drop all samples
		sr.mu.Unlock()
		return 0, nil
	}

	toWrite := samplesNeeded
	if toWrite > available {
		toWrite = available
	}

	n, err := r.Write(samples[:toWrite])
	if err != nil && err != io.ErrShortWrite {
		sr.mu.Unlock()
		return 0, err
	}

	framesWritten := n / sr.channels
	if n > 0 {
		sr.signalNotEmpty()
	}
	sr.mu.Unlock()

	return framesWritten, nil
}

// writeSamplesInt32 converts bytes to int32 samples and writes (no switches in loop).
// scratch is a preallocated buffer to avoid allocations.
func (sr *sampleRing) writeSamplesInt32(r *buffer.Ring[int32], pcmData []byte, format string, scratch []int32) (int, error) {
	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < sampleSize*sr.channels {
		return 0, fmt.Errorf("insufficient PCM data")
	}

	frames := len(pcmData) / (sampleSize * sr.channels)
	if frames == 0 {
		return 0, nil
	}

	samplesNeeded := frames * sr.channels
	if len(scratch) < samplesNeeded {
		// Fallback to pool if scratch too small
		samples := int32Pool.Get(samplesNeeded)
		defer int32Pool.Put(samples)
		return sr.writeSamplesInt32WithBuffer(r, pcmData, format, samples[:samplesNeeded], frames)
	}

	return sr.writeSamplesInt32WithBuffer(r, pcmData, format, scratch[:samplesNeeded], frames)
}

// writeSamplesInt32WithBuffer does the actual conversion (no switches in loop).
func (sr *sampleRing) writeSamplesInt32WithBuffer(r *buffer.Ring[int32], pcmData []byte, format string, samples []int32, frames int) (int, error) {

	if format == "S32" {
		for i := 0; i < frames*sr.channels; i++ {
			samples[i] = int32(binary.LittleEndian.Uint32(pcmData[i*4 : (i+1)*4]))
		}
	} else if format == "S24" {
		// Convert 24-bit PCM to 32-bit int32 with proper sign extension
		// 24-bit samples are stored in 3 bytes (little-endian)
		// We expand them to 32-bit by sign-extending the MSB
		for i := 0; i < frames*sr.channels; i++ {
			// Read 3 bytes (little-endian)
			b0 := uint32(pcmData[i*3])   // LSB
			b1 := uint32(pcmData[i*3+1]) // Middle byte
			b2 := uint32(pcmData[i*3+2]) // MSB (sign bit is bit 7)

			// Combine bytes: b0 | (b1 << 8) | (b2 << 16)
			val := int32(b0 | (b1 << 8) | (b2 << 16))

			// Sign extend: if MSB of b2 is set (negative), extend sign to 32-bit
			if (b2 & 0x80) != 0 {
				// Sign extend: set bits 24-31 to 1
				val |= ^int32(0xFFFFFF) // 0xFFFFFF00
			}
			samples[i] = val
		}
	} else {
		return 0, fmt.Errorf("unsupported format conversion: %s to INT32", format)
	}

	// Write non-blocking - write as much as possible, drop remainder if buffer full
	samplesNeeded := frames * sr.channels
	sr.mu.Lock()
	available := r.Available()
	if available == 0 {
		// Buffer full - drop all samples
		sr.mu.Unlock()
		return 0, nil
	}

	toWrite := samplesNeeded
	if toWrite > available {
		toWrite = available
	}

	n, err := r.Write(samples[:toWrite])
	if err != nil && err != io.ErrShortWrite {
		sr.mu.Unlock()
		return 0, err
	}

	framesWritten := n / sr.channels
	if n > 0 {
		sr.signalNotEmpty()
	}
	sr.mu.Unlock()

	return framesWritten, nil
}

// writeSamplesFloat32 converts bytes to float32 samples and writes (no switches in loop).
// scratch is a preallocated buffer to avoid allocations.
func (sr *sampleRing) writeSamplesFloat32(r *buffer.Ring[float32], pcmData []byte, format string, scratch []float32) (int, error) {
	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < sampleSize*sr.channels {
		return 0, fmt.Errorf("insufficient PCM data")
	}

	frames := len(pcmData) / (sampleSize * sr.channels)
	if frames == 0 {
		return 0, nil
	}

	samplesNeeded := frames * sr.channels
	if len(scratch) < samplesNeeded {
		// Fallback to pool if scratch too small
		samples := float32Pool.Get(samplesNeeded)
		defer float32Pool.Put(samples)
		return sr.writeSamplesFloat32WithBuffer(r, pcmData, format, samples[:samplesNeeded], frames)
	}

	return sr.writeSamplesFloat32WithBuffer(r, pcmData, format, scratch[:samplesNeeded], frames)
}

// writeSamplesFloat32WithBuffer does the actual conversion (no switches in loop).
func (sr *sampleRing) writeSamplesFloat32WithBuffer(r *buffer.Ring[float32], pcmData []byte, format string, samples []float32, frames int) (int, error) {

	if format == "F32" {
		for i := 0; i < frames*sr.channels; i++ {
			bits := binary.LittleEndian.Uint32(pcmData[i*4 : (i+1)*4])
			samples[i] = float32(bits)
		}
	} else if format == "S16" {
		maxVal := float32(32768.0)
		for i := 0; i < frames*sr.channels; i++ {
			val := int16(binary.LittleEndian.Uint16(pcmData[i*2 : (i+1)*2]))
			samples[i] = float32(val) / maxVal
		}
	} else if format == "S32" {
		maxVal := float32(2147483648.0)
		for i := 0; i < frames*sr.channels; i++ {
			val := int32(binary.LittleEndian.Uint32(pcmData[i*4 : (i+1)*4]))
			samples[i] = float32(val) / maxVal
		}
	} else {
		return 0, fmt.Errorf("unsupported format conversion: %s to FP32", format)
	}

	// Write non-blocking - write as much as possible, drop remainder if buffer full
	samplesNeeded := frames * sr.channels
	sr.mu.Lock()
	available := r.Available()
	if available == 0 {
		// Buffer full - drop all samples
		sr.mu.Unlock()
		return 0, nil
	}

	toWrite := samplesNeeded
	if toWrite > available {
		toWrite = available
	}

	n, err := r.Write(samples[:toWrite])
	if err != nil && err != io.ErrShortWrite {
		sr.mu.Unlock()
		return 0, err
	}

	framesWritten := n / sr.channels
	if n > 0 {
		sr.signalNotEmpty()
	}
	sr.mu.Unlock()

	return framesWritten, nil
}

// writeSamplesUint8 converts bytes to uint8 samples and writes (no switches in loop).
// scratch is a preallocated buffer to avoid allocations.
func (sr *sampleRing) writeSamplesUint8(r *buffer.Ring[uint8], pcmData []byte, format string, scratch []uint8) (int, error) {
	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < sampleSize*sr.channels {
		return 0, fmt.Errorf("insufficient PCM data")
	}

	frames := len(pcmData) / (sampleSize * sr.channels)
	if frames == 0 {
		return 0, nil
	}

	samplesNeeded := frames * sr.channels
	if len(scratch) < samplesNeeded {
		// Fallback to pool if scratch too small
		samples := uint8Pool.Get(samplesNeeded)
		defer uint8Pool.Put(samples)
		return sr.writeSamplesUint8WithBuffer(r, pcmData, format, samples[:samplesNeeded], frames)
	}

	return sr.writeSamplesUint8WithBuffer(r, pcmData, format, scratch[:samplesNeeded], frames)
}

// writeSamplesUint8WithBuffer does the actual conversion (no switches in loop).
func (sr *sampleRing) writeSamplesUint8WithBuffer(r *buffer.Ring[uint8], pcmData []byte, format string, samples []uint8, frames int) (int, error) {

	if format == "U8" {
		copy(samples, pcmData[:frames*sr.channels])
	} else {
		return 0, fmt.Errorf("unsupported format conversion: %s to UINT8", format)
	}

	// Write non-blocking - write as much as possible, drop remainder if buffer full
	samplesNeeded := frames * sr.channels
	sr.mu.Lock()
	available := r.Available()
	if available == 0 {
		// Buffer full - drop all samples
		sr.mu.Unlock()
		return 0, nil
	}

	toWrite := samplesNeeded
	if toWrite > available {
		toWrite = available
	}

	n, err := r.Write(samples[:toWrite])
	if err != nil && err != io.ErrShortWrite {
		sr.mu.Unlock()
		return 0, err
	}

	framesWritten := n / sr.channels
	if n > 0 {
		sr.signalNotEmpty()
	}
	sr.mu.Unlock()

	return framesWritten, nil
}

// ReadSamples reads samples from the ring buffer and converts to bytes.
// Uses type dispatch pattern - no switches in hot loops.
// scratch buffers are preallocated to avoid allocations.
func (sr *sampleRing) ReadSamples(pcmData []byte, format string, maxFrames int, scratchInt16 []int16, scratchInt32 []int32, scratchFloat32 []float32, scratchUint8 []uint8) (int, error) {
	switch r := sr.ring.(type) {
	case *buffer.Ring[int16]:
		return sr.readSamplesInt16(r, pcmData, format, maxFrames, scratchInt16)
	case *buffer.Ring[int32]:
		return sr.readSamplesInt32(r, pcmData, format, maxFrames, scratchInt32)
	case *buffer.Ring[float32]:
		return sr.readSamplesFloat32(r, pcmData, format, maxFrames, scratchFloat32)
	case *buffer.Ring[uint8]:
		return sr.readSamplesUint8(r, pcmData, format, maxFrames, scratchUint8)
	default:
		return 0, fmt.Errorf("unsupported sample type: %v", sr.dtype)
	}
}

// readSamplesInt16 reads int16 samples and converts to bytes (no switches in loop).
// scratch is a preallocated buffer to avoid allocations.
// Reads up to maxFrames (or all available if maxFrames is 0), non-blocking.
// For blocking reads, use ReadFrames instead.
func (sr *sampleRing) readSamplesInt16(r *buffer.Ring[int16], pcmData []byte, format string, maxFrames int, scratch []int16) (int, error) {
	sr.mu.Lock()
	availableSamples := r.Len()
	sr.mu.Unlock()

	if availableSamples == 0 {
		return 0, nil
	}

	framesAvailable := availableSamples / sr.channels
	if framesAvailable == 0 {
		return 0, nil
	}

	// Limit to maxFrames if specified
	if maxFrames > 0 && framesAvailable > maxFrames {
		framesAvailable = maxFrames
	}

	samplesNeeded := framesAvailable * sr.channels
	if len(scratch) < samplesNeeded {
		samples := int16Pool.Get(samplesNeeded)
		defer int16Pool.Put(samples)
		return sr.readSamplesInt16WithBuffer(r, pcmData, format, samples[:samplesNeeded], framesAvailable)
	}
	return sr.readSamplesInt16WithBuffer(r, pcmData, format, scratch[:samplesNeeded], framesAvailable)
}

// readSamplesInt16Blocking blocks until maxFrames are read
func (sr *sampleRing) readSamplesInt16Blocking(r *buffer.Ring[int16], pcmData []byte, format string, samples []int16, maxFrames int) (int, error) {
	offset := 0
	samplesNeeded := maxFrames * sr.channels

	for offset < samplesNeeded {
		select {
		case <-sr.ctx.Done():
			return offset / sr.channels, sr.ctx.Err()
		default:
		}

		sr.mu.Lock()
		available := r.Len()
		if available == 0 {
			sr.mu.Unlock()
			select {
			case <-sr.notEmpty:
				// Buffer has data, continue loop
			case <-sr.ctx.Done():
				return offset / sr.channels, sr.ctx.Err()
			}
			continue
		}

		toRead := samplesNeeded - offset
		if toRead > available {
			toRead = available
		}

		n, err := r.Read(samples[offset : offset+toRead])
		if err != nil && err != io.EOF {
			sr.mu.Unlock()
			return offset / sr.channels, err
		}

		offset += n
		if n > 0 {
			sr.signalNotFull()
		}
		sr.mu.Unlock()
	}

	// Convert to bytes
	framesRead := maxFrames
	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < framesRead*sr.channels*sampleSize {
		return 0, fmt.Errorf("insufficient output buffer")
	}

	if format == "S16" {
		for i := 0; i < framesRead*sr.channels; i++ {
			binary.LittleEndian.PutUint16(pcmData[i*2:(i+1)*2], uint16(samples[i]))
		}
	} else {
		return 0, fmt.Errorf("unsupported format conversion: INT16 to %s", format)
	}

	return framesRead, nil
}

// readSamplesInt16WithBuffer does the actual conversion (no switches in loop).
func (sr *sampleRing) readSamplesInt16WithBuffer(r *buffer.Ring[int16], pcmData []byte, format string, samples []int16, framesAvailable int) (int, error) {
	sr.mu.Lock()
	n, err := r.Read(samples)
	if err != nil && err != io.EOF {
		sr.mu.Unlock()
		return 0, fmt.Errorf("failed to read from sampleRing: %w", err)
	}

	framesRead := n / sr.channels
	// Signal that buffer is no longer full
	if n > 0 {
		sr.signalNotFull()
	}
	sr.mu.Unlock()

	if framesRead == 0 {
		return 0, nil
	}

	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < framesRead*sr.channels*sampleSize {
		return 0, fmt.Errorf("insufficient output buffer")
	}

	if format == "S16" {
		for i := 0; i < framesRead*sr.channels; i++ {
			binary.LittleEndian.PutUint16(pcmData[i*2:(i+1)*2], uint16(samples[i]))
		}
	} else {
		return 0, fmt.Errorf("unsupported format conversion: INT16 to %s", format)
	}

	return framesRead, nil
}

// readSamplesInt32 reads int32 samples and converts to bytes (no switches in loop).
// scratch is a preallocated buffer to avoid allocations.
// Reads up to maxFrames (or all available if maxFrames is 0), non-blocking.
// For blocking reads, use ReadFrames instead.
func (sr *sampleRing) readSamplesInt32(r *buffer.Ring[int32], pcmData []byte, format string, maxFrames int, scratch []int32) (int, error) {
	sr.mu.Lock()
	availableSamples := r.Len()
	sr.mu.Unlock()

	if availableSamples == 0 {
		return 0, nil
	}

	framesAvailable := availableSamples / sr.channels
	if framesAvailable == 0 {
		return 0, nil
	}

	// Limit to maxFrames if specified
	if maxFrames > 0 && framesAvailable > maxFrames {
		framesAvailable = maxFrames
	}

	samplesNeeded := framesAvailable * sr.channels
	if len(scratch) < samplesNeeded {
		samples := int32Pool.Get(samplesNeeded)
		defer int32Pool.Put(samples)
		return sr.readSamplesInt32WithBuffer(r, pcmData, format, samples[:samplesNeeded], framesAvailable)
	}
	return sr.readSamplesInt32WithBuffer(r, pcmData, format, scratch[:samplesNeeded], framesAvailable)
}

// readSamplesInt32Blocking blocks until maxFrames are read
func (sr *sampleRing) readSamplesInt32Blocking(r *buffer.Ring[int32], pcmData []byte, format string, samples []int32, maxFrames int) (int, error) {
	offset := 0
	samplesNeeded := maxFrames * sr.channels

	for offset < samplesNeeded {
		select {
		case <-sr.ctx.Done():
			return offset / sr.channels, sr.ctx.Err()
		default:
		}

		sr.mu.Lock()
		available := r.Len()
		if available == 0 {
			sr.mu.Unlock()
			select {
			case <-sr.notEmpty:
				// Buffer has data, continue loop
			case <-sr.ctx.Done():
				return offset / sr.channels, sr.ctx.Err()
			}
			continue
		}

		toRead := samplesNeeded - offset
		if toRead > available {
			toRead = available
		}

		n, err := r.Read(samples[offset : offset+toRead])
		if err != nil && err != io.EOF {
			sr.mu.Unlock()
			return offset / sr.channels, err
		}

		offset += n
		if n > 0 {
			sr.signalNotFull()
		}
		sr.mu.Unlock()
	}

	// Convert to bytes
	framesRead := maxFrames
	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < framesRead*sr.channels*sampleSize {
		return 0, fmt.Errorf("insufficient output buffer")
	}

	if format == "S32" {
		for i := 0; i < framesRead*sr.channels; i++ {
			binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], uint32(samples[i]))
		}
	} else if format == "S24" {
		for i := 0; i < framesRead*sr.channels; i++ {
			val := samples[i]
			pcmData[i*3] = byte(val)
			pcmData[i*3+1] = byte(val >> 8)
			pcmData[i*3+2] = byte(val >> 16)
		}
	} else {
		return 0, fmt.Errorf("unsupported format conversion: INT32 to %s", format)
	}

	return framesRead, nil
}

// readSamplesInt32WithBuffer does the actual conversion (no switches in loop).
func (sr *sampleRing) readSamplesInt32WithBuffer(r *buffer.Ring[int32], pcmData []byte, format string, samples []int32, framesAvailable int) (int, error) {
	sr.mu.Lock()
	n, err := r.Read(samples)
	if err != nil && err != io.EOF {
		sr.mu.Unlock()
		return 0, fmt.Errorf("failed to read from sampleRing: %w", err)
	}

	framesRead := n / sr.channels
	// Signal that buffer is no longer full
	if n > 0 {
		sr.signalNotFull()
	}
	sr.mu.Unlock()

	if framesRead == 0 {
		return 0, nil
	}

	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < framesRead*sr.channels*sampleSize {
		return 0, fmt.Errorf("insufficient output buffer")
	}

	if format == "S32" {
		for i := 0; i < framesRead*sr.channels; i++ {
			binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], uint32(samples[i]))
		}
	} else if format == "S24" {
		// Convert 32-bit int32 back to 24-bit PCM
		// Take only the lower 24 bits (sign-extended values are already correct)
		for i := 0; i < framesRead*sr.channels; i++ {
			val := samples[i]
			// Write 3 bytes (little-endian), taking only lower 24 bits
			pcmData[i*3] = byte(val)         // LSB
			pcmData[i*3+1] = byte(val >> 8)  // Middle byte
			pcmData[i*3+2] = byte(val >> 16) // MSB (bits 16-23, sign bit preserved)
		}
	} else {
		return 0, fmt.Errorf("unsupported format conversion: INT32 to %s", format)
	}

	return framesRead, nil
}

// readSamplesFloat32 reads float32 samples and converts to bytes (no switches in loop).
// scratch is a preallocated buffer to avoid allocations.
// Reads up to maxFrames (or all available if maxFrames is 0), non-blocking.
// For blocking reads, use ReadFrames instead.
func (sr *sampleRing) readSamplesFloat32(r *buffer.Ring[float32], pcmData []byte, format string, maxFrames int, scratch []float32) (int, error) {
	sr.mu.Lock()
	availableSamples := r.Len()
	sr.mu.Unlock()

	if availableSamples == 0 {
		return 0, nil
	}

	framesAvailable := availableSamples / sr.channels
	if framesAvailable == 0 {
		return 0, nil
	}

	// Limit to maxFrames if specified
	if maxFrames > 0 && framesAvailable > maxFrames {
		framesAvailable = maxFrames
	}

	samplesNeeded := framesAvailable * sr.channels
	if len(scratch) < samplesNeeded {
		samples := float32Pool.Get(samplesNeeded)
		defer float32Pool.Put(samples)
		return sr.readSamplesFloat32WithBuffer(r, pcmData, format, samples[:samplesNeeded], framesAvailable)
	}
	return sr.readSamplesFloat32WithBuffer(r, pcmData, format, scratch[:samplesNeeded], framesAvailable)
}

// readSamplesFloat32Blocking blocks until maxFrames are read
func (sr *sampleRing) readSamplesFloat32Blocking(r *buffer.Ring[float32], pcmData []byte, format string, samples []float32, maxFrames int) (int, error) {
	offset := 0
	samplesNeeded := maxFrames * sr.channels

	for offset < samplesNeeded {
		select {
		case <-sr.ctx.Done():
			return offset / sr.channels, sr.ctx.Err()
		default:
		}

		sr.mu.Lock()
		available := r.Len()
		if available == 0 {
			sr.mu.Unlock()
			select {
			case <-sr.notEmpty:
				// Buffer has data, continue loop
			case <-sr.ctx.Done():
				return offset / sr.channels, sr.ctx.Err()
			}
			continue
		}

		toRead := samplesNeeded - offset
		if toRead > available {
			toRead = available
		}

		n, err := r.Read(samples[offset : offset+toRead])
		if err != nil && err != io.EOF {
			sr.mu.Unlock()
			return offset / sr.channels, err
		}

		offset += n
		if n > 0 {
			sr.signalNotFull()
		}
		sr.mu.Unlock()
	}

	// Convert to bytes
	framesRead := maxFrames
	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < framesRead*sr.channels*sampleSize {
		return 0, fmt.Errorf("insufficient output buffer")
	}

	if format == "F32" {
		for i := 0; i < framesRead*sr.channels; i++ {
			bits := uint32(samples[i])
			binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], bits)
		}
	} else if format == "S16" {
		maxVal := float32(32767.0)
		for i := 0; i < framesRead*sr.channels; i++ {
			val := int16(samples[i] * maxVal)
			if val > 32767 {
				val = 32767
			} else if val < -32768 {
				val = -32768
			}
			binary.LittleEndian.PutUint16(pcmData[i*2:(i+1)*2], uint16(val))
		}
	} else if format == "S32" {
		maxVal := float32(2147483648.0)
		for i := 0; i < framesRead*sr.channels; i++ {
			val := int32(samples[i] * maxVal)
			if val > 2147483647 {
				val = 2147483647
			} else if val < -2147483648 {
				val = -2147483648
			}
			binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], uint32(val))
		}
	} else {
		return 0, fmt.Errorf("unsupported format conversion: FP32 to %s", format)
	}

	return framesRead, nil
}

// readSamplesFloat32WithBuffer does the actual conversion (no switches in loop).
func (sr *sampleRing) readSamplesFloat32WithBuffer(r *buffer.Ring[float32], pcmData []byte, format string, samples []float32, framesAvailable int) (int, error) {
	sr.mu.Lock()
	n, err := r.Read(samples)
	if err != nil && err != io.EOF {
		sr.mu.Unlock()
		return 0, fmt.Errorf("failed to read from sampleRing: %w", err)
	}

	framesRead := n / sr.channels
	// Signal that buffer is no longer full
	if n > 0 {
		sr.signalNotFull()
	}
	sr.mu.Unlock()

	if framesRead == 0 {
		return 0, nil
	}

	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < framesRead*sr.channels*sampleSize {
		return 0, fmt.Errorf("insufficient output buffer")
	}

	if format == "F32" {
		for i := 0; i < framesRead*sr.channels; i++ {
			bits := uint32(samples[i])
			binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], bits)
		}
	} else if format == "S16" {
		maxVal := float32(32767.0)
		for i := 0; i < framesRead*sr.channels; i++ {
			val := int16(samples[i] * maxVal)
			if val > 32767 {
				val = 32767
			} else if val < -32768 {
				val = -32768
			}
			binary.LittleEndian.PutUint16(pcmData[i*2:(i+1)*2], uint16(val))
		}
	} else if format == "S32" {
		maxVal := float32(2147483647.0)
		for i := 0; i < framesRead*sr.channels; i++ {
			val := int32(samples[i] * maxVal)
			if val > 2147483647 {
				val = 2147483647
			} else if val < -2147483648 {
				val = -2147483648
			}
			binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], uint32(val))
		}
	} else {
		return 0, fmt.Errorf("unsupported format conversion: FP32 to %s", format)
	}

	return framesRead, nil
}

// readSamplesUint8 reads uint8 samples and converts to bytes (no switches in loop).
// scratch is a preallocated buffer to avoid allocations.
// readSamplesUint8 reads uint8 samples and converts to bytes (no switches in loop).
// scratch is a preallocated buffer to avoid allocations.
// Reads up to maxFrames (or all available if maxFrames is 0), non-blocking.
// For blocking reads, use ReadFrames instead.
func (sr *sampleRing) readSamplesUint8(r *buffer.Ring[uint8], pcmData []byte, format string, maxFrames int, scratch []uint8) (int, error) {
	sr.mu.Lock()
	availableSamples := r.Len()
	sr.mu.Unlock()

	if availableSamples == 0 {
		return 0, nil
	}

	framesAvailable := availableSamples / sr.channels
	if framesAvailable == 0 {
		return 0, nil
	}

	// Limit to maxFrames if specified
	if maxFrames > 0 && framesAvailable > maxFrames {
		framesAvailable = maxFrames
	}

	samplesNeeded := framesAvailable * sr.channels
	if len(scratch) < samplesNeeded {
		samples := uint8Pool.Get(samplesNeeded)
		defer uint8Pool.Put(samples)
		return sr.readSamplesUint8WithBuffer(r, pcmData, format, samples[:samplesNeeded], framesAvailable)
	}
	return sr.readSamplesUint8WithBuffer(r, pcmData, format, scratch[:samplesNeeded], framesAvailable)
}

// readSamplesUint8Blocking blocks until maxFrames are read
func (sr *sampleRing) readSamplesUint8Blocking(r *buffer.Ring[uint8], pcmData []byte, format string, samples []uint8, maxFrames int) (int, error) {
	offset := 0
	samplesNeeded := maxFrames * sr.channels

	for offset < samplesNeeded {
		select {
		case <-sr.ctx.Done():
			return offset / sr.channels, sr.ctx.Err()
		default:
		}

		sr.mu.Lock()
		available := r.Len()
		if available == 0 {
			sr.mu.Unlock()
			select {
			case <-sr.notEmpty:
				// Buffer has data, continue loop
			case <-sr.ctx.Done():
				return offset / sr.channels, sr.ctx.Err()
			}
			continue
		}

		toRead := samplesNeeded - offset
		if toRead > available {
			toRead = available
		}

		n, err := r.Read(samples[offset : offset+toRead])
		if err != nil && err != io.EOF {
			sr.mu.Unlock()
			return offset / sr.channels, err
		}

		offset += n
		if n > 0 {
			sr.signalNotFull()
		}
		sr.mu.Unlock()
	}

	// Convert to bytes
	framesRead := maxFrames
	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < framesRead*sr.channels*sampleSize {
		return 0, fmt.Errorf("insufficient output buffer")
	}

	if format == "U8" {
		copy(pcmData, samples[:framesRead*sr.channels])
	} else {
		return 0, fmt.Errorf("unsupported format conversion: UINT8 to %s", format)
	}

	return framesRead, nil
}

// readSamplesUint8WithBuffer does the actual conversion (no switches in loop).
func (sr *sampleRing) readSamplesUint8WithBuffer(r *buffer.Ring[uint8], pcmData []byte, format string, samples []uint8, framesAvailable int) (int, error) {
	sr.mu.Lock()
	n, err := r.Read(samples)
	if err != nil && err != io.EOF {
		sr.mu.Unlock()
		return 0, fmt.Errorf("failed to read from sampleRing: %w", err)
	}

	framesRead := n / sr.channels
	// Signal that buffer is no longer full
	if n > 0 {
		sr.signalNotFull()
	}
	sr.mu.Unlock()

	if framesRead == 0 {
		return 0, nil
	}

	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < framesRead*sr.channels*sampleSize {
		return 0, fmt.Errorf("insufficient output buffer")
	}

	if format == "U8" {
		copy(pcmData, samples[:framesRead*sr.channels])
	} else {
		return 0, fmt.Errorf("unsupported format conversion: UINT8 to %s", format)
	}

	return framesRead, nil
}

// byteRing is a ring buffer for audio bytes (used for output devices).
// Conversion from tensor to bytes happens in WriteFrames.
// Read reads bytes directly (non-blocking).
type byteRing struct {
	ring     *buffer.Ring[byte]
	format   string
	channels int
	mu       sync.Mutex
	notEmpty chan struct{} // Signaled when buffer becomes non-empty (buffered, size 1)
	notFull  chan struct{} // Signaled when buffer becomes non-full (buffered, size 1)
	ctx      context.Context
}

// newByteRing creates a new byte ring buffer for output devices.
// The context is used to cancel blocking operations when the marshaller is closed.
func newByteRing(format string, channels, capacityBytes int, ctx context.Context) (*byteRing, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	backing := make([]byte, 0, capacityBytes)
	ring := buffer.NewRing(backing)

	return &byteRing{
		ring:     ring,
		format:   format,
		channels: channels,
		ctx:      ctx,
		notEmpty: make(chan struct{}, 1), // Buffered channel, size 1
		notFull:  make(chan struct{}, 1), // Buffered channel, size 1
	}, nil
}

// signalNotEmpty signals that the buffer is no longer empty (non-blocking).
func (br *byteRing) signalNotEmpty() {
	select {
	case br.notEmpty <- struct{}{}:
	default:
		// Channel already has a signal pending, no need to send again
	}
}

// signalNotFull signals that the buffer is no longer full (non-blocking).
func (br *byteRing) signalNotFull() {
	select {
	case br.notFull <- struct{}{}:
	default:
		// Channel already has a signal pending, no need to send again
	}
}

// unblock wakes up all waiting goroutines. Should be called when context is cancelled.
func (br *byteRing) unblock() {
	br.signalNotEmpty()
	br.signalNotFull()
}

// Len returns the number of bytes currently in the buffer.
func (br *byteRing) Len() int {
	br.mu.Lock()
	defer br.mu.Unlock()
	return br.ring.Len()
}

// Cap returns the capacity of the ring buffer in bytes.
func (br *byteRing) Cap() int {
	br.mu.Lock()
	defer br.mu.Unlock()
	return br.ring.Cap()
}

// Available returns the number of bytes that can be written.
func (br *byteRing) Available() int {
	br.mu.Lock()
	defer br.mu.Unlock()
	return br.ring.Available()
}

// Empty returns true if the buffer is empty.
func (br *byteRing) Empty() bool {
	return br.Len() == 0
}

// Full returns true if the buffer is full.
func (br *byteRing) Full() bool {
	return br.Available() == 0
}

// Reset clears the buffer.
func (br *byteRing) Reset() {
	br.mu.Lock()
	defer br.mu.Unlock()
	br.ring.Reset()
}

// Format returns the audio format string.
func (br *byteRing) Format() string {
	return br.format
}

// Channels returns the number of channels.
func (br *byteRing) Channels() int {
	return br.channels
}

// WriteFrames writes frames from a tensor directly to the byte ring buffer.
// Converts from tensor type to bytes automatically.
// Blocks until all data is written (waits if buffer is full).
// Returns the number of frames written.
func (br *byteRing) WriteFrames(t types.Tensor) (int, error) {
	if t == nil || t.Empty() {
		return 0, fmt.Errorf("tensor is nil or empty")
	}

	// Validate tensor shape
	shape := t.Shape()
	if len(shape) != 2 {
		return 0, fmt.Errorf("tensor must be 2D [frames, channels] for audio")
	}

	tensorChannels := shape[1]
	if tensorChannels != br.channels {
		return 0, fmt.Errorf("channel count mismatch: tensor has %d channels, byteRing expects %d", tensorChannels, br.channels)
	}

	frames := shape[0]
	if frames == 0 {
		return 0, nil
	}

	// Convert tensor to PCM bytes
	tensorData := t.Data()
	if tensorData == nil {
		return 0, fmt.Errorf("tensor data is nil")
	}

	sampleSize := sampleSizeFromFormat(br.format)
	bytesNeeded := frames * br.channels * sampleSize

	// Convert tensor data to bytes
	pcmData, err := tensorToPCM(tensorData, br.format, br.channels, frames)
	if err != nil {
		return 0, fmt.Errorf("failed to convert tensor to PCM: %w", err)
	}

	// Write all bytes, blocking if buffer is full
	offset := 0
	for offset < bytesNeeded {
		select {
		case <-br.ctx.Done():
			return offset / (br.channels * sampleSize), br.ctx.Err()
		default:
		}

		br.mu.Lock()
		// Check if buffer has space
		available := br.ring.Available()

		if available == 0 {
			// Buffer is full, wait for space
			br.mu.Unlock()
			select {
			case <-br.notFull:
				// Buffer has space, continue loop
			case <-br.ctx.Done():
				return offset / (br.channels * sampleSize), br.ctx.Err()
			}
			continue
		}

		// Calculate how much we can write
		toWrite := bytesNeeded - offset
		if toWrite > available {
			toWrite = available
		}

		// Write data
		n, err := br.ring.Write(pcmData[offset : offset+toWrite])
		if err != nil && err != io.ErrShortWrite {
			br.mu.Unlock()
			return offset / (br.channels * sampleSize), err
		}

		offset += n
		// Signal that buffer is no longer empty
		if n > 0 {
			br.signalNotEmpty()
		}
		br.mu.Unlock()
	}

	return frames, nil
}

// Read reads bytes directly from the ring buffer (non-blocking).
// Returns the number of bytes read.
func (br *byteRing) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}

	br.mu.Lock()
	defer br.mu.Unlock()

	// Non-blocking read - return what's available
	n, err := br.ring.Read(p)
	if err != nil && err != io.EOF {
		return n, err
	}

	// Signal that buffer is no longer full
	if n > 0 {
		br.signalNotFull()
	}

	return n, nil
}
