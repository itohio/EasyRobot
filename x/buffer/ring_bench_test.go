package buffer

import (
	"encoding/binary"
	"testing"

	"github.com/smallnest/ringbuffer"
)

// Typical audio scenarios:
// - Sample rate: 48000 Hz
// - Frame size: 1024 frames
// - Channels: 2 (stereo)
// - Buffer: 3 seconds = 48000 * 3 = 144000 samples
// - For int16: 144000 * 2 bytes = 288000 bytes
// - For float32: 144000 * 4 bytes = 576000 bytes

const (
	audioSampleRate = 48000
	audioFrameSize  = 1024
	audioChannels   = 2
	audioBufferSec  = 3
)

// BenchmarkRing_Byte_WriteRead benchmarks buffer.Ring[byte] for byte operations
func BenchmarkRing_Byte_WriteRead(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec * 2 // bytes for int16
	backing := make([]byte, 0, capacity)
	rb := NewRing(backing)

	// Pre-generate test data
	data := make([]byte, audioFrameSize*audioChannels*2) // 1024 frames * 2 channels * 2 bytes
	for i := range data {
		data[i] = byte(i % 256)
	}

	readBuf := make([]byte, len(data))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Write
		_, _ = rb.Write(data)
		// Read
		_, _ = rb.Read(readBuf)
	}
}

// BenchmarkRingBuffer_Byte_WriteRead benchmarks external ringbuffer.RingBuffer for byte operations
func BenchmarkRingBuffer_Byte_WriteRead(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec * 2 // bytes for int16
	rb := ringbuffer.New(capacity)

	// Pre-generate test data
	data := make([]byte, audioFrameSize*audioChannels*2) // 1024 frames * 2 channels * 2 bytes
	for i := range data {
		data[i] = byte(i % 256)
	}

	readBuf := make([]byte, len(data))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Write
		_, _ = rb.Write(data)
		// Read
		_, _ = rb.Read(readBuf)
	}
}

// BenchmarkRing_Int16_WriteRead benchmarks buffer.Ring[int16] for direct int16 operations
// This avoids byte conversion overhead
func BenchmarkRing_Int16_WriteRead(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec // samples
	backing := make([]int16, 0, capacity)
	rb := NewRing(backing)

	// Pre-generate test data (int16 samples directly)
	data := make([]int16, audioFrameSize*audioChannels) // 1024 frames * 2 channels
	for i := range data {
		data[i] = int16(i % 32767)
	}

	readBuf := make([]int16, len(data))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Write
		_, _ = rb.Write(data)
		// Read
		_, _ = rb.Read(readBuf)
	}
}

// BenchmarkRingBuffer_Int16_WithConversion benchmarks external ringbuffer with int16->byte conversion
// This simulates what we'd need to do with ringbuffer.RingBuffer for int16 samples
func BenchmarkRingBuffer_Int16_WithConversion(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec * 2 // bytes
	rb := ringbuffer.New(capacity)

	// Pre-generate test data (int16 samples)
	samples := make([]int16, audioFrameSize*audioChannels)
	for i := range samples {
		samples[i] = int16(i % 32767)
	}

	// Convert to bytes (overhead we want to avoid)
	data := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(data[i*2:], uint16(s))
	}

	readBuf := make([]byte, len(data))
	outputSamples := make([]int16, len(samples))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Write (with conversion overhead)
		_, _ = rb.Write(data)
		// Read
		_, _ = rb.Read(readBuf)
		// Convert back (overhead we want to avoid)
		for j := range outputSamples {
			outputSamples[j] = int16(binary.LittleEndian.Uint16(readBuf[j*2:]))
		}
	}
}

// BenchmarkRing_Float32_WriteRead benchmarks buffer.Ring[float32] for direct float32 operations
func BenchmarkRing_Float32_WriteRead(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec // samples
	backing := make([]float32, 0, capacity)
	rb := NewRing(backing)

	// Pre-generate test data (float32 samples directly)
	data := make([]float32, audioFrameSize*audioChannels)
	for i := range data {
		data[i] = float32(i) / 32767.0
	}

	readBuf := make([]float32, len(data))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Write
		_, _ = rb.Write(data)
		// Read
		_, _ = rb.Read(readBuf)
	}
}

// BenchmarkRingBuffer_Float32_WithConversion benchmarks external ringbuffer with float32->byte conversion
func BenchmarkRingBuffer_Float32_WithConversion(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec * 4 // bytes
	rb := ringbuffer.New(capacity)

	// Pre-generate test data (float32 samples)
	samples := make([]float32, audioFrameSize*audioChannels)
	for i := range samples {
		samples[i] = float32(i) / 32767.0
	}

	// Convert to bytes (overhead)
	data := make([]byte, len(samples)*4)
	for i, s := range samples {
		binary.LittleEndian.PutUint32(data[i*4:], uint32(s))
	}

	readBuf := make([]byte, len(data))
	outputSamples := make([]float32, len(samples))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Write (with conversion overhead)
		_, _ = rb.Write(data)
		// Read
		_, _ = rb.Read(readBuf)
		// Convert back (overhead)
		for j := range outputSamples {
			outputSamples[j] = float32(binary.LittleEndian.Uint32(readBuf[j*4:]))
		}
	}
}

// BenchmarkRing_Int16_WriteItem benchmarks single item writes (for small chunks)
func BenchmarkRing_Int16_WriteItem(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec
	backing := make([]int16, 0, capacity)
	rb := NewRing(backing)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = rb.WriteItem(int16(i % 32767))
	}
}

// BenchmarkRingBuffer_Int16_WriteItem benchmarks single byte writes with conversion
func BenchmarkRingBuffer_Int16_WriteItem(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec * 2
	rb := ringbuffer.New(capacity)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		sample := int16(i % 32767)
		bytes := make([]byte, 2)
		binary.LittleEndian.PutUint16(bytes, uint16(sample))
		_, _ = rb.Write(bytes)
	}
}

// BenchmarkRing_Int16_ReadItem benchmarks single item reads
func BenchmarkRing_Int16_ReadItem(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec
	backing := make([]int16, 0, capacity)
	rb := NewRing(backing)

	// Pre-fill buffer
	data := make([]int16, capacity)
	for i := range data {
		data[i] = int16(i % 32767)
	}
	_, _ = rb.Write(data)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = rb.ReadItem()
	}
}

// BenchmarkRingBuffer_Int16_ReadItem benchmarks single byte reads with conversion
func BenchmarkRingBuffer_Int16_ReadItem(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec * 2
	rb := ringbuffer.New(capacity)

	// Pre-fill buffer
	data := make([]byte, capacity)
	for i := 0; i < capacity; i += 2 {
		binary.LittleEndian.PutUint16(data[i:], uint16(i/2%32767))
	}
	_, _ = rb.Write(data)

	readBuf := make([]byte, 2)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = rb.Read(readBuf)
		_ = int16(binary.LittleEndian.Uint16(readBuf))
	}
}

// BenchmarkRing_Int16_Concurrent benchmarks concurrent write/read operations
func BenchmarkRing_Int16_Concurrent(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec
	backing := make([]int16, 0, capacity)
	rb := NewRing(backing)

	data := make([]int16, audioFrameSize*audioChannels)
	for i := range data {
		data[i] = int16(i % 32767)
	}

	readBuf := make([]int16, len(data))

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// Write
			_, _ = rb.Write(data)
			// Read
			_, _ = rb.Read(readBuf)
		}
	})
}

// BenchmarkRingBuffer_Int16_Concurrent benchmarks concurrent operations with conversion
func BenchmarkRingBuffer_Int16_Concurrent(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec * 2
	rb := ringbuffer.New(capacity)

	samples := make([]int16, audioFrameSize*audioChannels)
	for i := range samples {
		samples[i] = int16(i % 32767)
	}

	data := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(data[i*2:], uint16(s))
	}

	readBuf := make([]byte, len(data))
	outputSamples := make([]int16, len(samples))

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// Write (with conversion)
			_, _ = rb.Write(data)
			// Read
			_, _ = rb.Read(readBuf)
			// Convert back
			for j := range outputSamples {
				outputSamples[j] = int16(binary.LittleEndian.Uint16(readBuf[j*2:]))
			}
		}
	})
}

// BenchmarkRing_Int16_SmallChunks benchmarks small chunk operations (typical audio callback size)
func BenchmarkRing_Int16_SmallChunks(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec
	backing := make([]int16, 0, capacity)
	rb := NewRing(backing)

	// Small chunk (typical callback size: 128-512 samples)
	chunkSize := 256
	data := make([]int16, chunkSize)
	for i := range data {
		data[i] = int16(i % 32767)
	}

	readBuf := make([]int16, chunkSize)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = rb.Write(data)
		_, _ = rb.Read(readBuf)
	}
}

// BenchmarkRingBuffer_Int16_SmallChunks benchmarks small chunks with conversion
func BenchmarkRingBuffer_Int16_SmallChunks(b *testing.B) {
	capacity := audioSampleRate * audioChannels * audioBufferSec * 2
	rb := ringbuffer.New(capacity)

	chunkSize := 256
	samples := make([]int16, chunkSize)
	for i := range samples {
		samples[i] = int16(i % 32767)
	}

	data := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(data[i*2:], uint16(s))
	}

	readBuf := make([]byte, len(data))
	outputSamples := make([]int16, len(samples))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = rb.Write(data)
		_, _ = rb.Read(readBuf)
		for j := range outputSamples {
			outputSamples[j] = int16(binary.LittleEndian.Uint16(readBuf[j*2:]))
		}
	}
}
