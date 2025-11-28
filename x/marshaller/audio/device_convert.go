package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/itohio/EasyRobot/x/buffer"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// pcmToTensor converts PCM bytes to tensor data based on format.
func pcmToTensor(pcmData []byte, format string, channels int, frames int, dtype tensortypes.DataType) (any, error) {
	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < frames*channels*sampleSize {
		return nil, fmt.Errorf("insufficient PCM data: got %d bytes, need %d", len(pcmData), frames*channels*sampleSize)
	}

	switch dtype {
	case tensortypes.INT16:
		data := make([]int16, frames*channels)
		if format == "S16" {
			for i := 0; i < frames*channels; i++ {
				data[i] = int16(binary.LittleEndian.Uint16(pcmData[i*2 : (i+1)*2]))
			}
		} else {
			return nil, fmt.Errorf("unsupported format conversion: %s to INT16", format)
		}
		return data, nil

	case tensortypes.INT32:
		data := make([]int32, frames*channels)
		if format == "S32" {
			for i := 0; i < frames*channels; i++ {
				data[i] = int32(binary.LittleEndian.Uint32(pcmData[i*4 : (i+1)*4]))
			}
		} else if format == "S24" {
			// 24-bit samples stored in 32-bit ints
			for i := 0; i < frames*channels; i++ {
				// Read 3 bytes, sign-extend to 32-bit
				b0 := uint32(pcmData[i*3])
				b1 := uint32(pcmData[i*3+1])
				b2 := uint32(pcmData[i*3+2])
				val := int32((b0 | (b1 << 8) | (b2 << 16)))
				if (b2 & 0x80) != 0 {
					val |= ^int32(0xFFFFFF)
				}
				data[i] = val
			}
		} else {
			return nil, fmt.Errorf("unsupported format conversion: %s to INT32", format)
		}
		return data, nil

	case tensortypes.FP32:
		data := make([]float32, frames*channels)
		if format == "F32" {
			for i := 0; i < frames*channels; i++ {
				bits := binary.LittleEndian.Uint32(pcmData[i*4 : (i+1)*4])
				data[i] = float32(bits)
			}
		} else if format == "S16" {
			maxVal := float32(32768.0)
			for i := 0; i < frames*channels; i++ {
				val := int16(binary.LittleEndian.Uint16(pcmData[i*2 : (i+1)*2]))
				data[i] = float32(val) / maxVal
			}
		} else if format == "S32" {
			maxVal := float32(2147483648.0)
			for i := 0; i < frames*channels; i++ {
				val := int32(binary.LittleEndian.Uint32(pcmData[i*4 : (i+1)*4]))
				data[i] = float32(val) / maxVal
			}
		} else {
			return nil, fmt.Errorf("unsupported format conversion: %s to FP32", format)
		}
		return data, nil

	case tensortypes.UINT8:
		data := make([]uint8, frames*channels)
		if format == "U8" {
			copy(data, pcmData[:frames*channels])
		} else {
			return nil, fmt.Errorf("unsupported format conversion: %s to UINT8", format)
		}
		return data, nil

	default:
		return nil, fmt.Errorf("unsupported tensor dtype: %v", dtype)
	}
}

// tensorToPCM converts tensor data to PCM bytes based on format.
func tensorToPCM(tensorData any, format string, channels int, frames int) ([]byte, error) {
	sampleSize := sampleSizeFromFormat(format)
	pcmData := make([]byte, frames*channels*sampleSize)

	switch data := tensorData.(type) {
	case []int16:
		if format == "S16" {
			for i := 0; i < frames*channels; i++ {
				binary.LittleEndian.PutUint16(pcmData[i*2:(i+1)*2], uint16(data[i]))
			}
		} else {
			return nil, fmt.Errorf("unsupported format conversion: INT16 to %s", format)
		}

	case []int32:
		if format == "S32" {
			for i := 0; i < frames*channels; i++ {
				binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], uint32(data[i]))
			}
		} else if format == "S24" {
			// Convert 32-bit to 24-bit
			for i := 0; i < frames*channels; i++ {
				val := data[i]
				pcmData[i*3] = byte(val)
				pcmData[i*3+1] = byte(val >> 8)
				pcmData[i*3+2] = byte(val >> 16)
			}
		} else {
			return nil, fmt.Errorf("unsupported format conversion: INT32 to %s", format)
		}

	case []float32:
		if format == "F32" {
			for i := 0; i < frames*channels; i++ {
				bits := uint32(data[i])
				binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], bits)
			}
		} else if format == "S16" {
			maxVal := float32(32767.0)
			for i := 0; i < frames*channels; i++ {
				val := int16(data[i] * maxVal)
				if val > 32767 {
					val = 32767
				} else if val < -32768 {
					val = -32768
				}
				binary.LittleEndian.PutUint16(pcmData[i*2:(i+1)*2], uint16(val))
			}
		} else if format == "S32" {
			maxVal := float32(2147483647.0)
			for i := 0; i < frames*channels; i++ {
				val := int32(data[i] * maxVal)
				if val > 2147483647 {
					val = 2147483647
				} else if val < -2147483648 {
					val = -2147483648
				}
				binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], uint32(val))
			}
		} else {
			return nil, fmt.Errorf("unsupported format conversion: FP32 to %s", format)
		}

	case []uint8:
		if format == "U8" {
			copy(pcmData, data[:frames*channels])
		} else {
			return nil, fmt.Errorf("unsupported format conversion: UINT8 to %s", format)
		}

	default:
		return nil, fmt.Errorf("unsupported tensor data type: %T", tensorData)
	}

	return pcmData, nil
}

func sampleSizeFromFormat(format string) int {
	switch format {
	case "U8":
		return 1
	case "S16":
		return 2
	case "S24":
		return 3
	case "S32", "F32":
		return 4
	default:
		return 2 // Default to 16-bit
	}
}

// byteRingToSampleRing converts bytes from byteRing to samples and writes to sampleRing.
// Uses sampleRing.WriteSamples which handles conversion internally.
func byteRingToSampleRing(byteRing *buffer.Ring[byte], sampleRing *sampleRing, format string, channels int, scratchInt16 []int16, scratchInt32 []int32, scratchFloat32 []float32, scratchUint8 []uint8) error {
	sampleSize := sampleSizeFromFormat(format)
	availableBytes := byteRing.Len()
	if availableBytes == 0 {
		return nil
	}

	// Read chunk from byteRing (use pooled buffer)
	chunkSize := 4096 * sampleSize * channels
	if chunkSize > availableBytes {
		chunkSize = availableBytes
	}

	framesInChunk := chunkSize / (sampleSize * channels)
	if framesInChunk == 0 {
		return nil
	}
	chunkSize = framesInChunk * sampleSize * channels

	pcmData := bytePool.Get(chunkSize)
	defer bytePool.Put(pcmData)

	n, err := byteRing.Read(pcmData)
	if err != nil && err != io.EOF {
		return fmt.Errorf("failed to read from byteRing: %w", err)
	}
	if n == 0 {
		return nil
	}

	// Use sampleRing.WriteSamples to convert and write (handles conversion internally)
	_, err = sampleRing.WriteSamples(pcmData[:n], format, scratchInt16, scratchInt32, scratchFloat32, scratchUint8)
	if err != nil {
		return fmt.Errorf("failed to write samples: %w", err)
	}

	return nil
}

// Old conversion functions removed - now using sampleRing.WriteSamples directly
// which handles all conversions internally with type dispatch pattern.

// sampleRingToByteRing converts samples from sampleRing to bytes and writes to byteRing.
// This implements: sampleRing -> sample converter -> byteRing
// Uses type dispatch pattern - no switches in hot loops.
func sampleRingToByteRing(sampleRing *sampleRing, byteRing *buffer.Ring[byte], format string, channels int, maxFrames int) error {
	dtype := sampleRing.Dtype()

	// Type dispatch at entry - no switches in hot loops
	switch dtype {
	case tensortypes.INT16:
		return sampleRingToByteRingInt16(sampleRing, byteRing, format, channels, maxFrames)
	case tensortypes.INT32:
		return sampleRingToByteRingInt32(sampleRing, byteRing, format, channels, maxFrames)
	case tensortypes.FP32:
		return sampleRingToByteRingFloat32(sampleRing, byteRing, format, channels, maxFrames)
	case tensortypes.UINT8:
		return sampleRingToByteRingUint8(sampleRing, byteRing, format, channels, maxFrames)
	default:
		return fmt.Errorf("unsupported sample type: %v", dtype)
	}
}

// sampleRingToByteRingInt16 converts int16 samples to bytes (no switches in loop).
func sampleRingToByteRingInt16(sampleRing *sampleRing, byteRing *buffer.Ring[byte], format string, channels int, maxFrames int) error {
	availableSamples := sampleRing.Len()
	if availableSamples == 0 {
		return nil
	}

	framesAvailable := availableSamples / channels
	if framesAvailable == 0 {
		return nil
	}

	if maxFrames > 0 && framesAvailable > maxFrames {
		framesAvailable = maxFrames
	}

	samplesNeeded := framesAvailable * channels
	buf := int16Pool.Get(samplesNeeded)
	defer int16Pool.Put(buf)
	n, err := sampleRing.Read(buf)
	if err != nil && err != io.EOF {
		return fmt.Errorf("failed to read from sampleRing: %w", err)
	}

	framesRead := n / channels
	if framesRead == 0 {
		return nil
	}

	samples := buf[:n]
	sampleSize := sampleSizeFromFormat(format)
	pcmData := bytePool.Get(framesRead * channels * sampleSize)
	defer bytePool.Put(pcmData)

	// Convert int16 samples to PCM bytes (no switches in loop)
	if format == "S16" {
		for i := 0; i < framesRead*channels; i++ {
			binary.LittleEndian.PutUint16(pcmData[i*2:(i+1)*2], uint16(samples[i]))
		}
	} else {
		return fmt.Errorf("unsupported format conversion: INT16 to %s", format)
	}

	// Write to byteRing - handle partial writes
	written, err := byteRing.Write(pcmData)
	if err != nil && err != io.ErrShortWrite {
		return fmt.Errorf("failed to write to byteRing: %w", err)
	}

	// If we got a short write, that's okay - we'll write more on next call
	// The byteRing will be consumed by malgo callback
	if written < len(pcmData) {
		// Buffer is full, but that's okay - malgo will consume it
		// Return nil to continue (we'll write more on next iteration)
	}

	return nil
}

// sampleRingToByteRingInt32 converts int32 samples to bytes (no switches in loop).
func sampleRingToByteRingInt32(sampleRing *sampleRing, byteRing *buffer.Ring[byte], format string, channels int, maxFrames int) error {
	availableSamples := sampleRing.Len()
	if availableSamples == 0 {
		return nil
	}

	framesAvailable := availableSamples / channels
	if framesAvailable == 0 {
		return nil
	}

	if maxFrames > 0 && framesAvailable > maxFrames {
		framesAvailable = maxFrames
	}

	samplesNeeded := framesAvailable * channels
	buf := int32Pool.Get(samplesNeeded)
	defer int32Pool.Put(buf)
	n, err := sampleRing.Read(buf)
	if err != nil && err != io.EOF {
		return fmt.Errorf("failed to read from sampleRing: %w", err)
	}

	framesRead := n / channels
	if framesRead == 0 {
		return nil
	}

	samples := buf[:n]
	sampleSize := sampleSizeFromFormat(format)
	pcmData := bytePool.Get(framesRead * channels * sampleSize)
	defer bytePool.Put(pcmData)

	// Convert int32 samples to PCM bytes (no switches in loop)
	if format == "S32" {
		for i := 0; i < framesRead*channels; i++ {
			binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], uint32(samples[i]))
		}
	} else if format == "S24" {
		for i := 0; i < framesRead*channels; i++ {
			val := samples[i]
			pcmData[i*3] = byte(val)
			pcmData[i*3+1] = byte(val >> 8)
			pcmData[i*3+2] = byte(val >> 16)
		}
	} else {
		return fmt.Errorf("unsupported format conversion: INT32 to %s", format)
	}

	// Write to byteRing - handle partial writes
	written, err := byteRing.Write(pcmData)
	if err != nil && err != io.ErrShortWrite {
		return fmt.Errorf("failed to write to byteRing: %w", err)
	}

	// If we got a short write, that's okay - we'll write more on next call
	// The byteRing will be consumed by malgo callback
	if written < len(pcmData) {
		// Buffer is full, but that's okay - malgo will consume it
		// Return nil to continue (we'll write more on next iteration)
	}

	return nil
}

// sampleRingToByteRingFloat32 converts float32 samples to bytes (no switches in loop).
func sampleRingToByteRingFloat32(sampleRing *sampleRing, byteRing *buffer.Ring[byte], format string, channels int, maxFrames int) error {
	availableSamples := sampleRing.Len()
	if availableSamples == 0 {
		return nil
	}

	framesAvailable := availableSamples / channels
	if framesAvailable == 0 {
		return nil
	}

	if maxFrames > 0 && framesAvailable > maxFrames {
		framesAvailable = maxFrames
	}

	samplesNeeded := framesAvailable * channels
	buf := float32Pool.Get(samplesNeeded)
	defer float32Pool.Put(buf)
	n, err := sampleRing.Read(buf)
	if err != nil && err != io.EOF {
		return fmt.Errorf("failed to read from sampleRing: %w", err)
	}

	framesRead := n / channels
	if framesRead == 0 {
		return nil
	}

	samples := buf[:n]
	sampleSize := sampleSizeFromFormat(format)
	pcmData := bytePool.Get(framesRead * channels * sampleSize)
	defer bytePool.Put(pcmData)

	// Convert float32 samples to PCM bytes (no switches in loop)
	// Type dispatch: format determines output format
	if format == "F32" {
		// Direct float32 to bytes (IEEE 754)
		for i := range framesRead * channels {
			bits := math.Float32bits(samples[i])
			binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], bits)
		}
	} else if format == "S16" {
		// Convert float32 [-1, 1] to int16
		for i := range framesRead * channels {
			val := samples[i]
			if val > 1.0 {
				val = 1.0
			} else if val < -1.0 {
				val = -1.0
			}
			intVal := int16(val * 32767.0)
			binary.LittleEndian.PutUint16(pcmData[i*2:(i+1)*2], uint16(intVal))
		}
	} else if format == "S32" {
		// Convert float32 [-1, 1] to int32
		for i := range framesRead * channels {
			val := samples[i]
			if val > 1.0 {
				val = 1.0
			} else if val < -1.0 {
				val = -1.0
			}
			intVal := int32(val * 2147483647.0)
			binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], uint32(intVal))
		}
	} else {
		return fmt.Errorf("unsupported format conversion: FP32 to %s", format)
	}

	// Write to byteRing - handle partial writes
	written, err := byteRing.Write(pcmData)
	if err != nil && err != io.ErrShortWrite {
		return fmt.Errorf("failed to write to byteRing: %w", err)
	}

	// If we got a short write, that's okay - we'll write more on next call
	// The byteRing will be consumed by malgo callback
	if written < len(pcmData) {
		// Buffer is full, but that's okay - malgo will consume it
		// Return nil to continue (we'll write more on next iteration)
	}

	return nil
}

// sampleRingToByteRingUint8 converts uint8 samples to bytes (no switches in loop).
func sampleRingToByteRingUint8(sampleRing *sampleRing, byteRing *buffer.Ring[byte], format string, channels int, maxFrames int) error {
	availableSamples := sampleRing.Len()
	if availableSamples == 0 {
		return nil
	}

	framesAvailable := availableSamples / channels
	if framesAvailable == 0 {
		return nil
	}

	if maxFrames > 0 && framesAvailable > maxFrames {
		framesAvailable = maxFrames
	}

	samplesNeeded := framesAvailable * channels
	buf := uint8Pool.Get(samplesNeeded)
	defer uint8Pool.Put(buf)
	n, err := sampleRing.Read(buf)
	if err != nil && err != io.EOF {
		return fmt.Errorf("failed to read from sampleRing: %w", err)
	}

	framesRead := n / channels
	if framesRead == 0 {
		return nil
	}

	samples := buf[:n]
	sampleSize := sampleSizeFromFormat(format)
	pcmData := bytePool.Get(framesRead * channels * sampleSize)
	defer bytePool.Put(pcmData)

	// Convert uint8 samples to PCM bytes (no switches in loop)
	if format == "U8" {
		copy(pcmData, samples)
	} else {
		return fmt.Errorf("unsupported format conversion: UINT8 to %s", format)
	}

	// Write to byteRing - handle partial writes
	written, err := byteRing.Write(pcmData)
	if err != nil && err != io.ErrShortWrite {
		return fmt.Errorf("failed to write to byteRing: %w", err)
	}

	// If we got a short write, that's okay - we'll write more on next call
	// The byteRing will be consumed by malgo callback
	if written < len(pcmData) {
		// Buffer is full, but that's okay - malgo will consume it
		// Return nil to continue (we'll write more on next iteration)
	}

	return nil
}
