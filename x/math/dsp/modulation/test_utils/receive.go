package test_utils

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
)

// InspectReceiver is a callback function called after processing each chunk.
// It receives:
//   - startSampleIndex, endSampleIndex: sample indices in the input samples buffer
//   - startBufferIndex, endBufferIndex: byte indices in the output decoded buffer
//   - receiver: the receiver interface for accessing stats
type InspectReceiver func(startSampleIndex, endSampleIndex int, startBufferIndex, endBufferIndex int, receiver modulation.Receiver)

// ReceiveOptions configures the Receive function behavior
type ReceiveOptions struct {
	// ChunkParts is the number of chunks to split processing into for streaming simulation
	ChunkParts int
	// Inspect is an optional callback called after processing each chunk for validation/inspection
	Inspect InspectReceiver
}

// DefaultReceiveOptions returns default options
func DefaultReceiveOptions() ReceiveOptions {
	return ReceiveOptions{
		ChunkParts: 10,
	}
}

// Receive processes samples through a demodulator using chunked processing to simulate streaming.
//
// Parameters:
//   - sampleRate: Sample rate in Hz (currently unused, reserved for future channel distortion)
//   - demodulator: Demodulator processor (chain.Processor[byte, float32])
//   - dst: Destination buffer writer (typically buffer.Bits)
//   - samples: Complete samples buffer to process
//   - opts: Optional configuration (nil uses defaults)
//
// Returns:
//   - totalDecoded: Total number of bytes/symbols decoded
func Receive(
	sampleRate float32,
	demodulator chain.Processor[byte, float32],
	dst buffer.Writer[byte],
	samples []float32,
	opts *ReceiveOptions,
) int {
	if opts == nil {
		def := DefaultReceiveOptions()
		opts = &def
	}

	// Simulate streaming by splitting into chunks
	// Each chunk is a slice view of the same buffer (no copying!)
	chunks := splitIntoChunksFromBuffer(samples, opts.ChunkParts)
	totalDecoded := 0
	chunkStartIndex := 0
	decodedStartIndex := 0

	for _, chunk := range chunks {
		// chunk is already a slice view, use it directly
		reader := buffer.New(chunk)

		// Get decoded buffer length before processing
		decodedLengthBefore := dst.Len()

		decoded := demodulator.Process(dst, reader)

		// Get decoded buffer length after processing
		decodedLengthAfter := dst.Len()

		totalDecoded += decoded

		// Call inspection callback if provided
		if opts.Inspect != nil {
			if rx, ok := demodulator.(modulation.Receiver); ok {
				// Sample indices: where samples were read from
				startSampleIndex := chunkStartIndex
				endSampleIndex := chunkStartIndex + len(chunk)
				// Buffer indices: where bytes were written in the decoded buffer
				startBufferIndex := decodedStartIndex
				endBufferIndex := decodedStartIndex + (decodedLengthAfter - decodedLengthBefore)
				opts.Inspect(startSampleIndex, endSampleIndex, startBufferIndex, endBufferIndex, rx)
			}
		}

		chunkStartIndex += len(chunk)
		decodedStartIndex += decodedLengthAfter - decodedLengthBefore
	}

	// Print receiver statistics if available
	if rx, ok := demodulator.(modulation.Receiver); ok {
		quality := rx.Quality()
		transmission := rx.Transmission()
		params := rx.Params()
		fmt.Printf("Receiver Quality: SNR=%.2f dB, SignalStrength=%.2f dB\n", quality.SNR, quality.SignalStrength)
		fmt.Printf("Receiver Transmission: Locked=%v, TransmissionStart=%d, TransmissionEnd=%d, PacketStart=%d, PreambleIndex=%d, LockCount=%d, UnlockCount=%d\n",
			transmission.IsLocked, transmission.TransmissionStart, transmission.TransmissionEnd, transmission.PacketStart, transmission.PreambleStart, transmission.LockCount, transmission.UnlockCount)
		fmt.Printf("Receiver Params: BaudRate=%.2f Hz, SampleRate=%.2f Hz, SignalStrength=%.2f dB, Bandwidth=%.2f Hz, CarrierFrequency=%.2f Hz\n",
			params.BaudRate, params.SampleRate, params.SignalStrength, params.Bandwidth, params.CarrierFrequency)
	}

	return totalDecoded
}

// splitIntoChunksFromBuffer splits a buffer into chunk slices (views, no copying)
// All chunks are views into the same underlying buffer
func splitIntoChunksFromBuffer(samples []float32, parts int) [][]float32 {
	if parts <= 1 {
		return [][]float32{samples}
	}
	chunks := make([][]float32, 0, parts)
	chunkSize := len(samples) / parts
	for i := range parts {
		start := i * chunkSize
		end := start + chunkSize
		if i == parts-1 {
			end = len(samples)
		}
		// Use slice view (no copying) - all chunks point to same underlying buffer
		chunks = append(chunks, samples[start:end])
	}
	return chunks
}
