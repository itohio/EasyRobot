package modulation

import (
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Modulator interface for digital modulators
type Modulator interface {
	// Encode encodes bytes into amplitude vector
	// Returns false when encoding is complete
	// Returns expected amplitudes length if amplitudes is nil.
	Encode(amplitudes vecTypes.Vector, data []byte) int

	// Reset resets the encoder state
	Reset()
}

// Demodulator interface for digital demodulators
type Demodulator interface {
	// Decode decodes amplitude vector into bytes
	// Returns false when decoding is complete
	// Returns expected data length if data is nil.
	Decode(data []byte, amplitudes vecTypes.Vector) int

	// Reset resets the decoder state
	Reset()

	// OnIsEndOfPacket sets a callback to check if packet is complete
	OnIsEndOfPacket(callback func([]byte) bool)
}

// ByteEncoder is an interface for byte-level encoders that can be composed.
// Encoders implementing this interface transform []byte to []byte.
type ByteEncoder interface {
	// Encode encodes input bytes to output bytes using destination pattern.
	// Returns the number of bytes written to dst.
	// Returns expected dst length if dst is nil.
	Encode(dst, src []byte) int

	// Reset resets the encoder state.
	Reset()
}

// ByteDecoder is an interface for byte-level decoders that can be composed.
// Decoders implementing this interface transform []byte to []byte.
type ByteDecoder interface {
	// Decode decodes input bytes to output bytes using destination pattern.
	// Returns the number of bytes written to dst.
	// Returns expected dst length if dst is nil.
	Decode(dst, src []byte) int

	// Reset resets the decoder state.
	Reset()
}

// FrequencyDetector detects which frequency is present in a signal sample
// This is a reusable component for FSK demodulation
type FrequencyDetector interface {
	// DetectFrequency detects which frequency has the highest magnitude in the samples
	// Returns the index of the detected frequency and its magnitude
	DetectFrequency(samples []float32) (freqIdx int, magnitude float32)

	// Reset resets the detector state
	Reset()
}
