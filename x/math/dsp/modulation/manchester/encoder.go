package manchester

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// Encoder handles Manchester line coding encoding.
// Manchester encoding: each bit is encoded as a transition.
// Standard Manchester: 0 = low-to-high (01), 1 = high-to-low (10)
// Implements chain.Processor[byte, byte] - reads bytes, writes bytes (1 byte -> 2 bytes)
type Encoder struct {
	// IEEE 802.3 variant: if true, 0 = high-to-low (10), 1 = low-to-high (01)
	ieee8023 bool
	// Lookup table reference (16 bytes)
	encodeTable *[16]byte
}

// NewEncoder creates a new Manchester encoder with standard encoding.
// Standard: 0 = 01, 1 = 10
func NewEncoder() *Encoder {
	return &Encoder{
		ieee8023:    false,
		encodeTable: &encodeTable4Std,
	}
}

// NewEncoderIEEE8023 creates a new Manchester encoder with IEEE 802.3 encoding.
// IEEE 802.3: 0 = 10, 1 = 01
func NewEncoderIEEE8023() *Encoder {
	return &Encoder{
		ieee8023:    true,
		encodeTable: &encodeTable4IEEE,
	}
}

// Process processes input bytes to Manchester-encoded bytes.
// Each input byte produces 2 output bytes (1 byte -> 2 bytes).
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface
func (m *Encoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	// Read available data from source
	// Use a reasonable buffer size for reading
	var srcBuf [256]byte
	totalRead := 0
	var readErr error

	// Read all available data
	for readErr == nil {
		n, err := src.Read(srcBuf[:])
		if n > 0 {
			// If dst is nil, calculate and return size
			if dst == nil {
				totalRead += n
				if err == io.EOF {
					break
				}
				continue
			}

			// Prepare output buffer
			var dstBuf [512]byte // 2x the input buffer size
			dstIdx := 0

			// Process each input byte
			for i := 0; i < n; i++ {
				b := srcBuf[i]

				// Process high 4 bits -> first output byte
				high4 := (b >> 4) & 0x0F
				dstBuf[dstIdx] = m.encodeTable[high4]

				// Process low 4 bits -> second output byte
				low4 := b & 0x0F
				dstBuf[dstIdx+1] = m.encodeTable[low4]

				dstIdx += 2
			}

			// Write output
			_, writeErr := dst.Write(dstBuf[:dstIdx])
			if writeErr != nil {
				// Return bytes written so far
				return totalRead * 2
			}
			totalRead += n
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
	}

	// If dst was nil, return calculated size
	if dst == nil {
		return totalRead * 2
	}

	// Return total bytes written (2x input bytes)
	return totalRead * 2
}

// Reset resets the encoder state (no-op for Manchester encoding).
// Reset implements chain.Processor interface
func (m *Encoder) Reset() {
	// No state to reset
}

// Ensure Encoder implements chain.Processor[byte, byte]
var _ chain.Processor[byte, byte] = (*Encoder)(nil)
