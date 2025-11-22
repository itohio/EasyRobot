package audio

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
)

// ProtocolHeader represents the header prepended to each packet
type ProtocolHeader struct {
	Magic      [4]byte // Magic bytes to identify packet start: "AUDO"
	Version    uint8   // Protocol version
	DataLength uint32  // Length of data payload in bytes
}

const (
	// ProtocolMagic is the magic bytes that identify a packet start
	ProtocolMagic = "AUDO"
	// ProtocolVersion is the current protocol version
	ProtocolVersion = 1
	// HeaderSize is the size of the protocol header in bytes
	HeaderSize = 9 // 4 (magic) + 1 (version) + 4 (length)
	// CRCSize is the size of the CRC checksum in bytes
	CRCSize = 4 // CRC32
)

// ProtocolPacket represents a complete protocol packet
type ProtocolPacket struct {
	Header  ProtocolHeader
	Payload []byte
	CRC     uint32
}

// ProtocolMarshaller wraps data with a protocol header before encoding
type ProtocolMarshaller struct {
	header ProtocolHeader
}

// NewProtocolMarshaller creates a new protocol marshaller
func NewProtocolMarshaller() *ProtocolMarshaller {
	return &ProtocolMarshaller{}
}

// MarshalData wraps the provided data with a protocol header and CRC
func (pm *ProtocolMarshaller) MarshalData(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("data cannot be empty")
	}

	// Create header
	pm.header = ProtocolHeader{
		Version:    ProtocolVersion,
		DataLength: uint32(len(data)),
	}
	copy(pm.header.Magic[:], ProtocolMagic)

	// Calculate CRC32 of payload
	crc := crc32.ChecksumIEEE(data)

	// Serialize header
	headerBytes := make([]byte, HeaderSize)
	copy(headerBytes[0:4], pm.header.Magic[:])
	headerBytes[4] = pm.header.Version
	binary.BigEndian.PutUint32(headerBytes[5:9], pm.header.DataLength)

	// Serialize CRC
	crcBytes := make([]byte, CRCSize)
	binary.BigEndian.PutUint32(crcBytes, crc)

	// Combine header + payload + CRC
	packet := make([]byte, HeaderSize+len(data)+CRCSize)
	copy(packet[0:HeaderSize], headerBytes)
	copy(packet[HeaderSize:HeaderSize+len(data)], data)
	copy(packet[HeaderSize+len(data):], crcBytes)

	return packet, nil
}

// ProtocolUnmarshaller extracts data from protocol packets
type ProtocolUnmarshaller struct {
	buffer       []byte
	header       *ProtocolHeader
	expectedData uint32 // Expected payload size from header
	receivedData uint32 // Amount of payload data received so far
	state        unmarshalState
}

type unmarshalState int

const (
	stateSearchingHeader unmarshalState = iota // Looking for magic bytes
	stateReadingPayload                        // Found header, reading payload
)

// NewProtocolUnmarshaller creates a new protocol unmarshaller
func NewProtocolUnmarshaller() *ProtocolUnmarshaller {
	return &ProtocolUnmarshaller{
		buffer: make([]byte, 0, 4096),
		state:  stateSearchingHeader,
	}
}

// UnmarshalData processes incoming bytes and extracts complete packets
// Returns: (completePacket, remainingBytes, error)
func (pu *ProtocolUnmarshaller) UnmarshalData(data []byte) ([]byte, []byte, error) {
	pu.buffer = append(pu.buffer, data...)

	for {
		switch pu.state {
		case stateSearchingHeader:
			// Look for magic bytes
			headerStart := pu.findMagicBytes()
			if headerStart == -1 {
				// No header found, keep searching
				// Keep last 3 bytes in case magic spans buffer boundary
				if len(pu.buffer) > 3 {
					pu.buffer = pu.buffer[len(pu.buffer)-3:]
				}
				return nil, pu.buffer, nil
			}

			// Found potential header start, skip any bytes before it
			if headerStart > 0 {
				pu.buffer = pu.buffer[headerStart:]
			}

			// Check if we have enough bytes for header
			if len(pu.buffer) < HeaderSize {
				return nil, pu.buffer, nil // Need more data
			}

			// Parse header
			pu.header = &ProtocolHeader{}
			copy(pu.header.Magic[:], pu.buffer[0:4])
			pu.header.Version = pu.buffer[4]
			pu.header.DataLength = binary.BigEndian.Uint32(pu.buffer[5:9])

			// Validate version
			if pu.header.Version != ProtocolVersion {
				// Invalid version, skip one byte and continue searching
				pu.buffer = pu.buffer[1:]
				continue
			}

			// Move to reading payload state
			pu.state = stateReadingPayload
			pu.expectedData = pu.header.DataLength
			pu.receivedData = 0
			pu.buffer = pu.buffer[HeaderSize:] // Remove header from buffer

		case stateReadingPayload:
			// Check if we have enough data for complete packet (payload + CRC)
			totalNeeded := int(pu.expectedData) + CRCSize
			if len(pu.buffer) < totalNeeded {
				// Not enough data yet
				return nil, pu.buffer, nil
			}

			// Extract payload
			payload := pu.buffer[0:pu.expectedData]
			crcBytes := pu.buffer[pu.expectedData : pu.expectedData+CRCSize]
			receivedCRC := binary.BigEndian.Uint32(crcBytes)

			// Calculate and verify CRC
			calculatedCRC := crc32.ChecksumIEEE(payload)
			if receivedCRC != calculatedCRC {
				// CRC mismatch, reset and search for new header
				pu.Reset()
				// Skip one byte and continue searching
				if len(pu.buffer) > 1 {
					pu.buffer = pu.buffer[1:]
				} else {
					pu.buffer = pu.buffer[:0]
				}
				continue
			}

			// Valid packet received!
			remaining := pu.buffer[totalNeeded:]
			pu.Reset() // Reset for next packet

			return payload, remaining, nil
		}
	}
}

// findMagicBytes searches for the magic bytes in the buffer
// Returns the index where magic bytes start, or -1 if not found
func (pu *ProtocolUnmarshaller) findMagicBytes() int {
	for i := 0; i <= len(pu.buffer)-4; i++ {
		if pu.buffer[i] == ProtocolMagic[0] &&
			pu.buffer[i+1] == ProtocolMagic[1] &&
			pu.buffer[i+2] == ProtocolMagic[2] &&
			pu.buffer[i+3] == ProtocolMagic[3] {
			return i
		}
	}
	return -1
}

// Reset clears the internal buffer and state
func (pu *ProtocolUnmarshaller) Reset() {
	pu.buffer = pu.buffer[:0]
	pu.header = nil
	pu.expectedData = 0
	pu.receivedData = 0
	pu.state = stateSearchingHeader
}

// IsEndOfPacket creates a callback function for decoders to check if a packet is complete
// This callback accumulates decoded bytes and checks for complete protocol packets
func (pu *ProtocolUnmarshaller) IsEndOfPacket() func([]byte) bool {
	// Create a separate buffer for the callback to accumulate data
	callbackBuffer := make([]byte, 0, 4096)

	return func(decoded []byte) bool {
		// Accumulate decoded bytes
		callbackBuffer = append(callbackBuffer, decoded...)

		// Try to extract a complete packet
		packet, remaining, err := pu.UnmarshalData(callbackBuffer)
		if err != nil {
			return false
		}

		if packet != nil {
			// Complete packet found! Keep remaining bytes for next packet
			callbackBuffer = remaining
			return true
		}

		// Not enough data yet, keep accumulating
		return false
	}
}

