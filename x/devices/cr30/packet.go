package cr30

import (
	"encoding/binary"
	"errors"
	"math"
)

const (
	packetSize    = 60
	payloadStart  = 4
	payloadEnd    = 56
	payloadSize   = 52
	markerOffset  = 58
	checksumOffset = 59

	startAA = 0xAA
	startBB = 0xBB

	markerAA = 0xFF
	markerBB = 0x00 // or 0xFF
)

var (
	ErrInvalidPacketSize = errors.New("cr30: packet must be exactly 60 bytes")
	ErrInvalidStartByte  = errors.New("cr30: start byte must be 0xAA or 0xBB")
	ErrInvalidMarker     = errors.New("cr30: invalid marker byte")
	ErrInvalidChecksum   = errors.New("cr30: invalid checksum")
	ErrPayloadTooLarge   = errors.New("cr30: payload must be at most 52 bytes")
)

// Packet represents a CR30 protocol packet (60 bytes).
type Packet struct {
	data [packetSize]byte
}

// NewPacket creates a new empty packet.
func NewPacket() *Packet {
	p := &Packet{}
	p.data[markerOffset] = markerAA // Default marker
	return p
}

// PacketFromBytes creates a packet from raw bytes.
func PacketFromBytes(data []byte) (*Packet, error) {
	if len(data) != packetSize {
		return nil, ErrInvalidPacketSize
	}
	p := &Packet{}
	copy(p.data[:], data)
	return p, nil
}

// Start returns the start byte (0xAA or 0xBB).
func (p *Packet) Start() byte {
	return p.data[0]
}

// SetStart sets the start byte and updates marker accordingly.
func (p *Packet) SetStart(start byte) error {
	if start != startAA && start != startBB {
		return ErrInvalidStartByte
	}
	p.data[0] = start
	// Update marker based on start byte
	if start == startAA {
		p.data[markerOffset] = markerAA
	} else if p.data[markerOffset] != markerBB && p.data[markerOffset] != markerAA {
		p.data[markerOffset] = markerAA
	}
	return nil
}

// Cmd returns the command byte.
func (p *Packet) Cmd() byte {
	return p.data[1]
}

// SetCmd sets the command byte.
func (p *Packet) SetCmd(cmd byte) {
	p.data[1] = cmd
}

// Subcmd returns the subcommand byte.
func (p *Packet) Subcmd() byte {
	return p.data[2]
}

// SetSubcmd sets the subcommand byte.
func (p *Packet) SetSubcmd(subcmd byte) {
	p.data[2] = subcmd
}

// Param returns the parameter byte.
func (p *Packet) Param() byte {
	return p.data[3]
}

// SetParam sets the parameter byte.
func (p *Packet) SetParam(param byte) {
	p.data[3] = param
}

// Payload returns the payload bytes (52 bytes).
func (p *Packet) Payload() []byte {
	return p.data[payloadStart:payloadEnd]
}

// SetPayload sets the payload bytes.
func (p *Packet) SetPayload(data []byte) error {
	if len(data) > payloadSize {
		return ErrPayloadTooLarge
	}
	// Clear payload area
	for i := payloadStart; i < payloadEnd; i++ {
		p.data[i] = 0
	}
	// Set new payload
	copy(p.data[payloadStart:], data)
	return nil
}

// Marker returns the marker byte.
func (p *Packet) Marker() byte {
	return p.data[markerOffset]
}

// SetMarker sets the marker byte.
func (p *Packet) SetMarker(marker byte) error {
	start := p.Start()
	if start == startAA && marker != markerAA {
		return ErrInvalidMarker
	}
	if start == startBB && marker != markerBB && marker != markerAA {
		return ErrInvalidMarker
	}
	p.data[markerOffset] = marker
	return nil
}

// Checksum returns the checksum byte.
func (p *Packet) Checksum() byte {
	return p.data[checksumOffset]
}

// CalculateChecksum calculates and sets the checksum.
func (p *Packet) CalculateChecksum() byte {
	sum := 0
	for i := 0; i < markerOffset; i++ {
		sum += int(p.data[i])
	}
	if p.Start() == startBB {
		sum = (sum - 1) % 256
	} else {
		sum = sum % 256
	}
	if sum < 0 {
		sum += 256
	}
	p.data[checksumOffset] = byte(sum)
	return p.data[checksumOffset]
}

// VerifyChecksum verifies the packet checksum.
func (p *Packet) VerifyChecksum() bool {
	original := p.Checksum()
	calculated := p.CalculateChecksum()
	return original == calculated
}

// IsValid checks if the packet has valid structure.
func (p *Packet) IsValid() bool {
	start := p.Start()
	if start != startAA && start != startBB {
		return false
	}
	if start == startAA && p.Marker() != markerAA {
		return false
	}
	if start == startBB && p.Marker() != markerBB && p.Marker() != markerAA {
		return false
	}
	return true
}

// Bytes returns the packet as a byte slice.
func (p *Packet) Bytes() []byte {
	p.CalculateChecksum()
	return p.data[:]
}

// PacketBuilder builds CR30 protocol packets.
type PacketBuilder struct{}

// NewPacketBuilder creates a new packet builder.
func NewPacketBuilder() *PacketBuilder {
	return &PacketBuilder{}
}

// BuildPacket builds a complete packet with checksum.
func (b *PacketBuilder) BuildPacket(start, cmd, subcmd, param byte, data []byte) ([]byte, error) {
	p := NewPacket()
	if err := p.SetStart(start); err != nil {
		return nil, err
	}
	p.SetCmd(cmd)
	p.SetSubcmd(subcmd)
	p.SetParam(param)
	if data != nil {
		if err := p.SetPayload(data); err != nil {
			return nil, err
		}
	}
	return p.Bytes(), nil
}

// BuildHandshakePacket builds a handshake packet (0xAA start).
func (b *PacketBuilder) BuildHandshakePacket(cmd, subcmd, param byte, data []byte) ([]byte, error) {
	return b.BuildPacket(startAA, cmd, subcmd, param, data)
}

// BuildCommandPacket builds a command packet (0xBB start).
func (b *PacketBuilder) BuildCommandPacket(cmd, subcmd, param byte, data []byte) ([]byte, error) {
	return b.BuildPacket(startBB, cmd, subcmd, param, data)
}

// PacketParser parses CR30 protocol packets and maintains state for multi-packet operations.
type PacketParser struct {
	spdBytes      []byte
	chunksReceived map[byte]bool
	chunkInfo     []ChunkInfo
}

// NewPacketParser creates a new packet parser.
func NewPacketParser() *PacketParser {
	return &PacketParser{
		chunksReceived: make(map[byte]bool),
		chunkInfo:     make([]ChunkInfo, 0),
	}
}

// ResetSPDCollection resets SPD chunk collection state.
func (p *PacketParser) ResetSPDCollection() {
	p.spdBytes = make([]byte, 0)
	p.chunksReceived = make(map[byte]bool)
	p.chunkInfo = make([]ChunkInfo, 0)
}

// IsValidPacket checks if a packet has valid structure.
func (p *PacketParser) IsValidPacket(data []byte) bool {
	if len(data) != packetSize {
		return false
	}
	packet, err := PacketFromBytes(data)
	if err != nil {
		return false
	}
	return packet.IsValid()
}

// ParsePacket parses bytes into a Packet object.
func (p *PacketParser) ParsePacket(data []byte) (*Packet, error) {
	if len(data) != packetSize {
		return nil, ErrInvalidPacketSize
	}
	packet, err := PacketFromBytes(data)
	if err != nil {
		return nil, err
	}
	if !packet.IsValid() {
		return nil, ErrInvalidStartByte
	}
	return packet, nil
}

// ExtractPayload extracts payload bytes from a packet.
func (p *PacketParser) ExtractPayload(data []byte) ([]byte, error) {
	packet, err := p.ParsePacket(data)
	if err != nil {
		return nil, err
	}
	return packet.Payload(), nil
}

// ChunkInfo contains information about a parsed SPD chunk.
type ChunkInfo struct {
	Subcmd      byte
	Payload     []byte
	SPDBytes    []byte
	SPDFloats   []float32
	SPDStart    int
	SPDCount    int
}

// ParseSPDChunk parses an SPD chunk packet and accumulates data.
func (p *PacketParser) ParseSPDChunk(data []byte) (*ChunkInfo, error) {
	packet, err := p.ParsePacket(data)
	if err != nil {
		return nil, err
	}

	subcmd := packet.Subcmd()
	if p.chunksReceived[subcmd] {
		return nil, errors.New("cr30: chunk already received")
	}

	payload := packet.Payload()
	info := &ChunkInfo{
		Subcmd:  subcmd,
		Payload: make([]byte, len(payload)),
	}
	copy(info.Payload, payload)

	switch subcmd {
	case 0x10:
		// Chunk 0x10: Contains first SPD bytes (starts at offset 2)
		spdData := payload[2:50]
		p.spdBytes = append(p.spdBytes, spdData...)
		info.SPDBytes = make([]byte, len(spdData))
		copy(info.SPDBytes, spdData)
		info.SPDStart = 2
		info.SPDCount = len(spdData)
		info.SPDFloats = parseFloats(spdData)

	case 0x11, 0x12:
		// Chunks 0x11, 0x12: Pure SPD data (starts at offset 2)
		spdData := payload[2:50]
		p.spdBytes = append(p.spdBytes, spdData...)
		info.SPDBytes = make([]byte, len(spdData))
		copy(info.SPDBytes, spdData)
		info.SPDStart = 2
		info.SPDCount = len(spdData)
		info.SPDFloats = parseFloats(spdData)

	case 0x13:
		// Chunk 0x13: Final chunk (may have different structure)
		info.SPDCount = 0

	default:
		return nil, errors.New("cr30: unknown SPD chunk subcmd")
	}

	p.chunksReceived[subcmd] = true
	p.chunkInfo = append(p.chunkInfo, *info)

	return info, nil
}

// GetAccumulatedSPD returns accumulated SPD bytes from all chunks.
func (p *PacketParser) GetAccumulatedSPD() []byte {
	return p.spdBytes
}

// IsSPDComplete checks if all expected SPD chunks have been received.
func (p *PacketParser) IsSPDComplete() bool {
	expected := map[byte]bool{0x10: true, 0x11: true, 0x12: true, 0x13: true}
	for subcmd := range expected {
		if !p.chunksReceived[subcmd] {
			return false
		}
	}
	return true
}

// parseFloats parses little-endian float32 values from bytes.
func parseFloats(data []byte) []float32 {
	if len(data)%4 != 0 {
		return nil
	}
	count := len(data) / 4
	floats := make([]float32, count)
	for i := 0; i < count; i++ {
		offset := i * 4
		bits := binary.LittleEndian.Uint32(data[offset : offset+4])
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}

