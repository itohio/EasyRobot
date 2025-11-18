package main

import (
	"encoding/binary"
	"fmt"

	"github.com/itohio/EasyRobot/x/math/protocol/peg"
)

// CRCValidator validates CRC fields in packets
type CRCValidator struct{}

// NewCRCValidator creates a new CRC validator
func NewCRCValidator() *CRCValidator {
	return &CRCValidator{}
}

// HasCRC checks if the state has a potential CRC field
// It looks for the last uint8 or uint16 field as a potential CRC
func (v *CRCValidator) HasCRC(state peg.State) bool {
	fields := state.Fields()
	if len(fields) == 0 {
		return false
	}

	// Check the last field - CRC is typically at the end
	lastField := fields[len(fields)-1]
	return lastField.Type == peg.FieldUint8 || lastField.Type == peg.FieldUint16LE || lastField.Type == peg.FieldUint16BE
}

// Validate validates the CRC in a packet state
// It attempts to validate CRC8 or CRC16 if the last field matches CRC types
func (v *CRCValidator) Validate(state peg.State) (bool, string) {
	packet := state.Packet()
	fields := state.Fields()

	if len(fields) == 0 {
		return true, "" // No fields, assume valid
	}

	// Check the last field as potential CRC
	lastField := fields[len(fields)-1]
	crcOffset := lastField.Offset

	if crcOffset >= len(packet) {
		return true, "" // Field offset beyond packet, skip validation
	}

	// Try CRC8 if last field is uint8
	if lastField.Type == peg.FieldUint8 {
		return v.validateCRC8(packet, crcOffset)
	}

	// Try CRC16 if last field is uint16
	if lastField.Type == peg.FieldUint16LE || lastField.Type == peg.FieldUint16BE {
		isBigEndian := lastField.Type == peg.FieldUint16BE
		return v.validateCRC16(packet, crcOffset, isBigEndian)
	}

	// No CRC field detected
	return true, ""
}

func (v *CRCValidator) validateCRC8(data []byte, offset int) (bool, string) {
	if offset >= len(data) {
		return false, "CRC8 field offset beyond packet length"
	}

	actualCRC := uint8(data[offset])
	crcBytes := data[:offset]
	expectedCRC := crc8(crcBytes)

	if actualCRC == expectedCRC {
		return true, ""
	}

	return false, fmt.Sprintf("CRC8 mismatch: expected %d, got %d", expectedCRC, actualCRC)
}

func (v *CRCValidator) validateCRC16(data []byte, offset int, isBigEndian bool) (bool, string) {
	if offset+2 > len(data) {
		return false, "CRC16 field offset beyond packet length"
	}

	var actualCRC uint16
	if isBigEndian {
		actualCRC = binary.BigEndian.Uint16(data[offset : offset+2])
	} else {
		actualCRC = binary.LittleEndian.Uint16(data[offset : offset+2])
	}

	crcBytes := data[:offset]
	expectedCRC := crc16Cumulative(crcBytes)

	if actualCRC == expectedCRC {
		return true, ""
	}

	return false, fmt.Sprintf("CRC16 mismatch: expected %d, got %d", expectedCRC, actualCRC)
}

// crc16Cumulative calculates CRC16 as cumulative sum
func crc16Cumulative(b []byte) uint16 {
	var sum uint32
	for _, v := range b {
		sum += uint32(v)
	}
	return uint16(sum & 0xFFFF)
}

// crc8 calculates simple CRC8 (sum)
func crc8(b []byte) uint8 {
	var sum uint8
	for _, v := range b {
		sum += v
	}
	return sum
}
