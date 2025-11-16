package ld06

import (
	"encoding/binary"
	"math"
)

// CRC8 calculation for LD06 protocol
// Polynomial: 0x31 (CRC-8-CCITT)
func crc8(b []byte) uint8 {
	var crc uint8 = 0
	for _, v := range b {
		crc ^= v
		for i := 0; i < 8; i++ {
			if crc&0x80 != 0 {
				crc = (crc << 1) ^ 0x31
			} else {
				crc <<= 1
			}
		}
	}
	return crc
}

// BuildMeasurementPacket is a helper (used in tests) to construct a valid LD06 packet.
// startAngleDeg and endAngleDeg are in degrees.
func BuildMeasurementPacket(radarSpeedHz float64, startAngleDeg, endAngleDeg float64, distancesMm []uint16, intensities []uint8) []byte {
	if len(distancesMm) != len(intensities) {
		return nil
	}
	n := len(distancesMm)
	if n > 255 {
		return nil // data length is 1 byte
	}

	// Packet structure:
	// 0: start (0x54)
	// 1: data length (n)
	// 2-3: radar speed (0.01 Hz units)
	// 4-5: start angle (0.01° units, int16)
	// 6..(6+3n-1): data points [dist_LSB, dist_MSB, intensity] × n
	// (6+3n)..(6+3n+1): end angle (0.01° units, int16)
	// (6+3n+2)..(6+3n+3): timestamp (0.1 ms units, uint16) - use 0 for tests
	// (6+3n+4): CRC8

	packetLen := 6 + 3*n + 2 + 2 + 1
	buf := make([]byte, packetLen)

	buf[0] = 0x54
	buf[1] = byte(n)

	// Radar speed (0.01 Hz units)
	speedCentiHz := uint16(math.Round(radarSpeedHz * 100.0))
	binary.LittleEndian.PutUint16(buf[2:4], speedCentiHz)

	// Start angle (0.01° units, int16)
	startAngleCenti := int16(math.Round(startAngleDeg * 100.0))
	binary.LittleEndian.PutUint16(buf[4:6], uint16(startAngleCenti))

	// Data points
	for i := 0; i < n; i++ {
		offset := 6 + 3*i
		dist := distancesMm[i]
		buf[offset] = byte(dist & 0xFF)        // LSB
		buf[offset+1] = byte((dist >> 8) & 0xFF) // MSB
		buf[offset+2] = intensities[i]
	}

	// End angle (0.01° units, int16)
	endAngleCenti := int16(math.Round(endAngleDeg * 100.0))
	binary.LittleEndian.PutUint16(buf[6+3*n:6+3*n+2], uint16(endAngleCenti))

	// Timestamp (0.1 ms units, uint16) - 0 for tests
	binary.LittleEndian.PutUint16(buf[6+3*n+2:6+3*n+4], 0)

	// CRC8 over all bytes except CRC itself
	crc := crc8(buf[:packetLen-1])
	buf[packetLen-1] = crc

	return buf
}

