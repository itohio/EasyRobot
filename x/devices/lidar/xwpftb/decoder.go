package xwpftb

import (
	"encoding/binary"
	"math"
)

func crc16Cumulative(b []byte) uint16 {
	var sum uint32
	for _, v := range b {
		sum += uint32(v)
	}
	return uint16(sum & 0xFFFF)
}

// BuildMeasurementFrame is a helper (used in tests) to construct a valid frame.
func BuildMeasurementFrame(startAngleDeg float64, distancesMm []float64) []byte {
	m := len(distancesMm)
	payloadLen := 5 + 3*m
	totalLen := 8 + payloadLen + 2
	buf := make([]byte, totalLen)

	buf[0] = 0xAA
	binary.LittleEndian.PutUint16(buf[1:3], uint16(totalLen))
	buf[3] = 0x01
	buf[4] = 0x61
	buf[5] = 0xAD
	binary.LittleEndian.PutUint16(buf[6:8], uint16(payloadLen))

	payload := buf[8 : 8+payloadLen]
	payload[0] = 0 // rpm byte (unused here)
	// Offset angle (ignored)
	payload[1] = 0
	payload[2] = 0
	// Start angle centi-deg
	startCenti := uint16(math.Round(startAngleDeg * 100.0))
	binary.LittleEndian.PutUint16(payload[3:5], startCenti)

	const sliceSpan = 24.0
	step := sliceSpan / float64(m)

	for i := 0; i < m; i++ {
		p := 5 + 3*i
		payload[p] = 0 // quality
		distRaw := uint16(math.Round(distancesMm[i] / 0.25))
		payload[p+1] = byte(distRaw & 0xFF)
		payload[p+2] = byte(distRaw >> 8)
		_ = step // angles are derived by decoder
	}

	crc := crc16Cumulative(buf[:len(buf)-2])
	binary.LittleEndian.PutUint16(buf[len(buf)-2:], crc)
	return buf
}
