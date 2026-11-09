// Package telemetry frames rover state for the radio link.
package telemetry

import "errors"

const syncByte = 0xA5

// ErrChecksum is returned when a frame fails its CRC16 check.
var ErrChecksum = errors.New("telemetry: checksum mismatch")

// EncodeFrame wraps payload as sync, length, payload, CRC16-CCITT (big endian).
// The CRC covers length and payload, not the sync byte.
func EncodeFrame(payload []byte) []byte {
	out := make([]byte, 0, len(payload)+4)
	out = append(out, syncByte, byte(len(payload)))
	out = append(out, payload...)
	crc := crc16(out[1:])
	return append(out, byte(crc>>8), byte(crc))
}

// DecodeFrame validates and unwraps one frame.
func DecodeFrame(frame []byte) ([]byte, error) {
	if len(frame) < 4 || frame[0] != syncByte || int(frame[1]) != len(frame)-4 {
		return nil, errors.New("telemetry: malformed frame")
	}
	body := frame[1 : len(frame)-2]
	want := uint16(frame[len(frame)-2])<<8 | uint16(frame[len(frame)-1])
	if crc16(body) != want {
		return nil, ErrChecksum
	}
	return body[1:], nil
}

func crc16(b []byte) uint16 {
	crc := uint16(0xFFFF)
	for _, x := range b {
		crc ^= uint16(x) << 8
		for range 8 {
			if crc&0x8000 != 0 {
				crc = crc<<1 ^ 0x1021
			} else {
				crc <<= 1
			}
		}
	}
	return crc
}
