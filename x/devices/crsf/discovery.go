package crsf

import "encoding/binary"

// DeviceInfo describes a CRSF device on the bus (Type 0x29 response to Ping 0x28).
type DeviceInfo struct {
	Dest            Address
	Origin          Address
	Name            string
	SerialNumber    uint32
	HardwareVersion uint32
	SoftwareVersion uint32
	ParamCount      uint8
	ProtocolVersion uint8
}

// EncodePing creates a broadcast Ping frame addressed from origin to dest.
func EncodePing(dest, origin Address) *Frame {
	return &Frame{
		Type:    TypePingDevices,
		Payload: []byte{byte(dest), byte(origin)},
	}
}

// DecodePing extracts dest and origin from a Ping frame.
func DecodePing(f *Frame) (dest, origin Address, err error) {
	if f.Type != TypePingDevices {
		return 0, 0, ErrTypeMismatch
	}
	if len(f.Payload) < 2 {
		return 0, 0, ErrFrameTooShort
	}
	return Address(f.Payload[0]), Address(f.Payload[1]), nil
}

// DecodeDeviceInfo parses a DeviceInfo extended frame.
func DecodeDeviceInfo(f *Frame) (*DeviceInfo, error) {
	if f.Type != TypeDeviceInfo {
		return nil, ErrTypeMismatch
	}
	p := f.Payload
	if len(p) < 2+1+12+2 { // dest+origin + null-terminated name(min 1) + serial+hw+sw + count+proto
		return nil, ErrFrameTooShort
	}

	di := &DeviceInfo{
		Dest:   Address(p[0]),
		Origin: Address(p[1]),
	}

	nameEnd := 2
	for nameEnd < len(p) && p[nameEnd] != 0 {
		nameEnd++
	}
	di.Name = string(p[2:nameEnd])
	pos := nameEnd + 1 // skip null terminator

	if pos+14 > len(p) {
		return nil, ErrFrameTooShort
	}
	di.SerialNumber = binary.BigEndian.Uint32(p[pos : pos+4])
	di.HardwareVersion = binary.BigEndian.Uint32(p[pos+4 : pos+8])
	di.SoftwareVersion = binary.BigEndian.Uint32(p[pos+8 : pos+12])
	di.ParamCount = p[pos+12]
	di.ProtocolVersion = p[pos+13]
	return di, nil
}

// EncodeDeviceInfo creates a DeviceInfo response frame.
func EncodeDeviceInfo(di *DeviceInfo) *Frame {
	nameBytes := len(di.Name) + 1 // null-terminated
	p := make([]byte, 2+nameBytes+14)
	p[0] = byte(di.Dest)
	p[1] = byte(di.Origin)
	copy(p[2:], di.Name)
	// p[2+len(di.Name)] = 0 already zero

	pos := 2 + nameBytes
	binary.BigEndian.PutUint32(p[pos:pos+4], di.SerialNumber)
	binary.BigEndian.PutUint32(p[pos+4:pos+8], di.HardwareVersion)
	binary.BigEndian.PutUint32(p[pos+8:pos+12], di.SoftwareVersion)
	p[pos+12] = di.ParamCount
	p[pos+13] = di.ProtocolVersion
	return &Frame{Type: TypeDeviceInfo, Payload: p}
}
