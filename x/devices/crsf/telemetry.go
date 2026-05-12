package crsf

import (
	"encoding/binary"
	"math"
)

// LinkStatistics represents uplink/downlink RF quality (Type 0x14).
type LinkStatistics struct {
	UplinkRSSI1     int8  // dBm (negated in wire format)
	UplinkRSSI2     int8  // dBm
	UplinkQuality   uint8 // 0..100 %
	UplinkSNR       int8  // dB
	ActiveAntenna   uint8
	RFProfile       uint8
	UplinkPower     uint8 // enum index (see RFPowerDBm)
	DownlinkRSSI    int8  // dBm
	DownlinkQuality uint8 // 0..100 %
	DownlinkSNR     int8  // dB
}

// RFPowerDBm converts the UplinkPower enum to dBm.
var RFPowerDBm = [...]int{0, 10, 14, 20, 27, 30, 33, 24, 17}

func DecodeLinkStats(f *Frame) (*LinkStatistics, error) {
	if f.Type != TypeLinkStats {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 10 {
		return nil, ErrFrameTooShort
	}
	p := f.Payload
	return &LinkStatistics{
		UplinkRSSI1:     -int8(p[0]),
		UplinkRSSI2:     -int8(p[1]),
		UplinkQuality:   p[2],
		UplinkSNR:       int8(p[3]),
		ActiveAntenna:   p[4],
		RFProfile:       p[5],
		UplinkPower:     p[6],
		DownlinkRSSI:    -int8(p[7]),
		DownlinkQuality: p[8],
		DownlinkSNR:     int8(p[9]),
	}, nil
}

func EncodeLinkStats(ls *LinkStatistics) *Frame {
	p := []byte{
		byte(-ls.UplinkRSSI1),
		byte(-ls.UplinkRSSI2),
		ls.UplinkQuality,
		byte(ls.UplinkSNR),
		ls.ActiveAntenna,
		ls.RFProfile,
		ls.UplinkPower,
		byte(-ls.DownlinkRSSI),
		ls.DownlinkQuality,
		byte(ls.DownlinkSNR),
	}
	return &Frame{Type: TypeLinkStats, Payload: p}
}

// LinkStatsRX provides per-antenna receiver statistics (Type 0x1C).
type LinkStatsRX struct {
	RSSIDB     int8
	RSSIPercent uint8
	LinkQuality uint8
	SNR        int8
	RFPower    uint8
}

func DecodeLinkStatsRX(f *Frame) (*LinkStatsRX, error) {
	if f.Type != TypeLinkStatsRX {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 5 {
		return nil, ErrFrameTooShort
	}
	p := f.Payload
	return &LinkStatsRX{
		RSSIDB:      -int8(p[0]),
		RSSIPercent: p[1],
		LinkQuality: p[2],
		SNR:         int8(p[3]),
		RFPower:     p[4],
	}, nil
}

func EncodeLinkStatsRX(ls *LinkStatsRX) *Frame {
	p := []byte{
		byte(-ls.RSSIDB),
		ls.RSSIPercent,
		ls.LinkQuality,
		byte(ls.SNR),
		ls.RFPower,
	}
	return &Frame{Type: TypeLinkStatsRX, Payload: p}
}

// LinkStatsTX provides transmitter-side statistics (Type 0x1D).
type LinkStatsTX struct {
	RSSIDB     int8
	RSSIPercent uint8
	LinkQuality uint8
	SNR        int8
	RFPower    uint8
	FPS        uint16
}

func DecodeLinkStatsTX(f *Frame) (*LinkStatsTX, error) {
	if f.Type != TypeLinkStatsTX {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 7 {
		return nil, ErrFrameTooShort
	}
	p := f.Payload
	return &LinkStatsTX{
		RSSIDB:      -int8(p[0]),
		RSSIPercent: p[1],
		LinkQuality: p[2],
		SNR:         int8(p[3]),
		RFPower:     p[4],
		FPS:         binary.BigEndian.Uint16(p[5:7]),
	}, nil
}

func EncodeLinkStatsTX(ls *LinkStatsTX) *Frame {
	p := make([]byte, 7)
	p[0] = byte(-ls.RSSIDB)
	p[1] = ls.RSSIPercent
	p[2] = ls.LinkQuality
	p[3] = byte(ls.SNR)
	p[4] = ls.RFPower
	binary.BigEndian.PutUint16(p[5:7], ls.FPS)
	return &Frame{Type: TypeLinkStatsTX, Payload: p}
}

// Battery represents battery sensor telemetry (Type 0x08).
type Battery struct {
	Voltage      float32 // Volts
	Current      float32 // Amps
	CapacityUsed uint32  // mAh
	Remaining    uint8   // percent
}

func DecodeBattery(f *Frame) (*Battery, error) {
	if f.Type != TypeBattery {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 8 {
		return nil, ErrFrameTooShort
	}
	p := f.Payload
	return &Battery{
		Voltage:      float32(binary.BigEndian.Uint16(p[0:2])) / 10.0,
		Current:      float32(binary.BigEndian.Uint16(p[2:4])) / 10.0,
		CapacityUsed: uint32(p[4])<<16 | uint32(p[5])<<8 | uint32(p[6]),
		Remaining:    p[7],
	}, nil
}

func EncodeBattery(b *Battery) *Frame {
	p := make([]byte, 8)
	binary.BigEndian.PutUint16(p[0:2], uint16(b.Voltage*10))
	binary.BigEndian.PutUint16(p[2:4], uint16(b.Current*10))
	p[4] = byte(b.CapacityUsed >> 16)
	p[5] = byte(b.CapacityUsed >> 8)
	p[6] = byte(b.CapacityUsed)
	p[7] = b.Remaining
	return &Frame{Type: TypeBattery, Payload: p}
}

// GPS represents GPS position telemetry (Type 0x02).
type GPS struct {
	Latitude   float64 // degrees
	Longitude  float64 // degrees
	Speed      float32 // km/h
	Heading    float32 // degrees
	Altitude   float32 // meters (MSL)
	Satellites uint8
}

func DecodeGPS(f *Frame) (*GPS, error) {
	if f.Type != TypeGPS {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 15 {
		return nil, ErrFrameTooShort
	}
	p := f.Payload
	return &GPS{
		Latitude:   float64(int32(binary.BigEndian.Uint32(p[0:4]))) / 1e7,
		Longitude:  float64(int32(binary.BigEndian.Uint32(p[4:8]))) / 1e7,
		Speed:      float32(binary.BigEndian.Uint16(p[8:10])) / 100.0,
		Heading:    float32(binary.BigEndian.Uint16(p[10:12])) / 100.0,
		Altitude:   float32(binary.BigEndian.Uint16(p[12:14])) - 1000.0,
		Satellites: p[14],
	}, nil
}

func EncodeGPS(g *GPS) *Frame {
	p := make([]byte, 15)
	binary.BigEndian.PutUint32(p[0:4], uint32(int32(g.Latitude*1e7)))
	binary.BigEndian.PutUint32(p[4:8], uint32(int32(g.Longitude*1e7)))
	binary.BigEndian.PutUint16(p[8:10], uint16(g.Speed*100))
	binary.BigEndian.PutUint16(p[10:12], uint16(g.Heading*100))
	binary.BigEndian.PutUint16(p[12:14], uint16(g.Altitude+1000))
	p[14] = g.Satellites
	return &Frame{Type: TypeGPS, Payload: p}
}

// Attitude represents craft orientation (Type 0x1E).
type Attitude struct {
	Pitch float32 // radians
	Roll  float32 // radians
	Yaw   float32 // radians
}

func DecodeAttitude(f *Frame) (*Attitude, error) {
	if f.Type != TypeAttitude {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 6 {
		return nil, ErrFrameTooShort
	}
	p := f.Payload
	const lsb = 1e-4 // 100 µrad
	return &Attitude{
		Pitch: float32(int16(binary.BigEndian.Uint16(p[0:2]))) * lsb,
		Roll:  float32(int16(binary.BigEndian.Uint16(p[2:4]))) * lsb,
		Yaw:   float32(int16(binary.BigEndian.Uint16(p[4:6]))) * lsb,
	}, nil
}

func EncodeAttitude(a *Attitude) *Frame {
	p := make([]byte, 6)
	const scale = 1e4
	binary.BigEndian.PutUint16(p[0:2], uint16(int16(a.Pitch*scale)))
	binary.BigEndian.PutUint16(p[2:4], uint16(int16(a.Roll*scale)))
	binary.BigEndian.PutUint16(p[4:6], uint16(int16(a.Yaw*scale)))
	return &Frame{Type: TypeAttitude, Payload: p}
}

// BaroAltitude represents barometric altitude + vario (Type 0x09).
type BaroAltitude struct {
	Altitude      float32 // meters relative to start
	VerticalSpeed float32 // m/s
}

func DecodeBaroAltitude(f *Frame) (*BaroAltitude, error) {
	if f.Type != TypeBaroAltitude {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 3 {
		return nil, ErrFrameTooShort
	}
	p := f.Payload
	packed := binary.BigEndian.Uint16(p[0:2])
	vPacked := int8(p[2])

	var altDm int32
	if packed&0x8000 != 0 {
		altDm = int32(packed&0x7FFF) * 10
	} else {
		altDm = int32(packed) - 10000
	}

	const kl = 100.0
	const kr = 0.026

	var vSpeed float32
	if vPacked != 0 {
		sign := float32(1)
		abs := float32(vPacked)
		if vPacked < 0 {
			sign = -1
			abs = -abs
		}
		vSpeed = sign * float32(math.Exp(float64(abs*kr))-1) * kl / 100.0
	}

	return &BaroAltitude{
		Altitude:      float32(altDm) / 10.0,
		VerticalSpeed: vSpeed,
	}, nil
}

func EncodeBaroAltitude(ba *BaroAltitude) *Frame {
	p := make([]byte, 3)

	altDm := int32(ba.Altitude * 10)
	var packed uint16
	switch {
	case altDm < -10000:
		packed = 0
	case altDm < 0x8000-10000:
		packed = uint16(altDm + 10000)
	default:
		packed = uint16((altDm+5)/10) | 0x8000
	}
	binary.BigEndian.PutUint16(p[0:2], packed)

	const kl = 100.0
	const kr = 0.026
	vCmS := ba.VerticalSpeed * 100
	var vPacked int8
	if vCmS != 0 {
		sign := float64(1)
		abs := float64(vCmS)
		if vCmS < 0 {
			sign = -1
			abs = -abs
		}
		vPacked = int8(sign * math.Log(abs/kl+1) / kr)
	}
	p[2] = byte(vPacked)

	return &Frame{Type: TypeBaroAltitude, Payload: p}
}

// FlightMode represents the current flight mode string (Type 0x21).
type FlightMode struct {
	Mode string
}

func DecodeFlightMode(f *Frame) (*FlightMode, error) {
	if f.Type != TypeFlightMode {
		return nil, ErrTypeMismatch
	}
	s := string(f.Payload)
	if n := len(s); n > 0 && s[n-1] == 0 {
		s = s[:n-1]
	}
	return &FlightMode{Mode: s}, nil
}

func EncodeFlightMode(fm *FlightMode) *Frame {
	p := make([]byte, len(fm.Mode)+1)
	copy(p, fm.Mode)
	return &Frame{Type: TypeFlightMode, Payload: p}
}

// Heartbeat represents a heartbeat frame (Type 0x0B).
type Heartbeat struct {
	Origin Address
}

func DecodeHeartbeat(f *Frame) (*Heartbeat, error) {
	if f.Type != TypeHeartbeat {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 2 {
		return nil, ErrFrameTooShort
	}
	return &Heartbeat{Origin: Address(binary.BigEndian.Uint16(f.Payload[0:2]))}, nil
}

func EncodeHeartbeat(hb *Heartbeat) *Frame {
	p := make([]byte, 2)
	binary.BigEndian.PutUint16(p, uint16(hb.Origin))
	return &Frame{Type: TypeHeartbeat, Payload: p}
}
