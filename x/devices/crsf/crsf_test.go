package crsf

import (
	"bytes"
	"math"
	"testing"
)

func TestCRC8KnownVector(t *testing.T) {
	// Construct a known CRSF channels frame manually:
	// Type=0x16, 22-byte all-zero payload → compute expected CRC
	data := make([]byte, 23) // type + 22 bytes payload
	data[0] = byte(TypeChannelsPacked)
	expected := CRC8(data)

	// Build via Frame and confirm CRC matches
	f := &Frame{Type: TypeChannelsPacked, Payload: make([]byte, 22)}
	var buf [MaxFrameSize]byte
	n, err := f.Marshal(buf[:])
	if err != nil {
		t.Fatal(err)
	}
	if buf[n-1] != expected {
		t.Errorf("CRC = 0x%02X, want 0x%02X", buf[n-1], expected)
	}
}

func TestCRC8Empty(t *testing.T) {
	if got := CRC8(nil); got != 0 {
		t.Errorf("CRC8(nil) = 0x%02X, want 0x00", got)
	}
}

func TestCRC8SingleByte(t *testing.T) {
	// CRC8(0x00) = crc8tab[0 ^ 0x00] = crc8tab[0] = 0x00
	if got := CRC8([]byte{0x00}); got != 0x00 {
		t.Errorf("CRC8(0x00) = 0x%02X, want 0x00", got)
	}
	// CRC8(0x01) = crc8tab[0 ^ 0x01] = crc8tab[1] = 0xD5
	if got := CRC8([]byte{0x01}); got != 0xD5 {
		t.Errorf("CRC8(0x01) = 0x%02X, want 0xD5", got)
	}
}

func TestFrameMarshalUnmarshal(t *testing.T) {
	original := &Frame{
		Type:    TypeChannelsPacked,
		Payload: make([]byte, 22),
	}
	original.Payload[0] = 0xAB

	var buf [MaxFrameSize]byte
	n, err := original.Marshal(buf[:])
	if err != nil {
		t.Fatal(err)
	}

	decoded := &Frame{}
	consumed, err := decoded.Unmarshal(buf[:n])
	if err != nil {
		t.Fatal(err)
	}
	if consumed != n {
		t.Errorf("consumed %d, want %d", consumed, n)
	}
	if decoded.Type != original.Type {
		t.Errorf("type = 0x%02X, want 0x%02X", decoded.Type, original.Type)
	}
	if !bytes.Equal(decoded.Payload, original.Payload) {
		t.Error("payload mismatch")
	}
}

func TestFrameCRCReject(t *testing.T) {
	f := &Frame{Type: TypeHeartbeat, Payload: []byte{0xEA, 0x00}}
	var buf [MaxFrameSize]byte
	n, _ := f.Marshal(buf[:])

	buf[n-1] ^= 0xFF

	decoded := &Frame{}
	_, err := decoded.Unmarshal(buf[:n])
	if err != ErrCRCMismatch {
		t.Errorf("expected ErrCRCMismatch, got %v", err)
	}
}

func TestFrameInvalidSync(t *testing.T) {
	buf := []byte{0x42, 0x04, 0x16, 0x00, 0x00}
	f := &Frame{}
	_, err := f.Unmarshal(buf)
	if err != ErrInvalidSync {
		t.Errorf("expected ErrInvalidSync, got %v", err)
	}
}

func TestFrameInvalidLength(t *testing.T) {
	buf := []byte{SyncByte, 0x00, 0x16, 0x00} // length 0 is invalid
	f := &Frame{}
	_, err := f.Unmarshal(buf)
	if err != ErrInvalidLength {
		t.Errorf("expected ErrInvalidLength, got %v", err)
	}
}

func TestChannelsPackUnpack(t *testing.T) {
	ch := &Channels{}
	ch.Ch[0] = 172
	ch.Ch[1] = 992
	ch.Ch[2] = 1811
	ch.Ch[15] = 1500

	payload := MarshalChannels(ch)
	if len(payload) != 22 {
		t.Fatalf("payload len = %d, want 22", len(payload))
	}

	got, err := UnmarshalChannels(payload)
	if err != nil {
		t.Fatal(err)
	}
	for i := range NumChannels {
		if got.Ch[i] != ch.Ch[i] {
			t.Errorf("Ch[%d] = %d, want %d", i, got.Ch[i], ch.Ch[i])
		}
	}
}

func TestChannelsBoundary(t *testing.T) {
	ch := &Channels{}
	ch.Ch[0] = ChannelMin
	ch.Ch[1] = ChannelMax
	ch.Ch[15] = 0x7FF // max 11-bit

	payload := MarshalChannels(ch)
	got, err := UnmarshalChannels(payload)
	if err != nil {
		t.Fatal(err)
	}
	if got.Ch[0] != ChannelMin {
		t.Errorf("Ch[0] = %d, want %d", got.Ch[0], ChannelMin)
	}
	if got.Ch[1] != ChannelMax {
		t.Errorf("Ch[1] = %d, want %d", got.Ch[1], ChannelMax)
	}
	if got.Ch[15] != 0x7FF {
		t.Errorf("Ch[15] = %d, want %d", got.Ch[15], 0x7FF)
	}
}

func TestChannelsMicroseconds(t *testing.T) {
	tests := []struct {
		us  uint16
		tol uint16 // ±1µs tolerance from integer quantization
	}{
		{1500, 0}, {1000, 0}, {2000, 0}, {988, 1}, {2012, 1},
	}
	ch := &Channels{}
	for _, tt := range tests {
		ch.SetMicroseconds(0, tt.us)
		got := ch.Microseconds(0)
		diff := int(got) - int(tt.us)
		if diff < 0 {
			diff = -diff
		}
		if diff > int(tt.tol) {
			t.Errorf("round-trip %dµs → %dµs (tolerance ±%d)", tt.us, got, tt.tol)
		}
	}
}

func TestChannelsPackTo(t *testing.T) {
	ch := &Channels{}
	ch.Ch[3] = 992
	var buf [22]byte
	n := MarshalChannelsTo(buf[:], ch)
	if n != 22 {
		t.Fatalf("MarshalChannelsTo returned %d, want 22", n)
	}
	got, err := UnmarshalChannels(buf[:])
	if err != nil {
		t.Fatal(err)
	}
	if got.Ch[3] != 992 {
		t.Errorf("Ch[3] = %d, want 992", got.Ch[3])
	}
}

func assertFloat32(t *testing.T, name string, got, want, eps float32) {
	t.Helper()
	if diff := got - want; diff > eps || diff < -eps {
		t.Errorf("%s = %f, want %f (diff %f)", name, got, want, diff)
	}
}

func assertFloat64(t *testing.T, name string, got, want, eps float64) {
	t.Helper()
	if diff := got - want; diff > eps || diff < -eps {
		t.Errorf("%s = %f, want %f (diff %f)", name, got, want, diff)
	}
}

func TestBatteryRoundTrip(t *testing.T) {
	original := &Battery{
		Voltage:      11.4,
		Current:      5.2,
		CapacityUsed: 1200,
		Remaining:    72,
	}
	f := EncodeBattery(original)
	got, err := DecodeBattery(f)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32(t, "Voltage", got.Voltage, 11.4, 0.15)
	assertFloat32(t, "Current", got.Current, 5.2, 0.15)
	if got.CapacityUsed != 1200 {
		t.Errorf("CapacityUsed = %d, want 1200", got.CapacityUsed)
	}
	if got.Remaining != 72 {
		t.Errorf("Remaining = %d, want 72", got.Remaining)
	}
}

func TestGPSRoundTrip(t *testing.T) {
	original := &GPS{
		Latitude:   54.6872,
		Longitude:  25.2797,
		Speed:      12.5,
		Heading:    180.0,
		Altitude:   150.0,
		Satellites: 12,
	}
	f := EncodeGPS(original)
	got, err := DecodeGPS(f)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat64(t, "Latitude", got.Latitude, 54.6872, 1e-6)
	assertFloat64(t, "Longitude", got.Longitude, 25.2797, 1e-6)
	assertFloat32(t, "Speed", got.Speed, 12.5, 0.02)
	assertFloat32(t, "Heading", got.Heading, 180.0, 0.02)
	assertFloat32(t, "Altitude", got.Altitude, 150.0, 1.0)
	if got.Satellites != 12 {
		t.Errorf("Satellites = %d, want 12", got.Satellites)
	}
}

func TestAttitudeRoundTrip(t *testing.T) {
	original := &Attitude{
		Pitch: 0.5,
		Roll:  -0.3,
		Yaw:   3.14,
	}
	f := EncodeAttitude(original)
	got, err := DecodeAttitude(f)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32(t, "Pitch", got.Pitch, 0.5, 1e-3)
	assertFloat32(t, "Roll", got.Roll, -0.3, 1e-3)
	assertFloat32(t, "Yaw", got.Yaw, 3.14, 1e-3)
}

func TestLinkStatsRoundTrip(t *testing.T) {
	original := &LinkStatistics{
		UplinkRSSI1:     -80,
		UplinkRSSI2:     -90,
		UplinkQuality:   95,
		UplinkSNR:       5,
		ActiveAntenna:   1,
		RFProfile:       2,
		UplinkPower:     3,
		DownlinkRSSI:    -75,
		DownlinkQuality: 88,
		DownlinkSNR:     -2,
	}
	f := EncodeLinkStats(original)
	got, err := DecodeLinkStats(f)
	if err != nil {
		t.Fatal(err)
	}
	if got.UplinkRSSI1 != -80 {
		t.Errorf("UplinkRSSI1 = %d, want -80", got.UplinkRSSI1)
	}
	if got.UplinkQuality != 95 {
		t.Errorf("UplinkQuality = %d, want 95", got.UplinkQuality)
	}
	if got.DownlinkRSSI != -75 {
		t.Errorf("DownlinkRSSI = %d, want -75", got.DownlinkRSSI)
	}
	if got.DownlinkSNR != -2 {
		t.Errorf("DownlinkSNR = %d, want -2", got.DownlinkSNR)
	}
}

func TestFlightModeRoundTrip(t *testing.T) {
	original := &FlightMode{Mode: "ACRO"}
	f := EncodeFlightMode(original)
	got, err := DecodeFlightMode(f)
	if err != nil {
		t.Fatal(err)
	}
	if got.Mode != "ACRO" {
		t.Errorf("Mode = %q, want %q", got.Mode, "ACRO")
	}
}

func TestHeartbeatRoundTrip(t *testing.T) {
	original := &Heartbeat{Origin: AddrFlightController}
	f := EncodeHeartbeat(original)
	got, err := DecodeHeartbeat(f)
	if err != nil {
		t.Fatal(err)
	}
	if got.Origin != AddrFlightController {
		t.Errorf("Origin = 0x%02X, want 0x%02X", got.Origin, AddrFlightController)
	}
}

func TestBaroAltitudeRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		alt  float32
		vs   float32
	}{
		{"zero", 0, 0},
		{"positive low", 50.0, 2.0},
		{"negative", -100.0, -5.0},
		{"high meter mode", 2500.0, 10.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := &BaroAltitude{Altitude: tt.alt, VerticalSpeed: tt.vs}
			f := EncodeBaroAltitude(original)
			got, err := DecodeBaroAltitude(f)
			if err != nil {
				t.Fatal(err)
			}
			// Altitude: dm resolution in low range, 1m in high range
			altEps := float32(1.0)
			if math.Abs(float64(tt.alt)) > 2000 {
				altEps = 5.0
			}
			assertFloat32(t, "Altitude", got.Altitude, tt.alt, altEps)
			// Vario: logarithmic encoding has limited precision (~3cm/s at low, worse at higher)
			vsEps := float32(0.2)
			if math.Abs(float64(tt.vs)) > 3 {
				vsEps = 0.5
			}
			if math.Abs(float64(tt.vs)) > 10 {
				vsEps = 2.0
			}
			assertFloat32(t, "VerticalSpeed", got.VerticalSpeed, tt.vs, vsEps)
		})
	}
}

func TestDecoderStream(t *testing.T) {
	ch := &Channels{}
	ch.Ch[0] = 992
	ch.Ch[2] = 500

	f := EncodeChannels(ch)
	var buf [MaxFrameSize]byte
	n, _ := f.Marshal(buf[:])

	// Prefix with garbage then two valid frames
	stream := append([]byte{0xFF, 0x00, 0x42}, buf[:n]...)
	stream = append(stream, buf[:n]...)

	dec := NewDecoder(bytes.NewReader(stream))

	for i := range 2 {
		frame, err := dec.ReadFrame()
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		decoded, err := DecodeChannels(frame)
		if err != nil {
			t.Fatal(err)
		}
		if decoded.Ch[0] != 992 || decoded.Ch[2] != 500 {
			t.Errorf("frame %d: channels mismatch", i)
		}
	}
}

func TestEncoderDecoder(t *testing.T) {
	var pipe bytes.Buffer
	enc := NewEncoder(&pipe)
	dec := NewDecoder(&pipe)

	bat := &Battery{Voltage: 12.6, Current: 3.0, Remaining: 85}
	if err := enc.WriteBattery(bat); err != nil {
		t.Fatal(err)
	}

	frame, err := dec.ReadFrame()
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeBattery(frame)
	if err != nil {
		t.Fatal(err)
	}
	if got.Remaining != 85 {
		t.Errorf("remaining = %d, want 85", got.Remaining)
	}
	assertFloat32(t, "Voltage", got.Voltage, 12.6, 0.15)
}

func TestTypeMismatch(t *testing.T) {
	f := &Frame{Type: TypeBattery, Payload: make([]byte, 10)}
	if _, err := DecodeChannels(f); err != ErrTypeMismatch {
		t.Errorf("expected ErrTypeMismatch, got %v", err)
	}
	if _, err := DecodeGPS(f); err != ErrTypeMismatch {
		t.Errorf("expected ErrTypeMismatch, got %v", err)
	}
}

func BenchmarkCRC8(b *testing.B) {
	data := make([]byte, 22)
	b.ReportAllocs()
	for range b.N {
		CRC8(data)
	}
}

func BenchmarkPackChannels(b *testing.B) {
	ch := &Channels{}
	ch.Ch[0] = 992
	var buf [22]byte
	b.ReportAllocs()
	for range b.N {
		clear(buf[:])
		packChannels(buf[:], ch)
	}
}

func BenchmarkUnpackChannels(b *testing.B) {
	payload := MarshalChannels(&Channels{})
	ch := &Channels{}
	b.ReportAllocs()
	for range b.N {
		unpackChannels(payload, ch)
	}
}
