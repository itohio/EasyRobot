package ld06

import (
	"context"
	"testing"
	"time"

	mat "github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
)

type stubSerial struct {
	data []byte
	pos  int
}

func (s *stubSerial) Read(p []byte) (int, error) {
	if s.pos >= len(s.data) {
		time.Sleep(5 * time.Millisecond)
		return 0, nil
	}
	n := copy(p, s.data[s.pos:])
	s.pos += n
	return n, nil
}

func (s *stubSerial) Write(p []byte) (int, error) { return len(p), nil }
func (s *stubSerial) Buffered() int               { return 0 }

func TestCRC8(t *testing.T) {
	// Test CRC8 calculation
	b := []byte{0x54, 0x01, 0x00, 0x00}
	crc := crc8(b)
	if crc == 0 {
		t.Fatal("CRC should not be zero")
	}

	// Test that CRC validates correctly
	packet := append(b, crc)
	if crc8(packet[:len(packet)-1]) != packet[len(packet)-1] {
		t.Fatal("CRC validation failed")
	}
}

func TestParseAndAssembleFullRotation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	ser := &stubSerial{}
	dev := New(ctx, ser, nil, 0, 3600) // no motor, no target points

	var stream []byte
	// Build packets that form a full rotation
	// Packet 1: 0° to 120°
	dist1 := make([]uint16, 10)
	int1 := make([]uint8, 10)
	for i := range dist1 {
		dist1[i] = uint16(1000 + i*10)
		int1[i] = 100
	}
	pkt1 := BuildMeasurementPacket(10.0, 0.0, 120.0, dist1, int1)
	stream = append(stream, pkt1...)

	// Packet 2: 120° to 240°
	dist2 := make([]uint16, 10)
	int2 := make([]uint8, 10)
	for i := range dist2 {
		dist2[i] = uint16(2000 + i*10)
		int2[i] = 100
	}
	pkt2 := BuildMeasurementPacket(10.0, 120.0, 240.0, dist2, int2)
	stream = append(stream, pkt2...)

	// Packet 3: 240° to 10° (wraparound - triggers rotation complete)
	dist3 := make([]uint16, 10)
	int3 := make([]uint8, 10)
	for i := range dist3 {
		dist3[i] = uint16(3000 + i*10)
		int3[i] = 100
	}
	pkt3 := BuildMeasurementPacket(10.0, 240.0, 10.0, dist3, int3)
	stream = append(stream, pkt3...)

	ser.data = stream

	scanReceived := make(chan bool, 1)
	dev.OnRead(func(m matTypes.Matrix) {
		if m.Cols() > 0 {
			scanReceived <- true
		}
	})

	_ = dev.Configure(false)

	select {
	case <-scanReceived:
		// Success
		dst := mat.New(2, 100)
		n := dev.Read(dst)
		if n == 0 {
			t.Fatal("expected scan data")
		}
		if n != 30 {
			t.Fatalf("expected 30 points, got %d", n)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("timeout waiting for scan")
	}
}
