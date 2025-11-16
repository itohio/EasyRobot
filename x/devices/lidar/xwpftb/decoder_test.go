package xwpftb

import (
	"context"
	"testing"
	"time"

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

func TestCRC16Cumulative(t *testing.T) {
	// Simple known sequence
	b := []byte{0x01, 0x02, 0x03}
	got := crc16Cumulative(b)
	want := uint16(0x0006)
	if got != want {
		t.Fatalf("crc got %04x want %04x", got, want)
	}
}

func TestParseAndAssembleFullRotation(t *testing.T) {
	var stream []byte
	// Build 15 slices to complete rotation
	for i := 0; i < 15; i++ {
		start := float64(i) * 24.0
		// 10 samples in each slice
		dist := make([]float64, 10)
		for j := range dist {
			dist[j] = 1000.0 + float64(i*10+j)
		}
		frame := BuildMeasurementFrame(start, dist)
		stream = append(stream, frame...)
	}

	ss := &stubSerial{data: stream}
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	dev := New(ctx, ss, nil, 0, 256)
	defer dev.Close()

	done := make(chan struct{}, 1)
	var gotCols int
	dev.OnRead(func(m matTypes.Matrix) {
		gotCols = m.Cols()
		done <- struct{}{}
	})

	select {
	case <-done:
		if gotCols != 15*10 {
			t.Fatalf("unexpected columns: %d (want %d)", gotCols, 15*10)
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("timeout waiting for scan")
	}
}
