package camera

import (
	"errors"
	"testing"

	"github.com/itohio/EasyRobot/drivers/camera/sensor/ov2640"
	"github.com/itohio/EasyRobot/drivers/camera/sensor/ov3660"
)

func TestFrameResultJPEG(t *testing.T) {
	j := encodeJPEG(t)
	pad := func(n int) []byte { // the JPEG followed by sensor padding, n bytes in all
		b := make([]byte, n)
		copy(b, j)
		for i := len(j); i < n; i++ {
			b[i] = 0x55
		}
		return b
	}
	buf := pad(len(j) + 512)
	for _, tt := range []struct {
		name  string
		got   int
		ended bool
		err   error
	}{
		{"frame ended after the JPEG", len(buf), true, nil},
		{"padding overran the buffer, EOI inside (review C4)", len(buf), false, nil},
	} {
		data, err := frameResult(true, buf, tt.got, 0, tt.ended)
		if err != tt.err || len(data) != len(j) {
			t.Errorf("%s: %d bytes %v, want %d nil", tt.name, len(data), err, len(j))
		}
	}
	short := pad(len(j))[:len(j)/2] // the buffer holds only half the JPEG
	if _, err := frameResult(true, short, len(short), 0, false); err != ErrBufSize {
		t.Errorf("JPEG larger than the buffer: %v, want ErrBufSize", err)
	}
	if _, err := frameResult(true, buf, len(j)/2, 0, false); err != ErrTimeout {
		t.Errorf("frame stopped mid-JPEG with room left: %v, want ErrTimeout", err)
	}
	noSOI := append([]byte{0x00}, buf...)
	if _, err := frameResult(true, noSOI, len(noSOI), 0, true); err != ErrTimeout {
		t.Errorf("no SOI: %v, want ErrTimeout", err)
	}
}

func TestFrameResultRaw(t *testing.T) {
	buf := make([]byte, 1024)
	if data, err := frameResult(false, buf, 1000, 1000, true); err != nil || len(data) != 1000 {
		t.Errorf("complete raw frame: %d %v", len(data), err)
	}
	for _, tt := range []struct {
		got   int
		ended bool
	}{{999, true}, {1001, true}, {1000, false}} {
		if _, err := frameResult(false, buf, tt.got, 1000, tt.ended); err != ErrTimeout {
			t.Errorf("got %d ended %v: %v, want ErrTimeout", tt.got, tt.ended, err)
		}
	}
}

func TestDriverModeErrorsMapToErrFormat(t *testing.T) {
	for _, driverErr := range []error{ov3660.ErrMode, ov2640.ErrMode} {
		if got := modeErr(driverErr, driverErr); got != ErrFormat {
			t.Errorf("%v → %v, want ErrFormat", driverErr, got)
		}
		other := errors.New("sccb nak")
		if modeErr(other, driverErr) != other || modeErr(nil, driverErr) != nil {
			t.Error("other errors must pass through")
		}
	}
}

func TestDescriptorChain(t *testing.T) {
	for _, tt := range []struct {
		n, nodes, last int
	}{{nodeBytes * 3, 3, nodeBytes}, {nodeBytes*2 + 100, 3, 100}, {1, 1, 1}, {153600, 39, 153600 - 38*nodeBytes}} {
		if got := chainNodes(tt.n); got != tt.nodes {
			t.Errorf("n %d: %d nodes, want %d", tt.n, got, tt.nodes)
		}
		total := 0
		for i := range tt.nodes {
			total += nodeSize(tt.n, i)
		}
		if last := nodeSize(tt.n, tt.nodes-1); last != tt.last || total != tt.n {
			t.Errorf("n %d: last node %d, total %d; want %d, %d", tt.n, last, total, tt.last, tt.n)
		}
	}
	if w := descWord(4032); w != 1<<31|4032 || descLength(w) != 0 {
		t.Errorf("descriptor word %#x (length %d), want owner=DMA, size 4032, length 0", w, descLength(w))
	}
	if got := descLength(1<<31 | 1<<30 | 1234<<12 | 4032); got != 1234 {
		t.Errorf("received length %d, want 1234", got)
	}
	// A full node reports 4032, above 2048: the 12-bit length field must be
	// read whole (verify V-BATCH-2 fix 2: a 0x7FF mask survived).
	if got := descLength(1<<31 | nodeBytes<<12 | nodeBytes); got != nodeBytes {
		t.Errorf("full node length %d, want %d", got, nodeBytes)
	}
}

func TestEOFIndex(t *testing.T) {
	const base = 0x3FC90000
	for _, tt := range []struct {
		eof  uint32
		used int
		want int
	}{
		{base, 3, 0}, {base + 24, 3, 2}, {base + 36, 3, -1}, // beyond the chain
		{base + 13, 3, -1}, // not a descriptor boundary
		{0, 3, -1},         // no EOF
		{base - 12, 3, -1},
	} {
		if got := eofIndex(tt.eof, base, tt.used); got != tt.want {
			t.Errorf("eof %#x used %d: %d, want %d", tt.eof, tt.used, got, tt.want)
		}
	}
}
