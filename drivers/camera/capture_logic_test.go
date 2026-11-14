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

// JPEG buffer size follows the configured resolution (review C3): W·H/4,
// whole descriptors, at least one, at most the backend maximum.
func TestJPEGBufSize(t *testing.T) {
	const max = 384 * nodeBytes
	for _, tt := range []struct {
		w, h uint16
		want int
	}{
		{2048, 1536, 196 * nodeBytes}, // 786,432 B → 195.04 nodes, rounded up
		{1024, 768, 49 * nodeBytes},   // 196,608 B: fits internal SRAM
		{320, 240, 5 * nodeBytes},
		{126, 128, nodeBytes}, // W·H/4 = 4,032 exactly: one descriptor, not two (review C3-3)
		{16, 16, nodeBytes},   // at least one descriptor
		{0, 0, nodeBytes},     // even for an empty frame (verify: the floor was never reached)
		{4096, 4096, max},     // capped
	} {
		if got := jpegBufSize(tt.w, tt.h, max); got != tt.want {
			t.Errorf("%dx%d: %d, want %d", tt.w, tt.h, got, tt.want)
		}
	}
	// Review C3-2: W·H is computed without overflowing the target's 32-bit
	// int for the whole uint16 range (65535² > 2³¹). Red only where int is 32
	// bits: run with tinygo test -target=wasm.
	const big = 266297 * nodeBytes // 65535²/4 = 1,073,709,056 B → 266,296.9 nodes
	if got := jpegBufSize(65535, 65535, 1<<31-1); got != big {
		t.Errorf("65535x65535 uncapped: %d, want %d", got, big)
	}
}

// GDMA RX status after a capture (IN_INT_RAW), review R-ESP32CAM-E7 P1: a
// FIFO overflow or underflow is ErrOverflow whatever else is set; EOF ends
// the frame; descriptor empty/error ends the transfer without a frame end.
func TestDMAOutcome(t *testing.T) {
	const (
		inDone    = 1 << 0
		sucEOF    = 1 << 1
		dscrErr   = 1 << 3
		dscrEmpty = 1 << 4
		fullWM    = 1 << 5
		ovfL1     = 1 << 6
		udfL1     = 1 << 7
		ovfL3     = 1 << 8
		udfL3     = 1 << 9
	)
	for _, tt := range []struct {
		raw   uint32
		ended bool
		err   error
	}{
		{sucEOF, true, nil},
		{sucEOF | fullWM, true, nil},     // watermark is not an error
		{dscrEmpty, false, nil},          // buffer ran out: frameResult decides (ErrBufSize/JPEG)
		{inDone | dscrEmpty, false, nil}, // IN_DONE (a descriptor filled) is not a frame end (verify fix C)
		{dscrErr, false, nil},
		{0, false, nil}, // deadline
		{sucEOF | ovfL1, true, ErrOverflow},
		{sucEOF | udfL1, true, ErrOverflow},
		{ovfL3, false, ErrOverflow},
		{dscrEmpty | udfL3, false, ErrOverflow},
	} {
		ended, err := dmaOutcome(tt.raw)
		if ended != tt.ended || err != tt.err {
			t.Errorf("raw %#x: ended %v err %v, want %v %v", tt.raw, ended, err, tt.ended, tt.err)
		}
	}
	// Each status that must end the wait, one bit at a time (verify: a
	// dmaDone checking only bit 8 of the four fault bits survived).
	for _, bit := range []uint{1, 3, 4, 6, 7, 8, 9} {
		if !dmaDone(1 << bit) {
			t.Errorf("dmaDone(bit %d) = false, want true", bit)
		}
	}
	for _, raw := range []uint32{0, 1 << 0, 1 << 2, 1 << 5} { // none, IN_DONE, ERR_EOF, watermark
		if dmaDone(raw) {
			t.Errorf("dmaDone(%#x) = true, want false", raw)
		}
	}
	for _, bit := range []uint{6, 7, 8, 9} {
		if _, err := dmaOutcome(1 << bit); err != ErrOverflow {
			t.Errorf("dmaOutcome(bit %d): %v, want ErrOverflow", bit, err)
		}
	}
}
