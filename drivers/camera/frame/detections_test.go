package frame

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"reflect"
	"testing"
)

func sampleDetections() Detections {
	return Detections{
		Seq: 7, Width: 320, Height: 240,
		Times: StageTimes{Capture: 81_000, Letterbox: 412_000, Detect: 32_790_000, Send: 13_400_000},
		Dets: []Detection{
			{Class: 63, Score: 37224, X0: 55, Y0: 13, X1: 307, Y1: 287},
			{Class: 0, Score: 65535, X0: -4, Y0: 0, X1: 320, Y1: 240},
		},
	}
}

func TestDetectionsRoundTrip(t *testing.T) {
	for _, d := range []Detections{sampleDetections(), {Seq: 0xFFFFFFFF, Width: 1, Height: 1}} {
		b := AppendDetections(nil, &d)
		if want := DetHeaderSize + DetSize*len(d.Dets) + 4; len(b) != want {
			t.Fatalf("%d bytes, want %d", len(b), want)
		}
		got, n, err := ParseDetections(b)
		if err != nil {
			t.Fatal(err)
		}
		if n != len(b) {
			t.Errorf("consumed %d of %d bytes", n, len(b))
		}
		if len(d.Dets) == 0 {
			d.Dets = nil
		}
		if !reflect.DeepEqual(got, d) {
			t.Errorf("round trip:\n got %+v\nwant %+v", got, d)
		}
	}
}

// The wire layout is the one doc.go specifies (golden vector there).
func TestDetectionsGolden(t *testing.T) {
	d := Detections{Seq: 7, Width: 320, Height: 240,
		Times: StageTimes{Capture: 1, Letterbox: 2, Detect: 3, Send: 4},
		Dets:  []Detection{{Class: 63, Score: 0x8000, X0: -1, Y0: 2, X1: 300, Y1: 200}}}
	want := []byte{
		0x45, 0x52, 0x49, 0x44, 0x07, 0x00, 0x00, 0x00, 0x01, 0x00, 0x40, 0x01, 0xf0, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
		0x3f, 0x00, 0x00, 0x80, 0xff, 0xff, 0x02, 0x00, 0x2c, 0x01, 0xc8, 0x00, 0xc0, 0xb5, 0xda, 0xed,
	}
	if got := AppendDetections(nil, &d); !bytes.Equal(got, want) {
		t.Errorf("wire bytes\n got % x\nwant % x", got, want)
	}
}

func TestParseDetectionsRejects(t *testing.T) {
	good := AppendDetections(nil, ptr(sampleDetections()))
	mutate := func(f func(b []byte) []byte) []byte { return f(append([]byte(nil), good...)) }
	recrc := func(b []byte) []byte {
		n := len(b) - 4
		binary.LittleEndian.PutUint32(b[n:], crc32.ChecksumIEEE(b[:n]))
		return b
	}
	tooMany := make([]byte, DetHeaderSize)
	copy(tooMany, good[:DetHeaderSize])
	binary.LittleEndian.PutUint16(tooMany[8:], MaxDetections+1)
	cases := []struct {
		name string
		b    []byte
		want error
	}{
		{"bad magic", mutate(func(b []byte) []byte { b[3] = 'M'; return recrc(b) }), ErrBadMagic},
		{"bad crc", mutate(func(b []byte) []byte { b[len(b)-1] ^= 1; return b }), ErrCRC},
		{"payload bit flip", mutate(func(b []byte) []byte { b[DetHeaderSize+5] ^= 0x10; return b }), ErrCRC},
		{"count above max", tooMany, ErrCount},
		{"truncated header", good[:DetHeaderSize-1], ErrShort},
		{"truncated detections", good[:len(good)-5], ErrShort},
		{"truncated crc", good[:len(good)-1], ErrShort},
		{"zero size image", mutate(func(b []byte) []byte { b[10], b[11] = 0, 0; return recrc(b) }), ErrMeta},
	}
	for _, c := range cases {
		if _, _, err := ParseDetections(c.b); !errors.Is(err, c.want) {
			t.Errorf("%s: %v, want %v", c.name, err, c.want)
		}
	}
}

// Old receivers that know only image chunks still decode the image; the
// record reaches them as text.
func TestDetectionsInStreamAreTextToImageDecoder(t *testing.T) {
	var s bytes.Buffer
	s.WriteString("log line\n")
	if err := NewEncoder(&s).WriteImage(Meta{Width: 2, Height: 2, Format: Y8, Sensor: SensorOV2640}, 5, []byte{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	d := sampleDetections()
	d.Seq = 0
	rec := AppendDetections(nil, &d)
	s.Write(rec)
	var text bytes.Buffer
	dec := NewDecoder(&s)
	dec.Text = &text
	img, err := dec.Next()
	if err != nil || !bytes.Equal(img.Data, []byte{1, 2, 3, 4}) {
		t.Fatalf("image: %v %v", img.Data, err)
	}
	if _, err := dec.Next(); err != io.EOF {
		t.Fatalf("after the image: %v, want EOF", err)
	}
	want := append([]byte("log line\n"), rec...)
	if !bytes.Equal(text.Bytes(), want) {
		t.Errorf("text = % x\nwant % x", text.Bytes(), want)
	}
	got, _, err := ParseDetections(text.Bytes()[len("log line\n"):])
	if err != nil || got.Seq != img.Seq {
		t.Errorf("record from text: seq %d, %v; image seq %d", got.Seq, err, img.Seq)
	}
}

func TestAppendDetectionsAllocatesNothing(t *testing.T) {
	d := sampleDetections()
	buf := make([]byte, 0, 256)
	if n := testing.AllocsPerRun(100, func() { buf = AppendDetections(buf[:0], &d) }); n != 0 {
		t.Errorf("%v allocations per call", n)
	}
}

func ptr[T any](v T) *T { return &v }
