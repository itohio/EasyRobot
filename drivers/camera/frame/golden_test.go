package frame

import (
	"bytes"
	"encoding/hex"
	"strings"
	"testing"
)

func unhex(t *testing.T, s string) []byte {
	t.Helper()
	b, err := hex.DecodeString(strings.ReplaceAll(s, " ", ""))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// The vectors in the package doc (doc.go); their CRCs were checked with an
// independent CRC-32 implementation (Python zlib.crc32).
func TestGoldenVectors(t *testing.T) {
	img := unhex(t, "45 52 49 4d 00 00 00 00 00 00 01 00 14 00 07 00 44 33 22 11 3a 5c bd cc"+
		"02 00 02 00 01 01 00 00 04 00 00 00 02 00 00 00"+
		"01 02 03 04")
	var b bytes.Buffer
	if err := NewEncoder(&b).WriteImage(Meta{Width: 2, Height: 2, Format: Y8, Sensor: SensorOV2640}, 0x11223344, []byte{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b.Bytes(), img) {
		t.Fatalf("encoder\n got % x\nwant % x", b.Bytes(), img)
	}
	got, err := NewDecoder(bytes.NewReader(img)).Next()
	if err != nil || got.Seq != 0 || got.TS != 0x11223344 || got.Meta.Sensor != SensorOV2640 || !bytes.Equal(got.Data, []byte{1, 2, 3, 4}) {
		t.Fatalf("decoder: %+v, %v", got, err)
	}

	for _, tt := range []struct {
		c    Control
		wire string
	}{
		{Control{Kind: Nak, Seq: 7, Chunk: 3}, "45 52 49 4e 07 00 00 00 03 00 01 00 23 01 53 86"},
		{Control{Kind: Ack, Seq: 7}, "45 52 49 4e 07 00 00 00 00 00 02 00 0e fd cb bf"},
	} {
		var w [ControlSize]byte
		tt.c.Put(&w)
		if want := unhex(t, tt.wire); !bytes.Equal(w[:], want) {
			t.Errorf("%+v\n got % x\nwant % x", tt.c, w[:], want)
		}
		var p ControlParser
		var parsed Control
		var ok bool
		for _, x := range w {
			parsed, ok = p.Feed(x)
		}
		if !ok || parsed != tt.c {
			t.Errorf("parser: %+v, %v; want %+v", parsed, ok, tt.c)
		}
	}
}
