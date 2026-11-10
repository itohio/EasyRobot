package camera

import (
	"bytes"
	"image"
	"image/jpeg"
	"testing"
)

func encodeJPEG(t *testing.T) []byte {
	t.Helper()
	img := image.NewGray(image.Rect(0, 0, 64, 48))
	for i := range img.Pix {
		img.Pix[i] = uint8(i * 7)
	}
	var b bytes.Buffer
	if err := jpeg.Encode(&b, img, &jpeg.Options{Quality: 90}); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestJPEGSpanCutsSensorPadding(t *testing.T) {
	j := encodeJPEG(t)
	// DMA buffer: the JPEG, then padding that contains an FF D9 of its own.
	frame := append(append([]byte{}, j...), 0x00, 0x00, 0xFF, 0xD9, 0x55, 0x55)
	got, ok := jpegSpan(frame)
	if !ok || !bytes.Equal(got, j) {
		t.Fatalf("span ok=%v len %d, want the %d-byte JPEG", ok, len(got), len(j))
	}
	if _, err := jpeg.Decode(bytes.NewReader(got)); err != nil {
		t.Errorf("span does not decode: %v", err)
	}
}

func TestJPEGSpanSkipsFFD9InsideSegments(t *testing.T) {
	frame := []byte{
		0xFF, 0xD8,
		0xFF, 0xDB, 0x00, 0x06, 0x00, 0xFF, 0xD9, 0x01, // DQT payload with FF D9 bytes
		0xFF, 0xDA, 0x00, 0x04, 0xFF, 0xD9, // SOS header with FF D9 bytes
		0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD0, 0x56, // entropy data: stuffed FF, restart marker
		0xFF, 0xD9, // EOI
		0xAA, 0xAA,
	}
	got, ok := jpegSpan(frame)
	if !ok || len(got) != len(frame)-2 {
		t.Fatalf("span ok=%v len %d, want %d", ok, len(got), len(frame)-2)
	}
}

func TestJPEGSpanRejectsBrokenFrames(t *testing.T) {
	j := encodeJPEG(t)
	for name, b := range map[string][]byte{
		"empty":                           nil,
		"no SOI":                          append([]byte{0x00}, j...),
		"truncated":                       j[:len(j)/2],
		"no EOI":                          j[:len(j)-2],
		"bad segment len":                 {0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x01, 0xFF, 0xD9},
		"segment overrun":                 {0xFF, 0xD8, 0xFF, 0xDB, 0x40, 0x00, 0xFF, 0xD9},
		"header only, no SOS (review C6)": {0xFF, 0xD8, 0xFF, 0xD9},
		"EOI before SOS":                  {0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x04, 0x01, 0x02, 0xFF, 0xD9},
	} {
		if got, ok := jpegSpan(b); ok {
			t.Errorf("%s: accepted %d bytes", name, len(got))
		}
	}
}

func TestSwap16MakesRGB565LittleEndian(t *testing.T) {
	// Big-endian RGB565 on the bus: pure red 0xF800, then pure blue 0x001F.
	b := []byte{0xF8, 0x00, 0x00, 0x1F, 0x77}
	swap16(b)
	if !bytes.Equal(b, []byte{0x00, 0xF8, 0x1F, 0x00, 0x77}) {
		t.Errorf("got % x", b)
	}
	if red := uint16(b[0]) | uint16(b[1])<<8; red != 0xF800 {
		t.Errorf("first pixel read little-endian = %#04x, want red 0xF800", red)
	}
}
