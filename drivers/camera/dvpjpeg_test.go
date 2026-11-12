package camera

import (
	"encoding/binary"
	"errors"
	"testing"
	"unsafe"
)

// dvpLayout lays a byte stream out as the K210 DVP stores it: each 32-bit
// word holds four stream bytes in reverse order. The result is 4-byte
// aligned.
func dvpLayout(stream []byte) []byte {
	w := make([]uint32, (len(stream)+3)/4)
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(w))), 4*len(w))
	for i := 0; i+4 <= len(stream); i += 4 {
		binary.NativeEndian.PutUint32(b[i:], binary.LittleEndian.Uint32([]byte{stream[i+3], stream[i+2], stream[i+1], stream[i]}))
	}
	return b
}

// TestDVPJPEG: the stream comes back in order, cut right after FF D9 at
// every byte offset within a word (also when FF and D9 straddle words);
// frames without either marker are ErrIncomplete.
func TestDVPJPEG(t *testing.T) {
	jpeg := []byte{0xFF, 0xD8, 0xFF, 0xDB, 1, 2, 0xFF, 0x00, 3, 4, 5, 0xFF, 0xD9}
	for pad := range 4 {
		stream := append(append([]byte{}, jpeg[:2]...), make([]byte, pad)...)
		stream = append(stream, jpeg[2:]...)
		want := len(stream)
		stream = append(stream, make([]byte, 64-len(stream))...)
		b := dvpLayout(stream)
		n, err := dvpJPEG(b)
		if err != nil || n != want || string(b[:n]) != string(stream[:want]) {
			t.Errorf("pad %d: n %d err %v bytes % x, want % x", pad, n, err, b[:n], stream[:want])
		}
	}
	for name, stream := range map[string][]byte{
		"no FF D9": append([]byte{0xFF, 0xD8, 1, 2}, make([]byte, 60)...),
		"no FF D8": append([]byte{0, 0, 0xFF, 0xD9}, make([]byte, 60)...),
	} {
		if _, err := dvpJPEG(dvpLayout(stream)); !errors.Is(err, ErrIncomplete) {
			t.Errorf("%s: err %v, want ErrIncomplete", name, err)
		}
	}
}
