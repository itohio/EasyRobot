package camera

import (
	"errors"
	"unsafe"
)

// ErrIncomplete is returned by Capture for a JPEG frame without its start
// (FF D8) or end (FF D9) marker.
var ErrIncomplete = errors.New("camera: incomplete JPEG frame")

// dvpJPEG turns a K210 DVP JPEG capture in b (4-byte aligned, a multiple of
// 4 long) in place into the JPEG byte stream and returns its length up to
// and including FF D9. The DVP stores four sensor bytes per 32-bit word in
// reverse order (ER-K210-K6 evidence). With the OV2640's sensor-timed HREF
// every line is one full 256-byte chunk, so the lines are contiguous and
// need no reassembly (ER-K210-REC evidence, R3). Words are reversed only up
// to the end marker.
func dvpJPEG(b []byte) (int, error) {
	words := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/4)
	var prev byte
	for i, w := range words {
		w = w>>24 | (w>>8)&0xFF00 | (w<<8)&0xFF0000 | w<<24
		words[i] = w
		for k := range 4 {
			c := b[4*i+k]
			if prev == 0xFF && c == 0xD9 {
				if b[0] != 0xFF || b[1] != 0xD8 {
					return 0, ErrIncomplete
				}
				return 4*i + k + 1, nil
			}
			prev = c
		}
	}
	return 0, ErrIncomplete
}
