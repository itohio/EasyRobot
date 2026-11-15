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
// reverse order (observed on the K210). With the OV2640's sensor-timed HREF
// every line is one full 256-byte chunk, so the lines are contiguous and
// need no reassembly (observed). Words are reversed only up
// to the end marker.
func dvpJPEG(b []byte) (int, error) {
	words := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/4)
	var prev byte
	for i, w := range words {
		w = w>>24 | (w>>8)&0xFF00 | (w<<8)&0xFF0000 | w<<24
		words[i] = w
		for k := range 4 {
			// The byte just stored, from the register: b is uncached on the
			// K210, so reading it back costs a bus load.
			c := byte(w >> (8 * k))
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

// captureJPEG runs one JPEG capture into b (the buffer the DVP writes, 8-byte
// aligned) and returns the frame's length: it zeroes b, lets capture fill
// it, and extracts the JPEG (dvpJPEG). The DVP writes only the lines the
// sensor sends, so without the clear a cut frame would end in the previous
// frame's bytes, which dvpJPEG could read as the end marker. Zero bytes
// never form FF D9.
func captureJPEG(b []byte, capture func() error) (int, error) {
	clearJPEG(b)
	if err := capture(); err != nil {
		return 0, err
	}
	return dvpJPEG(b)
}

// clearJPEG zeroes b, 8 bytes per store where b is 8-byte aligned (the DVP
// buffer always is): b is uncached on the K210, where the byte-wise clear
// costs a store per byte.
func clearJPEG(b []byte) {
	p := unsafe.Pointer(unsafe.SliceData(b))
	if uintptr(p)&7 != 0 {
		clear(b)
		return
	}
	clear(unsafe.Slice((*uint64)(p), len(b)/8))
	clear(b[len(b)&^7:])
}
