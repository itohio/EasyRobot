package camera

// Host-independent helpers used by capture backends (ESP32-S3); no build
// tag, so they are tested on the host.

// jpegSpan returns b from its SOI (FF D8 at offset 0) through its EOI
// (FF D9). A JPEG without a scan (SOS) before the EOI is rejected. It walks the marker segments up to the first SOS by their
// lengths, so table bytes cannot be mistaken for an EOI, then scans the
// entropy-coded data, where a literal FF is always followed by 00 or a
// restart marker. Sensors pad the frame after the EOI (OV3660, M3 evidence).
func jpegSpan(b []byte) ([]byte, bool) {
	if len(b) < 4 || b[0] != 0xFF || b[1] != 0xD8 {
		return nil, false
	}
	i := 2
	for {
		for i < len(b) && b[i] == 0xFF && i+1 < len(b) && b[i+1] == 0xFF {
			i++ // fill bytes
		}
		if i+4 > len(b) || b[i] != 0xFF {
			return nil, false
		}
		m := b[i+1]
		if m == 0xD9 { // EOI before any scan: a header without image data
			return nil, false
		}
		if m == 0x01 || m >= 0xD0 && m <= 0xD7 { // standalone markers
			i += 2
			continue
		}
		n := int(b[i+2])<<8 | int(b[i+3])
		if n < 2 {
			return nil, false
		}
		i += 2 + n
		if m == 0xDA { // SOS: entropy-coded data follows
			break
		}
	}
	for ; i+1 < len(b); i++ {
		if b[i] == 0xFF && b[i+1] == 0xD9 {
			return b[:i+2], true
		}
	}
	return nil, false
}

// swap16 swaps the bytes of each 16-bit word in place (big- to
// little-endian RGB565). A trailing odd byte is left alone.
func swap16(b []byte) {
	for i := 0; i+1 < len(b); i += 2 {
		b[i], b[i+1] = b[i+1], b[i]
	}
}
