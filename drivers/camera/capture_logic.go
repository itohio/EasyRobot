package camera

// Host-independent capture decisions shared by the backends (no build tag,
// tested on the host).

// frameResult turns what the DMA delivered into the frame data or an error.
// buf is the capture buffer, got the bytes received, ended whether the frame
// ended (VSYNC/EOF) before the buffer or the deadline ran out.
//   - JPEG: a complete JPEG (SOI … SOS … EOI) inside the received bytes is
//     the frame, even if the sensor's padding after the EOI overran the
//     buffer. Otherwise a full buffer means the JPEG is larger than it
//     (ErrBufSize), anything else an incomplete frame (ErrTimeout).
//   - Raw: the frame ended with exactly frameLen bytes, else ErrTimeout.
func frameResult(jpeg bool, buf []byte, got, frameLen int, ended bool) ([]byte, error) {
	if jpeg {
		if data, ok := jpegSpan(buf[:got]); ok {
			return data, nil
		}
		if !ended && got == len(buf) {
			return nil, ErrBufSize
		}
		return nil, ErrTimeout
	}
	if !ended || got != frameLen {
		return nil, ErrTimeout
	}
	return buf[:frameLen], nil
}

// modeErr maps the sensor driver's "mode not supported" sentinel (driverErr,
// e.g. ov3660.ErrMode) to the shared ErrFormat (contract m8); other errors
// pass through. Each backend passes its own driver's sentinel, so a
// firmware build links only the drivers it uses. The ESP32-S3 backend asks
// ov3660.Supported first instead: an interface comparison links TinyGo's
// reflectlite there (~9 KB).
func modeErr(err, driverErr error) error {
	if err != nil && err == driverErr {
		return ErrFormat
	}
	return err
}

// GDMA descriptor chain arithmetic (ESP32-S3 backend; pure, host-tested).
// A descriptor's first word is size[11:0] | length[23:12] | suc_eof[30] |
// owner[31]; descriptors are 12 bytes (dw0, buffer, next).
const (
	nodeBytes = 4032 // descriptor buffer: ≤ 4095, multiple of 4 and of the 32 B cache line
	descSize  = 12
)

// chainNodes is the number of descriptors covering n bytes.
func chainNodes(n int) int { return (n + nodeBytes - 1) / nodeBytes }

// nodeSize is the buffer size of descriptor i of a chain covering n bytes.
func nodeSize(n, i int) int {
	if i == chainNodes(n)-1 {
		return n - i*nodeBytes
	}
	return nodeBytes
}

// descWord is a fresh descriptor's first word: owned by the DMA, size bytes
// of buffer, nothing received yet.
func descWord(size int) uint32 { return 1<<31 | uint32(size) }

// descLength is the byte count the DMA wrote into a descriptor.
func descLength(dw0 uint32) int { return int(dw0>>12) & 0xFFF }

// eofIndex is the index of the descriptor at eofAddr (IN_SUC_EOF_DES_ADDR) in
// a table of used descriptors at base, or -1 if eofAddr is none of them.
func eofIndex(eofAddr, base uint32, used int) int {
	if eofAddr < base || (eofAddr-base)%descSize != 0 {
		return -1
	}
	i := int((eofAddr - base) / descSize)
	if i >= used {
		return -1
	}
	return i
}
