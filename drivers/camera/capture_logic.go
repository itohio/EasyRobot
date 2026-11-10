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
