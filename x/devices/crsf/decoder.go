package crsf

import "io"

// Decoder reads CRSF frames from a byte stream.
// It handles sync-byte scanning and frame reassembly.
type Decoder struct {
	r   io.Reader
	buf [MaxFrameSize]byte
	pos int // valid bytes in buf
}

func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{r: r}
}

// ReadFrame reads the next valid frame from the stream.
// Scans forward to the next sync byte on framing errors.
// The returned Frame owns its Payload (copied from the internal buffer).
func (d *Decoder) ReadFrame() (*Frame, error) {
	for {
		if err := d.ensureBytes(2); err != nil {
			return nil, err
		}

		if !d.isSyncByte(d.buf[0]) {
			d.advance(1)
			continue
		}

		frameLen := int(d.buf[1])
		if frameLen < 2 || frameLen > 62 {
			d.advance(1)
			continue
		}

		total := 2 + frameLen
		if err := d.ensureBytes(total); err != nil {
			return nil, err
		}

		var f Frame
		if _, err := f.Unmarshal(d.buf[:total]); err != nil {
			d.advance(1)
			continue
		}

		// Copy payload out before advancing (Unmarshal returns a sub-slice of buf)
		if len(f.Payload) > 0 {
			owned := make([]byte, len(f.Payload))
			copy(owned, f.Payload)
			f.Payload = owned
		}

		d.advance(total)
		return &f, nil
	}
}

func (d *Decoder) ensureBytes(n int) error {
	for d.pos < n {
		read, err := d.r.Read(d.buf[d.pos:])
		d.pos += read
		if err != nil {
			if d.pos >= n {
				return nil
			}
			return err
		}
	}
	return nil
}

func (d *Decoder) advance(n int) {
	remaining := d.pos - n
	if remaining > 0 {
		copy(d.buf[:remaining], d.buf[n:d.pos])
	}
	d.pos = remaining
}

func (d *Decoder) isSyncByte(b byte) bool {
	return b == SyncByte || isDeviceAddress(b)
}
