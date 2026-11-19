package frame

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
)

const (
	// DetMagic is the detection record magic; on the wire it reads "ERID".
	DetMagic uint32 = 0x44495245
	// DetHeaderSize is the size of a detection record header in bytes.
	DetHeaderSize = 32
	// DetSize is the size of one detection in a record in bytes.
	DetSize = 12
	// MaxDetections is the largest detection count of one record.
	MaxDetections = 64
)

// Errors returned for detection records (with ErrShort, ErrBadMagic and
// ErrMeta).
var (
	ErrCount = errors.New("frame: detection count above MaxDetections")
	ErrCRC   = errors.New("frame: CRC mismatch")
)

// StageTimes are the device's stage durations for one frame, in µs.
type StageTimes struct {
	Capture   uint32 // camera capture
	Letterbox uint32 // conversion to the detector's input
	Detect    uint32 // network and decode
	Send      uint32 // writing the previous frame (image chunks and record); 0 for the first
}

// Detection is one detected object: a class id (COCO for YOLOX), a score
// in Q0.16 (score = Score/65535) and a box in the pixel coordinates of the
// image with the record's frame_seq (corners; it may extend past the image).
type Detection struct {
	Class          uint8
	Score          uint16
	X0, Y0, X1, Y1 int16
}

// Detections is a detection record: what a detector found in the image
// with frame_seq Seq (Width × Height pixels), and the stage times.
type Detections struct {
	Seq           uint32
	Width, Height uint16
	Times         StageTimes
	Dets          []Detection // at most MaxDetections
}

// AppendDetections appends the wire form of d to b and returns it. d.Dets
// must hold at most MaxDetections entries (more are not written). It does
// not allocate when b has room for DetHeaderSize + DetSize·n + 4 bytes.
func AppendDetections(b []byte, d *Detections) []byte {
	n := min(len(d.Dets), MaxDetections)
	start := len(b)
	b = binary.LittleEndian.AppendUint32(b, DetMagic)
	b = binary.LittleEndian.AppendUint32(b, d.Seq)
	b = binary.LittleEndian.AppendUint16(b, uint16(n))
	b = binary.LittleEndian.AppendUint16(b, d.Width)
	b = binary.LittleEndian.AppendUint16(b, d.Height)
	b = binary.LittleEndian.AppendUint16(b, 0)
	b = binary.LittleEndian.AppendUint32(b, d.Times.Capture)
	b = binary.LittleEndian.AppendUint32(b, d.Times.Letterbox)
	b = binary.LittleEndian.AppendUint32(b, d.Times.Detect)
	b = binary.LittleEndian.AppendUint32(b, d.Times.Send)
	for _, x := range d.Dets[:n] {
		b = append(b, x.Class, 0)
		b = binary.LittleEndian.AppendUint16(b, x.Score)
		b = binary.LittleEndian.AppendUint16(b, uint16(x.X0))
		b = binary.LittleEndian.AppendUint16(b, uint16(x.Y0))
		b = binary.LittleEndian.AppendUint16(b, uint16(x.X1))
		b = binary.LittleEndian.AppendUint16(b, uint16(x.Y1))
	}
	return binary.LittleEndian.AppendUint32(b, crc32.ChecksumIEEE(b[start:]))
}

// ParseDetections decodes the detection record at the start of b and
// returns it with the number of bytes it occupies. It returns ErrShort for
// a truncated record, ErrBadMagic, ErrCount for a count above
// MaxDetections, ErrMeta for a zero width or height and ErrCRC for a CRC
// mismatch; a receiver scanning a stream rescans from magic+1 after any of
// them.
func ParseDetections(b []byte) (Detections, int, error) {
	if len(b) < DetHeaderSize {
		return Detections{}, 0, ErrShort
	}
	if binary.LittleEndian.Uint32(b) != DetMagic {
		return Detections{}, 0, ErrBadMagic
	}
	n := int(binary.LittleEndian.Uint16(b[8:]))
	if n > MaxDetections {
		return Detections{}, 0, ErrCount
	}
	size := DetHeaderSize + DetSize*n + 4
	if len(b) < size {
		return Detections{}, 0, ErrShort
	}
	if crc32.ChecksumIEEE(b[:size-4]) != binary.LittleEndian.Uint32(b[size-4:]) {
		return Detections{}, 0, ErrCRC
	}
	d := Detections{
		Seq:    binary.LittleEndian.Uint32(b[4:]),
		Width:  binary.LittleEndian.Uint16(b[10:]),
		Height: binary.LittleEndian.Uint16(b[12:]),
		Times: StageTimes{
			Capture:   binary.LittleEndian.Uint32(b[16:]),
			Letterbox: binary.LittleEndian.Uint32(b[20:]),
			Detect:    binary.LittleEndian.Uint32(b[24:]),
			Send:      binary.LittleEndian.Uint32(b[28:]),
		},
	}
	if d.Width == 0 || d.Height == 0 {
		return Detections{}, 0, ErrMeta
	}
	if n > 0 {
		d.Dets = make([]Detection, n)
	}
	for i := range d.Dets {
		p := b[DetHeaderSize+DetSize*i:]
		d.Dets[i] = Detection{
			Class: p[0],
			Score: binary.LittleEndian.Uint16(p[2:]),
			X0:    int16(binary.LittleEndian.Uint16(p[4:])),
			Y0:    int16(binary.LittleEndian.Uint16(p[6:])),
			X1:    int16(binary.LittleEndian.Uint16(p[8:])),
			Y1:    int16(binary.LittleEndian.Uint16(p[10:])),
		}
	}
	return d, size, nil
}
