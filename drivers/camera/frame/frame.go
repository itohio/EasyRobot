package frame

import (
	"encoding/binary"
	"errors"
)

const (
	// Magic is the chunk magic; on the wire it reads "ERIM".
	Magic uint32 = 0x4D495245
	// HeaderSize is the size of a chunk header in bytes.
	HeaderSize = 24
	// MetaSize is the size of the image meta block in bytes.
	MetaSize = 16
	// MaxPayload is the largest payload of one chunk in bytes.
	MaxPayload = 4096
	// MaxChunks is the largest chunk count of one image (chunk_cnt is u16).
	MaxChunks = 0xFFFF
	// MaxDataLen is the largest image data length one image can carry.
	MaxDataLen = MaxChunks*MaxPayload - MetaSize
	// Version is the meta version byte this package writes and accepts.
	Version = 1
)

// Header flag bits. Other bits are written as 0 and ignored on receive
// (new flag bits are additive in v1).
const (
	FlagFirst uint16 = 1 << 0 // first chunk of an image
	FlagLast  uint16 = 1 << 1 // last chunk of an image
	FlagMeta  uint16 = 1 << 2 // payload starts with a Meta (first chunk only)
)

// Format is the pixel format of an image.
type Format uint8

// Pixel formats of format v1.
const (
	Y8           Format = 1 // 8-bit grey
	RGB565LE     Format = 2 // RGB565, little-endian 16-bit words
	RGB565BE     Format = 3 // RGB565, big-endian 16-bit words
	YUYV         Format = 4 // YUV 4:2:2, byte order Y0 U Y1 V
	JPEG         Format = 5 // JPEG file; DataLen is the exact file length
	RGB888Planar Format = 6 // three planes: R, then G, then B
)

// Known reports whether f is one of the v1 formats.
func (f Format) Known() bool { return f >= Y8 && f <= RGB888Planar }

// BytesPerPixel returns the bytes per pixel of f: 0 for JPEG and unknown
// formats, 3 for RGB888Planar (summed over the three planes).
func (f Format) BytesPerPixel() int {
	switch f {
	case Y8:
		return 1
	case RGB565LE, RGB565BE, YUYV:
		return 2
	case RGB888Planar:
		return 3
	}
	return 0
}

// String returns the format name, or "unknown".
func (f Format) String() string {
	switch f {
	case Y8:
		return "Y8"
	case RGB565LE:
		return "RGB565LE"
	case RGB565BE:
		return "RGB565BE"
	case YUYV:
		return "YUYV"
	case JPEG:
		return "JPEG"
	case RGB888Planar:
		return "RGB888Planar"
	}
	return "unknown"
}

// Sensor identifies the camera sensor that produced an image.
type Sensor uint8

// Sensor ids of format v1.
const (
	SensorUnknown Sensor = 0
	SensorOV7740  Sensor = 1
	SensorOV2640  Sensor = 2
	SensorOV3660  Sensor = 3
	SensorOV5640  Sensor = 4
)

// String returns the sensor name, or "unknown".
func (s Sensor) String() string {
	switch s {
	case SensorOV7740:
		return "OV7740"
	case SensorOV2640:
		return "OV2640"
	case SensorOV3660:
		return "OV3660"
	case SensorOV5640:
		return "OV5640"
	}
	return "unknown"
}

// Errors returned by this package.
var (
	ErrShort    = errors.New("frame: buffer too short")
	ErrVersion  = errors.New("frame: unsupported meta version")
	ErrMeta     = errors.New("frame: invalid image meta")
	ErrDataLen  = errors.New("frame: data length does not match meta")
	ErrTooLarge = errors.New("frame: image too large for one frame")
	ErrBadMagic = errors.New("frame: bad magic")
	ErrNoImage  = errors.New("frame: no image written yet")
	ErrChunk    = errors.New("frame: chunk index outside the last image")
)

// Header is a decoded chunk header.
type Header struct {
	Seq        uint32 // frame_seq: +1 per image, wraps
	ChunkIdx   uint16 // 0 .. ChunkCnt-1
	ChunkCnt   uint16 // ≥ 1
	PayloadLen uint16 // 1 .. MaxPayload
	Flags      uint16 // Flag* bits
	TS         uint32 // device clock at capture start, µs, low 32 bits
	CRC        uint32 // CRC-32/IEEE over header bytes 0..19, then the payload
}

// Put writes h, with the magic, into b.
func (h *Header) Put(b *[HeaderSize]byte) {
	binary.LittleEndian.PutUint32(b[0:], Magic)
	binary.LittleEndian.PutUint32(b[4:], h.Seq)
	binary.LittleEndian.PutUint16(b[8:], h.ChunkIdx)
	binary.LittleEndian.PutUint16(b[10:], h.ChunkCnt)
	binary.LittleEndian.PutUint16(b[12:], h.PayloadLen)
	binary.LittleEndian.PutUint16(b[14:], h.Flags)
	binary.LittleEndian.PutUint32(b[16:], h.TS)
	binary.LittleEndian.PutUint32(b[20:], h.CRC)
}

// ParseHeader decodes a chunk header from b. It checks the magic only; use
// [Header.Valid] for the receiver bounds.
func ParseHeader(b []byte) (Header, error) {
	if len(b) < HeaderSize {
		return Header{}, ErrShort
	}
	if binary.LittleEndian.Uint32(b) != Magic {
		return Header{}, ErrBadMagic
	}
	return Header{
		Seq:        binary.LittleEndian.Uint32(b[4:]),
		ChunkIdx:   binary.LittleEndian.Uint16(b[8:]),
		ChunkCnt:   binary.LittleEndian.Uint16(b[10:]),
		PayloadLen: binary.LittleEndian.Uint16(b[12:]),
		Flags:      binary.LittleEndian.Uint16(b[14:]),
		TS:         binary.LittleEndian.Uint32(b[16:]),
		CRC:        binary.LittleEndian.Uint32(b[20:]),
	}, nil
}

// Valid reports whether h passes the receiver's header checks: payload
// length 1..MaxPayload, chunk count ≥ 1, chunk index < chunk count, the
// first flag set exactly on chunk 0, the last flag set exactly on the last
// chunk, and the meta flag set only on (and required for) chunk 0.
func (h *Header) Valid() bool {
	if h.PayloadLen == 0 || h.PayloadLen > MaxPayload || h.ChunkCnt == 0 || h.ChunkIdx >= h.ChunkCnt {
		return false
	}
	first := h.ChunkIdx == 0
	last := h.ChunkIdx == h.ChunkCnt-1
	if (h.Flags&FlagFirst != 0) != first || (h.Flags&FlagLast != 0) != last || (h.Flags&FlagMeta != 0) != first {
		return false
	}
	return !first || h.PayloadLen >= MetaSize
}

// Meta describes the image carried by one frame. It is the first MetaSize
// bytes of the first chunk's payload.
type Meta struct {
	Width, Height uint16
	Format        Format
	// Stride is the row pitch in bytes; 0 means tight (Width × bytes per
	// pixel). For RGB888Planar it counts all three planes, so each plane
	// row is Stride/3 bytes. Ignored for JPEG.
	Stride uint16
	// DataLen is the image byte count over all chunks, meta excluded. For
	// JPEG it is the exact file length.
	DataLen uint32
	Sensor  Sensor
}

// Put writes m into b with the version byte set to [Version] and the
// reserved bytes zeroed.
func (m *Meta) Put(b *[MetaSize]byte) {
	binary.LittleEndian.PutUint16(b[0:], m.Width)
	binary.LittleEndian.PutUint16(b[2:], m.Height)
	b[4] = byte(m.Format)
	b[5] = Version
	binary.LittleEndian.PutUint16(b[6:], m.Stride)
	binary.LittleEndian.PutUint32(b[8:], m.DataLen)
	b[12] = byte(m.Sensor)
	b[13], b[14], b[15] = 0, 0, 0
}

// MarshalBinary returns the 16-byte wire form of m.
func (m Meta) MarshalBinary() ([]byte, error) {
	var b [MetaSize]byte
	m.Put(&b)
	return b[:], nil
}

// UnmarshalBinary decodes m from the first MetaSize bytes of b. It rejects
// a version byte other than [Version]; reserved bytes are ignored.
func (m *Meta) UnmarshalBinary(b []byte) error {
	if len(b) < MetaSize {
		return ErrShort
	}
	if b[5] != Version {
		return ErrVersion
	}
	*m = Meta{
		Width:   binary.LittleEndian.Uint16(b[0:]),
		Height:  binary.LittleEndian.Uint16(b[2:]),
		Format:  Format(b[4]),
		Stride:  binary.LittleEndian.Uint16(b[6:]),
		DataLen: binary.LittleEndian.Uint32(b[8:]),
		Sensor:  Sensor(b[12]),
	}
	return nil
}

// RowBytes returns the row pitch in bytes: Stride, or Width × bytes per
// pixel when Stride is 0. It returns 0 for JPEG and unknown formats.
func (m *Meta) RowBytes() int {
	bpp := m.Format.BytesPerPixel()
	if bpp == 0 {
		return 0
	}
	if m.Stride != 0 {
		return int(m.Stride)
	}
	return int(m.Width) * bpp
}

// Validate checks m for sanity: width, height and DataLen ≥ 1 and DataLen
// ≤ [MaxDataLen]; for the raw formats a stride of 0 or ≥ the tight row,
// RGB888Planar strides divisible by 3, even YUYV widths, and DataLen ≥
// Height × RowBytes. Unknown formats pass with only the generic checks,
// since new formats are additive in v1.
func (m *Meta) Validate() error {
	if m.Width == 0 || m.Height == 0 || m.DataLen == 0 || m.DataLen > MaxDataLen {
		return ErrMeta
	}
	bpp := m.Format.BytesPerPixel()
	if bpp == 0 {
		return nil
	}
	tight := int(m.Width) * bpp
	if m.Stride != 0 && int(m.Stride) < tight {
		return ErrMeta
	}
	if m.Format == RGB888Planar && m.Stride%3 != 0 {
		return ErrMeta
	}
	if m.Format == YUYV && m.Width%2 != 0 {
		return ErrMeta
	}
	if uint64(m.DataLen) < uint64(m.Height)*uint64(m.RowBytes()) {
		return ErrMeta
	}
	return nil
}

// Image is one decoded image.
type Image struct {
	Seq  uint32 // frame_seq
	TS   uint32 // frame_ts of the first chunk, µs
	Meta Meta
	// Data holds Meta.DataLen bytes. Each image gets its own slice.
	Data []byte
}
