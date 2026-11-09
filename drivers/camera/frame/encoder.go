package frame

import (
	"hash/crc32"
	"io"
)

// Encoder writes images as frame v1 chunks to an io.Writer. It keeps the
// frame sequence number and fixed header/meta buffers, so WriteImage does
// not allocate. An Encoder is not safe for concurrent use.
type Encoder struct {
	w    io.Writer
	seq  uint32
	hdr  [HeaderSize]byte
	meta [MetaSize]byte

	// The last image written, for Resend.
	last    Header // Seq, ChunkCnt, TS of the last image
	data    []byte // caller's buffer of the last image
	hasLast bool
}

// NewEncoder returns an Encoder writing to w whose first image gets
// sequence number 0.
func NewEncoder(w io.Writer) *Encoder { return &Encoder{w: w} }

// Seq returns the sequence number the next image will carry.
func (e *Encoder) Seq() uint32 { return e.seq }

// WriteImage writes data as one image. m.DataLen must be 0 (it is then set
// to len(data)) or equal len(data); the meta must pass [Meta.Validate] and
// m.Format must be a v1 format. ts is the capture start in device µs.
//
// The image is split into chunks of at most MaxPayload payload bytes, the
// first of which starts with the 16-byte meta. The sequence number advances
// once per image that passes validation, even if a write fails, so the
// receiver counts a failed image as lost.
//
// The Encoder keeps a reference to data for [Encoder.Resend]: the caller must
// not change it until the next WriteImage.
func (e *Encoder) WriteImage(m Meta, ts uint32, data []byte) error {
	if uint64(len(data)) > MaxDataLen {
		return ErrTooLarge
	}
	if m.DataLen == 0 {
		m.DataLen = uint32(len(data))
	} else if m.DataLen != uint32(len(data)) {
		return ErrDataLen
	}
	if !m.Format.Known() {
		return ErrMeta
	}
	if err := m.Validate(); err != nil {
		return err
	}
	m.Put(&e.meta)

	total := MetaSize + len(data)
	cnt := (total + MaxPayload - 1) / MaxPayload
	e.last = Header{Seq: e.seq, ChunkCnt: uint16(cnt), TS: ts}
	e.data = data
	e.hasLast = true
	e.seq++
	for i := range cnt {
		if err := e.writeChunk(i); err != nil {
			return err
		}
	}
	return nil
}

// Resend writes chunk idx of the last image again, byte for byte as the
// first time. It returns ErrNoImage if no image was written and ErrChunk if
// idx is not a chunk of the last image.
func (e *Encoder) Resend(idx int) error {
	if !e.hasLast {
		return ErrNoImage
	}
	if idx < 0 || idx >= int(e.last.ChunkCnt) {
		return ErrChunk
	}
	return e.writeChunk(idx)
}

// writeChunk writes chunk i of the last image: chunk 0 carries the meta and
// the first MaxPayload-MetaSize data bytes, chunk i ≥ 1 the data from
// i·MaxPayload − MetaSize on.
func (e *Encoder) writeChunk(i int) error {
	cnt := int(e.last.ChunkCnt)
	h := e.last
	h.ChunkIdx = uint16(i)
	off := 0
	room := MaxPayload - MetaSize
	if i == 0 {
		h.Flags = FlagFirst | FlagMeta
	} else {
		off = i*MaxPayload - MetaSize
		room = MaxPayload
	}
	if i == cnt-1 {
		h.Flags |= FlagLast
	}
	n := len(e.data) - off
	if n > room {
		n = room
	}
	piece := e.data[off : off+n]
	plen := n
	if i == 0 {
		plen += MetaSize
	}
	h.PayloadLen = uint16(plen)
	h.CRC = 0
	h.Put(&e.hdr)

	crc := crc32.Update(0, crc32.IEEETable, e.hdr[:20])
	if i == 0 {
		crc = crc32.Update(crc, crc32.IEEETable, e.meta[:])
	}
	crc = crc32.Update(crc, crc32.IEEETable, piece)
	e.hdr[20] = byte(crc)
	e.hdr[21] = byte(crc >> 8)
	e.hdr[22] = byte(crc >> 16)
	e.hdr[23] = byte(crc >> 24)

	if _, err := e.w.Write(e.hdr[:]); err != nil {
		return err
	}
	if i == 0 {
		if _, err := e.w.Write(e.meta[:]); err != nil {
			return err
		}
	}
	if n > 0 {
		if _, err := e.w.Write(piece); err != nil {
			return err
		}
	}
	return nil
}
