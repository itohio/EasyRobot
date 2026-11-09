package frame

import (
	"encoding/binary"
	"hash/crc32"
)

// Control messages go from the receiver back to the sender, for chunk
// resend (frame format v1, additive). Layout, little-endian, 16 bytes:
//
//	0  u32 magic  0x4E495245, bytes on the wire "ERIN"
//	4  u32 frame_seq of the image concerned
//	8  u16 chunk_idx (Nak) or 0 (Ack)
//	10 u16 kind: 1 Nak (resend this chunk), 2 Ack (image complete, go on)
//	12 u32 CRC-32/IEEE over bytes 0..11
//
// A sender that supports resend keeps the last image until it gets Ack for
// its frame_seq (or a timeout of its choosing) and answers each Nak with the
// chunk again, unchanged. Senders without resend never read the port.
const (
	ControlMagic uint32 = 0x4E495245
	ControlSize         = 16
)

// ControlKind is the kind of a Control message.
type ControlKind uint16

const (
	Nak ControlKind = 1
	Ack ControlKind = 2
)

// Control is one receiver → sender message.
type Control struct {
	Kind  ControlKind
	Seq   uint32
	Chunk uint16
}

// Put writes c into b.
func (c Control) Put(b *[ControlSize]byte) {
	binary.LittleEndian.PutUint32(b[0:], ControlMagic)
	binary.LittleEndian.PutUint32(b[4:], c.Seq)
	binary.LittleEndian.PutUint16(b[8:], c.Chunk)
	binary.LittleEndian.PutUint16(b[10:], uint16(c.Kind))
	binary.LittleEndian.PutUint32(b[12:], crc32.ChecksumIEEE(b[:12]))
}

// ControlParser finds Control messages in a byte stream, one byte at a
// time, without allocating (for firmware reading its UART).
type ControlParser struct {
	buf [ControlSize]byte
	n   int
}

// Feed adds one byte and reports a complete, CRC-valid message.
func (p *ControlParser) Feed(b byte) (Control, bool) {
	if p.n == ControlSize {
		copy(p.buf[:], p.buf[1:])
		p.n--
	}
	p.buf[p.n] = b
	p.n++
	if p.n < ControlSize {
		return Control{}, false
	}
	if binary.LittleEndian.Uint32(p.buf[0:]) != ControlMagic ||
		binary.LittleEndian.Uint32(p.buf[12:]) != crc32.ChecksumIEEE(p.buf[:12]) {
		return Control{}, false
	}
	c := Control{
		Seq:   binary.LittleEndian.Uint32(p.buf[4:]),
		Chunk: binary.LittleEndian.Uint16(p.buf[8:]),
		Kind:  ControlKind(binary.LittleEndian.Uint16(p.buf[10:])),
	}
	p.n = 0
	return c, true
}
