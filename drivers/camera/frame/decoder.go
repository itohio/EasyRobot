package frame

import (
	"bufio"
	"encoding/binary"
	"hash/crc32"
	"io"
)

// Stats counts what a Decoder has seen.
type Stats struct {
	Images    uint64 // complete images returned by Next
	CRCErrors uint64 // chunks whose header passed the bounds but whose CRC did not match
	Dropped   uint64 // images with at least one valid chunk that did not complete
	Lost      uint64 // images never seen at all, from frame_seq gaps
	TextBytes uint64 // bytes passed to Text (not part of a valid chunk)
}

// Decoder reads frame v1 chunks from a byte stream and reassembles images.
// Bytes that are not part of a valid chunk go to Text. A Decoder is not safe
// for concurrent use.
type Decoder struct {
	// Text, if not nil, receives every byte that is not part of a valid
	// chunk (boot banners, logs), in stream order.
	Text io.Writer

	r     *bufio.Reader
	stats Stats

	// Image under assembly.
	active  bool
	seq     uint32
	cnt     uint16
	next    uint16 // next expected chunk index
	ts      uint32
	meta    Meta
	data    []byte
	pending int // image data bytes still expected

	// Sequence bookkeeping over valid chunks.
	seqSeen bool   // lastSeq is set
	lastSeq uint32 // frame_seq of the last valid chunk
	seqDone bool   // lastSeq completed or was already counted as dropped
}

// bufSize holds a whole chunk so a bad one can be rescanned from magic+1.
const bufSize = 8192

// NewDecoder returns a Decoder reading from r through an internal buffer.
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{r: bufio.NewReaderSize(r, bufSize)}
}

// Stats returns the counters so far.
func (d *Decoder) Stats() Stats { return d.stats }

// Next returns the next complete image. It returns io.EOF once the input is
// exhausted (remaining bytes are flushed to Text and an unfinished image is
// counted as dropped). Other read errors are returned as is; Next may be
// called again after them and resumes with the buffered bytes.
//
// Receiver rules (format v1): scan for the magic; a header failing
// [Header.Valid] is rejected and the scan resumes at magic+1; a CRC
// mismatch drops the image under assembly and the scan resumes at magic+1;
// a missing or out-of-order chunk index, a changed chunk count, or a new
// frame_seq before the last chunk drops the image; an image completes when
// its last chunk arrives and exactly MetaSize + DataLen bytes were received.
// First chunks whose meta fails [Meta.UnmarshalBinary] or [Meta.Validate]
// drop the image.
func (d *Decoder) Next() (Image, error) {
	for {
		h, cb, err := d.scan(true)
		if err != nil {
			return Image{}, err
		}
		img, done := d.chunk(&h, cb[HeaderSize:])
		d.r.Discard(len(cb)) // len(cb) bytes are buffered: cannot fail
		if done {
			return img, nil
		}
	}
}

// Chunk is one CRC-valid chunk, for receivers that reassemble themselves
// (chunk resend, see [Assembler]).
type Chunk struct {
	Header
	Payload []byte // a copy; owned by the caller
}

// NextChunk returns the next CRC-valid chunk in stream order, applying the
// same scanning, text and CRC rules as Next but no image assembly; Images,
// Dropped and Lost are left to the caller. Use either Next or NextChunk on a
// Decoder, not both.
func (d *Decoder) NextChunk() (Chunk, error) {
	h, cb, err := d.scan(false)
	if err != nil {
		return Chunk{}, err
	}
	c := Chunk{Header: h, Payload: append([]byte(nil), cb[HeaderSize:]...)}
	d.r.Discard(len(cb))
	return c, nil
}

// scan finds the next CRC-valid chunk and returns its header and the
// buffered chunk bytes (header + payload, not yet discarded). A CRC
// mismatch drops the image under assembly when drop is set.
func (d *Decoder) scan(drop bool) (Header, []byte, error) {
	for {
		p, err := d.r.Peek(4)
		if len(p) < 4 {
			return Header{}, nil, d.endOfInput(err)
		}
		if binary.LittleEndian.Uint32(p) != Magic {
			if err := d.skipToCandidate(); err != nil {
				return Header{}, nil, err
			}
			continue
		}

		hb, err := d.r.Peek(HeaderSize)
		if len(hb) < HeaderSize {
			if err == io.EOF {
				if err := d.text(1); err != nil {
					return Header{}, nil, err
				}
				continue
			}
			return Header{}, nil, err
		}
		h, _ := ParseHeader(hb)
		if !h.Valid() {
			if err := d.text(1); err != nil {
				return Header{}, nil, err
			}
			continue
		}

		n := HeaderSize + int(h.PayloadLen)
		cb, err := d.r.Peek(n)
		if len(cb) < n {
			if err == io.EOF {
				if err := d.text(1); err != nil {
					return Header{}, nil, err
				}
				continue
			}
			return Header{}, nil, err
		}
		crc := crc32.Update(0, crc32.IEEETable, cb[:20])
		crc = crc32.Update(crc, crc32.IEEETable, cb[HeaderSize:n])
		if crc != h.CRC {
			d.stats.CRCErrors++
			if drop {
				d.drop()
			}
			if err := d.text(1); err != nil {
				return Header{}, nil, err
			}
			continue
		}
		return h, cb[:n], nil
	}
}

// chunk applies one CRC-valid chunk to the image under assembly. payload is
// only valid until the next read from d.r.
func (d *Decoder) chunk(h *Header, payload []byte) (Image, bool) {
	if !d.seqSeen || h.Seq != d.lastSeq {
		if d.active {
			d.drop() // new frame_seq before last
		} else if d.seqSeen && !d.seqDone {
			d.stats.Dropped++ // only orphan chunks of lastSeq were seen
		}
		if d.seqSeen {
			gap := h.Seq - d.lastSeq
			if gap > 1 && gap < 1<<31 {
				d.stats.Lost += uint64(gap - 1)
			}
		}
		d.seqSeen = true
		d.lastSeq = h.Seq
		d.seqDone = false
	}

	if h.ChunkIdx == 0 {
		if d.active {
			d.drop() // first chunk again within the same frame_seq
		}
		var m Meta
		if m.UnmarshalBinary(payload) != nil || m.Validate() != nil {
			d.markDropped()
			return Image{}, false
		}
		hint := int(m.DataLen)
		if hint > 1<<20 {
			hint = 1 << 20
		}
		d.active = true
		d.seq = h.Seq
		d.cnt = h.ChunkCnt
		d.next = 0
		d.ts = h.TS
		d.meta = m
		d.data = make([]byte, 0, hint)
		d.pending = int(m.DataLen)
		payload = payload[MetaSize:]
	} else if !d.active || h.ChunkIdx != d.next || h.ChunkCnt != d.cnt {
		d.drop()
		d.markDropped()
		return Image{}, false
	}

	if len(payload) > d.pending {
		d.drop()
		return Image{}, false
	}
	d.data = append(d.data, payload...)
	d.pending -= len(payload)
	d.next++

	if h.ChunkIdx != h.ChunkCnt-1 {
		return Image{}, false
	}
	if d.pending != 0 {
		d.drop()
		return Image{}, false
	}
	img := Image{Seq: d.seq, TS: d.ts, Meta: d.meta, Data: d.data}
	d.active = false
	d.data = nil
	d.seqDone = true
	d.stats.Images++
	return img, true
}

// drop abandons the image under assembly, counting it once.
func (d *Decoder) drop() {
	if !d.active {
		return
	}
	d.active = false
	d.data = nil
	d.markDropped()
}

// markDropped counts lastSeq as dropped unless it already completed or was
// counted.
func (d *Decoder) markDropped() {
	if d.seqSeen && !d.seqDone {
		d.stats.Dropped++
		d.seqDone = true
	}
}

// skipToCandidate passes buffered bytes to Text up to the next byte that
// could start the magic, always at least one byte.
func (d *Decoder) skipToCandidate() error {
	b, _ := d.r.Peek(d.r.Buffered())
	i := 1
	for i < len(b) && b[i] != byte(Magic&0xFF) {
		i++
	}
	return d.text(i)
}

// text passes the next n buffered bytes to Text and discards them.
func (d *Decoder) text(n int) error {
	b, _ := d.r.Peek(n)
	var err error
	if d.Text != nil {
		_, err = d.Text.Write(b)
	}
	d.stats.TextBytes += uint64(len(b))
	d.r.Discard(len(b))
	return err
}

// endOfInput handles a short Peek: at io.EOF it flushes the rest to Text and
// counts an unfinished image; other errors are returned unchanged.
func (d *Decoder) endOfInput(err error) error {
	if err != io.EOF {
		return err
	}
	if n := d.r.Buffered(); n > 0 {
		if werr := d.text(n); werr != nil {
			return werr
		}
	}
	if d.active {
		d.drop()
	} else {
		d.markDropped()
	}
	return io.EOF
}
