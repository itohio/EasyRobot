package frame

// Assembler rebuilds images from chunks that may arrive out of order, as
// they do with chunk resend: the receiver adds every CRC-valid chunk, asks
// Missing which indices to Nak, and gets the image once all chunks are in.
// It holds one image at a time; a chunk of a different frame_seq abandons
// the image under assembly. Chunks of the image completed last are ignored:
// resends can arrive after the image is already whole. Not safe for
// concurrent use.
type Assembler struct {
	active  bool
	seq     uint32
	cnt     uint16
	ts      uint32
	got     int
	chunks  [][]byte // payload per index; nil = missing
	done    bool     // doneSeq is set
	doneSeq uint32   // frame_seq of the image completed last
	Abandon uint64   // images given up because another frame_seq arrived
}

// Current reports the frame_seq and chunk count of the image under
// assembly.
func (a *Assembler) Current() (seq uint32, cnt uint16, ok bool) {
	return a.seq, a.cnt, a.active
}

// Add stores a chunk. It returns the image when this chunk completed it.
// Duplicates, and chunks of the image completed last, are ignored. A chunk
// whose chunk count differs from the image under assembly restarts it.
func (a *Assembler) Add(c Chunk) (Image, bool) {
	if a.done && c.Seq == a.doneSeq {
		return Image{}, false
	}
	if !a.active || c.Seq != a.seq || c.ChunkCnt != a.cnt {
		if a.active {
			a.Abandon++
		}
		a.active, a.seq, a.cnt, a.got = true, c.Seq, c.ChunkCnt, 0
		a.chunks = make([][]byte, c.ChunkCnt)
	}
	if a.chunks[c.ChunkIdx] != nil {
		return Image{}, false
	}
	if c.ChunkIdx == 0 {
		a.ts = c.TS
	}
	a.chunks[c.ChunkIdx] = c.Payload
	a.got++
	if a.got < int(a.cnt) {
		return Image{}, false
	}
	return a.finish()
}

// Missing appends the indices still missing from the image under assembly
// to dst.
func (a *Assembler) Missing(dst []uint16) []uint16 {
	if !a.active {
		return dst
	}
	for i, p := range a.chunks {
		if p == nil {
			dst = append(dst, uint16(i))
		}
	}
	return dst
}

// Reset gives up the image under assembly (counted in Abandon).
func (a *Assembler) Reset() {
	if a.active {
		a.Abandon++
	}
	a.active, a.chunks = false, nil
}

// finish joins the chunks and checks the meta. An image whose meta or
// length is wrong is abandoned.
func (a *Assembler) finish() (Image, bool) {
	defer func() { a.active, a.chunks = false, nil }()
	var m Meta
	if len(a.chunks[0]) < MetaSize || m.UnmarshalBinary(a.chunks[0]) != nil || m.Validate() != nil {
		a.Abandon++
		return Image{}, false
	}
	data := make([]byte, 0, m.DataLen)
	data = append(data, a.chunks[0][MetaSize:]...)
	for _, p := range a.chunks[1:] {
		data = append(data, p...)
	}
	if uint32(len(data)) != m.DataLen {
		a.Abandon++
		return Image{}, false
	}
	a.done, a.doneSeq = true, a.seq
	return Image{Seq: a.seq, TS: a.ts, Meta: m, Data: data}, true
}
