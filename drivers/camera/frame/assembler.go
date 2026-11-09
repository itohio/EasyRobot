package frame

// Assembler rebuilds images from chunks that may arrive out of order, as
// they do with chunk resend: the receiver adds every CRC-valid chunk, asks
// Missing which indices to Nak, and gets the image once all chunks are in.
// It holds one image at a time; a chunk of a different frame_seq abandons
// the image under assembly. Chunks of the image finished last (completed or
// given up) are ignored, because resends can arrive after it is whole; that
// image is identified by frame_seq and frame_ts together, so a rebooted
// device's new image with a reused frame_seq is not mistaken for it.
// Not safe for concurrent use.
type Assembler struct {
	active  bool
	seq     uint32
	cnt     uint16
	ts      uint32
	got     int
	chunks  [][]byte // payload per index; nil = missing
	done    bool     // doneSeq/doneTS are set
	doneSeq uint32   // frame_seq of the image finished last
	doneTS  uint32   // its frame_ts
	Abandon uint64   // images dropped: another frame_seq arrived, or a bad meta or length
}

// Current reports the frame_seq and chunk count of the image under
// assembly.
func (a *Assembler) Current() (seq uint32, cnt uint16, ok bool) {
	return a.seq, a.cnt, a.active
}

// Add stores a chunk. It returns the image when this chunk completed it.
// Chunks failing [Header.Valid], duplicates, and chunks of the image
// finished last are ignored. A chunk whose chunk count or frame_ts differs
// from the image under assembly restarts it.
func (a *Assembler) Add(c Chunk) (Image, bool) {
	if c.ChunkCnt == 0 || c.ChunkIdx >= c.ChunkCnt {
		return Image{}, false
	}
	if a.done && c.Seq == a.doneSeq && c.TS == a.doneTS {
		return Image{}, false
	}
	if !a.active || c.Seq != a.seq || c.ChunkCnt != a.cnt || c.TS != a.ts {
		if a.active {
			a.Abandon++
		}
		a.active, a.seq, a.cnt, a.ts, a.got = true, c.Seq, c.ChunkCnt, c.TS, 0
		a.chunks = make([][]byte, c.ChunkCnt)
	}
	if a.chunks[c.ChunkIdx] != nil {
		return Image{}, false
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

// GiveUp abandons the image under assembly and remembers it as finished, so
// late resends of it are ignored. It is the receiver's decision, not counted
// in Abandon.
func (a *Assembler) GiveUp() {
	if a.active {
		a.done, a.doneSeq, a.doneTS = true, a.seq, a.ts
	}
	a.active, a.chunks = false, nil
}

// finish joins the chunks and checks the meta and the length before
// allocating. An image whose meta or length is wrong is abandoned.
func (a *Assembler) finish() (Image, bool) {
	defer func() { a.active, a.chunks = false, nil }()
	a.done, a.doneSeq, a.doneTS = true, a.seq, a.ts
	var m Meta
	if len(a.chunks[0]) < MetaSize || m.UnmarshalBinary(a.chunks[0]) != nil || m.Validate() != nil {
		a.Abandon++
		return Image{}, false
	}
	n := len(a.chunks[0]) - MetaSize
	for _, p := range a.chunks[1:] {
		n += len(p)
	}
	if uint64(n) != uint64(m.DataLen) {
		a.Abandon++
		return Image{}, false
	}
	data := make([]byte, 0, n)
	data = append(data, a.chunks[0][MetaSize:]...)
	for _, p := range a.chunks[1:] {
		data = append(data, p...)
	}
	return Image{Seq: a.seq, TS: a.ts, Meta: m, Data: data}, true
}
