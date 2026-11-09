package frame

import (
	"bytes"
	"errors"
	"io"
	"runtime"
	"testing"
)

func testImageData(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*7 + i>>8)
	}
	return b
}

// chunksOf decodes s into its CRC-valid chunks with NextChunk.
func chunksOf(t *testing.T, s []byte) []Chunk {
	t.Helper()
	d := NewDecoder(bytes.NewReader(s))
	var out []Chunk
	for {
		c, err := d.NextChunk()
		if err == io.EOF {
			return out
		}
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, c)
	}
}

func TestResendRepeatsChunksByteForByte(t *testing.T) {
	data := testImageData(160 * 120) // Y8 QQVGA: 5 chunks
	var first bytes.Buffer
	e := NewEncoder(&first)
	m := Meta{Width: 160, Height: 120, Format: Y8}
	if err := e.WriteImage(m, 42, data); err != nil {
		t.Fatal(err)
	}
	orig := chunksOf(t, first.Bytes())
	if len(orig) != 5 {
		t.Fatalf("%d chunks, want 5", len(orig))
	}
	for i := range orig {
		var again bytes.Buffer
		e.w = &again
		if err := e.Resend(i); err != nil {
			t.Fatal(err)
		}
		got := chunksOf(t, again.Bytes())
		if len(got) != 1 || got[0].Header != orig[i].Header || !bytes.Equal(got[0].Payload, orig[i].Payload) {
			t.Fatalf("resent chunk %d differs from the original", i)
		}
	}
	if err := e.Resend(5); !errors.Is(err, ErrChunk) {
		t.Fatalf("Resend(5) err = %v, want ErrChunk", err)
	}
	if err := NewEncoder(io.Discard).Resend(0); !errors.Is(err, ErrNoImage) {
		t.Fatalf("Resend before any image: err = %v, want ErrNoImage", err)
	}
}

func TestAssemblerRebuildsOutOfOrderWithResends(t *testing.T) {
	data := testImageData(320 * 240 * 2) // RGB565 QVGA: 38 chunks
	var buf bytes.Buffer
	e := NewEncoder(&buf)
	if err := e.WriteImage(Meta{Width: 320, Height: 240, Format: RGB565LE}, 7, data); err != nil {
		t.Fatal(err)
	}
	all := chunksOf(t, buf.Bytes())
	var a Assembler
	// Lose chunks 0, 13 and the last on the first pass.
	lost := map[uint16]bool{0: true, 13: true, uint16(len(all) - 1): true}
	for _, c := range all {
		if lost[c.ChunkIdx] {
			continue
		}
		if _, done := a.Add(c); done {
			t.Fatal("completed with chunks missing")
		}
	}
	miss := a.Missing(nil)
	if len(miss) != 3 || miss[0] != 0 || miss[1] != 13 || miss[2] != uint16(len(all)-1) {
		t.Fatalf("Missing = %v, want [0 13 %d]", miss, len(all)-1)
	}
	var img Image
	var done bool
	for _, i := range miss {
		img, done = a.Add(all[i])
	}
	if !done || img.Seq != 0 || img.TS != 7 || !bytes.Equal(img.Data, data) || img.Meta.Width != 320 {
		t.Fatalf("done=%v seq=%d ts=%d len=%d", done, img.Seq, img.TS, len(img.Data))
	}
	if a.Abandon != 0 {
		t.Fatalf("Abandon = %d", a.Abandon)
	}
	// A late resend of the completed image is ignored: no new image starts,
	// nothing is reported missing.
	a.Add(all[1])
	if _, _, ok := a.Current(); ok || len(a.Missing(nil)) != 0 {
		t.Fatal("a late chunk of the completed image started a new assembly")
	}
	// The next image starts normally; a different seq then abandons it.
	a.Add(Chunk{Header: Header{Seq: 1, ChunkCnt: 2, ChunkIdx: 1}, Payload: []byte{1}})
	a.Add(Chunk{Header: Header{Seq: 2, ChunkCnt: 2, ChunkIdx: 1}, Payload: []byte{1}})
	if a.Abandon != 1 {
		t.Fatalf("Abandon = %d after a new seq, want 1", a.Abandon)
	}
}

// imageChunks encodes one small Y8 image with seq/ts and returns its chunks.
func imageChunks(t *testing.T, seq, ts uint32, w, h uint16) []Chunk {
	t.Helper()
	var buf bytes.Buffer
	e := NewEncoder(&buf)
	e.seq = seq
	if err := e.WriteImage(Meta{Width: w, Height: h, Format: Y8}, ts, testImageData(int(w)*int(h))); err != nil {
		t.Fatal(err)
	}
	return chunksOf(t, buf.Bytes())
}

func TestAssemblerGiveUpIgnoresLateResends(t *testing.T) {
	all := imageChunks(t, 4, 100, 160, 120) // 5 chunks
	var a Assembler
	a.Add(all[0])
	a.GiveUp()
	for _, c := range all {
		if _, done := a.Add(c); done {
			t.Fatal("a given-up image came back to life")
		}
	}
	if _, _, ok := a.Current(); ok || a.Abandon != 0 {
		t.Fatalf("active=%v Abandon=%d after late resends of a given-up image", ok, a.Abandon)
	}
}

func TestAssemblerRebootReusingSeqIsNotIgnored(t *testing.T) {
	var a Assembler
	for _, c := range imageChunks(t, 0, 5_000_000, 32, 4) {
		a.Add(c)
	}
	// The device reboots: seq restarts at 0 with a new timestamp.
	var got bool
	for _, c := range imageChunks(t, 0, 1000, 32, 4) {
		_, got = a.Add(c)
	}
	if !got {
		t.Fatal("the rebooted device's seq-0 image was dropped as a late resend")
	}
}

func TestAssemblerChunkCountChangeRestarts(t *testing.T) {
	small := imageChunks(t, 3, 9, 32, 4)    // 1 chunk
	large := imageChunks(t, 3, 9, 160, 120) // 5 chunks, same seq and ts
	var a Assembler
	a.Add(large[1])
	if img, done := a.Add(small[0]); !done || len(img.Data) != 32*4 {
		t.Fatalf("done=%v len=%d: a different chunk count must restart the image", done, len(img.Data))
	}
	if a.Abandon != 1 {
		t.Fatalf("Abandon = %d, want 1 (the 5-chunk image)", a.Abandon)
	}
}

func TestAssemblerRejectsLengthMismatchWithoutAllocatingIt(t *testing.T) {
	// One CRC-valid chunk whose meta claims far more data than it carries.
	var meta [MetaSize]byte
	m := Meta{Width: 65520, Height: 4096, Format: Y8, DataLen: 65520 * 4096} // valid meta, ≤ MaxDataLen
	m.Put(&meta)
	c := Chunk{Header: Header{Seq: 1, ChunkCnt: 1}, Payload: append(meta[:], 1, 2, 3)}
	var a Assembler
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	if _, done := a.Add(c); done {
		t.Fatal("accepted a length mismatch")
	}
	runtime.ReadMemStats(&after)
	if a.Abandon != 1 {
		t.Fatalf("Abandon = %d, want 1", a.Abandon)
	}
	if got := after.TotalAlloc - before.TotalAlloc; got > 1<<20 { // never the claimed 268 MB
		t.Fatalf("rejecting the image allocated %d bytes", got)
	}
}

func TestAssemblerIgnoresInvalidChunks(t *testing.T) {
	var a Assembler
	for _, h := range []Header{{ChunkCnt: 0}, {ChunkCnt: 2, ChunkIdx: 2}, {ChunkCnt: 1, ChunkIdx: 9}} {
		if _, done := a.Add(Chunk{Header: h, Payload: []byte{1}}); done {
			t.Fatalf("%+v completed an image", h)
		}
	}
	if _, _, ok := a.Current(); ok {
		t.Fatal("an invalid chunk started an assembly")
	}
}

func TestControlParserFindsMessagesInNoise(t *testing.T) {
	var p ControlParser
	var stream []byte
	stream = append(stream, "ERI\x00garbage ERIN"...) // partial magics
	var b [ControlSize]byte
	want := []Control{{Kind: Nak, Seq: 9, Chunk: 3}, {Kind: Ack, Seq: 9}}
	for _, c := range want {
		c.Put(&b)
		stream = append(stream, b[:]...)
		stream = append(stream, 0xFF, 'E')
	}
	bad := b
	bad[13] ^= 1 // CRC broken: must be ignored
	stream = append(stream, bad[:]...)
	var got []Control
	for _, x := range stream {
		if c, ok := p.Feed(x); ok {
			got = append(got, c)
		}
	}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}
