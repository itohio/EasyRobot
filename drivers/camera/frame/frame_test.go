package frame

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"testing"
	"testing/iotest"
)

// pattern returns n deterministic, non-trivial bytes.
func pattern(n int, salt byte) []byte {
	b := make([]byte, n)
	x := uint32(2463534242) ^ uint32(salt)
	for i := range b {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		b[i] = byte(x)
	}
	return b
}

type testImage struct {
	meta Meta
	ts   uint32
	data []byte
}

func encodeImages(t *testing.T, imgs ...testImage) []byte {
	t.Helper()
	var buf bytes.Buffer
	e := NewEncoder(&buf)
	for _, im := range imgs {
		if err := e.WriteImage(im.meta, im.ts, im.data); err != nil {
			t.Fatalf("WriteImage: %v", err)
		}
	}
	return buf.Bytes()
}

// splitChunks splits a text-free stream into its chunks.
func splitChunks(t *testing.T, s []byte) [][]byte {
	t.Helper()
	var out [][]byte
	for len(s) > 0 {
		h, err := ParseHeader(s)
		if err != nil {
			t.Fatalf("ParseHeader: %v", err)
		}
		n := HeaderSize + int(h.PayloadLen)
		out = append(out, s[:n:n])
		s = s[n:]
	}
	return out
}

func join(parts ...[]byte) []byte {
	var b []byte
	for _, p := range parts {
		b = append(b, p...)
	}
	return b
}

func decodeAll(t *testing.T, r io.Reader) ([]Image, Stats, []byte) {
	t.Helper()
	var text bytes.Buffer
	d := NewDecoder(r)
	d.Text = &text
	var imgs []Image
	for {
		img, err := d.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		imgs = append(imgs, img)
	}
	return imgs, d.Stats(), text.Bytes()
}

func seqs(imgs []Image) []uint32 {
	s := make([]uint32, len(imgs))
	for i, im := range imgs {
		s[i] = im.Seq
	}
	return s
}

func equalSeqs(a []uint32, b ...uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func y8(w, h int, salt byte) testImage {
	return testImage{meta: Meta{Width: uint16(w), Height: uint16(h), Format: Y8, Sensor: SensorOV7740}, ts: uint32(salt) * 1000, data: pattern(w*h, salt)}
}

func jpegImg(n int, salt byte) testImage {
	return testImage{meta: Meta{Width: 640, Height: 480, Format: JPEG, Sensor: SensorOV2640}, ts: uint32(salt), data: pattern(n, salt)}
}

func TestRoundTripFormats(t *testing.T) {
	cases := []struct {
		name string
		meta Meta
		n    int
	}{
		{"Y8", Meta{Width: 16, Height: 8, Format: Y8, Sensor: SensorOV7740}, 16 * 8},
		{"Y8 stride", Meta{Width: 15, Height: 8, Format: Y8, Stride: 16}, 16 * 8},
		{"RGB565LE", Meta{Width: 32, Height: 24, Format: RGB565LE, Sensor: SensorOV7740}, 32 * 24 * 2},
		{"RGB565BE", Meta{Width: 32, Height: 24, Format: RGB565BE, Sensor: SensorOV2640}, 32 * 24 * 2},
		{"YUYV", Meta{Width: 32, Height: 24, Format: YUYV, Sensor: SensorOV3660}, 32 * 24 * 2},
		{"JPEG", Meta{Width: 1600, Height: 1200, Format: JPEG, Sensor: SensorOV5640}, 9001},
		{"RGB888Planar", Meta{Width: 20, Height: 10, Format: RGB888Planar}, 20 * 10 * 3},
		{"RGB888Planar stride", Meta{Width: 20, Height: 10, Format: RGB888Planar, Stride: 66}, 66 * 10},
		{"RGB565 big", Meta{Width: 320, Height: 240, Format: RGB565LE}, 320 * 240 * 2},
	}
	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data := pattern(tc.n, byte(i))
			s := encodeImages(t, testImage{tc.meta, 0xDEADBEEF, data})
			imgs, st, text := decodeAll(t, bytes.NewReader(s))
			if len(imgs) != 1 {
				t.Fatalf("images = %d, want 1 (stats %+v)", len(imgs), st)
			}
			want := tc.meta
			want.DataLen = uint32(tc.n)
			got := imgs[0]
			if got.Meta != want || got.Seq != 0 || got.TS != 0xDEADBEEF || !bytes.Equal(got.Data, data) {
				t.Fatalf("got seq=%d ts=%#x meta=%+v len=%d, want meta=%+v len=%d", got.Seq, got.TS, got.Meta, len(got.Data), want, tc.n)
			}
			if len(text) != 0 || st != (Stats{Images: 1}) {
				t.Fatalf("text=%q stats=%+v", text, st)
			}
		})
	}
}

func TestRoundTripOneByteReader(t *testing.T) {
	a, b := y8(160, 120, 1), jpegImg(5000, 2)
	s := join([]byte("boot\r\n"), encodeImages(t, a, b), []byte("bye\n"))
	imgs, st, text := decodeAll(t, iotest.OneByteReader(bytes.NewReader(s)))
	if !equalSeqs(seqs(imgs), 0, 1) || !bytes.Equal(imgs[0].Data, a.data) || !bytes.Equal(imgs[1].Data, b.data) {
		t.Fatalf("seqs %v", seqs(imgs))
	}
	if string(text) != "boot\r\nbye\n" || st.Images != 2 {
		t.Fatalf("text %q stats %+v", text, st)
	}
}

func TestChunkBoundaries(t *testing.T) {
	cases := []struct {
		name  string
		img   testImage
		plens []int
	}{
		{"fills first chunk", jpegImg(MaxPayload-MetaSize, 1), []int{4096}},
		{"one byte more", jpegImg(MaxPayload-MetaSize+1, 2), []int{4096, 1}},
		{"two full chunks", jpegImg(2*MaxPayload-MetaSize, 3), []int{4096, 4096}},
		{"one byte", jpegImg(1, 4), []int{17}},
		{"160x120 Y8", y8(160, 120, 5), []int{4096, 4096, 4096, 4096, 2832}},
		{"320x240 RGB565", testImage{Meta{Width: 320, Height: 240, Format: RGB565LE}, 7, pattern(320*240*2, 6)}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := encodeImages(t, tc.img)
			chunks := splitChunks(t, s)
			total := MetaSize + len(tc.img.data)
			wantCnt := (total + MaxPayload - 1) / MaxPayload
			if len(chunks) != wantCnt {
				t.Fatalf("chunks = %d, want %d", len(chunks), wantCnt)
			}
			sum := 0
			for i, c := range chunks {
				h, _ := ParseHeader(c)
				if !h.Valid() || int(h.ChunkIdx) != i || int(h.ChunkCnt) != wantCnt {
					t.Fatalf("chunk %d header %+v", i, h)
				}
				if tc.plens != nil && int(h.PayloadLen) != tc.plens[i] {
					t.Fatalf("chunk %d payload_len = %d, want %d", i, h.PayloadLen, tc.plens[i])
				}
				if i < wantCnt-1 && h.PayloadLen != MaxPayload {
					t.Fatalf("chunk %d not full: %d", i, h.PayloadLen)
				}
				if got := crc32.ChecksumIEEE(join(c[:20], c[HeaderSize:])); got != h.CRC {
					t.Fatalf("chunk %d crc %#x, ChecksumIEEE %#x", i, h.CRC, got)
				}
				sum += int(h.PayloadLen)
			}
			if sum != total {
				t.Fatalf("payload sum %d, want %d", sum, total)
			}
			imgs, st, _ := decodeAll(t, bytes.NewReader(s))
			if len(imgs) != 1 || !bytes.Equal(imgs[0].Data, tc.img.data) || st.Images != 1 {
				t.Fatalf("decode: %d images, stats %+v", len(imgs), st)
			}
		})
	}
}

func TestTextPassthrough(t *testing.T) {
	a, b := y8(160, 120, 1), jpegImg(100, 2)
	ca := encodeImages(t, a)
	cb := encodeImages(t, a, b)[len(ca):] // b as seq 1
	cases := []struct {
		name  string
		parts [][]byte
		text  string
	}{
		{"before between after", [][]byte{[]byte("K210 boot\r\n"), ca, []byte("log: 1\n"), cb, []byte("tail")}, "K210 boot\r\nlog: 1\ntail"},
		{"magic-like text", [][]byte{[]byte("ERIM ERI E\x00ERIMxxxxxxxxxxxxxxxxxxxxxxxxx"), ca, []byte("EERIM"), cb, []byte("ER")}, "ERIM ERI E\x00ERIMxxxxxxxxxxxxxxxxxxxxxxxxx" + "EERIM" + "ER"},
		{"binary text", [][]byte{{0, 0xFF, 0x45, 0x45, 0x52}, ca, cb, {0x45, 0x52, 0x49}}, "\x00\xff\x45\x45\x52\x45\x52\x49"},
		{"no text", [][]byte{ca, cb}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			imgs, st, text := decodeAll(t, bytes.NewReader(join(tc.parts...)))
			if !equalSeqs(seqs(imgs), 0, 1) {
				t.Fatalf("seqs %v stats %+v", seqs(imgs), st)
			}
			if string(text) != tc.text || st.TextBytes != uint64(len(tc.text)) {
				t.Fatalf("text %q (%d), want %q", text, st.TextBytes, tc.text)
			}
			if st.Dropped != 0 || st.CRCErrors != 0 || st.Lost != 0 {
				t.Fatalf("stats %+v", st)
			}
		})
	}
}

func TestCorruptByte(t *testing.T) {
	cases := []struct {
		name     string
		imgs     []testImage
		chunk    int // index into all chunks
		off      int // byte offset in that chunk
		wantSeqs []uint32
		want     Stats
	}{
		{"middle chunk payload", []testImage{y8(160, 120, 1), y8(160, 120, 2), y8(160, 120, 3)}, 7, 100, []uint32{0, 2}, Stats{Images: 2, CRCErrors: 1, Dropped: 1}},
		{"first chunk meta", []testImage{y8(160, 120, 1), y8(160, 120, 2), y8(160, 120, 3)}, 5, HeaderSize + 2, []uint32{0, 2}, Stats{Images: 2, CRCErrors: 1, Dropped: 1}},
		{"last chunk", []testImage{y8(160, 120, 1), y8(160, 120, 2), y8(160, 120, 3)}, 9, HeaderSize + 7, []uint32{0, 2}, Stats{Images: 2, CRCErrors: 1, Dropped: 1}},
		{"header ts", []testImage{y8(160, 120, 1), y8(160, 120, 2), y8(160, 120, 3)}, 6, 17, []uint32{0, 2}, Stats{Images: 2, CRCErrors: 1, Dropped: 1}},
		// A single-chunk image leaves no valid chunk: it shows up as a seq gap.
		{"single chunk image", []testImage{jpegImg(50, 1), jpegImg(50, 2), jpegImg(50, 3)}, 1, HeaderSize + 20, []uint32{0, 2}, Stats{Images: 2, CRCErrors: 1, Lost: 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			chunks := splitChunks(t, encodeImages(t, tc.imgs...))
			bad := append([]byte(nil), chunks[tc.chunk]...)
			bad[tc.off] ^= 0x10
			chunks[tc.chunk] = bad
			imgs, st, _ := decodeAll(t, bytes.NewReader(join(chunks...)))
			if !equalSeqs(seqs(imgs), tc.wantSeqs...) {
				t.Fatalf("seqs %v, want %v", seqs(imgs), tc.wantSeqs)
			}
			for _, im := range imgs {
				if !bytes.Equal(im.Data, tc.imgs[im.Seq].data) {
					t.Fatalf("image %d data mismatch", im.Seq)
				}
			}
			st.TextBytes = 0 // the corrupt chunk is rescanned as text
			if st != tc.want {
				t.Fatalf("stats %+v, want %+v", st, tc.want)
			}
		})
	}
}

func TestBadHeaderResync(t *testing.T) {
	hdr := func(h Header) []byte {
		var b [HeaderSize]byte
		h.Put(&b)
		binary.LittleEndian.PutUint32(b[20:], crc32.ChecksumIEEE(b[:20]))
		return b[:]
	}
	img := y8(64, 64, 9)
	good := encodeImages(t, img)
	cases := []struct {
		name string
		bad  []byte
	}{
		{"payload_len 4097", join(hdr(Header{ChunkCnt: 1, PayloadLen: 4097, Flags: FlagFirst | FlagLast | FlagMeta}), pattern(4097, 1))},
		{"payload_len 0", hdr(Header{ChunkCnt: 1, PayloadLen: 0, Flags: FlagFirst | FlagLast | FlagMeta})},
		{"chunk_cnt 0", hdr(Header{ChunkCnt: 0, PayloadLen: 20, Flags: FlagFirst | FlagLast | FlagMeta})},
		{"chunk_idx = chunk_cnt", hdr(Header{ChunkIdx: 2, ChunkCnt: 2, PayloadLen: 20, Flags: FlagLast})},
		{"truncated header", []byte("ERIM\x01\x02")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := join([]byte("a"), tc.bad, []byte("b"), good, []byte("c"))
			imgs, st, text := decodeAll(t, bytes.NewReader(s))
			if len(imgs) != 1 || !bytes.Equal(imgs[0].Data, img.data) {
				t.Fatalf("images %d stats %+v", len(imgs), st)
			}
			wantText := join([]byte("a"), tc.bad, []byte("bc"))
			if !bytes.Equal(text, wantText) {
				t.Fatalf("text len %d, want %d", len(text), len(wantText))
			}
			if st.CRCErrors != 0 || st.Dropped != 0 || st.Images != 1 {
				t.Fatalf("stats %+v", st)
			}
		})
	}
}

func TestChunkSequenceErrors(t *testing.T) {
	imgs3 := []testImage{y8(160, 120, 1), y8(160, 120, 2), y8(160, 120, 3)} // 5 chunks each
	cases := []struct {
		name     string
		order    []int // chunk indices of the original stream, in send order
		wantSeqs []uint32
		want     Stats
	}{
		{"missing middle chunk", []int{0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14}, []uint32{0, 2}, Stats{Images: 2, Dropped: 1}},
		{"out of order", []int{0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13, 14}, []uint32{0, 2}, Stats{Images: 2, Dropped: 1}},
		{"new seq before last", []int{0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, []uint32{1, 2}, Stats{Images: 2, Dropped: 1}},
		{"missing first chunk", []int{0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14}, []uint32{0, 2}, Stats{Images: 2, Dropped: 1}},
		{"duplicate chunk", []int{0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, []uint32{1, 2}, Stats{Images: 2, Dropped: 1}},
		{"seq gap", []int{0, 1, 2, 3, 4, 10, 11, 12, 13, 14}, []uint32{0, 2}, Stats{Images: 2, Lost: 1}},
		{"truncated at end", []int{0, 1, 2, 3, 4, 5, 6}, []uint32{0}, Stats{Images: 1, Dropped: 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			chunks := splitChunks(t, encodeImages(t, imgs3...))
			var s []byte
			for _, i := range tc.order {
				s = append(s, chunks[i]...)
			}
			imgs, st, _ := decodeAll(t, bytes.NewReader(s))
			if !equalSeqs(seqs(imgs), tc.wantSeqs...) {
				t.Fatalf("seqs %v, want %v", seqs(imgs), tc.wantSeqs)
			}
			if st != tc.want {
				t.Fatalf("stats %+v, want %+v", st, tc.want)
			}
		})
	}
}

func TestSeqGapCounted(t *testing.T) {
	var all []testImage
	for i := range 8 {
		all = append(all, jpegImg(5000, byte(i)))
	}
	chunks := splitChunks(t, encodeImages(t, all...)) // 2 chunks per image
	var s []byte
	for _, img := range []int{0, 1, 4, 7} {
		s = append(s, chunks[2*img]...)
		s = append(s, chunks[2*img+1]...)
	}
	imgs, st, _ := decodeAll(t, bytes.NewReader(s))
	if !equalSeqs(seqs(imgs), 0, 1, 4, 7) || st != (Stats{Images: 4, Lost: 4}) {
		t.Fatalf("seqs %v stats %+v", seqs(imgs), st)
	}
}

func TestSeqWrap(t *testing.T) {
	var buf bytes.Buffer
	e := NewEncoder(&buf)
	e.seq = 0xFFFFFFFF
	for i := range 3 {
		if err := e.WriteImage(jpegImg(10, byte(i)).meta, 0, pattern(10, byte(i))); err != nil {
			t.Fatal(err)
		}
	}
	imgs, st, _ := decodeAll(t, &buf)
	if !equalSeqs(seqs(imgs), 0xFFFFFFFF, 0, 1) || st != (Stats{Images: 3}) {
		t.Fatalf("seqs %v stats %+v", seqs(imgs), st)
	}
}

func TestJPEGExactLength(t *testing.T) {
	for _, n := range []int{1, 17, MaxPayload - MetaSize, MaxPayload, 12345} {
		data := pattern(n, byte(n))
		if n >= 3 {
			data[0], data[1], data[n-1] = 0xFF, 0xD8, 0xD9
		}
		s := join(encodeImages(t, testImage{Meta{Width: 800, Height: 600, Format: JPEG}, 1, data}), []byte{0xFF, 0xD9, '\n'})
		imgs, st, text := decodeAll(t, bytes.NewReader(s))
		if len(imgs) != 1 || len(imgs[0].Data) != n || imgs[0].Meta.DataLen != uint32(n) || !bytes.Equal(imgs[0].Data, data) {
			t.Fatalf("n=%d: images %d stats %+v", n, len(imgs), st)
		}
		if !bytes.Equal(text, []byte{0xFF, 0xD9, '\n'}) {
			t.Fatalf("n=%d: text %q", n, text)
		}
	}
}

func TestMagicInsideData(t *testing.T) {
	// The image data is itself a valid frame stream plus raw magics.
	inner := encodeImages(t, y8(64, 64, 3), jpegImg(100, 4))
	data := join(bytes.Repeat([]byte("ERIM"), 50), inner, []byte("ERIM"))
	outer := testImage{Meta{Width: 1, Height: 1, Format: JPEG}, 5, data}
	s := encodeImages(t, outer, outer)
	imgs, st, text := decodeAll(t, bytes.NewReader(s))
	if !equalSeqs(seqs(imgs), 0, 1) || !bytes.Equal(imgs[0].Data, data) || !bytes.Equal(imgs[1].Data, data) {
		t.Fatalf("seqs %v stats %+v", seqs(imgs), st)
	}
	if len(text) != 0 || st != (Stats{Images: 2}) {
		t.Fatalf("text %d stats %+v", len(text), st)
	}
}

func TestBadMetaDropsImage(t *testing.T) {
	cases := []struct {
		name  string
		patch func(meta []byte)
	}{
		{"version 2", func(m []byte) { m[5] = 2 }},
		{"data_len too small for Y8", func(m []byte) { binary.LittleEndian.PutUint16(m[2:], 121) }},
		{"zero width", func(m []byte) { binary.LittleEndian.PutUint16(m[0:], 0) }},
		{"stride below width", func(m []byte) { binary.LittleEndian.PutUint16(m[6:], 159) }},
		{"data_len mismatch", func(m []byte) { binary.LittleEndian.PutUint32(m[8:], 160*120+1) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			chunks := splitChunks(t, encodeImages(t, y8(160, 120, 1), y8(160, 120, 2)))
			c := append([]byte(nil), chunks[0]...)
			tc.patch(c[HeaderSize : HeaderSize+MetaSize])
			binary.LittleEndian.PutUint32(c[20:], crc32.ChecksumIEEE(join(c[:20], c[HeaderSize:])))
			chunks[0] = c
			imgs, st, _ := decodeAll(t, bytes.NewReader(join(chunks...)))
			if !equalSeqs(seqs(imgs), 1) || st != (Stats{Images: 1, Dropped: 1}) {
				t.Fatalf("seqs %v stats %+v", seqs(imgs), st)
			}
		})
	}
}

func TestDecoderErrors(t *testing.T) {
	d := NewDecoder(bytes.NewReader(nil))
	if _, err := d.Next(); err != io.EOF {
		t.Fatalf("empty: %v", err)
	}
	if _, err := d.Next(); err != io.EOF {
		t.Fatalf("empty again: %v", err)
	}
	boom := errors.New("boom")
	s := encodeImages(t, y8(160, 120, 1))
	d = NewDecoder(io.MultiReader(bytes.NewReader(s[:5000]), iotest.ErrReader(boom)))
	if _, err := d.Next(); err != boom {
		t.Fatalf("read error: %v", err)
	}
}

func TestMetaWireLayout(t *testing.T) {
	m := Meta{Width: 0x0140, Height: 0x00F0, Format: RGB565BE, Stride: 0x0280, DataLen: 0x00025800, Sensor: SensorOV5640}
	want := []byte{0x40, 0x01, 0xF0, 0x00, 3, 1, 0x80, 0x02, 0x00, 0x58, 0x02, 0x00, 4, 0, 0, 0}
	got, _ := m.MarshalBinary()
	if !bytes.Equal(got, want) {
		t.Fatalf("meta % x, want % x", got, want)
	}
	var back Meta
	if err := back.UnmarshalBinary(append(want[:13:13], 9, 9, 9)); err != nil || back != m {
		t.Fatalf("unmarshal %+v %v", back, err)
	}
	bad := append([]byte(nil), want...)
	bad[5] = 0
	if err := back.UnmarshalBinary(bad); err != ErrVersion {
		t.Fatalf("version 0: %v", err)
	}
	if err := back.UnmarshalBinary(want[:15]); err != ErrShort {
		t.Fatalf("short: %v", err)
	}

	h := Header{Seq: 0x01020304, ChunkIdx: 1, ChunkCnt: 3, PayloadLen: 0x1000, Flags: 0, TS: 0xA0B0C0D0, CRC: 0x11223344}
	var hb [HeaderSize]byte
	h.Put(&hb)
	wantH := []byte{'E', 'R', 'I', 'M', 4, 3, 2, 1, 1, 0, 3, 0, 0x00, 0x10, 0, 0, 0xD0, 0xC0, 0xB0, 0xA0, 0x44, 0x33, 0x22, 0x11}
	if !bytes.Equal(hb[:], wantH) {
		t.Fatalf("header % x, want % x", hb, wantH)
	}
	if back, err := ParseHeader(hb[:]); err != nil || back != h {
		t.Fatalf("ParseHeader %+v %v", back, err)
	}
}

func TestFormat(t *testing.T) {
	cases := []struct {
		f    Format
		bpp  int
		name string
	}{
		{Y8, 1, "Y8"}, {RGB565LE, 2, "RGB565LE"}, {RGB565BE, 2, "RGB565BE"}, {YUYV, 2, "YUYV"},
		{JPEG, 0, "JPEG"}, {RGB888Planar, 3, "RGB888Planar"}, {0, 0, "unknown"}, {7, 0, "unknown"},
	}
	for _, tc := range cases {
		if tc.f.BytesPerPixel() != tc.bpp || tc.f.String() != tc.name || tc.f.Known() != (tc.name != "unknown") {
			t.Errorf("format %d: bpp %d name %q", tc.f, tc.f.BytesPerPixel(), tc.f.String())
		}
	}
}

func TestEncoderRejects(t *testing.T) {
	e := NewEncoder(io.Discard)
	cases := []struct {
		name string
		meta Meta
		n    int
		want error
	}{
		{"data_len mismatch", Meta{Width: 4, Height: 4, Format: Y8, DataLen: 15}, 16, ErrDataLen},
		{"short data", Meta{Width: 4, Height: 4, Format: Y8}, 15, ErrMeta},
		{"unknown format", Meta{Width: 4, Height: 4, Format: 9}, 16, ErrMeta},
		{"empty", Meta{Width: 4, Height: 4, Format: JPEG}, 0, ErrMeta},
		{"odd YUYV", Meta{Width: 3, Height: 2, Format: YUYV}, 12, ErrMeta},
		{"planar stride not /3", Meta{Width: 4, Height: 2, Format: RGB888Planar, Stride: 13}, 26, ErrMeta},
	}
	for _, tc := range cases {
		if err := e.WriteImage(tc.meta, 0, make([]byte, tc.n)); err != tc.want {
			t.Errorf("%s: %v, want %v", tc.name, err, tc.want)
		}
	}
	if e.Seq() != 0 {
		t.Fatalf("rejected images advanced seq to %d", e.Seq())
	}
}

type countWriter struct{ n int }

func (w *countWriter) Write(p []byte) (int, error) { w.n += len(p); return len(p), nil }

func TestEncoderZeroAllocs(t *testing.T) {
	w := &countWriter{}
	e := NewEncoder(w)
	m := Meta{Width: 320, Height: 240, Format: RGB565LE, Sensor: SensorOV7740}
	data := pattern(320*240*2, 1)
	allocs := testing.AllocsPerRun(50, func() {
		if err := e.WriteImage(m, 123, data); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("WriteImage allocates %v times per image", allocs)
	}
	chunks := (MetaSize + len(data) + MaxPayload - 1) / MaxPayload
	if want := 51 * (chunks*HeaderSize + MetaSize + len(data)); w.n != want {
		t.Fatalf("wrote %d bytes, want %d", w.n, want)
	}
}

type failWriter struct{ left int }

func (w *failWriter) Write(p []byte) (int, error) {
	if w.left <= 0 {
		return 0, io.ErrShortWrite
	}
	w.left--
	return len(p), nil
}

func TestEncoderWriteError(t *testing.T) {
	e := NewEncoder(&failWriter{left: 2})
	if err := e.WriteImage(y8(160, 120, 1).meta, 0, pattern(160*120, 1)); err != io.ErrShortWrite {
		t.Fatalf("err %v", err)
	}
	if e.Seq() != 1 {
		t.Fatalf("seq %d after failed image, want 1", e.Seq())
	}
}
