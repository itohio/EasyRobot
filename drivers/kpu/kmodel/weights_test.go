package kmodel

import (
	"bytes"
	"encoding/hex"
	"errors"
	"testing"
)

func TestEncodeWeightsOrder(t *testing.T) {
	// Conv 3×3, 2 in, 2 out: TFLite OHWI value = 100·o + 10·(ky·3+kx) + i.
	w := make([]uint16, 2*3*3*2)
	for o := range 2 {
		for ky := range 3 {
			for kx := range 3 {
				for i := range 2 {
					w[((o*3+ky)*3+kx)*2+i] = uint16(100*o + 10*(ky*3+kx) + i)
				}
			}
		}
	}
	got, err := EncodeWeights(w, 3, 2, 2, false, true)
	if err != nil {
		t.Fatal(err)
	}
	var want []byte // OIHW
	for o := range 2 {
		for i := range 2 {
			for tap := range 9 {
				want = append(want, byte(100*o+10*tap+i))
			}
		}
	}
	if !bytes.Equal(got, want) {
		t.Errorf("conv OIHW:\n got %v\nwant %v", got, want)
	}

	// Depthwise 3×3, 3 channels: TFLite [KH][KW][C] value = 10·tap + c.
	d := make([]uint16, 9*3)
	for tap := range 9 {
		for c := range 3 {
			d[tap*3+c] = uint16(10*tap + c)
		}
	}
	got, err = EncodeWeights(d, 3, 3, 3, true, true)
	if err != nil {
		t.Fatal(err)
	}
	want = want[:0]
	for c := range 3 {
		for tap := range 9 {
			want = append(want, byte(10*tap+c))
		}
	}
	if !bytes.Equal(got, want) {
		t.Errorf("depthwise (c, ky, kx):\n got %v\nwant %v", got, want)
	}

	// 16-bit: little-endian.
	got, err = EncodeWeights([]uint16{0x1234, 0xabcd}, 1, 2, 1, false, false)
	if err != nil || !bytes.Equal(got, []byte{0x34, 0x12, 0xcd, 0xab}) {
		t.Errorf("16-bit: %x, %v", got, err)
	}
}

func TestEncodeWeightsRejects(t *testing.T) {
	cases := map[string]func() ([]byte, error){
		"8-bit value 256":    func() ([]byte, error) { return EncodeWeights([]uint16{256}, 1, 1, 1, false, true) },
		"kernel 5":           func() ([]byte, error) { return EncodeWeights(make([]uint16, 25), 5, 1, 1, false, true) },
		"short weights":      func() ([]byte, error) { return EncodeWeights(make([]uint16, 17), 3, 2, 1, false, true) },
		"depthwise in ≠ out": func() ([]byte, error) { return EncodeWeights(make([]uint16, 9*2), 3, 2, 4, true, true) },
		"channels 1025":      func() ([]byte, error) { return EncodeWeights(make([]uint16, 1025), 1, 1025, 1, false, true) },
		"channels 0":         func() ([]byte, error) { return EncodeWeights(nil, 1, 0, 1, false, true) },
	}
	for name, f := range cases {
		if _, err := f(); !errors.Is(err, ErrFormat) {
			t.Errorf("%s: err %v, want ErrFormat", name, err)
		}
	}
	if _, err := EncodeWeights([]uint16{255, 0xffff}, 1, 2, 1, false, false); err != nil {
		t.Errorf("16-bit 0xffff rejected: %v", err)
	}
}

// Golden blobs from nncase rc5 (ncc compiled tiny1/tiny2): the 8-bit weights in TFLite order, as ncc quantised them, and
// ncc's weight blob for the same layer. EncodeWeights must reproduce ncc's
// bytes exactly (byte identity checked in-repo).
var goldenWeights = []struct {
	name      string
	k, ic, oc int
	dw        bool
	in, ncc   string // hex
}{
	{"tiny1 layer 0: conv 3×3, 3→8", 3, 3, 8, false,
		"334e122d430f584d20454a1f3c2f04575d26252c00393f1571632776cc7362eb6775c77770d66e57ff5b72cf7287a7847abe798ca187f4dcc0f0d6bbead2bdf2dac1eed6bde9d0baffe3c9fde0c6f5dbc427ade321aee87699a508b8f400b9fc709da859a9af53a5b9938e92c3e49cbcdea484809741007f550e75989d85add094a7cf9c857f93e5f4dce7f3dcecfce1eaf9e1ecf7e1f0fee7e7f7e5e8f5e0efffe95b387b37006b6652828387887076867e818faab198bac1a0a2a39e6e487d6f508095899238006d42096e8e8c8f594b8061497d8e9a91",
		"332d58453c572539714e434d4a2f5d2c3f63120f201f0426001527766275705772877a8cccebc7d6ffcfa7bea17367776e5b72847987f4f0eaf2eee9fffdf5dcd6d2dad6d0e3e0dbc0bbbdc1bdbac9c6c4272176080070595393adae99b8b99da9a58ee3e8a5f4fca8afb992c3bc84415598ada785e4de80000e9dd0cf7f9ca4977f7585949c93e5e7eceaecf0e7e8eff4f3fcf9f7fef7f5ffdcdce1e1e1e7e5e0e95b376683707eaabaa2380052877681b1c1a37b6b8288868f98a09e6e6f9538428e59618e48508900098c4b499a7d80926d6e8f807d91"},
	{"tiny2 layer 1: depthwise 3×3 ×8", 3, 8, 8, true,
		"80c38fd0c90ad9cc53c7bac7d547ffdda3da8bccc60bb7bdd5fb64c2cb229acb00df12ffff216cffffac59fcc300e793b0ff21d4d133cbcb75df00c38f18b1d876b530ccda1cd4c3",
		"8053a3d500ffb07576c3c7dafbdfacffdfb58fba8b641259210030d0c7ccc2fffcd4c3ccc9d5c6cbffc3d18fda0a470b22210033181cd9ffb79a6ce7cbb1d4ccddbdcbff93cbd8c3"},
}

func TestEncodeWeightsGoldenNcc(t *testing.T) {
	for _, g := range goldenWeights {
		in, err := hex.DecodeString(g.in)
		if err != nil {
			t.Fatal(err)
		}
		want, err := hex.DecodeString(g.ncc)
		if err != nil {
			t.Fatal(err)
		}
		w := make([]uint16, len(in))
		for i, v := range in {
			w[i] = uint16(v)
		}
		got, err := EncodeWeights(w, g.k, g.ic, g.oc, g.dw, true)
		if err != nil {
			t.Fatalf("%s: %v", g.name, err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s: blob differs from ncc's\ngot  %x\nwant %x", g.name, got, want)
		}
	}
}
