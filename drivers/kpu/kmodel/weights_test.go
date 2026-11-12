package kmodel

import (
	"bytes"
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
