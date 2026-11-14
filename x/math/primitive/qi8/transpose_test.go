package qi8

import (
	"bytes"
	"testing"
)

// naiveTranspose: out[o0][o1][o2] = in[i…] with out axis k = in axis perm[k].
func naiveTranspose(src []int8, in Shape, perm [3]int) ([]int8, Shape) {
	d := [3]int{in.H, in.W, in.C}
	od := [3]int{d[perm[0]], d[perm[1]], d[perm[2]]}
	dst := make([]int8, len(src))
	var i [3]int
	for o0 := range od[0] {
		for o1 := range od[1] {
			for o2 := range od[2] {
				i[perm[0]], i[perm[1]], i[perm[2]] = o0, o1, o2
				dst[(o0*od[1]+o1)*od[2]+o2] = src[(i[0]*d[1]+i[1])*d[2]+i[2]]
			}
		}
	}
	return dst, Shape{H: od[0], W: od[1], C: od[2]}
}

func TestTransposeAllPerms(t *testing.T) {
	in := Shape{H: 3, W: 4, C: 5}
	src := make([]int8, in.Len())
	for i := range src {
		src[i] = int8(i*37 + 11)
	}
	for _, perm := range [][3]int{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}} {
		p := TransposeParams{In: in, Perm: perm}
		want, ws := naiveTranspose(src, in, perm)
		if s, err := p.OutShape(); err != nil || s != ws {
			t.Fatalf("perm %v: OutShape %v %v, want %v", perm, s, err, ws)
		}
		dst := make([]int8, len(src))
		if err := Transpose(dst, src, p); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(u8s(dst), u8s(want)) {
			t.Errorf("perm %v: differs from the naive transpose", perm)
		}
	}
}

func TestTransposeErrors(t *testing.T) {
	in := Shape{H: 2, W: 3, C: 4}
	for _, perm := range [][3]int{{0, 0, 1}, {0, 1, 3}, {-1, 1, 2}, {2, 2, 2}} {
		if err := Transpose(make([]int8, 24), make([]int8, 24), TransposeParams{In: in, Perm: perm}); err != ErrShape {
			t.Errorf("perm %v: %v, want ErrShape", perm, err)
		}
	}
	if err := Transpose(make([]int8, 23), make([]int8, 24), TransposeParams{In: in, Perm: [3]int{2, 1, 0}}); err != ErrLength {
		t.Errorf("short dst: %v, want ErrLength", err)
	}
	if err := Transpose(nil, nil, TransposeParams{In: Shape{H: 0, W: 1, C: 1}, Perm: [3]int{0, 1, 2}}); err != ErrShape {
		t.Errorf("empty shape: %v, want ErrShape", err)
	}
}

func TestTransposeNoAlloc(t *testing.T) {
	p := TransposeParams{In: Shape{H: 13, W: 13, C: 80}, Perm: [3]int{2, 0, 1}}
	src, dst := make([]int8, p.In.Len()), make([]int8, p.In.Len())
	if n := testing.AllocsPerRun(20, func() { _ = Transpose(dst, src, p) }); n != 0 {
		t.Errorf("Transpose allocates %v times per run", n)
	}
}

func u8s(b []int8) []byte {
	o := make([]byte, len(b))
	for i, v := range b {
		o[i] = byte(v)
	}
	return o
}
