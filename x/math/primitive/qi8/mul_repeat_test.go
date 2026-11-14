package qi8

import "testing"

// Repeat > 1 broadcasts each b value over Repeat consecutive a values (an
// H×W×1 operand times an H×W×C one, as YOLOX's objectness × class scores);
// the result equals Mul against b expanded by hand.
func TestMulRepeat(t *testing.T) {
	rq, err := QuantizeMul(0.02, 0.03, 0.01)
	if err != nil {
		t.Fatal(err)
	}
	const hw, c = 6, 5
	a, b := make([]int8, hw*c), make([]int8, hw)
	for i := range a {
		a[i] = int8(i*29 - 70)
	}
	for i := range b {
		b[i] = int8(i*41 - 90)
	}
	exp := make([]int8, len(a))
	for i := range exp {
		exp[i] = b[i/c]
	}
	for _, round := range []Rounding{RoundExact, RoundTFLite} {
		p := MulParams{ZPA: -3, ZPB: 7, ZPOut: 2, RQ: rq, Min: -128, Max: 127, Round: round}
		want := make([]int8, len(a))
		if err := Mul(want, a, exp, p); err != nil {
			t.Fatal(err)
		}
		p.Repeat = c
		got := make([]int8, len(a))
		if err := Mul(got, a, b, p); err != nil {
			t.Fatal(err)
		}
		if string(u8s(got)) != string(u8s(want)) {
			t.Errorf("round %d: per-pixel broadcast differs from the expanded operand", round)
		}
	}
	p := MulParams{RQ: rq, Min: -128, Max: 127, Repeat: c}
	if err := Mul(make([]int8, hw*c), a, make([]int8, hw+1), p); err != ErrLength {
		t.Errorf("b not a multiple: %v, want ErrLength", err)
	}
	if err := Mul(make([]int8, 7), make([]int8, 7), make([]int8, 1), MulParams{RQ: rq, Min: -128, Max: 127, Repeat: 2}); err != ErrLength {
		t.Errorf("len(a) not a multiple of Repeat: %v, want ErrLength", err)
	}
}
