package qi8

import (
	"errors"
	"testing"
)

// Shapes whose element counts overflow int (or exceed MaxElements) must be
// rejected with ErrShape before any length check or indexing, on every
// target. 2^21·2^21·2^22 = 2^64 wraps to 0 on a 64-bit int, so a nil slice
// would pass a naive length check and the kernel would then index out of
// range.
func TestHugeShapesRejected(t *testing.T) {
	rq := []Requant{{Mult: 1 << 30, Shift: 0}}
	wrap := Shape{H: 1 << 21, W: 1 << 21, C: 1 << 22}
	big32 := Shape{H: 1 << 16, W: 1 << 16, C: 1} // 2^32 elements: wraps a 32-bit int
	cases := []struct {
		name string
		run  func() error
	}{
		{"conv wrap", func() error {
			return Conv2D(nil, nil, nil, nil, rq, Conv2DParams{In: wrap, OutC: 1, KH: 1, KW: 1, StrideH: 1, StrideW: 1, ActMin: -128, ActMax: 127})
		}},
		{"conv 2^32", func() error {
			return Conv2D(nil, nil, nil, nil, rq, Conv2DParams{In: big32, OutC: 1, KH: 1, KW: 1, StrideH: 1, StrideW: 1, ActMin: -128, ActMax: 127})
		}},
		{"conv out wrap", func() error {
			// input fits, but outH·outW·OutC = 2^15·2^15·2^30 does not.
			return Conv2D(nil, nil, nil, nil, rq, Conv2DParams{In: Shape{H: 1 << 15, W: 1 << 15, C: 1}, OutC: 1 << 30, KH: 1, KW: 1, StrideH: 1, StrideW: 1, ActMin: -128, ActMax: 127})
		}},
		{"depthwise wrap", func() error {
			return DepthwiseConv2D(nil, nil, nil, nil, rq, DepthwiseConv2DParams{In: wrap, KH: 1, KW: 1, StrideH: 1, StrideW: 1, ActMin: -128, ActMax: 127})
		}},
		{"maxpool wrap", func() error {
			return MaxPool2D(nil, nil, Pool2DParams{In: wrap, KH: 1, KW: 1, StrideH: 1, StrideW: 1})
		}},
		{"avgpool 2^32", func() error {
			return AvgPool2D(nil, nil, Pool2DParams{In: big32, KH: 1, KW: 1, StrideH: 1, StrideW: 1})
		}},
		{"kernel wrap", func() error {
			return MaxPool2D(nil, nil, Pool2DParams{In: Shape{H: 4, W: 4, C: 1}, KH: MaxElements + 1, KW: 1, StrideH: 1, StrideW: 1})
		}},
		{"fc weights wrap", func() error {
			// In is at the reduction bound; In·Out overflows MaxElements.
			return FullyConnected(nil, nil, nil, nil, rq, FCParams{In: MaxTaps, Out: 1 << 30, ActMin: -128, ActMax: 127})
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var err error
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("panicked: %v", r)
					}
				}()
				err = c.run()
			}()
			if !errors.Is(err, ErrShape) {
				t.Fatalf("err = %v, want ErrShape", err)
			}
		})
	}
	if _, err := (Conv2DParams{In: wrap, OutC: 1, KH: 1, KW: 1, StrideH: 1, StrideW: 1}).OutShape(); !errors.Is(err, ErrShape) {
		t.Fatalf("OutShape err = %v, want ErrShape", err)
	}
}

// The int32 accumulator holds at most MaxTaps taps of |x−zp|·|w| ≤ 255·128;
// longer reductions must be rejected instead of wrapping silently.
func TestReductionLengthBound(t *testing.T) {
	if MaxTaps != 65793 {
		t.Fatalf("MaxTaps = %d, want 65793 (floor((2^31−1)/(255·128)))", MaxTaps)
	}
	rq := []Requant{{Mult: 1 << 30, Shift: 0}}
	n := MaxTaps + 1
	src := make([]int8, n)
	w := make([]int8, n)
	dst := make([]int8, 1)
	if err := FullyConnected(dst, src, w, nil, rq, FCParams{In: n, Out: 1, ActMin: -128, ActMax: 127}); !errors.Is(err, ErrShape) {
		t.Fatalf("FC In=MaxTaps+1: err = %v, want ErrShape", err)
	}
	if err := FullyConnected(dst, src[:MaxTaps], w[:MaxTaps], nil, rq, FCParams{In: MaxTaps, Out: 1, ActMin: -128, ActMax: 127}); err != nil {
		t.Fatalf("FC In=MaxTaps: %v", err)
	}
	if err := Conv2D(dst, src, w, nil, rq, Conv2DParams{In: Shape{H: 1, W: 1, C: n}, OutC: 1, KH: 1, KW: 1, StrideH: 1, StrideW: 1, ActMin: -128, ActMax: 127}); !errors.Is(err, ErrShape) {
		t.Fatalf("Conv K=MaxTaps+1: err = %v, want ErrShape", err)
	}
	// Worst case at the bound must not wrap: x−zp = 255, w = −128 at every tap.
	for i := 0; i < MaxTaps; i++ {
		src[i], w[i] = 127, -128
	}
	unit := []Requant{{Mult: 1 << 30, Shift: 1}} // M' = 1
	if err := FullyConnected(dst, src[:MaxTaps], w[:MaxTaps], nil, unit, FCParams{In: MaxTaps, Out: 1, InZP: -128, ActMin: -128, ActMax: 127}); err != nil {
		t.Fatal(err)
	}
	if dst[0] != -128 {
		t.Fatalf("worst-case acc wrapped: dst = %d, want -128 (saturated negative)", dst[0])
	}
}

// TestReductionLengthBoundWZP: with a weight zero point |w − WZP| ≤ 255, so
// the bound that keeps Σ (x−zp)(w−WZP) inside int32 is MaxTapsWZP; longer
// reductions are ErrShape for every weighted op, and the worst case at the
// bound does not wrap.
func TestReductionLengthBoundWZP(t *testing.T) {
	if MaxTapsWZP != 33025 {
		t.Fatalf("MaxTapsWZP = %d, want 33025 (floor((2^31−1)/(255·255)))", MaxTapsWZP)
	}
	rq := []Requant{{Mult: 1 << 30, Shift: 0}}
	n := MaxTapsWZP + 1
	src, w, dst := make([]int8, n), make([]int8, n), make([]int8, 1)
	act := func(p *FCParams) { p.ActMin, p.ActMax = -128, 127 }
	fc := FCParams{In: n, Out: 1, WZP: 127}
	act(&fc)
	if err := FullyConnected(dst, src, w, nil, rq, fc); !errors.Is(err, ErrShape) {
		t.Fatalf("FC In=MaxTapsWZP+1, WZP≠0: err = %v, want ErrShape", err)
	}
	fc.WZP = 0
	if err := FullyConnected(dst, src, w, nil, rq, fc); err != nil {
		t.Fatalf("FC In=MaxTapsWZP+1, WZP=0: %v", err)
	}
	if err := Conv2D(dst, src, w, nil, rq, Conv2DParams{In: Shape{H: 1, W: 1, C: n}, OutC: 1, KH: 1, KW: 1, StrideH: 1, StrideW: 1, WZP: -1, ActMin: -128, ActMax: 127}); !errors.Is(err, ErrShape) {
		t.Fatalf("Conv K=MaxTapsWZP+1, WZP≠0: err = %v, want ErrShape", err)
	}
	dp := DepthwiseConv2DParams{In: Shape{H: 1, W: n, C: 1}, KH: 1, KW: n, StrideH: 1, StrideW: 1, WZP: 5, ActMin: -128, ActMax: 127}
	if err := DepthwiseConv2D(dst, src, w, nil, rq, dp); !errors.Is(err, ErrShape) {
		t.Fatalf("Depthwise taps=MaxTapsWZP+1, WZP≠0: err = %v, want ErrShape", err)
	}
	// Worst case at the bound: x−zp = 255 and w−WZP = −255 at every tap.
	for i := range MaxTapsWZP {
		src[i], w[i] = 127, -128
	}
	unit := []Requant{{Mult: 1 << 30, Shift: 1}} // M' = 1
	fc = FCParams{In: MaxTapsWZP, Out: 1, InZP: -128, WZP: 127}
	act(&fc)
	if err := FullyConnected(dst, src[:MaxTapsWZP], w[:MaxTapsWZP], nil, unit, fc); err != nil {
		t.Fatal(err)
	}
	if dst[0] != -128 {
		t.Fatalf("worst-case acc wrapped: dst = %d, want -128 (saturated negative)", dst[0])
	}
}
