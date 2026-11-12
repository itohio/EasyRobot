//go:build qi8ctest && cgo

package qi8

import (
	"math"
	"math/rand"
	"testing"
)

// TestCRequantizeMatchesGo: the C requantisation the PIE kernels use gives
// the Go reference's byte for every rounding mode on the int32 edges, every
// shift, int8 zero points and clamps, and random inputs. Run with
// `go test -tags qi8ctest` (cgo; the C is built with UBSan, so a signed
// overflow aborts the test).
func TestCRequantizeMatchesGo(t *testing.T) {
	// Review R-NN-P3 P1: the TFLite path added zp in int32 and wrapped.
	big := Requant{Mult: math.MaxInt32, Shift: 0}
	if got := cRequantizeMode(math.MaxInt32, big, 127, -128, 127, RoundTFLite); got != 127 {
		t.Fatalf("acc 2^31-1, M≈1, zp 127: C gives %d, Go %d", got, requantizeMode(math.MaxInt32, big, 127, -128, 127, RoundTFLite))
	}

	accs := []int32{math.MinInt32, math.MinInt32 + 1, -1 << 30, -65537, -1000, -1, 0, 1, 999, 65536, 1 << 30, math.MaxInt32 - 1, math.MaxInt32}
	mults := []int32{multMin, multMin + 1, 1518500250, math.MaxInt32}
	zps := []int64{-128, -1, 0, 1, 127}
	clamps := [][2]int64{{-128, 127}, {0, 127}, {-128, 0}, {-5, 6}}
	n, bad := 0, 0
	check := func(acc int32, r Requant, zp int64, c [2]int64, mode Rounding) {
		n++
		want := requantizeMode(acc, r, zp, c[0], c[1], mode)
		if got := cRequantizeMode(acc, r, zp, c[0], c[1], mode); got != want {
			if bad < 10 {
				t.Errorf("acc %d %+v zp %d clamp %v mode %d: C %d, Go %d", acc, r, zp, c, mode, got, want)
			}
			bad++
		}
	}
	for _, mode := range []Rounding{RoundExact, RoundTFLite} {
		for _, acc := range accs {
			for _, zp := range zps {
				for _, c := range clamps {
					check(acc, Requant{}, zp, c, mode)
					for _, m := range mults {
						for s := int32(MinShift); s <= MaxShift; s++ {
							check(acc, Requant{Mult: m, Shift: s}, zp, c, mode)
						}
					}
				}
			}
		}
	}
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 200000; i++ {
		r := Requant{Mult: multMin + rng.Int31n(math.MaxInt32-multMin) + 1, Shift: MinShift + rng.Int31n(MaxShift-MinShift+1)}
		lo := int64(rng.Intn(256) - 128)
		hi := lo + int64(rng.Intn(int(128-lo)))
		check(int32(rng.Uint32()), r, int64(rng.Intn(256)-128), [2]int64{lo, hi}, Rounding(i&1))
	}
	if bad > 0 {
		t.Fatalf("%d of %d cases differ", bad, n)
	}
	t.Logf("%d cases equal", n)
}
