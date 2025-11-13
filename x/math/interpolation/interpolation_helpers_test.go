package interpolation

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
)

const floatTolerance = 1e-5

func approxEqual(a, b float32) bool {
	return math.Abs(float64(a-b)) <= floatTolerance
}

func approxEqualSlice(t *testing.T, actual, expected []float32) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("length mismatch: got %d want %d", len(actual), len(expected))
	}
	for i := range actual {
		if !approxEqual(actual[i], expected[i]) {
			t.Fatalf("index %d: got %f want %f", i, actual[i], expected[i])
		}
	}
}

func TestBezier1d(t *testing.T) {
	tests := []struct {
		name     string
		points   [4]float32
		t        float32
		expected float32
	}{
		{
			name:     "start",
			points:   [4]float32{0, 1, 2, 3},
			t:        0,
			expected: 0,
		},
		{
			name:     "mid",
			points:   [4]float32{0, 1, 2, 3},
			t:        0.5,
			expected: 1.5,
		},
		{
			name:     "end",
			points:   [4]float32{0, 1, 2, 3},
			t:        1,
			expected: 3,
		},
		{
			name:     "constant",
			points:   [4]float32{2, 2, 2, 2},
			t:        0.3,
			expected: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Bezier1d(tc.points[0], tc.points[1], tc.points[2], tc.points[3], tc.t)
			if !approxEqual(got, tc.expected) {
				t.Fatalf("got %f want %f", got, tc.expected)
			}
		})
	}
}

func TestBezierVector(t *testing.T) {
	t.Run("2d", func(t *testing.T) {
		points := [4][2]float32{
			{0, 0},
			{1, 2},
			{2, 4},
			{3, 6},
		}

		got := Bezier2d(points[0], points[1], points[2], points[3], 0.5)
		want := [2]float32{1.5, 3}
		if !approxEqual(got[0], want[0]) || !approxEqual(got[1], want[1]) {
			t.Fatalf("got %v want %v", got, want)
		}
	})

	t.Run("3d", func(t *testing.T) {
		points := [4][3]float32{
			{0, 0, 0},
			{1, 2, 3},
			{2, 4, 6},
			{3, 6, 9},
		}

		got := Bezier3d(points[0], points[1], points[2], points[3], 0.25)
		want := [3]float32{0.75, 1.5, 2.25}
		for i := range got {
			if !approxEqual(got[i], want[i]) {
				t.Fatalf("index %d: got %f want %f", i, got[i], want[i])
			}
		}
	})

	t.Run("generic", func(t *testing.T) {
		p1 := []float32{0, 0}
		p2 := []float32{1, 2}
		p3 := []float32{2, 4}
		p4 := []float32{3, 6}

		got := Bezier(p1, p2, p3, p4, 0.5)
		want := []float32{1.5, 3}
		approxEqualSlice(t, got, want)
	})

	t.Run("mismatched-length", func(t *testing.T) {
		p1 := []float32{0, 0}
		p2 := []float32{1}
		p3 := []float32{2, 4}
		p4 := []float32{3, 6}

		if got := Bezier(p1, p2, p3, p4, 0.5); got != nil {
			t.Fatalf("expected nil for mismatched input, got %v", got)
		}
	})
}

func TestCosine1D(t *testing.T) {
	tests := []struct {
		name     string
		a, b, t  float32
		expected float32
	}{
		{name: "start", a: 2, b: 10, t: 0, expected: 2},
		{name: "end", a: 2, b: 10, t: 1, expected: 10},
		{name: "mid", a: 2, b: 10, t: 0.5, expected: 6},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if got := Cosine1D(tc.a, tc.b, tc.t); !approxEqual(got, tc.expected) {
				t.Fatalf("got %f want %f", got, tc.expected)
			}
		})
	}
}

func TestLerpD(t *testing.T) {
	tests := []struct {
		name     string
		a, d, t  float32
		expected float32
	}{
		{name: "zero", a: 5, d: 0, t: 0.4, expected: 5},
		{name: "positive", a: 1, d: 4, t: 0.25, expected: 2},
		{name: "negative", a: 3, d: -6, t: 0.5, expected: 0},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if got := LerpD(tc.a, tc.d, tc.t); !approxEqual(got, tc.expected) {
				t.Fatalf("got %f want %f", got, tc.expected)
			}
		})
	}
}

func TestCubicSpline1DProperties(t *testing.T) {
	points := []struct {
		name           string
		p1, p2, p3, p4 float32
	}{
		{name: "ascending", p1: 1, p2: 2, p3: 4, p4: 8},
		{name: "descending", p1: 8, p2: 4, p3: 2, p4: 1},
		{name: "constant", p1: 3, p2: 3, p3: 3, p4: 3},
	}

	for _, tc := range points {
		tc := tc
		t.Run(tc.name+"_endpoints", func(t *testing.T) {
			if got := CubicSpline1D(tc.p1, tc.p2, tc.p3, tc.p4, 0); !approxEqual(got, tc.p2) {
				t.Fatalf("t=0: got %f want %f", got, tc.p2)
			}
			if got := CubicSpline1D(tc.p1, tc.p2, tc.p3, tc.p4, 1); !approxEqual(got, tc.p3) {
				t.Fatalf("t=1: got %f want %f", got, tc.p3)
			}
		})

		t.Run(tc.name+"_constant", func(t *testing.T) {
			if tc.p1 == tc.p2 && tc.p2 == tc.p3 && tc.p3 == tc.p4 {
				mid := CubicSpline1D(tc.p1, tc.p2, tc.p3, tc.p4, 0.5)
				if !approxEqual(mid, tc.p2) {
					t.Fatalf("constant case: got %f want %f", mid, tc.p2)
				}
			}
		})
	}
}

func TestCubicCatmulRomSpline1D(t *testing.T) {
	p1, p2, p3, p4 := float32(-1), float32(0), float32(1), float32(2)

	if got := CubicCatmulRomSpline1D(p1, p2, p3, p4, 0); !approxEqual(got, p2) {
		t.Fatalf("t=0: got %f want %f", got, p2)
	}

	if got := CubicCatmulRomSpline1D(p1, p2, p3, p4, 1); !approxEqual(got, p3) {
		t.Fatalf("t=1: got %f want %f", got, p3)
	}

	if got := CubicCatmulRomSpline1D(p1, p2, p3, p4, 0.5); !approxEqual(got, 0.5) {
		t.Fatalf("t=0.5: got %f want %f", got, 0.5)
	}
}

func TestCubicHermiteSpline1D(t *testing.T) {
	p1, p2, p3, p4 := float32(0), float32(1), float32(2), float32(10)

	t.Run("endpoints", func(t *testing.T) {
		start := CubicHermiteSpline1D(p1, p2, p3, p4, 0, 0.5, 0)
		end := CubicHermiteSpline1D(p1, p2, p3, p4, 1, 0.5, 0)

		if !approxEqual(start, 3) {
			t.Fatalf("t=0: got %f want %f", start, float32(3))
		}
		if !approxEqual(end, p3) {
			t.Fatalf("t=1: got %f want %f", end, p3)
		}
	})

	t.Run("tension_effect", func(t *testing.T) {
		base := CubicHermiteSpline1D(p1, p2, p3, p4, 0.5, 0, 0)
		tensioned := CubicHermiteSpline1D(p1, p2, p3, p4, 0.5, 0.5, 0)
		if !(tensioned < base) {
			t.Fatalf("expected increased tension to reduce midpoint value: base=%f tensioned=%f", base, tensioned)
		}
	})

	t.Run("bias_effect", func(t *testing.T) {
		positive := CubicHermiteSpline1D(p1, p2, p3, p4, 0.5, 0.5, 0.5)
		negative := CubicHermiteSpline1D(p1, p2, p3, p4, 0.5, 0.5, -0.5)
		if !(positive > negative) {
			t.Fatalf("expected positive bias to favour first segment: positive=%f negative=%f", positive, negative)
		}
	})
}

func TestKrigingSamples(t *testing.T) {
	k := NewKriging(ExponentialVariogram(1))
	k.AddSample(0, 0, 1)
	k.AddSample(1, 1, 2)

	s := k.Samples()

	if len(s) != 2 {
		t.Fatalf("expected 2 samples, got %d", len(s))
	}

	if s[0].X != 0 || s[1].V != 2 {
		t.Fatalf("unexpected sample contents: %v", s)
	}

	s[0].V = 10
	if k.samples[0].V != 10 {
		t.Fatalf("expected Samples to share backing slice")
	}
}

func TestRBFSamples(t *testing.T) {
	r := NewRBF(GaussianKernel(), 1)
	r.AddSample(0, 0, 1)
	r.AddSample(1, 1, 2)

	s := r.Samples()
	if len(s) != 2 {
		t.Fatalf("expected 2 samples, got %d", len(s))
	}
	if s[1].X != 1 || s[1].Y != 1 || s[1].V != 2 {
		t.Fatalf("unexpected sample contents: %v", s)
	}

	s[1].V = 20
	if r.samples[1].V != 20 {
		t.Fatalf("expected Samples to share backing slice")
	}
}

func TestCopyMatrix(t *testing.T) {
	src := mat.Matrix{
		{1, 2},
		{3, 4},
	}
	dst := mat.New(len(src), len(src[0]))

	result := copyMatrix(src, dst)
	if result == nil {
		t.Fatal("expected non-nil result")
	}

	for i := range src {
		for j := range src[i] {
			if src[i][j] != dst[i][j] {
				t.Fatalf("dst[%d][%d]=%f want %f", i, j, dst[i][j], src[i][j])
			}
		}
	}

	if &result[0][0] != &dst[0][0] {
		t.Fatal("expected returned matrix to be destination")
	}
}
