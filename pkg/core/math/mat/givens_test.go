package mat

import (
	"testing"

	"github.com/chewxy/math32"
)

func TestG1(t *testing.T) {
	tests := []struct {
		name     string
		a, b     float32
		checkCS  bool
		checkSig bool
	}{
		{
			name:     "a > b",
			a:        5,
			b:        3,
			checkCS:  true,
			checkSig: true,
		},
		{
			name:     "b > a",
			a:        3,
			b:        5,
			checkCS:  true,
			checkSig: true,
		},
		{
			name:     "zero b",
			a:        5,
			b:        0,
			checkCS:  true,
			checkSig: true,
		},
		{
			name:     "zero a",
			a:        0,
			b:        5,
			checkCS:  true,
			checkSig: true,
		},
		{
			name:     "both zero",
			a:        0,
			b:        0,
			checkCS:  true,
			checkSig: true,
		},
		{
			name:     "3-4-5",
			a:        3,
			b:        4,
			checkCS:  true,
			checkSig: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cs, sn, sig := G1(tt.a, tt.b)

			// Check that rotation matrix has correct properties
			cs2 := cs * cs
			sn2 := sn * sn
			if math32.Abs(cs2+sn2-1.0) > 1e-5 {
				t.Errorf("G1(%v, %v): cs² + sn² should be 1, got %v + %v = %v",
					tt.a, tt.b, cs2, sn2, cs2+sn2)
			}

			// Check that rotation eliminates b component
			// [cs sn] [a] = [sig]
			// [-sn cs] [b]   [0 ]
			resultX := cs*tt.a + sn*tt.b
			resultY := -sn*tt.a + cs*tt.b

			if math32.Abs(resultX-sig) > 1e-5 {
				t.Errorf("G1(%v, %v): resultX should equal sig, got %v != %v",
					tt.a, tt.b, resultX, sig)
			}
			if math32.Abs(resultY) > 1e-5 && (tt.a != 0 || tt.b != 0) {
				t.Errorf("G1(%v, %v): resultY should be ~0, got %v",
					tt.a, tt.b, resultY)
			}

			// Check that sig = sqrt(a² + b²)
			expectedSig := math32.Sqrt(tt.a*tt.a + tt.b*tt.b)
			if math32.Abs(sig-expectedSig) > 1e-5 && (tt.a != 0 || tt.b != 0) {
				t.Errorf("G1(%v, %v): sig should equal sqrt(a²+b²), got %v != %v",
					tt.a, tt.b, sig, expectedSig)
			}
		})
	}
}

func TestG2(t *testing.T) {
	tests := []struct {
		name    string
		cs, sn  float32
		x, y    float32
		checkRot bool
	}{
		{
			name:     "identity rotation",
			cs:       1,
			sn:       0,
			x:        5,
			y:        3,
			checkRot: true,
		},
		{
			name:     "90 degree rotation",
			cs:       0,
			sn:       1,
			x:        5,
			y:        3,
			checkRot: true,
		},
		{
			name:     "45 degree rotation",
			cs:       math32.Cos(math32.Pi / 4),
			sn:       math32.Sin(math32.Pi / 4),
			x:        5,
			y:        0,
			checkRot: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := tt.x
			y := tt.y
			G2(tt.cs, tt.sn, &x, &y)

			if tt.checkRot {
				// Verify rotation preserves magnitude
				origMag := math32.Sqrt(tt.x*tt.x + tt.y*tt.y)
				newMag := math32.Sqrt(x*x + y*y)
				if math32.Abs(origMag-newMag) > 1e-5 {
					t.Errorf("G2: rotation should preserve magnitude, got %v != %v",
						newMag, origMag)
				}
			}
		})
	}
}

func TestG1G2Integration(t *testing.T) {
	// Test that G1 and G2 work together correctly
	a := float32(5)
	b := float32(3)

	cs, sn, sig := G1(a, b)

	// Apply rotation to original vector
	x := a
	y := b
	G2(cs, sn, &x, &y)

	// Result should be [sig, 0]
	if math32.Abs(x-sig) > 1e-5 {
		t.Errorf("G1+G2: x should equal sig, got %v != %v", x, sig)
	}
	if math32.Abs(y) > 1e-5 {
		t.Errorf("G1+G2: y should be ~0, got %v", y)
	}
}

