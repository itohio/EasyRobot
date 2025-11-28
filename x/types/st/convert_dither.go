package st

// Xorshift64* PRNG for dithering
// Fast, good quality random number generator for TPDF dithering
type xorshift64Star struct {
	state uint64
}

// newXorshift64Star creates a new Xorshift64* PRNG with seed
func newXorshift64Star(seed uint64) *xorshift64Star {
	if seed == 0 {
		seed = 1 // Avoid zero state
	}
	return &xorshift64Star{state: seed}
}

// next returns the next random uint64
func (x *xorshift64Star) next() uint64 {
	x.state ^= x.state >> 12
	x.state ^= x.state << 25
	x.state ^= x.state >> 27
	return x.state * 0x2545F4914F6CDD1D
}

// nextFloat32 returns a random float32 in [0, 1)
func (x *xorshift64Star) nextFloat32() float32 {
	// Use upper 32 bits for better quality
	return float32(x.next()>>32) / float32(1<<32)
}

// tpdfDither generates triangular PDF dither noise in [-1, 1] range
// TPDF = two uniform random numbers subtracted
func tpdfDither(rng *xorshift64Star) float32 {
	u1 := rng.nextFloat32()
	u2 := rng.nextFloat32()
	// TPDF: (u1 - u2) gives triangular distribution in [-1, 1]
	return u1 - u2
}

