package qi8

// TransposeParams describes TFLite TRANSPOSE on the three non-batch axes of
// an NHWC tensor (batch 1): output axis k is input axis Perm[k] (0 = H,
// 1 = W, 2 = C). Values are copied unchanged.
type TransposeParams struct {
	In   Shape
	Perm [3]int
}

func (p *TransposeParams) permValid() bool {
	var seen [3]bool
	for _, a := range p.Perm {
		if a < 0 || a > 2 || seen[a] {
			return false
		}
		seen[a] = true
	}
	return true
}

// OutShape is In with its axes permuted; ErrShape for an invalid shape or
// permutation.
func (p TransposeParams) OutShape() (Shape, error) {
	if !p.In.valid() || !p.permValid() {
		return Shape{}, ErrShape
	}
	d := [3]int{p.In.H, p.In.W, p.In.C}
	return Shape{H: d[p.Perm[0]], W: d[p.Perm[1]], C: d[p.Perm[2]]}, nil
}

// Transpose writes src (shape p.In) permuted into dst. ErrShape for an
// invalid shape or permutation, ErrLength when a length does not match.
// dst must not overlap src. It does not allocate.
func Transpose(dst, src []int8, p TransposeParams) error {
	n, ok := p.In.size()
	if !ok || !p.permValid() {
		return ErrShape
	}
	if len(src) != n || len(dst) != n {
		return ErrLength
	}
	d := [3]int{p.In.H, p.In.W, p.In.C}
	stride := [3]int{d[1] * d[2], d[2], 1} // input strides of H, W, C
	n0, n1, n2 := d[p.Perm[0]], d[p.Perm[1]], d[p.Perm[2]]
	s0, s1, s2 := stride[p.Perm[0]], stride[p.Perm[1]], stride[p.Perm[2]]
	o := 0
	for i0 := range n0 {
		b0 := i0 * s0
		for i1 := range n1 {
			b1 := b0 + i1*s1
			if s2 == 1 { // innermost axis unchanged: one contiguous run
				copy(dst[o:o+n2], src[b1:b1+n2])
				o += n2
				continue
			}
			for i2 := range n2 {
				dst[o] = src[b1+i2*s2]
				o++
			}
		}
	}
	return nil
}
