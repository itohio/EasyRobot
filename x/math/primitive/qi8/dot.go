package qi8

// Dot returns the raw int32 dot product sum(a[i]*b[i]) over len(a) elements
// (no zero points). It panics if len(b) < len(a). The int32 accumulator wraps
// on overflow (only possible for len(a) > 2^17).
func Dot(a, b []int8) int32 {
	b = b[:len(a)]
	var acc int32
	for i, v := range a {
		acc += int32(v) * int32(b[i])
	}
	return acc
}

// dotZP returns sum((x[i]-zp)*w[i]) over len(x) elements.
func dotZP(x, w []int8, zp int32) int32 {
	w = w[:len(x)]
	var acc int32
	i := 0
	for ; i+4 <= len(x); i += 4 {
		acc += (int32(x[i])-zp)*int32(w[i]) +
			(int32(x[i+1])-zp)*int32(w[i+1]) +
			(int32(x[i+2])-zp)*int32(w[i+2]) +
			(int32(x[i+3])-zp)*int32(w[i+3])
	}
	for ; i < len(x); i++ {
		acc += (int32(x[i]) - zp) * int32(w[i])
	}
	return acc
}

// FCParams describes a fully connected layer (GEMV): out = W·(x-InZP) + bias.
type FCParams struct {
	In, Out int // input and output vector lengths
	// InZP and OutZP are the activation zero points.
	InZP, OutZP int8
	// ActMin/ActMax is the fused activation clamp (see ActNone/ActReLU/ActReLU6).
	// Note the zero value [0, 0] clamps everything to 0.
	ActMin, ActMax int8
}

// FullyConnected computes, for every output o,
//
//	acc    = bias[o] + sum_i (src[i]-InZP) * weights[o*In+i]   (int32)
//	dst[o] = clamp(rq[o].Apply(acc) + OutZP, ActMin, ActMax)
//
// weights are [Out][In] row-major; bias is nil or len Out; rq has len Out.
func FullyConnected(dst, src, weights []int8, bias []int32, rq []Requant, p FCParams) error {
	if p.In <= 0 || p.Out <= 0 {
		return ErrShape
	}
	if len(src) != p.In || len(dst) != p.Out || len(weights) != p.In*p.Out {
		return ErrLength
	}
	if err := validBias(bias, p.Out); err != nil {
		return err
	}
	if err := validRequants(rq, p.Out); err != nil {
		return err
	}
	lo, hi, err := clampRange(p.ActMin, p.ActMax)
	if err != nil {
		return err
	}
	zp, zpOut := int32(p.InZP), int64(p.OutZP)
	for o := range dst {
		acc := dotZP(src, weights[o*p.In:(o+1)*p.In], zp)
		if len(bias) != 0 {
			acc += bias[o]
		}
		dst[o] = requantize(acc, rq[o], zpOut, lo, hi)
	}
	return nil
}
