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
	pl, err := p.plan()
	if err != nil {
		return err
	}
	if len(src) != pl.in || len(dst) != pl.out || len(weights) != pl.wLen {
		return ErrLength
	}
	if err := pl.setQuant(bias, rq, p); err != nil {
		return err
	}
	fullyConnectedGeneric(dst, src, weights, bias, rq, &pl)
	return nil
}

// fcPlan is the validated, resolved form of FCParams.
type fcPlan struct {
	in, out, wLen int

	inZP          int32
	outZP, lo, hi int64 // output zero point and activation clamp
}

// plan validates the sizes of p.
func (p FCParams) plan() (fcPlan, error) {
	if p.In <= 0 || p.Out <= 0 || p.In > MaxTaps {
		return fcPlan{}, ErrShape
	}
	wLen, ok := mul2(p.In, p.Out)
	if !ok {
		return fcPlan{}, ErrShape
	}
	return fcPlan{in: p.In, out: p.Out, wLen: wLen}, nil
}

// setQuant validates bias, requant and the activation range (in that order)
// and stores the quantisation constants in pl.
func (pl *fcPlan) setQuant(bias []int32, rq []Requant, p FCParams) error {
	if err := validBias(bias, pl.out); err != nil {
		return err
	}
	if err := validRequants(rq, pl.out); err != nil {
		return err
	}
	lo, hi, err := clampRange(p.ActMin, p.ActMax)
	if err != nil {
		return err
	}
	pl.inZP, pl.outZP, pl.lo, pl.hi = int32(p.InZP), int64(p.OutZP), lo, hi
	return nil
}

// fullyConnectedGeneric is the FullyConnected inner loop. All arguments are
// validated against pl.
func fullyConnectedGeneric(dst, src, weights []int8, bias []int32, rq []Requant, pl *fcPlan) {
	in, zp, zpOut, lo, hi := pl.in, pl.inZP, pl.outZP, pl.lo, pl.hi
	for o := range dst[:pl.out] {
		acc := dotZP(src, weights[o*in:(o+1)*in], zp)
		if len(bias) != 0 {
			acc += bias[o]
		}
		dst[o] = requantize(acc, rq[o], zpOut, lo, hi)
	}
}
