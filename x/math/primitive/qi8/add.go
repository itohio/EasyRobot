package qi8

import "math"

// AddRequant holds the two real multipliers of a requantised addition with a
// shared shift: realA = MultA·2^(Shift−31) (= scaleA/scaleOut) and likewise
// realB. Both Mults are in [1, 2^31), the larger in [2^30, 2^31), and Shift
// is in [MinShift, MaxShift] (total right shift 31−Shift in [1, 62]).
type AddRequant struct {
	MultA, MultB int32
	Shift        int32
}

// Valid reports whether r is inside the supported range.
func (r AddRequant) Valid() bool {
	return r.MultA >= 1 && r.MultB >= 1 && max(r.MultA, r.MultB) >= multMin &&
		r.Shift >= MinShift && r.Shift <= MaxShift
}

// QuantizeAddMultipliers converts realA = scaleA/scaleOut and
// realB = scaleB/scaleOut into an AddRequant. The larger multiplier is
// quantised like QuantizeMultiplier (frac·2^exp, Mult = round(frac·2^31),
// renormalised on rounding up to 2^31, Shift = exp); the smaller one is
// round(real·2^(31−Shift)) at the same shift (half away from zero, exact in
// float64). Returns ErrRequant for non-finite or non-positive values, a larger
// multiplier outside [2^-32, 2^30), or a smaller one that rounds to 0.
func QuantizeAddMultipliers(realA, realB float64) (AddRequant, error) {
	if !(realA > 0) || !(realB > 0) || math.IsInf(realA, 0) || math.IsInf(realB, 0) {
		return AddRequant{}, ErrRequant
	}
	r, err := QuantizeMultiplier(max(realA, realB))
	if err != nil {
		return AddRequant{}, err
	}
	small := math.Round(math.Ldexp(min(realA, realB), int(31-r.Shift)))
	if small < 1 {
		return AddRequant{}, ErrRequant
	}
	out := AddRequant{MultA: r.Mult, MultB: int32(small), Shift: r.Shift}
	if realA < realB {
		out.MultA, out.MultB = out.MultB, out.MultA
	}
	return out, nil
}

// AddParams describes an element-wise requantised addition of two int8
// tensors with their own zero points:
//
//	dst[i] = clamp(ZPOut + round((a[i]−ZPA)·realA + (b[i]−ZPB)·realB), Min, Max)
//
// with realA, realB from RQ. The sum is evaluated exactly in int64
// ((a−ZPA)·MultA + (b−ZPB)·MultB, |·| < 2^40) and rounded once, half away
// from zero, like Requant.Apply. Min/Max is the fused activation clamp.
type AddParams struct {
	ZPA, ZPB, ZPOut int8
	RQ              AddRequant
	Min, Max        int8
}

func (p *AddParams) valid() error {
	if !p.RQ.Valid() {
		return ErrRequant
	}
	if p.Min > p.Max {
		return ErrActivation
	}
	return nil
}

// Add computes the requantised element-wise sum (see AddParams). dst, a and
// b must have equal lengths (0 allowed); dst may alias a or b.
func Add(dst, a, b []int8, p AddParams) error {
	if len(a) != len(dst) || len(b) != len(dst) {
		return ErrLength
	}
	if err := p.valid(); err != nil {
		return err
	}
	addGeneric(dst, a, b, &p)
	return nil
}

// addGeneric is the Add inner loop; equal lengths, p valid.
//
// It evaluates the int64 definition exactly in 32-bit arithmetic (on the
// ESP32-S3 the direct int64 loop costs 2.4× as many cycles). With
// Mult = Mh·2^16 + Ml (Mh < 2^15, Ml < 2^16) and dA = a−ZPA, dB = b−ZPB
// (|d| ≤ 255):
//
//	acc = H·2^16 + L,  H = dA·MhA + dB·MhB (|H| < 2^24),  L = dA·MlA + dB·MlB (|L| < 2^25)
//
// Rounding half away from zero is floor((acc + 2^(s−1) − [acc<0]) / 2^s).
// For s ≥ 17 the half is a multiple of 2^16, so with
// Hn = H + 2^(s−17) + floor((L − [acc<0]) / 2^16) the result is
// floor(Hn / 2^(s−16)) (the remaining low part lies in [0, 2^16)); the sign
// of acc is the sign of H + floor(L / 2^16). |acc| < 2^40, so every s ≥ 41
// rounds to 0 and s is capped at 41 (|Hn| < 2^26). Total shifts s ≤ 16
// (multipliers ≥ 2^14) use the int64 loop.
func addGeneric(dst, a, b []int8, p *AddParams) {
	s := 31 - p.RQ.Shift // 1..62
	if s <= 16 {
		addGeneric64(dst, a, b, p)
		return
	}
	s = min(s, 41)
	n := len(dst)
	a, b = a[:n], b[:n]
	mhA, mlA := p.RQ.MultA>>16, p.RQ.MultA&0xFFFF
	mhB, mlB := p.RQ.MultB>>16, p.RQ.MultB&0xFFFF
	za, zb := int32(p.ZPA), int32(p.ZPB)
	halfH := int32(1) << uint(s-17)
	sh := uint(s - 16)
	zo, lo, hi := int32(p.ZPOut), int32(p.Min), int32(p.Max)
	for i := range dst {
		da, db := int32(a[i])-za, int32(b[i])-zb
		h := da*mhA + db*mhB
		l := da*mlA + db*mlB
		neg := int32(uint32(h+l>>16) >> 31) // acc < 0
		v := (h+halfH+(l-neg)>>16)>>sh + zo
		if v < lo {
			v = lo
		}
		if v > hi {
			v = hi
		}
		dst[i] = int8(v)
	}
}

// addGeneric64 is the direct int64 evaluation (any shift).
func addGeneric64(dst, a, b []int8, p *AddParams) {
	n := len(dst)
	a, b = a[:n], b[:n]
	ma, mb := int64(p.RQ.MultA), int64(p.RQ.MultB)
	za, zb := int64(p.ZPA), int64(p.ZPB)
	s := uint(31 - p.RQ.Shift) // 1..62
	half := int64(1) << (s - 1)
	zo, lo, hi := int64(p.ZPOut), int64(p.Min), int64(p.Max)
	for i := range dst {
		acc := (int64(a[i])-za)*ma + (int64(b[i])-zb)*mb
		var v int64
		if acc >= 0 {
			v = (acc + half) >> s
		} else {
			v = -((-acc + half) >> s)
		}
		v += zo
		if v < lo {
			v = lo
		}
		if v > hi {
			v = hi
		}
		dst[i] = int8(v)
	}
}
