//go:build unitv

package main

import "math"

// TinyGo 0.43 lowers math.Exp to libm's exp, and its picolibc for the K210
// is built with the medlow code model: exp's absolute references to its
// tables (__exp_data, VAL) cannot reach RAM at 0x80000000, so ld.lld fails
// with "relocation R_RISCV_HI20 out of range". The SSD decoder calls
// math.Exp for every box, so this program defines exp itself and the linker
// never pulls libm's (as the yolo example does).
//
// The algorithm is Go's math.exp (FreeBSD's e_exp.c, error < 1 ulp; its
// notice: "Copyright (C) 2004 by Sun Microsystems, Inc. All rights
// reserved. Permission to use, copy, modify, and distribute this software
// is freely granted, provided that this notice is preserved."). The decoder
// rounds the result to float32.
//
//export exp
func exp(x float64) float64 {
	const (
		ln2Hi     = 6.93147180369123816490e-01
		ln2Lo     = 1.90821492927058770002e-10
		log2e     = 1.44269504088896338700e+00
		overflow  = 7.09782712893383973096e+02
		underflow = -7.45133219101941108420e+02
		nearZero  = 1.0 / (1 << 28)

		p1 = 1.66666666666666657415e-01
		p2 = -2.77777777770155933842e-03
		p3 = 6.61375632143793436117e-05
		p4 = -1.65339022054652515390e-06
		p5 = 4.13813679705723846039e-08
	)
	switch {
	case x != x: // NaN
		return x
	case x > overflow:
		return math.Inf(1)
	case x < underflow:
		return 0
	case -nearZero < x && x < nearZero:
		return 1 + x
	}
	var k int
	switch {
	case x < 0:
		k = int(log2e*x - 0.5)
	case x > 0:
		k = int(log2e*x + 0.5)
	}
	hi := x - float64(k)*ln2Hi
	lo := float64(k) * ln2Lo
	r := hi - lo
	t := r * r
	c := r - t*(p1+t*(p2+t*(p3+t*(p4+t*p5))))
	y := 1 - ((lo - (r*c)/(2-c)) - hi)
	return scale2(y, k)
}

// scale2 returns y·2^k for y in [0.5, 2] and k in [−1075, 1024] by
// multiplying with exact powers of two (libm's ldexp has the same medlow
// problem).
func scale2(y float64, k int) float64 {
	for k > 1023 {
		y *= 0x1p1023
		k -= 1023
	}
	for k < -1022 {
		y *= 0x1p-1022
		k += 1022
	}
	return y * math.Float64frombits(uint64(k+1023)<<52)
}
