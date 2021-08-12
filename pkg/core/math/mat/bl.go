package mat

func multo(r1, c1, r2, c2 int, m1, m2, dst []float32) {
	for i := 0; i < r1; i++ {
		for j := 0; j < r2; j++ {
			var acc float32
			for k := 0; k < c1; k++ {
				acc += m1[i*c1+k] * m2[k*c2+j]
			}
			dst[i*r2+j] = acc
		}
	}
}

func muldiagto(r1, c1 int, v, m1, dst []float32) {
	for i := 0; i < r1; i++ {
		for j := 0; j < c1; j++ {
			dst[i*c1+j] = m1[i*c1+j] * v[j]
		}
	}
}

func mulvto(m1, v, dst []float32) {
	src := 0
	for i := range dst {
		var acc float32
		for _, pv := range v {
			acc += pv * m1[src]
			src++
		}
		dst[i] = acc
	}
}

func mulvtto(m1, v, dst []float32) {
	N := len(dst)
	for i := range dst {
		var acc float32
		src := i
		for _, pv := range v {
			acc += pv * m1[src]
			src += N
		}
		dst[i] = acc
	}
}
