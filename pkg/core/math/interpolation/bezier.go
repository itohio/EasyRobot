package interpolation

func Bezier1d(p1, p2, p3, p4, t float32) float32 {
	tt := t * t
	ttt := tt * t
	t1 := 1 - t
	tt1 := t1 * t1
	ttt1 := tt1 * t1

	return ttt1*p1 + 3*(t*tt1*p2+tt*t1*p3) + ttt*p4
}

func Bezier2d(p1, p2, p3, p4 [2]float32, t float32) [2]float32 {
	tt := t * t
	ttt := tt * t
	t1 := 1 - t
	tt1 := t1 * t1
	ttt1 := tt1 * t1
	txtt1 := t * tt1
	ttxt1 := tt * t1

	return [2]float32{
		ttt1*p1[0] + 3*(txtt1*p2[0]+ttxt1*p3[0]) + ttt*p4[0],
		ttt1*p1[1] + 3*(txtt1*p2[1]+ttxt1*p3[1]) + ttt*p4[1],
	}
}

func Bezier3d(p1, p2, p3, p4 [3]float32, t float32) [3]float32 {
	tt := t * t
	ttt := tt * t
	t1 := 1 - t
	tt1 := t1 * t1
	ttt1 := tt1 * t1
	txtt1 := t * tt1
	ttxt1 := tt * t1

	return [3]float32{
		ttt1*p1[0] + 3*(txtt1*p2[0]+ttxt1*p3[0]) + ttt*p4[0],
		ttt1*p1[1] + 3*(txtt1*p2[1]+ttxt1*p3[1]) + ttt*p4[1],
		ttt1*p1[2] + 3*(txtt1*p2[2]+ttxt1*p3[2]) + ttt*p4[2],
	}
}

func Bezier(p1, p2, p3, p4 []float32, t float32) []float32 {
	if len(p1) != len(p2) || len(p1) != len(p3) || len(p1) != len(p4) {
		return nil
	}

	tt := t * t
	ttt := tt * t
	t1 := 1 - t
	tt1 := t1 * t1
	ttt1 := tt1 * t1
	txtt1 := t * tt1
	ttxt1 := tt * t1

	r := make([]float32, len(p1))
	for i := range p1 {
		r[i] = ttt1*p1[i] + 3*(txtt1*p2[i]+ttxt1*p3[i]) + ttt*p4[i]
	}

	return r
}
