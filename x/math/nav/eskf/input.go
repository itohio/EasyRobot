package eskf

import "github.com/chewxy/math32"

// inputs holds what makes a sample usable: the accelerometer's clip limits
// in the samples' units (the sensor's ±0.999·AccelRange through the fit the
// samples arrive by, Prior) and the gyro's.
type inputs struct {
	off, scale [3]float32 // the fit the samples are corrected by: its offset (zero without one) and scale (1 without one)
	accOn      bool       // the accelerometer's range is checked
	accLo      [3]float32
	accHi      [3]float32
	gyroLim    float32 // 0.999·GyroRange; 0: no check
}

// fit takes a calibration's offset and scale: a non-finite offset counts
// as 0 (through it nothing would clip, or everything), a scale that isn't
// finite and positive as 1 (no scale fitted: a gyro-only or offset-only
// fit).
func (in *inputs) fit(off, scale [3]float32) {
	for i := range 3 {
		s := scale[i]
		if !(s > 0) || math32.IsInf(s, 0) {
			s = 1
		}
		o := off[i]
		if !(o-o == 0) {
			o = 0
		}
		in.off[i], in.scale[i] = o, s
	}
}

// ranges sets the clip limits from the sensor's full scales (0: no check).
func (in *inputs) ranges(accel, gyro float32) {
	r := 0.999 * accel
	for i := range 3 {
		in.accLo[i] = (-r - in.off[i]) / in.scale[i]
		in.accHi[i] = (r - in.off[i]) / in.scale[i]
	}
	in.accOn = accel > 0
	in.gyroLim = 0.999 * gyro
}

// clipped: whether the accelerometer's and the gyro's readings are at a
// full scale (the sensor's counts saturate there, +range·32767/32768 and
// −range), their true values unknown.
func (in *inputs) clipped(acc, gyro [3]float32) (ac, gc bool) {
	g := in.gyroLim
	for i := range 3 {
		ac = ac || in.accOn && (acc[i] >= in.accHi[i] || acc[i] <= in.accLo[i])
		gc = gc || g > 0 && (gyro[i] >= g || gyro[i] <= -g)
	}
	return ac, gc
}

// finiteSample: dt above zero and finite, no NaN or infinity in the
// readings.
func finiteSample(dt float32, acc, gyro [3]float32) bool {
	return dt > 0 && !math32.IsInf(dt, 0) && allFinite(acc) && allFinite(gyro)
}

func allFinite(v [3]float32) bool {
	for _, x := range v {
		if math32.IsNaN(x) || math32.IsInf(x, 0) {
			return false
		}
	}
	return true
}
