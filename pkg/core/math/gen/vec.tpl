// Generated code. DO NOT EDIT

package vec

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math"
)

type {{.name}} [{{.size}}]{{.type}}

{{if .size}}
{{else}}
func New(size int) {{.name}} {
	return make({{.name}}, size)
}

func NewFrom(v ...float32) {{.name}} {
	return v[:]
}
{{end}}

func (v {{.name_class}}) Sum() {{.type}} {
	var sum {{.type}}
	for _, val := range v {
		sum += val
	}
	return sum
}

{{if .size }}
func (v {{.name_class}}) Vector() Vector {
	return v[:]
}
{{end}}

func (v {{.name_class}}) Slice(start, end int) Vector {
	if end < 0 {
		end = len(v)
	}
	return v[start:end]
}

{{if .xy}}
func (v {{.name_class}}) XY() ({{.type}}, {{.type}}) {
	return v[0], v[1]
}
{{end}}
{{if .xyz}}
func (v {{.name_class}}) XYZ() ({{.type}}, {{.type}}, {{.type}}) {
	return v[0], v[1], v[2]
}
{{end}}
{{if .xyzw}}
func (v {{.name_class}}) XYZW() ({{.type}}, {{.type}}, {{.type}}, {{.type}}) {
	return v[0], v[1], v[2], v[3]
}
{{end}}

func (v {{.name_class}}) SumSqr() {{.type}} {
	var sum {{.type}}
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v {{.name_class}}) Magnitude() {{.type}} {
	return math32.Sqrt(v.SumSqr())
}

func (v {{.name_class}}) DistanceSqr(v1 {{.name}}) {{.type}} {
	return v.Clone().Sub(v1).SumSqr()
}

func (v {{.name_class}}) Distance(v1 {{.name}}) {{.type}} {
	return math32.Sqrt(v.DistanceSqr(v1))
}

{{ if .size }}
func (v {{.name_class}}) Clone() {{.name_ret}} {
	clone := {{.name}}{}
	copy(clone[:], v[:])
	return &clone
}
{{else}}
func (v {{.name_class}}) Clone() {{.name_ret}} {
	if v == nil {
		return nil
	}

	clone := make({{.name}}, len(v))
	copy(clone, v)
	return clone
}
{{end}}

func (v {{.name_class}}) CopyFrom(start int, v1 Vector) {{.name_ret}} {
	copy(v[start:], v1)
	return v
}

func (v {{.name_class}}) CopyTo(start int, v1 Vector) Vector {
	copy(v1, v[start:])
	return v1
}

func (v {{.name_class}}) Clamp(min, max {{.name}}) {{.name_ret}} {
	for i := range v {
		v[i] = math.Clamp(v[i], min[i], max[i])
	}
	return v
}

func (v {{.name_class}}) FillC(c {{.type}}) {{.name_ret}} {
	for i := range v {
		v[i] = c
	}
	return v
}

func (v {{.name_class}}) Neg() {{.name_ret}} {
	for i := range v {
		v[i] = -v[i]
	}
	return v
}

func (v {{.name_class}}) Add(v1 {{.name}}) {{.name_ret}} {
	for i := range v {
		v[i] += v1[i]
	}
	return v
}

func (v {{.name_class}}) AddC(c {{.type}}) {{.name_ret}} {
	for i := range v {
		v[i] += c
	}
	return v
}

func (v {{.name_class}}) Sub(v1 {{.name}}) {{.name_ret}} {
	for i := range v {
		v[i] -= v1[i]
	}
	return v
}

func (v {{.name_class}}) SubC(c {{.type}}) {{.name_ret}} {
	for i := range v {
		v[i] -= c
	}
	return v
}

func (v {{.name_class}}) MulC(c {{.type}}) {{.name_ret}} {
	for i := range v {
		v[i] *= c
	}
	return v
}

func (v {{.name_class}}) MulCAdd(c {{.type}}, v1 {{.name}}) {{.name_ret}} {
	for i := range v {
		v[i] += v1[i] * c
	}
	return v
}

func (v {{.name_class}}) MulCSub(c {{.type}}, v1 {{.name}}) {{.name_ret}} {
	for i := range v {
		v[i] -= v1[i] * c
	}
	return v
}

func (v {{.name_class}}) DivC(c {{.type}}) {{.name_ret}} {
	for i := range v {
		v[i] /= c
	}
	return v
}

func (v {{.name_class}}) DivCAdd(c {{.type}}, v1 {{.name}}) {{.name_ret}} {
	for i := range v {
		v[i] += v1[i] / c
	}
	return v
}

func (v {{.name_class}}) DivCSub(c {{.type}}, v1 {{.name}}) {{.name_ret}} {
	for i := range v {
		v[i] -= v1[i] / c
	}
	return v
}

func (v {{.name_class}}) Normal() {{.name_ret}} {
	d := v.Magnitude()
	return v.DivC(d)
}

func (v {{.name_class}}) NormalFast() {{.name_ret}} {
	d := v.SumSqr()
	return v.MulC(math.FastISqrt(d))
}

{{if .quaternion}}
func (v {{.name_class}}) Axis() Vector {
	return v[:3]
}

func (v {{.name_class}}) Theta() float32 {
	return v[3]
}

func (v {{.name_class}}) Conjugate() {{.name_ret}} {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	return v
}

func (v {{.name_class}}) Roll() float32 {
	return math32.Atan2(v[3]*v[0]+v[1]*v[2], 0.5-v[0]*v[0]-v[1]*v[1])
}
func (v {{.name_class}}) Pitch() float32 {
	return math32.Asin(-2.0 * (v[0]*v[2] - v[3]*v[1]))
}
func (v {{.name_class}}) Yaw() float32 {
	return math32.Atan2(v[0]*v[1]+v[3]*v[2], 0.5-v[1]*v[1]-v[2]*v[2])
}

func (a {{.name_class}}) Product(b Quaternion) {{.name_ret}} {
	x := a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1]
	y := a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0]
	z := a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3]
	w := a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2]
	a[0] = x
	a[1] = y
	a[2] = z
	a[3] = w
	return a
}

func (v {{.name_class}}) Slerp(v1 {{.name}}, time, spin {{.type}}) {{.name_ret}} {
	const SLERP_EPSILON = 1.0e-10
	var (
		k1, k2       {{.type}} // interpolation coefficions.
		angle        {{.type}} // angle between A and B
		angleSpin    {{.type}} // angle between A and B plus spin.
		sin_a, cos_a {{.type}} // sine, cosine of angle
	)

	flipk2 := 0
	cos_a = v.Dot(v1)
	if cos_a < 0.0 {
		cos_a = -cos_a
		flipk2 = -1
	} else {
		flipk2 = 1
	}

	if (1.0 - cos_a) < SLERP_EPSILON {
		k1 = 1.0 - time
		k2 = time
	} else { /* normal case */
		angle = math32.Acos(cos_a)
		sin_a = math32.Sin(angle)
		angleSpin = angle + spin*math32.Pi
		k1 = math32.Sin(angle-time*angleSpin) / sin_a
		k2 = math32.Sin(time*angleSpin) / sin_a
	}
	k2 *= {{.type}}(flipk2)

	v[0] = k1*v[0] + k2*v1[0]
	v[1] = k1*v[1] + k2*v1[1]
	v[2] = k1*v[2] + k2*v1[2]
	v[3] = k1*v[3] + k2*v1[3]
	return v
}

func (v {{.name_class}}) SlerpLong(v1 {{.name}}, time, spin {{.type}}) {{.name_ret}} {
	const SLERP_EPSILON = 1.0e-10
	var (
		k1, k2       {{.type}} // interpolation coefficions.
		angle        {{.type}} // angle between A and B
		angleSpin    {{.type}} // angle between A and B plus spin.
		sin_a, cos_a {{.type}} // sine, cosine of angle
	)

	cos_a = v.Dot(v1)

	if 1.0-math32.Abs(cos_a) < SLERP_EPSILON {
		k1 = 1.0 - time
		k2 = time
	} else { /* normal case */
		angle = math32.Acos(cos_a)
		sin_a = math32.Sin(angle)
		angleSpin = angle + spin*math32.Pi
		k1 = math32.Sin(angle-time*angleSpin) / sin_a
		k2 = math32.Sin(time*angleSpin) / sin_a
	}

	v[0] = k1*v[0] + k2*v1[0]
	v[1] = k1*v[1] + k2*v1[1]
	v[2] = k1*v[2] + k2*v1[2]
	v[3] = k1*v[3] + k2*v1[3]
	return v
}
{{end}}

func (v {{.name_class}}) Multiply(v1 {{.name}}) {{.name_ret}} {
	for i := range v {
		v[i] *= v1[i]
	}
	return v
}

func (v {{.name_class}}) Dot(v1 {{.name}}) {{.type}} {
	var sum {{.type}}
	for i := range v {
		sum += v[i] * v1[i]
	}
	return sum
}

{{if .cross}}
func (v {{.name_class}}) Cross(v1 {{.name}}) {{.name_ret}} {
	t := []{{.type}}{v[0], v[1], v[2]}
	v[0] = t[1]*v1[2] - t[2]*v1[1]
	v[1] = t[2]*v1[0] - t[0]*v1[2]
	v[2] = t[0]*v1[1] - t[1]*v1[0]
	return v
}
{{end}}

{{if .refract2d}}
func (v {{.name_class}}) {{.refract2d}}(n {{.name}}, ni, nt {{.type}}) ({{.name_ret}}, bool) {
	var (
		cos_V  {{.name}}
		sin_T  {{.name}}
		n_mult {{.type}}
	)
{{if .size}}
	N_dot_V := n.Dot(*v)
{{else}}
	N_dot_V := n.Dot(v)
{{end}}
	if N_dot_V > 0.0 {
		n_mult = ni / nt
	} else {
		n_mult = nt / ni
	}

	cos_V[0] = n[0] * N_dot_V
	cos_V[1] = n[1] * N_dot_V
	sin_T[0] = (cos_V[0] - v[0]) * (n_mult)
	sin_T[1] = (cos_V[1] - v[1]) * (n_mult)
	len_sin_T := sin_T.Dot(sin_T)
	if len_sin_T >= 1.0 {
		return v, false // internal reflection
	}
	N_dot_T := math32.Sqrt(1.0 - len_sin_T)
	if N_dot_V < 0.0 {
		N_dot_T = -N_dot_T
	}
	v[0] = sin_T[0] - n[0]*N_dot_T
	v[1] = sin_T[1] - n[1]*N_dot_T

	return v, true
}
{{end}}

{{if .refract3d}}
func (v {{.name_class}}) {{.refract3d}}(n {{.name}}, ni, nt {{.type}}) ({{.name_ret}}, bool) {
	var (
		sin_T  {{.name}} /* sin vect of the refracted vect */
		cos_V  {{.name}} /* cos vect of the incident vect */
		n_mult {{.type}}  /* ni over nt */
	)

{{if .size}}
	N_dot_V := n.Dot(*v)
{{else}}
	N_dot_V := n.Dot(v)
{{end}}
	if N_dot_V > 0.0 {
		n_mult = ni / nt
	} else {
		n_mult = nt / ni
	}
	cos_V[0] = n[0] * N_dot_V
	cos_V[1] = n[1] * N_dot_V
	cos_V[2] = n[2] * N_dot_V
	sin_T[0] = (cos_V[0] - v[0]) * (n_mult)
	sin_T[1] = (cos_V[1] - v[1]) * (n_mult)
	sin_T[2] = (cos_V[2] - v[2]) * (n_mult)
	len_sin_T := sin_T.Dot(sin_T)
	if len_sin_T >= 1.0 {
		return v, false // internal reflection
	}
	N_dot_T := math32.Sqrt(1.0 - len_sin_T)
	if N_dot_V < 0.0 {
		N_dot_T = -N_dot_T
	}
	v[0] = sin_T[0] - n[0]*N_dot_T
	v[1] = sin_T[1] - n[1]*N_dot_T
	v[2] = sin_T[2] - n[2]*N_dot_T

	return v, true
}
{{end}}

func (v {{.name_class}}) Reflect(n {{.name}}) {{.name_ret}} {
{{if .size}}
	N_dot_V := n.Dot(*v) * 2
{{else}}
	N_dot_V := n.Dot(v) * 2
{{end}}
	return v.Neg().MulCAdd(N_dot_V, n)
}

func (v {{.name_class}}) Interpolate(v1 {{.name}}, t {{.type}}) {{.name_ret}} {
{{if .size}}
	d := v1.Clone().Sub(*v)
	return v.MulCAdd(t, *d)
{{else}}
	d := v1.Clone().Sub(v)
	return v.MulCAdd(t, d)
{{end}}
}
