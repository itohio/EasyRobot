package rigidbody

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/vec"
)

type samplePoint struct {
	position       vec.Vector3D
	tangent        vec.Vector3D
	orientation    vec.Quaternion
	linearHint     vec.Vector3D
	angularHint    vec.Vector3D
	hasOrientation bool
	hasLinearHint  bool
	hasAngularHint bool
}

type pathPlanner interface {
	Length() float32
	Signature() uint64
	Sample(progress float32) samplePoint
	Project(position vec.Vector3D) float32
	Curvature(progress float32) float32
}

func normalizeQuaternion(q vec.Quaternion) vec.Quaternion {
	mag := math32.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
	if mag <= epsilonDistance {
		return identityQuaternion()
	}
	inv := 1 / mag
	return vec.Quaternion{q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv}
}

func identityQuaternion() vec.Quaternion {
	return vec.Quaternion{1, 0, 0, 0}
}

func quaternionFromTangent(t vec.Vector3D) vec.Quaternion {
	if vectorMagnitude(t) <= epsilonDistance {
		return identityQuaternion()
	}
	yaw := math32.Atan2(t[1], t[0])
	half := yaw * 0.5
	return vec.Quaternion{math32.Cos(half), 0, 0, math32.Sin(half)}
}

func yawFromQuaternion(q vec.Quaternion) float32 {
	qw := q[0]
	qx := q[1]
	qy := q[2]
	qz := q[3]
	sinyCosp := 2 * (qw*qz + qx*qy)
	cosyCosp := 1 - 2*(qy*qy+qz*qz)
	return math32.Atan2(sinyCosp, cosyCosp)
}

func vectorMagnitude(v vec.Vector3D) float32 {
	return math32.Sqrt(dot(v, v))
}

func interpolateVec(a, b vec.Vector3D, t float32) vec.Vector3D {
	return vec.Vector3D{
		a[0] + (b[0]-a[0])*t,
		a[1] + (b[1]-a[1])*t,
		a[2] + (b[2]-a[2])*t,
	}
}

func unitTangent(a, b vec.Vector3D) vec.Vector3D {
	diff := subtract(b, a)
	length := vectorMagnitude(diff)
	if length <= epsilonDistance {
		return vec.Vector3D{1, 0, 0}
	}
	return vec.Vector3D{
		diff[0] / length,
		diff[1] / length,
		diff[2] / length,
	}
}

func distance(a, b vec.Vector3D) float32 {
	return vectorMagnitude(subtract(b, a))
}

func subtract(a, b vec.Vector3D) vec.Vector3D {
	return vec.Vector3D{
		a[0] - b[0],
		a[1] - b[1],
		a[2] - b[2],
	}
}

func dot(a, b vec.Vector3D) float32 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func clampIndex(idx, length int) int {
	if idx < 0 {
		return 0
	}
	if idx >= length {
		return length - 1
	}
	return idx
}

func computeLengths(points []vec.Vector3D) ([]float32, float32) {
	lengths := make([]float32, len(points))
	var total float32
	for i := 1; i < len(points); i++ {
		total += distance(points[i-1], points[i])
		lengths[i] = total
	}
	return lengths, total
}

func searchSegment(lengths []float32, progress float32) int {
	lo := 0
	hi := len(lengths) - 2
	for lo <= hi {
		mid := (lo + hi) / 2
		if progress < lengths[mid] {
			hi = mid - 1
			continue
		}
		if progress > lengths[mid+1] {
			lo = mid + 1
			continue
		}
		return mid
	}
	if lo < 0 {
		return 0
	}
	if lo > len(lengths)-2 {
		return len(lengths) - 2
	}
	return lo
}

func curvature2D(p0, p1, p2 vec.Vector3D) float32 {
	x1, y1 := p0[0], p0[1]
	x2, y2 := p1[0], p1[1]
	x3, y3 := p2[0], p2[1]

	den := (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
	if math32.Abs(den) <= epsilonDistance {
		return 0
	}

	a := distance(p0, p1)
	b := distance(p1, p2)
	c := distance(p2, p0)
	if a <= epsilonDistance || b <= epsilonDistance || c <= epsilonDistance {
		return 0
	}

	area2 := math32.Abs(den)
	return (2 * area2) / (a * b * c)
}

func clampProgress(progress, total float32) float32 {
	if progress < 0 {
		return 0
	}
	if progress > total {
		return total
	}
	return progress
}

func hashFloat32(h uint64, value float32) uint64 {
	return (h ^ uint64(math32.Float32bits(value))) * 1099511628211
}

func hashVector(h uint64, v vec.Vector3D) uint64 {
	h = hashFloat32(h, v[0])
	h = hashFloat32(h, v[1])
	h = hashFloat32(h, v[2])
	return h
}

func hashQuaternion(h uint64, q vec.Quaternion) uint64 {
	h = hashFloat32(h, q[0])
	h = hashFloat32(h, q[1])
	h = hashFloat32(h, q[2])
	h = hashFloat32(h, q[3])
	return h
}

func projectPoint(p, a, b vec.Vector3D) vec.Vector3D {
	ab := subtract(b, a)
	ap := subtract(p, a)
	len2 := dot(ab, ab)
	if len2 <= epsilonDistance {
		return a
	}
	t := clampFloat(dot(ap, ab)/len2, 0, 1)
	return vec.Vector3D{
		a[0] + (b[0]-a[0])*t,
		a[1] + (b[1]-a[1])*t,
		a[2] + (b[2]-a[2])*t,
	}
}

func validateWaypoints(points []vec.Vector3D) error {
	if len(points) < 2 {
		return ErrInvalidPath
	}
	return nil
}

func interpolateQuaternion(a, b vec.Quaternion, t float32) vec.Quaternion {
	return normalizeQuaternion(a.Slerp(b, t, 0).(vec.Quaternion))
}

func interpolateVector(a, b vec.Vector3D, t float32) vec.Vector3D {
	return vec.Vector3D{
		a[0] + (b[0]-a[0])*t,
		a[1] + (b[1]-a[1])*t,
		a[2] + (b[2]-a[2])*t,
	}
}
