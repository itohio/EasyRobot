package walk

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/control/kinematics/joints/planar"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// legReach measures how far a hip-frame foot position is inside the workspace
// of a knee-down coxa–femur–tibia leg (the first candidate planar.New3DOF
// tries: coxa towards the foot, tibia ≤ 0).
//
// The workspace is the annulus sector rMin ≤ r ≤ rMax around the femur pivot
// (r from the tibia limits within [−π, 0]) with the coxa yaw and the femur
// pitch inside their limits. The margin is the smallest slack in metres:
// r − rMin, rMax − r, and the yaw and femur angular slacks times their lever
// arms (horizontal foot radius ρ and r). margin > 0 means planar.New3DOF
// solves the point exactly and in limits.
type legReach struct {
	l0, l1, l2 float32
	lim        [3][2]float32
	rMin, rMax float32
}

func newLegReach(c [3]planar.Config) (legReach, error) {
	lo, hi := math32.Max(c[2].Min, -math32.Pi), math32.Min(c[2].Max, 0)
	if !(lo < hi) {
		return legReach{}, fmt.Errorf("%w: walk: the tibia range [%g, %g] has no knee-down pose", kintypes.ErrInvalidConfig, c[2].Min, c[2].Max)
	}
	r := legReach{l0: c[0].Length, l1: c[1].Length, l2: c[2].Length}
	for j := range 3 {
		r.lim[j] = [2]float32{c[j].Min, c[j].Max}
	}
	r.rMin, r.rMax = r.knee(lo), r.knee(hi) // r decreases with |a2|
	return r, nil
}

// knee is the femur-pivot-to-foot distance for tibia angle a2.
func (r legReach) knee(a2 float32) float32 {
	return math32.Sqrt(r.l1*r.l1 + r.l2*r.l2 + 2*r.l1*r.l2*math32.Cos(a2))
}

func (r legReach) margin(p vec.Vector3D) float32 {
	rho := math32.Hypot(p[0], p[1])
	yaw := math32.Atan2(p[1], p[0])
	xp, z := rho-r.l0, p[2]
	d := math32.Hypot(xp, z)
	m := math32.Min(d-r.rMin, r.rMax-d)
	m = math32.Min(m, math32.Min(yaw-r.lim[0][0], r.lim[0][1]-yaw)*rho)
	c := (d*d - r.l1*r.l1 - r.l2*r.l2) / (2 * r.l1 * r.l2)
	a2 := -math32.Acos(math32.Max(-1, math32.Min(1, c)))
	a1 := math32.Atan2(z, xp) - math32.Atan2(r.l2*math32.Sin(a2), r.l1+r.l2*math32.Cos(a2))
	return math32.Min(m, math32.Min(a1-r.lim[1][0], r.lim[1][1]-a1)*d)
}
