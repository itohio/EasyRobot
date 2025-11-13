package mecanum

import (
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

type drive struct {
	wR  float32
	wBX float32
	wBY float32

	params [4]float32
	pos    [3]float32

	dimensions   kintypes.Dimensions
	capabilities kintypes.Capabilities
}

var _ kintypes.Bidirectional = (*drive)(nil)

// New returns a kinematics model for a mecanum-drive platform.
// wheelRadius is the radius of each wheel, while baseX and baseY represent
// the chassis dimensions between wheel centers along the X and Y axes.
func New(wheelRadius, baseX, baseY float32) *drive {
	return &drive{
		wR:  wheelRadius,
		wBX: baseX,
		wBY: baseY,
		dimensions: kintypes.Dimensions{
			StateRows:    4,
			StateCols:    1,
			ControlSize:  4,
			ActuatorSize: 4,
		},
		capabilities: kintypes.Capabilities{
			Holonomic:        true,
			Omnidirectional:  true,
			SupportsLateral:  true,
			SupportsVertical: false,
			ConstraintRank:   3,
		},
	}
}

func (d *drive) Dimensions() kintypes.Dimensions {
	return d.dimensions
}

func (d *drive) Capabilities() kintypes.Capabilities {
	return d.capabilities
}

func (*drive) ConstraintSet() kintypes.Constraints {
	return kintypes.Constraints{}
}

// Forward interprets the 4×1 `state` column as wheel angular rates
// `[ω_fl, ω_fr, ω_rl, ω_rr]` and writes the resulting chassis twist
// `[vx, vy, ω]` into `destination`.
func (d *drive) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := ensureColumn(state, len(d.params)); err != nil {
		return err
	}
	if err := ensureColumn(destination, len(d.pos)); err != nil {
		return err
	}

	stateView := state.View().(mat.Matrix)
	for i := range d.params {
		d.params[i] = stateView[i][0]
	}

	transform := transform3x4(d.wR, d.wBX, d.wBY)
	res := transform.MulVec(vec.Vector4D(d.params), nil).(vec.Vector3D)
	copy(d.pos[:], res[:])

	destView := destination.View().(mat.Matrix)
	for i := range d.pos {
		destView[i][0] = d.pos[i]
	}
	return nil
}

// Backward consumes a desired chassis twist from `destination` (`[vx, vy, ω]`)
// and writes wheel rates into the `controls` column vector `[ω_fl, ω_fr, ω_rl, ω_rr]`.
// Optional `state` input pre-populates wheel values before solving.
func (d *drive) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := ensureColumn(destination, len(d.pos)); err != nil {
		return err
	}
	if err := ensureColumn(controls, len(d.params)); err != nil {
		return err
	}
	if state != nil {
		if err := ensureColumn(state, len(d.params)); err != nil {
			return err
		}
		stateView := state.View().(mat.Matrix)
		for i := range d.params {
			d.params[i] = stateView[i][0]
		}
	}

	destView := destination.View().(mat.Matrix)
	for i := range d.pos {
		d.pos[i] = destView[i][0]
	}

	transform := transform4x3(d.wR, d.wBX, d.wBY)
	res := transform.MulVec(vec.Vector3D(d.pos), nil).(vec.Vector4D)
	copy(d.params[:], res[:])

	controlView := controls.View().(mat.Matrix)
	for i := range d.params {
		controlView[i][0] = d.params[i]
	}
	return nil
}

func transform3x4(wR, baseX, baseY float32) mat.Matrix3x4 {
	c := 2 / (baseX + baseY)
	m := mat.New3x4(
		1, 1, 1, 1,
		1, -1, -1, 1,
		-c, c, -c, c,
	)
	return m.MulC(wR / 4).(mat.Matrix3x4)
}

func transform4x3(wR, baseX, baseY float32) mat.Matrix4x3 {
	c := (baseX + baseY) * 0.5
	m := mat.New4x3(
		1, 1, -c,
		1, -1, c,
		1, -1, -c,
		1, 1, c,
	)
	return m.DivC(wR).(mat.Matrix4x3)
}

type vec3 [3]float32

type vec4 [4]float32

func vec3FromArray(arr [3]float32) vec3 {
	return vec3{arr[0], arr[1], arr[2]}
}

func vec4FromArray(arr [4]float32) vec4 {
	return vec4{arr[0], arr[1], arr[2], arr[3]}
}

func ensureColumn(m mattype.Matrix, rows int) error {
	if m == nil {
		return kintypes.ErrInvalidDimensions
	}
	if m.Rows() != rows || m.Cols() < 1 {
		return kintypes.ErrInvalidDimensions
	}
	return nil
}
