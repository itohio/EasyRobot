package dh

import (
	"errors"

	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

const effectorSize = 7

var (
	_ kintypes.Bidirectional = (*DenavitHartenberg)(nil)

	// ErrNoConvergence indicates the iterative inverse kinematics solver failed to
	// converge within the configured iteration/tolerance budget.
	ErrNoConvergence = errors.New("dh: inverse kinematics did not converge")
)

type DenavitHartenberg struct {
	c             []Config
	eps           float32
	maxIterations int
	params        []float32
	pos           [effectorSize]float32
	H0i           []mat.Matrix4x4
	jointTypes    []int
	constraints   kintypes.Constraints
	dimensions    kintypes.Dimensions
	capabilities  kintypes.Capabilities
}

func New(eps float32, maxIterations int, cfg ...Config) *DenavitHartenberg {
	dof := len(cfg)
	jointTypes := make([]int, dof)
	for i := range cfg {
		jointTypes[i] = cfg[i].Index
	}

	return &DenavitHartenberg{
		eps:           eps,
		maxIterations: maxIterations,
		c:             cfg,
		params:        make([]float32, dof),
		H0i:           make([]mat.Matrix4x4, dof+1),
		jointTypes:    jointTypes,
		dimensions: kintypes.Dimensions{
			StateRows:    dof,
			StateCols:    1,
			ControlSize:  dof,
			ActuatorSize: dof,
		},
		capabilities: kintypes.Capabilities{
			Holonomic:      true,
			Underactuated:  false,
			ConstraintRank: dof,
		},
	}
}

func (p *DenavitHartenberg) Dimensions() kintypes.Dimensions {
	return p.dimensions
}

func (p *DenavitHartenberg) Capabilities() kintypes.Capabilities {
	return p.capabilities
}

func (p *DenavitHartenberg) ConstraintSet() kintypes.Constraints {
	return p.constraints
}

// Forward interprets `state` as a DOF×1 joint parameter column vector and
// writes the end-effector pose `[x, y, z, qx, qy, qz, qw]` into `destination`.
func (p *DenavitHartenberg) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := p.loadState(state); err != nil {
		return err
	}
	if err := ensureColumn(destination, effectorSize); err != nil {
		return err
	}

	if err := p.forwardInternal(); err != nil {
		return err
	}

	view := destination.View().(mat.Matrix)
	for i := 0; i < effectorSize; i++ {
		view[i][0] = p.pos[i]
	}
	return nil
}

// Backward consumes a desired pose `[x, y, z, qx, qy, qz, qw]` from
// `destination` and writes solved joint parameters into the `controls`
// column vector. Optional `state` seeds the current joint values.
func (p *DenavitHartenberg) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := p.loadState(state); err != nil {
		return err
	}
	if err := p.loadEffector(destination); err != nil {
		return err
	}
	if err := ensureColumn(controls, len(p.params)); err != nil {
		return err
	}

	if err := p.inverseInternal(); err != nil {
		return err
	}

	view := controls.View().(mat.Matrix)
	for i, val := range p.params {
		view[i][0] = val
	}
	return nil
}

func (p *DenavitHartenberg) forwardInternal() error {
	identity := mat.Matrix4x4{}.Eye().(mat.Matrix4x4)
	H := identity
	p.H0i[0] = identity
	for i, cfg := range p.c {
		p.params[i] = cfg.Limit(p.params[i])
		if !cfg.CalculateTransform(p.params[i], &H) {
			return kintypes.ErrUnsupportedOperation
		}
		p.H0i[i+1] = (mat.Matrix4x4{}).Mul(p.H0i[i], H).(mat.Matrix4x4)
	}

	var posVec vec.Vector3D
	posVec = p.H0i[len(p.c)].Col3D(3, posVec)
	p.pos[0] = posVec[0]
	p.pos[1] = posVec[1]
	p.pos[2] = posVec[2]

	quat := p.H0i[len(p.c)].View().Quaternion()
	copy(p.pos[3:], quat.View().(vec.Vector))

	return nil
}

func (p *DenavitHartenberg) inverseInternal() error {
	eps2 := p.eps * p.eps
	target := vec.Vector3D{p.pos[0], p.pos[1], p.pos[2]}
	var actual vec.Vector3D
	var errVec vec.Vector3D

	for iter := 0; iter <= p.maxIterations; iter++ {
		if err := p.forwardInternal(); err != nil {
			return err
		}

		actual[0] = p.pos[0]
		actual[1] = p.pos[1]
		actual[2] = p.pos[2]

		errVec[0] = target[0] - actual[0]
		errVec[1] = target[1] - actual[1]
		errVec[2] = target[2] - actual[2]

		if errVec.SumSqr() < eps2 {
			return nil
		}

		if iter == p.maxIterations {
			return ErrNoConvergence
		}

		if err := p.ikSolverJacobianPos(errVec); err != nil {
			return err
		}
	}

	return ErrNoConvergence
}

// ikSolverJacobianPos implements Jacobian-based IK solver for position only.
// Equivalent to ik_solver_jacobian_pos in the C++ reference implementation.
func (p *DenavitHartenberg) ikSolverJacobianPos(v vec.Vector3D) error {
	dof := len(p.c)
	J := mat.New(3, dof)
	Jinv := mat.New(dof, 3)

	var dn vec.Vector3D
	dn = p.H0i[dof].Col3D(3, dn)

	var R, di, d vec.Vector3D
	R = vec.Vector3D{0, 0, 1}

	for i := 0; i < dof; i++ {
		R = p.H0i[i].Col3D(2, R)

		jointType := p.jointTypes[i]
		if jointType == 0 {
			di = p.H0i[i].Col3D(3, di)
			d[0] = dn[0] - di[0]
			d[1] = dn[1] - di[1]
			d[2] = dn[2] - di[2]

			linear := vec.Vector3D{
				R[1]*d[2] - R[2]*d[1],
				R[2]*d[0] - R[0]*d[2],
				R[0]*d[1] - R[1]*d[0],
			}
			J.SetColFromRow(i, 0, vec.Vector(linear[:]))
		} else if jointType == 3 {
			J.SetColFromRow(i, 0, vec.Vector(R[:]))
		} else {
			return kintypes.ErrUnsupportedOperation
		}
	}

	if err := J.PseudoInverse(Jinv); err != nil {
		return kintypes.ErrUnsupportedOperation
	}

	var vVec vec.Vector = vec.Vector{v[0], v[1], v[2]}
	deltaParams := make(vec.Vector, dof)
	Jinv.MulVec(vVec, deltaParams)

	for i := range p.params {
		p.params[i] += deltaParams[i]
	}

	return nil
}

func (p *DenavitHartenberg) loadState(state mattype.Matrix) error {
	if err := ensureColumn(state, len(p.params)); err != nil {
		return err
	}

	view := state.View().(mat.Matrix)
	for i := range p.params {
		p.params[i] = view[i][0]
	}
	return nil
}

func (p *DenavitHartenberg) loadEffector(destination mattype.Matrix) error {
	if err := ensureColumn(destination, effectorSize); err != nil {
		return err
	}
	view := destination.View().(mat.Matrix)
	for i := 0; i < effectorSize; i++ {
		p.pos[i] = view[i][0]
	}
	return nil
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
