package dh

import (
	"github.com/itohio/EasyRobot/pkg/control/kinematics"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

type DenavitHartenberg struct {
	c             []Config
	eps           float32
	maxIterations int
	params        []float32
	pos           [7]float32
	H0i           []mat.Matrix4x4
	jointTypes    []int
}

func New(eps float32, maxIterations int, cfg ...Config) kinematics.Kinematics {
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
	}
}

func (p *DenavitHartenberg) DOF() int {
	return len(p.c)
}

func (p *DenavitHartenberg) Params() vec.Vector {
	return p.params[:]
}

func (p *DenavitHartenberg) Effector() vec.Vector {
	return p.pos[:]
}

func (p *DenavitHartenberg) Forward() bool {
	H := mat.Matrix4x4{}
	H.Eye()
	p.H0i[0].Eye()
	for i, cfg := range p.c {
		if !cfg.CalculateTransform(p.params[i], &H) {
			return false
		}
		p.H0i[i+1].Mul(p.H0i[i], H)
	}

	// Extract position from column 3 using Col3D (matches C++ reference)
	var posVec vec.Vector3D
	posVec = p.H0i[len(p.c)].Col3D(3, posVec)
	p.pos[0] = posVec[0]
	p.pos[1] = posVec[1]
	p.pos[2] = posVec[2]

	// Extract quaternion from rotation matrix (if needed)
	copy(p.pos[3:], p.H0i[len(p.c)].Quaternion().Vector())

	return true
}

func (p *DenavitHartenberg) Inverse() bool {
	eps2 := p.eps * p.eps
	var error vec.Vector3D
	target := vec.Vector3D{p.pos[0], p.pos[1], p.pos[2]} // Target position

	var actual vec.Vector3D

	for iter := 0; iter <= p.maxIterations; iter++ {
		// Forward kinematics with current params
		if !p.Forward() {
			return false
		}

		// Extract actual position
		actual[0] = p.pos[0]
		actual[1] = p.pos[1]
		actual[2] = p.pos[2]

		// Calculate error
		error[0] = target[0] - actual[0]
		error[1] = target[1] - actual[1]
		error[2] = target[2] - actual[2]

		// Check convergence
		errSqr := error[0]*error[0] + error[1]*error[1] + error[2]*error[2]
		if errSqr < eps2 {
			return true
		}

		if iter == p.maxIterations {
			return false
		}

		// Jacobian-based IK update
		if !p.ikSolverJacobianPos(error) {
			return false
		}
	}

	return false
}

// ikSolverJacobianPos implements Jacobian-based IK solver for position only.
// Equivalent to ik_solver_jacobian_pos in C++ reference.
func (p *DenavitHartenberg) ikSolverJacobianPos(v vec.Vector3D) bool {
	dof := len(p.c)

	// Create Jacobian matrix (3 x DOF)
	J := mat.New(3, dof)

	// Create pseudo-inverse matrix (DOF x 3)
	Jinv := mat.New(dof, 3)

	// Extract end-effector position
	var dn vec.Vector3D
	dn = p.H0i[dof].Col3D(3, dn)

	var R, di, d vec.Vector3D
	R = vec.Vector3D{0, 0, 1} // Z-axis

	for i := 0; i < dof; i++ {
		// Extract rotation axis from column 2
		R = p.H0i[i].Col3D(2, R)

		jointType := p.jointTypes[i]

		if jointType == 0 {
			// Revolute joint
			// Extract joint position from column 3
			di = p.H0i[i].Col3D(3, di)

			// d = dn - di
			d[0] = dn[0] - di[0]
			d[1] = dn[1] - di[1]
			d[2] = dn[2] - di[2]

			// R.cross(d) = R × d (angular velocity contribution)
			// For revolute joints: linear = R × d, angular = R
			linear := vec.Vector3D{
				R[1]*d[2] - R[2]*d[1], // R × d
				R[2]*d[0] - R[0]*d[2],
				R[0]*d[1] - R[1]*d[0],
			}

			// Set linear velocity column (rows 0-2)
			J.SetColFromRow(i, 0, vec.Vector(linear[:]))
			// Angular velocity column would be at row 3, but we only have 3 rows for position IK
			// For position-only IK, we only use linear velocity

		} else if jointType == 3 {
			// Prismatic joint along Z
			// Linear velocity contribution is along rotation axis
			J.SetColFromRow(i, 0, vec.Vector(R[:]))
		} else {
			// Unsupported joint type
			return false
		}
	}

	// Compute pseudo-inverse
	if err := J.PseudoInverse(Jinv); err != nil {
		return false
	}

	// Compute parameter update: delta_params = Jinv * v
	// Then update: params = params + delta_params
	var vVec vec.Vector = vec.Vector{v[0], v[1], v[2]}
	deltaParams := make(vec.Vector, dof)
	Jinv.MulVec(vVec, deltaParams)

	// Add delta to current parameters
	for i := range p.params {
		p.params[i] += deltaParams[i]
	}

	return true
}
