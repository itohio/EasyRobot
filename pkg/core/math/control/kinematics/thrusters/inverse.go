package thrusters

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

const solveTolerance = 1e-4

func allocate(body Body, thrusters []Thruster, current, desired BodyState) ([]Command, error) {
	if err := validateBody(body); err != nil {
		return nil, err
	}
	if len(thrusters) == 0 {
		return nil, ErrInfeasible
	}
	delta := deltaState(desired, current)
	required := targetWrench(body, delta)
	matrix := allocationMatrix(thrusters)
	solution, err := solveAllocation(matrix, required)
	if err != nil {
		return nil, err
	}
	cmds, saturated := mapSolution(thrusters, solution)
	wrench, err := verifySolution(body, thrusters, cmds)
	if err != nil {
		return nil, err
	}
	if saturated || !closeWrench(required, wrench) {
		return nil, ErrInfeasible
	}
	return cmds, nil
}

func targetWrench(body Body, delta BodyState) [6]float32 {
	var wrench [6]float32
	for i := 0; i < 3; i++ {
		wrench[i] = delta.Force[i] + body.Mass*delta.LinearAccel[i]
	}
	var inertiaContribution vec.Vector3D
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			inertiaContribution[row] += body.Inertia[row][col] * delta.AngularAccel[col]
		}
		wrench[row+3] = delta.Torque[row] + inertiaContribution[row]
	}
	return wrench
}

func allocationMatrix(thrusters []Thruster) mat.Matrix {
	rows := 6
	cols := len(thrusters) * 2
	matrix := mat.New(rows, cols)
	for i, th := range thrusters {
		thrustCol := 2 * i
		torqueCol := thrustCol + 1
		moment := cross(th.Position, th.Direction)
		for axis := 0; axis < 3; axis++ {
			matrix[axis][thrustCol] = th.Direction[axis]
			matrix[axis+3][thrustCol] = moment[axis]
			matrix[axis+3][torqueCol] = th.TorqueAxis[axis]
		}
	}
	return matrix
}

func solveAllocation(W mat.Matrix, wrench [6]float32) ([]float32, error) {
	rows := len(W)
	if rows == 0 {
		return nil, ErrInfeasible
	}
	cols := len(W[0])
	pseudo := mat.New(cols, rows)
	if err := W.PseudoInverse(pseudo); err != nil {
		if err := dampedPseudoInverse(W, pseudo, 1e-3); err != nil {
			return nil, ErrInfeasible
		}
	}
	wrenchVec := vec.New(rows)
	for i := 0; i < rows && i < len(wrenchVec); i++ {
		wrenchVec[i] = wrench[i]
	}
	solutionVec := vec.New(cols)
	pseudo.MulVec(wrenchVec, solutionVec)
	solution := make([]float32, cols)
	copy(solution, solutionVec)
	return solution, nil
}

func dampedPseudoInverse(J mat.Matrix, dst mat.Matrix, lambda float32) error {
	rows := len(J)
	cols := len(J[0])
	JT := mat.New(cols, rows)
	JT.Transpose(J)
	JJT := mat.New(rows, rows)
	JJT.Mul(J, JT)
	lambda2 := lambda * lambda
	for i := 0; i < rows; i++ {
		JJT[i][i] += lambda2
	}
	JJTInv := mat.New(rows, rows)
	if err := JJT.Inverse(JJTInv); err != nil {
		return err
	}
	dst.Mul(JT, JJTInv)
	return nil
}

func mapSolution(thrusters []Thruster, sol []float32) ([]Command, bool) {
	cmds := make([]Command, len(thrusters))
	saturated := false
	for i, th := range thrusters {
		thrust := sol[2*i]
		torque := sol[2*i+1]
		cmd, ok := clampCommand(Command{Thrust: thrust, Torque: torque}, th)
		if !ok {
			saturated = true
		}
		cmds[i] = cmd
	}
	return cmds, saturated
}

func verifySolution(body Body, thrusters []Thruster, cmds []Command) ([6]float32, error) {
	tc := make([]ThrusterCommand, len(thrusters))
	for i := range thrusters {
		tc[i] = ThrusterCommand{
			Thruster: thrusters[i],
			Command:  cmds[i],
		}
	}
	result, err := applyForward(body, BodyState{}, tc)
	if err != nil {
		return [6]float32{}, err
	}
	return [6]float32{
		result.Force[0],
		result.Force[1],
		result.Force[2],
		result.Torque[0],
		result.Torque[1],
		result.Torque[2],
	}, nil
}

func closeWrench(a, b [6]float32) bool {
	for i := range a {
		if math32.Abs(a[i]-b[i]) > solveTolerance {
			return false
		}
	}
	return true
}
