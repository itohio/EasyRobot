package thrusters

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

const solveTolerance = 1e-4

func Inverse(body Body, thrusters []Thruster, current, desired BodyState) ([]Command, error) {
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

func allocationMatrix(thrusters []Thruster) [][]float32 {
	rows := 6
	cols := len(thrusters) * 2
	matrix := make([][]float32, rows)
	for r := range matrix {
		matrix[r] = make([]float32, cols)
	}
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

func solveAllocation(W [][]float32, wrench [6]float32) ([]float32, error) {
	rows := len(W)
	cols := len(W[0])
	A := make([][]float32, cols)
	for i := range A {
		A[i] = make([]float32, cols)
	}
	y := make([]float32, cols)
	for i := 0; i < cols; i++ {
		for j := 0; j < cols; j++ {
			var sum float32
			for r := 0; r < rows; r++ {
				sum += W[r][i] * W[r][j]
			}
			A[i][j] = sum
		}
		var sum float32
		for r := 0; r < rows; r++ {
			sum += W[r][i] * wrench[r]
		}
		y[i] = sum
	}
	return solveLinear(A, y)
}

func solveLinear(A [][]float32, b []float32) ([]float32, error) {
	n := len(A)
	aug := make([][]float32, n)
	for i := 0; i < n; i++ {
		row := make([]float32, n+1)
		copy(row, A[i])
		row[n] = b[i]
		aug[i] = row
	}
	for col := 0; col < n; col++ {
		pivot := col
		maxVal := math32.Abs(aug[pivot][col])
		for r := col + 1; r < n; r++ {
			if v := math32.Abs(aug[r][col]); v > maxVal {
				maxVal = v
				pivot = r
			}
		}
		if maxVal < solveTolerance {
			return nil, ErrInfeasible
		}
		if pivot != col {
			aug[col], aug[pivot] = aug[pivot], aug[col]
		}
		pivotVal := aug[col][col]
		invPivot := 1 / pivotVal
		for c := col; c <= n; c++ {
			aug[col][c] *= invPivot
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			factor := aug[r][col]
			if math32.Abs(factor) < solveTolerance {
				continue
			}
			for c := col; c <= n; c++ {
				aug[r][c] -= factor * aug[col][c]
			}
		}
	}
	solution := make([]float32, n)
	for i := 0; i < n; i++ {
		solution[i] = aug[i][n]
	}
	return solution, nil
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
	result, err := Forward(body, BodyState{}, tc)
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
