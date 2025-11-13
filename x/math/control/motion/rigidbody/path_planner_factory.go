package rigidbody

import (
	"fmt"

	mattype "github.com/itohio/EasyRobot/pkg/core/math/mat/types"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

func plannerFromPositions(points []vec.Vector3D) (pathPlanner, error) {
	return newPositionPlanner(points)
}

func plannerFromWaypointMatrix(path mattype.Matrix) (pathPlanner, error) {
	rows := path.Rows()
	cols := path.Cols()
	if cols < 2 {
		return nil, ErrInvalidPath
	}
	switch rows {
	case 3:
		return plannerFromPositions(extractPositions(path))
	case 6:
		return plannerPositionsWithVelocity(path)
	case 7:
		return plannerPose(path)
	case 14:
		return plannerPoseVelocity(path)
	default:
		return nil, fmt.Errorf("rigidbody: unsupported waypoint rows %d", rows)
	}
}

func extractPositions(path mattype.Matrix) []vec.Vector3D {
	cols := path.Cols()
	points := make([]vec.Vector3D, cols)
	column := vec.Vector(make([]float32, path.Rows()))
	for c := 0; c < cols; c++ {
		path.GetCol(c, column)
		points[c] = vec.Vector3D{column[0], column[1], column[2]}
	}
	return points
}

func plannerPositionsWithVelocity(path mattype.Matrix) (pathPlanner, error) {
	cols := path.Cols()
	column := vec.Vector(make([]float32, 6))
	points := make([]vec.Vector3D, cols)
	vel := make([]vec.Vector3D, cols)
	for c := 0; c < cols; c++ {
		path.GetCol(c, column)
		points[c] = vec.Vector3D{column[0], column[1], column[2]}
		vel[c] = vec.Vector3D{column[3], column[4], column[5]}
	}
	return newPositionVelocityPlanner(points, vel)
}

func plannerPose(path mattype.Matrix) (pathPlanner, error) {
	cols := path.Cols()
	column := vec.Vector(make([]float32, 7))
	points := make([]vec.Vector3D, cols)
	quats := make([]vec.Quaternion, cols)
	for c := 0; c < cols; c++ {
		path.GetCol(c, column)
		points[c] = vec.Vector3D{column[0], column[1], column[2]}
		quats[c] = vec.Quaternion{column[3], column[4], column[5], column[6]}
	}
	return newPosePlanner(points, quats)
}

func plannerPoseVelocity(path mattype.Matrix) (pathPlanner, error) {
	cols := path.Cols()
	column := vec.Vector(make([]float32, 14))
	points := make([]vec.Vector3D, cols)
	quats := make([]vec.Quaternion, cols)
	lin := make([]vec.Vector3D, cols)
	ang := make([]vec.Vector3D, cols)
	for c := 0; c < cols; c++ {
		path.GetCol(c, column)
		points[c] = vec.Vector3D{column[0], column[1], column[2]}
		quats[c] = vec.Quaternion{column[3], column[4], column[5], column[6]}
		lin[c] = vec.Vector3D{column[7], column[8], column[9]}
		ang[c] = vec.Vector3D{column[10], column[11], column[12]}
	}
	return newPoseVelocityPlanner(points, quats, lin, ang)
}
