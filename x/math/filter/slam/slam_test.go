package slam

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/filter"
	"github.com/itohio/EasyRobot/x/math/grid"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// Helper to create pose matrix from [px, py, heading]
func poseFromVector(px, py, heading float32) mat.Matrix3x3 {
	cosH := math32.Cos(heading)
	sinH := math32.Sin(heading)
	return mat.Matrix3x3{
		{cosH, -sinH, px},
		{sinH, cosH, py},
		{0, 0, 1},
	}
}

// Helper to extract [px, py, heading] from pose matrix
func poseToVector(pose mat.Matrix3x3) (px, py, heading float32) {
	px = pose[0][2]
	py = pose[1][2]
	heading = math32.Atan2(pose[1][0], pose[0][0])
	return
}

// TestRayCast_Simple tests basic ray casting on a simple map.
func TestRayCast_Simple(t *testing.T) {
	// Create a simple 10x10 map with an obstacle in the center
	mapGrid := mat.New(10, 10)
	mapGrid.Eye()
	mapGrid.MulC(0) // Initialize all cells to free

	// Add obstacles aligned with the test rays
	mapGrid[3][5] = 1.0 // Occupied cell directly east of pose
	mapGrid[5][3] = 1.0 // Occupied cell directly north of pose

	mapResolution := float32(0.1) // 10cm per cell
	mapOriginX := float32(0.0)
	mapOriginY := float32(0.0)
	maxRange := float32(2.0)

	// Pose: start at (0.3, 0.3), facing east (0 radians)
	pose := vec.NewFrom(0.3, 0.3, 0.0)

	// Cast ray eastward
	rayAngle := float32(0.0) // East
	distance := grid.RayCast(pose, rayAngle, mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange)

	// Expected distance: from (0.3, 0.3) to obstacle at (0.5, 0.3)
	// Ray is eastward, so hits cell (5, 3) which is at x=0.5
	// Distance should be approximately 0.2 meters
	if distance < 0.15 || distance > 0.25 {
		t.Errorf("Expected distance ~0.2m for eastward ray, got %f", distance)
	}

	// Cast ray northward (should hit the same obstacle from different direction)
	rayAngle = math32.Pi / 2.0 // North
	distance = grid.RayCast(pose, rayAngle, mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange)

	// Expected distance: from (0.3, 0.3) to obstacle at (0.3, 0.5)
	if distance < 0.15 || distance > 0.25 {
		t.Errorf("Expected distance ~0.2m for northward ray, got %f", distance)
	}
}

// TestRayCast_NoObstacle tests ray casting when no obstacle is found.
func TestRayCast_NoObstacle(t *testing.T) {
	// Create empty map (all free)
	mapGrid := mat.New(10, 10)
	mapGrid.Eye()
	mapGrid.MulC(0) // All free

	mapResolution := float32(0.1)
	mapOriginX := float32(0.0)
	mapOriginY := float32(0.0)
	maxRange := float32(2.0)

	pose := vec.NewFrom(0.3, 0.3, 0.0)
	rayAngle := float32(0.0) // East

	distance := grid.RayCast(pose, rayAngle, mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange)

	// Should return maxRange when no obstacle found
	if distance != maxRange {
		t.Errorf("Expected maxRange when no obstacle, got %f", distance)
	}
}

// TestRayCast_OutOfBounds tests ray casting when starting outside map bounds.
func TestRayCast_OutOfBounds(t *testing.T) {
	mapGrid := mat.New(10, 10)
	mapResolution := float32(0.1)
	mapOriginX := float32(0.0)
	mapOriginY := float32(0.0)
	maxRange := float32(2.0)

	// Pose outside map bounds
	pose := vec.NewFrom(10.0, 10.0, 0.0)
	rayAngle := float32(0.0)

	distance := grid.RayCast(pose, rayAngle, mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange)

	// Should return maxRange when out of bounds
	if distance != maxRange {
		t.Errorf("Expected maxRange when out of bounds, got %f", distance)
	}
}

// TestRayCastAll tests casting all rays.
func TestRayCastAll(t *testing.T) {
	// Create map with obstacle
	mapGrid := mat.New(10, 10)
	mapGrid.Eye()
	mapGrid.MulC(0)
	mapGrid[5][5] = 1.0 // Obstacle

	mapResolution := float32(0.1)
	mapOriginX := float32(0.0)
	mapOriginY := float32(0.0)
	maxRange := float32(2.0)

	pose := vec.NewFrom(0.3, 0.3, 0.0)
	rayAngles := vec.NewFrom(0.0, math32.Pi/2.0, math32.Pi, 3.0*math32.Pi/2.0) // 4 rays: E, N, W, S

	distances := grid.RayCastAll(pose, rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange)

	if len(distances) != 4 {
		t.Errorf("Expected 4 distances, got %d", len(distances))
	}

	// All rays should hit the obstacle (similar distances)
	for i, d := range distances {
		if d <= 0 || d > maxRange {
			t.Errorf("Ray %d: expected valid distance, got %f", i, d)
		}
	}
}

// TestSLAM_New tests SLAM filter creation.
func TestSLAM_New(t *testing.T) {
	// Create simple map
	mapGrid := mat.New(10, 10)
	mapGrid.Eye()
	mapGrid.MulC(0)

	// 4 rays: east, north, west, south
	rayAngles := vec.NewFrom(0.0, math32.Pi/2.0, math32.Pi, 3.0*math32.Pi/2.0)

	mapResolution := float32(0.1)
	mapOriginX := float32(0.0)
	mapOriginY := float32(0.0)

	slam := New(rayAngles, mapOriginX, mapOriginY, WithMap(mapGrid), WithResolution(mapResolution))

	if slam.numRays != 4 {
		t.Errorf("Expected 4 rays, got %d", slam.numRays)
	}

	if slam.mapRows != 10 || slam.mapCols != 10 {
		t.Errorf("Expected map size 10x10, got %dx%d", slam.mapRows, slam.mapCols)
	}

	// Check initial pose at map origin
	pose := slam.GetPose()
	px, py, heading := poseToVector(pose)
	if px != mapOriginX || py != mapOriginY || heading != 0.0 {
		t.Errorf("Expected initial pose at origin, got [%f, %f, %f]", px, py, heading)
	}
}

// TestSLAM_SetPose tests setting initial pose.
func TestSLAM_SetPose(t *testing.T) {
	mapGrid := mat.New(10, 10)
	mapGrid.Eye()
	mapGrid.MulC(0)

	rayAngles := vec.NewFrom(0.0)
	mapResolution := float32(0.1)
	mapOriginX := float32(0.0)
	mapOriginY := float32(0.0)

	slam := New(rayAngles, mapOriginX, mapOriginY, WithMap(mapGrid), WithResolution(mapResolution))

	newPose := poseFromVector(1.0, 2.0, math32.Pi/4.0)
	slam.SetPose(newPose)

	pose := slam.GetPose()
	px, py, heading := poseToVector(pose)
	if px != 1.0 || py != 2.0 || math32.Abs(heading-math32.Pi/4.0) > 1e-6 {
		t.Errorf("Expected pose [1.0, 2.0, π/4], got [%f, %f, %f]", px, py, heading)
	}
}

// TestSLAM_UpdateMeasurement tests updating with measurements.
func TestSLAM_UpdateMeasurement(t *testing.T) {
	// Create map with obstacle
	mapGrid := mat.New(10, 10)
	mapGrid.Eye()
	mapGrid.MulC(0)
	mapGrid[5][5] = 1.0 // Obstacle at center

	rayAngles := vec.NewFrom(0.0, math32.Pi/2.0) // 2 rays: east, north
	mapResolution := float32(0.1)
	mapOriginX := float32(0.0)
	mapOriginY := float32(0.0)

	slam := New(rayAngles, mapOriginX, mapOriginY, WithMap(mapGrid), WithResolution(mapResolution))
	slam.SetPose(poseFromVector(0.3, 0.3, 0.0))

	// Provide angle and distance measurements
	angles := vec.NewFrom(0.0, math32.Pi/2.0)
	distances := vec.NewFrom(0.2, 0.2)
	slam.UpdateMeasurement(angles, distances)

	// Pose should have been updated by EKF
	pose := slam.GetPose()
	px, py, _ := poseToVector(pose)
	if px < -10 || px > 10 || py < -10 || py > 10 {
		t.Errorf("Expected reasonable pose, got [%f, %f]", px, py)
	}

	// Check expected distances were computed
	expected := slam.GetExpectedDistances()
	if len(expected) != 2 {
		t.Errorf("Expected 2 expected distances, got %d", len(expected))
	}
}

// TestSLAM_FilterInterface tests Filter interface implementation.
func TestSLAM_FilterInterface(t *testing.T) {
	mapGrid := mat.New(10, 10)
	mapGrid.Eye()
	mapGrid.MulC(0)

	rayAngles := vec.NewFrom(0.0)
	mapResolution := float32(0.1)
	mapOriginX := float32(0.0)
	mapOriginY := float32(0.0)

	slam := New(rayAngles, mapOriginX, mapOriginY, WithMap(mapGrid), WithResolution(mapResolution))

	// Test Filter interface methods
	input := slam.Input()
	output := slam.Output()
	target := slam.GetTarget()

	if input.Rows() != 2 || input.Cols() != 1 {
		t.Errorf("Expected input size 2x1, got %dx%d", input.Rows(), input.Cols())
	}
	if output.Rows() != 3 || output.Cols() != 3 {
		t.Errorf("Expected output size 3x3, got %dx%d", output.Rows(), output.Cols())
	}
	if target.Rows() != 3 || target.Cols() != 3 {
		t.Errorf("Expected target size 3x3, got %dx%d", target.Rows(), target.Cols())
	}

	// Test Update method (Filter interface)
	// Create input matrix: 2 rows (angles, distances) x 1 column
	inputMat := mat.New(2, 1)
	inputMat[0][0] = 0.0 // angle
	inputMat[1][0] = 0.5 // distance
	slam.Update(0.01, inputMat)

	// Pose should have been updated
	pose := slam.Output()
	px, _, _ := poseToVector(pose)
	if px < -10 || px > 10 {
		t.Errorf("Expected reasonable pose after update, got px=%f", px)
	}
}

// TestSLAM_Reset tests filter reset functionality.
func TestSLAM_Reset(t *testing.T) {
	mapGrid := mat.New(10, 10)
	mapGrid.Eye()
	mapGrid.MulC(0)

	rayAngles := vec.NewFrom(0.0)
	mapResolution := float32(0.1)
	mapOriginX := float32(0.0)
	mapOriginY := float32(0.0)

	slam := New(rayAngles, mapOriginX, mapOriginY, WithMap(mapGrid), WithResolution(mapResolution))

	// Set pose to non-origin
	slam.SetPose(poseFromVector(1.0, 2.0, math32.Pi/4.0))

	// Reset
	slam.Reset()

	// Pose should be back at origin
	pose := slam.GetPose()
	px, py, heading := poseToVector(pose)
	if px != mapOriginX || py != mapOriginY || heading != 0.0 {
		t.Errorf("Expected pose at origin after reset, got [%f, %f, %f]", px, py, heading)
	}
}

// TestSLAM_GetResiduals tests residual computation.
func TestSLAM_GetResiduals(t *testing.T) {
	mapGrid := mat.New(10, 10)
	mapGrid.Eye()
	mapGrid.MulC(0)

	rayAngles := vec.NewFrom(0.0, math32.Pi/2.0)
	mapResolution := float32(0.1)
	mapOriginX := float32(0.0)
	mapOriginY := float32(0.0)

	slam := New(rayAngles, mapOriginX, mapOriginY, WithMap(mapGrid), WithResolution(mapResolution))

	// Update with measurements
	angles := vec.NewFrom(0.0, math32.Pi/2.0)
	distances := vec.NewFrom(0.5, 0.6)
	slam.UpdateMeasurement(angles, distances)

	// Get residuals
	residuals := slam.GetResiduals()
	if len(residuals) != 2 {
		t.Errorf("Expected 2 residuals, got %d", len(residuals))
	}

	// Residuals should be measured - expected
	for i, r := range residuals {
		if r < -10 || r > 10 {
			t.Errorf("Residual %d: expected reasonable value, got %f", i, r)
		}
	}
}

// TestSLAM_SetMaxRange tests setting maximum range.
func TestSLAM_SetMaxRange(t *testing.T) {
	mapGrid := mat.New(10, 10)
	mapGrid.Eye()
	mapGrid.MulC(0)

	rayAngles := vec.NewFrom(0.0)
	mapResolution := float32(0.1)
	mapOriginX := float32(0.0)
	mapOriginY := float32(0.0)

	slam := New(rayAngles, mapOriginX, mapOriginY, WithMap(mapGrid), WithResolution(mapResolution))

	maxRange := float32(5.0)
	slam.SetMaxRange(maxRange)

	if slam.maxRange != maxRange {
		t.Errorf("Expected maxRange %f, got %f", maxRange, slam.maxRange)
	}
}

// TestSLAM_InterfaceConformance tests that SLAM implements Filter interface.
func TestSLAM_InterfaceConformance(t *testing.T) {
	var _ filter.Filter[matTypes.Matrix, mat.Matrix3x3] = (*SLAM)(nil)
}
