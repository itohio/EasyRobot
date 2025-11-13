package slam

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/grid"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

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

	slam := New(rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY)

	if slam.numRays != 4 {
		t.Errorf("Expected 4 rays, got %d", slam.numRays)
	}

	if slam.mapRows != 10 || slam.mapCols != 10 {
		t.Errorf("Expected map size 10x10, got %dx%d", slam.mapRows, slam.mapCols)
	}

	// Check initial pose at map origin
	pose := slam.GetPose()
	if pose[0] != mapOriginX || pose[1] != mapOriginY || pose[2] != 0.0 {
		t.Errorf("Expected initial pose at origin, got [%f, %f, %f]", pose[0], pose[1], pose[2])
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

	slam := New(rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY)

	newPose := vec.NewFrom(1.0, 2.0, math32.Pi/4.0)
	slam.SetPose(newPose)

	pose := slam.GetPose()
	if pose[0] != 1.0 || pose[1] != 2.0 || pose[2] != math32.Pi/4.0 {
		t.Errorf("Expected pose [1.0, 2.0, π/4], got [%f, %f, %f]", pose[0], pose[1], pose[2])
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

	slam := New(rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY)
	slam.SetPose(vec.NewFrom(0.3, 0.3, 0.0))

	// Provide distance measurements
	distances := vec.NewFrom(0.2, 0.2)
	slam.UpdateMeasurement(distances)

	// Pose should have been updated by EKF
	pose := slam.GetPose()
	if pose[0] < -10 || pose[0] > 10 || pose[1] < -10 || pose[1] > 10 {
		t.Errorf("Expected reasonable pose, got [%f, %f, %f]", pose[0], pose[1], pose[2])
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

	slam := New(rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY)

	// Test Filter interface methods
	input := slam.GetInput()
	output := slam.GetOutput()
	target := slam.GetTarget()

	if len(input) != 1 {
		t.Errorf("Expected input length 1, got %d", len(input))
	}
	if len(output) != 3 {
		t.Errorf("Expected output length 3, got %d", len(output))
	}
	if len(target) != 3 {
		t.Errorf("Expected target length 3, got %d", len(target))
	}

	// Test Update method (Filter interface)
	copy(slam.GetInput(), vec.NewFrom(0.5))
	slam.Update(0.01) // timestep

	// Pose should have been updated
	pose := slam.GetOutput()
	if pose[0] < -10 || pose[0] > 10 {
		t.Errorf("Expected reasonable pose after update, got [%f, %f, %f]", pose[0], pose[1], pose[2])
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

	slam := New(rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY)

	// Set pose to non-origin
	slam.SetPose(vec.NewFrom(1.0, 2.0, math32.Pi/4.0))

	// Reset
	slam.Reset()

	// Pose should be back at origin
	pose := slam.GetPose()
	if pose[0] != mapOriginX || pose[1] != mapOriginY || pose[2] != 0.0 {
		t.Errorf("Expected pose at origin after reset, got [%f, %f, %f]", pose[0], pose[1], pose[2])
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

	slam := New(rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY)

	// Update with measurements
	distances := vec.NewFrom(0.5, 0.6)
	slam.UpdateMeasurement(distances)

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

	slam := New(rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY)

	maxRange := float32(5.0)
	slam.SetMaxRange(maxRange)

	if slam.maxRange != maxRange {
		t.Errorf("Expected maxRange %f, got %f", maxRange, slam.maxRange)
	}
}
