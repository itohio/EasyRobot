package slam

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/filter"
	"github.com/itohio/EasyRobot/pkg/core/math/filter/ekalman"
	"github.com/itohio/EasyRobot/pkg/core/math/grid"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

const (
	// DefaultMaxRange is the default maximum sensor range in meters
	DefaultMaxRange = 10.0
)

// SLAM implements a simple SLAM filter for robot localization using ray-based sensors.
type SLAM struct {
	// Ray configuration
	rayAngles vec.Vector          // Array of ray angles (radians) relative to robot heading
	rayDirs   *grid.RayDirections // Pre-computed ray directions for efficiency

	// Map
	mapGrid       mat.Matrix // Occupancy grid map (rows x cols)
	logOddsMap    mat.Matrix // Log-odds representation for map building (nil if not used)
	mapResolution float32    // Grid cell size in meters
	mapOriginX    float32    // X coordinate of map origin (meters)
	mapOriginY    float32    // Y coordinate of map origin (meters)

	// State
	pose vec.Vector // Robot pose [px, py, heading]

	// Localization filter
	ekf *ekalman.EKF // Extended Kalman Filter for pose estimation
	// Note: EKF is required because measurement function (ray casting) is nonlinear in pose.
	// KF cannot be substituted because h(pose) cannot be expressed as H * pose.

	// Temporary storage
	expectedDistances vec.Vector // Expected distances for each ray
	measuredDistances vec.Vector // Measured distances (input)

	// Configuration
	maxRange            float32 // Maximum sensor range (meters)
	enableMapping       bool    // Enable online map building
	useOptimizedRaycast bool    // Use optimized ray casting (default: true)

	// Filter interface
	Input  vec.Vector // Measured distances input
	Output vec.Vector // Estimated pose output [px, py, heading]
	Target vec.Vector // Target pose (optional)

	// Dimensions
	numRays int // Number of rays
	mapRows int // Map rows
	mapCols int // Map columns
}

// New creates a new SLAM filter.
//
// Parameters:
//   - rayAngles: Array of ray angles in radians (relative to robot heading)
//   - mapGrid: Occupancy grid map (rows x cols), values: 0=free, 1=occupied, 0.5=unknown
//   - mapResolution: Grid cell size in meters
//   - mapOriginX: X coordinate of map origin in world coordinates (meters)
//   - mapOriginY: Y coordinate of map origin in world coordinates (meters)
func New(
	rayAngles vec.Vector,
	mapGrid mat.Matrix,
	mapResolution, mapOriginX, mapOriginY float32,
) *SLAM {
	if len(rayAngles) == 0 {
		panic("slam: rayAngles must have at least one element")
	}
	if len(mapGrid) == 0 || len(mapGrid[0]) == 0 {
		panic("slam: mapGrid must be non-empty")
	}
	if mapResolution <= 0 {
		panic("slam: mapResolution must be positive")
	}

	numRays := len(rayAngles)
	mapRows := len(mapGrid)
	mapCols := len(mapGrid[0])

	// Pre-compute ray directions for efficiency (embedded optimization)
	rayDirs := grid.NewRayDirections(rayAngles)

	// Create EKF for pose estimation (3D state: px, py, heading)
	n, m := 3, numRays // state dim, measurement dim

	// State transition function: static pose (no motion model)
	// Can be extended with odometry later
	fFunc := func(x, u vec.Vector, dt float32) vec.Vector {
		next := vec.New(3)
		copy(next, x) // Static: pose doesn't change
		return next
	}

	// Measurement function: ray casting (optimized version)
	// Note: This is nonlinear in pose, so we need EKF (not KF)
	// h(x) = rayCastDistance(x, θ_i, M) cannot be expressed as H * x
	// The measurement function depends nonlinearly on:
	//   - px, py: Position affects which cells the ray traverses
	//   - heading: Orientation affects ray direction
	// Therefore, we need EKF to linearize around current pose estimate.
	// KF would require constant H matrix, which is not possible for ray casting.
	hFunc := func(x vec.Vector) vec.Vector {
		// Use optimized ray casting with pre-computed directions
		// Allocate result vector (EKF will use it)
		distances := vec.New(numRays)
		grid.RayCastAllOptimized(x, rayDirs, mapGrid, mapResolution, mapOriginX, mapOriginY, DefaultMaxRange, distances)
		return distances
	}

	// Process noise covariance (uncertainty in pose)
	Q := mat.New(3, 3)
	Q.Eye()
	Q.MulC(0.01) // Small process noise (static pose)

	// Measurement noise covariance (uncertainty in distance measurements)
	R := mat.New(numRays, numRays)
	R.Eye()
	R.MulC(0.1) // Distance measurement noise

	// Create EKF with numerical Jacobians
	ekf := ekalman.New(n, m, fFunc, hFunc, nil, nil, Q, R)

	s := &SLAM{
		rayAngles:           rayAngles,
		rayDirs:             rayDirs,
		mapGrid:             mapGrid,
		logOddsMap:          nil, // Created when map building is enabled
		mapResolution:       mapResolution,
		mapOriginX:          mapOriginX,
		mapOriginY:          mapOriginY,
		pose:                vec.New(3),
		ekf:                 ekf,
		expectedDistances:   vec.New(numRays),
		measuredDistances:   vec.New(numRays),
		maxRange:            DefaultMaxRange,
		enableMapping:       false, // Disabled by default
		useOptimizedRaycast: true,  // Use optimized ray casting by default
		Input:               vec.New(numRays),
		Output:              vec.New(3),
		Target:              vec.New(3),
		numRays:             numRays,
		mapRows:             mapRows,
		mapCols:             mapCols,
	}

	// Initialize pose at map origin
	s.pose[0] = mapOriginX
	s.pose[1] = mapOriginY
	s.pose[2] = 0.0

	copy(s.Output, s.pose)

	// Set initial pose in EKF
	s.ekf.SetState(s.pose)

	// Set initial covariance (uncertainty about initial pose)
	initialP := mat.New(3, 3)
	initialP.Eye()
	initialP.MulC(1.0) // 1 meter position uncertainty, 1 rad heading uncertainty
	s.ekf.SetCovariance(initialP)

	return s
}

// SetMaxRange sets the maximum sensor range in meters.
func (s *SLAM) SetMaxRange(maxRange float32) *SLAM {
	if maxRange <= 0 {
		panic("slam: maxRange must be positive")
	}
	s.maxRange = maxRange
	return s
}

// SetPose sets the initial robot pose.
// pose: [px, py, heading] in world coordinates
func (s *SLAM) SetPose(pose vec.Vector) *SLAM {
	if len(pose) < 3 {
		panic("slam: pose must have at least 3 elements [px, py, heading]")
	}
	copy(s.pose, pose)
	copy(s.Output, pose)
	s.ekf.SetState(pose)
	return s
}

// GetPose returns the current robot pose estimate.
// Returns: [px, py, heading] in world coordinates
func (s *SLAM) GetPose() vec.Vector {
	return s.Output
}

// SetMappingEnabled enables or disables online map building.
func (s *SLAM) SetMappingEnabled(enabled bool) *SLAM {
	s.enableMapping = enabled
	if enabled && s.logOddsMap == nil {
		// Initialize log-odds map
		s.logOddsMap = mat.New(s.mapRows, s.mapCols)
		// Initialize from current map
		for i := 0; i < s.mapRows; i++ {
			for j := 0; j < s.mapCols; j++ {
				p := s.mapGrid[i][j]
				if p < 0.01 {
					p = 0.01
				}
				if p > 0.99 {
					p = 0.99
				}
				s.logOddsMap[i][j] = math32.Log(p / (1.0 - p))
			}
		}
	}
	return s
}

// SetOptimizedRaycast enables or disables optimized ray casting.
// Optimized ray casting uses pre-computed directions and Bresenham algorithm.
func (s *SLAM) SetOptimizedRaycast(enabled bool) *SLAM {
	s.useOptimizedRaycast = enabled
	return s
}

// UpdateMeasurement updates the SLAM filter with distance measurements.
// distances: Vector of distance measurements for each ray (meters)
func (s *SLAM) UpdateMeasurement(distances vec.Vector) *SLAM {
	if len(distances) != s.numRays {
		panic("slam: distances vector must have same length as rayAngles")
	}

	// Copy measurements
	copy(s.measuredDistances, distances)
	copy(s.Input, distances)

	// Update map if mapping is enabled (before pose update for better map)
	if s.enableMapping && s.logOddsMap != nil {
		InverseSensorModelAll(
			s.mapGrid, s.pose, s.rayAngles, distances,
			s.mapResolution, s.mapOriginX, s.mapOriginY, s.maxRange,
			s.logOddsMap,
		)
	}

	// Update EKF with measurements
	s.ekf.UpdateMeasurement(distances)

	// Get updated pose from EKF
	pose := s.ekf.GetOutput()
	copy(s.pose, pose)
	copy(s.Output, pose)

	// Compute expected distances for reference (use optimized version)
	if s.useOptimizedRaycast {
		grid.RayCastAllOptimized(s.pose, s.rayDirs, s.mapGrid, s.mapResolution, s.mapOriginX, s.mapOriginY, s.maxRange, s.expectedDistances)
	} else {
		expected := grid.RayCastAll(s.pose, s.rayAngles, s.mapGrid, s.mapResolution, s.mapOriginX, s.mapOriginY, s.maxRange)
		copy(s.expectedDistances, expected)
	}

	return s
}

// Reset resets the filter state to the map origin.
func (s *SLAM) Reset() filter.Filter {
	s.pose[0] = s.mapOriginX
	s.pose[1] = s.mapOriginY
	s.pose[2] = 0.0
	copy(s.Output, s.pose)
	copy(s.Input, vec.New(s.numRays))
	copy(s.Target, s.pose)

	s.ekf.SetState(s.pose)
	initialP := mat.New(3, 3)
	initialP.Eye()
	initialP.MulC(1.0)
	s.ekf.SetCovariance(initialP)

	return s
}

// Update implements the Filter interface.
// This method performs pose update and expects distances to be set in Input.
func (s *SLAM) Update(timestep float32) filter.Filter {
	// Check if measurement is available
	hasMeasurement := false
	for i := range s.Input {
		if s.Input[i] != 0 {
			hasMeasurement = true
			break
		}
	}

	if hasMeasurement {
		s.UpdateMeasurement(s.Input)
	}

	return s
}

// GetInput returns the measurement input vector (distances).
func (s *SLAM) GetInput() vec.Vector {
	return s.Input
}

// GetOutput returns the estimated pose vector [px, py, heading].
func (s *SLAM) GetOutput() vec.Vector {
	return s.Output
}

// GetTarget returns the target pose vector.
func (s *SLAM) GetTarget() vec.Vector {
	return s.Target
}

// GetExpectedDistances returns the expected distances computed from the current pose estimate.
func (s *SLAM) GetExpectedDistances() vec.Vector {
	return s.expectedDistances
}

// GetResiduals returns the difference between measured and expected distances.
func (s *SLAM) GetResiduals() vec.Vector {
	residuals := vec.New(s.numRays)
	copy(residuals, s.measuredDistances)
	residuals.Sub(s.expectedDistances)
	return residuals
}
