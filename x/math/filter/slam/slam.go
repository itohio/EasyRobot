package slam

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/filter/ekalman"
	"github.com/itohio/EasyRobot/x/math/grid"
	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

const (
	// DefaultMaxRange is the default maximum sensor range in meters
	DefaultMaxRange = 10.0
	// DefaultResolution is the default map resolution in meters per cell
	DefaultResolution = 0.1
	// DefaultMapSize is the default map size (rows and columns)
	DefaultMapSize = 100
)

// Option configures a SLAM filter.
type Option func(*SLAM)

// WithMap sets a preallocated map matrix.
// This option overrides WithWidth and WithHeight.
func WithMap(mapGrid matTypes.Matrix) Option {
	return func(s *SLAM) {
		if mapGrid == nil {
			panic("slam: mapGrid cannot be nil")
		}
		rows := mapGrid.Rows()
		cols := mapGrid.Cols()
		if rows == 0 || cols == 0 {
			panic("slam: mapGrid must be non-empty")
		}
		s.mapGrid = mapGrid
		s.mapRows = rows
		s.mapCols = cols
	}
}

// WithWidth sets the map width (columns).
func WithWidth(width int) Option {
	return func(s *SLAM) {
		if width <= 0 {
			panic("slam: width must be positive")
		}
		s.mapCols = width
	}
}

// WithHeight sets the map height (rows).
func WithHeight(height int) Option {
	return func(s *SLAM) {
		if height <= 0 {
			panic("slam: height must be positive")
		}
		s.mapRows = height
	}
}

// WithResolution sets the map resolution (grid cell size in meters).
func WithResolution(resolution float32) Option {
	return func(s *SLAM) {
		if resolution <= 0 {
			panic("slam: resolution must be positive")
		}
		s.mapResolution = resolution
	}
}

// WithOnline enables or disables online map building.
func WithOnline(enabled bool) Option {
	return func(s *SLAM) {
		s.enableMapping = enabled
	}
}

// SLAM implements a SLAM filter for robot localization using ray-based sensors.
// It follows SOLID principles with clear separation of concerns:
// - Map management (SetMap, Map)
// - Pose estimation (via EKF)
// - Measurement processing (Update)
type SLAM struct {
	// Ray configuration
	rayAngles vec.Vector          // Ray angles (radians) relative to robot heading
	rayDirs   *grid.RayDirections // Pre-computed ray directions

	// Map - stored as interface for flexibility
	mapGrid       matTypes.Matrix // Occupancy grid map (rows x cols)
	logOddsMap    mat.Matrix      // Log-odds representation for map building (needs direct access)
	mapResolution float32         // Grid cell size in meters
	mapOriginX    float32         // X coordinate of map origin (meters)
	mapOriginY    float32         // Y coordinate of map origin (meters)

	// State
	pose mat.Matrix3x3 // Robot pose as 3x3 transformation matrix

	// Localization filter
	ekf *ekalman.EKF // Extended Kalman Filter for pose estimation

	// Temporary storage (pre-allocated to avoid allocations in hot path)
	expectedDistances vec.Vector // Expected distances for each ray
	measuredDistances vec.Vector // Measured distances (input)
	poseVec           vec.Vector // Pose as [px, py, heading] for EKF

	// Configuration
	maxRange            float32 // Maximum sensor range (meters)
	enableMapping       bool    // Enable online map building
	useOptimizedRaycast bool    // Use optimized ray casting

	// Filter interface
	inputMatrix matTypes.Matrix // Input matrix: 2 rows x numRays [angles; distances]
	outputPose  mat.Matrix3x3   // Output pose as 3x3 transformation matrix
	targetPose  mat.Matrix3x3   // Target pose

	// Dimensions
	numRays int // Number of rays
	mapRows int // Map rows
	mapCols int // Map columns
}

// New creates a new SLAM filter with the given options.
//
// Required:
//   - rayAngles: Array of ray angles in radians
//   - Either WithMap() or both WithWidth() and WithHeight()
//   - WithResolution(): Grid cell size in meters
//
// Optional:
//   - WithOnline(): Enable online map building (default: false)
//   - mapOriginX, mapOriginY: Map origin coordinates
func New(rayAngles vec.Vector, mapOriginX, mapOriginY float32, opts ...Option) *SLAM {
	if len(rayAngles) == 0 {
		panic("slam: rayAngles must have at least one element")
	}

	numRays := len(rayAngles)

	// Initialize with defaults
	s := &SLAM{
		rayAngles:           rayAngles,
		mapResolution:       DefaultResolution,
		mapOriginX:          mapOriginX,
		mapOriginY:          mapOriginY,
		maxRange:            DefaultMaxRange,
		enableMapping:       false,
		useOptimizedRaycast: true,
		numRays:             numRays,
		mapRows:             DefaultMapSize,
		mapCols:             DefaultMapSize,
		poseVec:             vec.New(3), // Pre-allocate for hot path
		expectedDistances:   vec.New(numRays),
		measuredDistances:   vec.New(numRays),
	}

	// Apply options
	for _, opt := range opts {
		opt(s)
	}

	// Initialize pose matrix
	identityPose := mat.Matrix3x3{
		{1, 0, mapOriginX},
		{0, 1, mapOriginY},
		{0, 0, 1},
	}
	s.pose = identityPose
	s.outputPose = identityPose
	s.targetPose = identityPose

	// Initialize pose vector
	s.poseVec[0] = mapOriginX
	s.poseVec[1] = mapOriginY
	s.poseVec[2] = 0.0

	// Create map if not provided
	if s.mapGrid == nil {
		if s.mapRows <= 0 || s.mapCols <= 0 {
			panic("slam: map dimensions must be set via WithMap() or WithWidth()/WithHeight()")
		}
		s.mapGrid = mat.New(s.mapRows, s.mapCols)
		// Initialize to free space (0.0) - mat.New initializes to zero
	}

	// Pre-compute ray directions
	s.rayDirs = grid.NewRayDirections(rayAngles)

	// Initialize EKF
	s.initEKF()

	// Initialize input matrix (2 rows x numRays)
	s.inputMatrix = mat.New(2, numRays)

	return s
}

// initEKF initializes the Extended Kalman Filter for pose estimation.
func (s *SLAM) initEKF() {
	n, m := 3, s.numRays // state dim, measurement dim

	// State transition: static pose (can be extended with odometry)
	fFunc := func(x, u vec.Vector, dt float32) vec.Vector {
		next := vec.New(3)
		copy(next, x)
		return next
	}

	// Measurement function: ray casting (nonlinear, requires EKF)
	// Need to convert matTypes.Matrix to mat.Matrix for grid functions
	hFunc := func(x vec.Vector) vec.Vector {
		distances := vec.New(s.numRays)
		mapMat := s.getMapMatrix() // Get mat.Matrix for grid functions
		grid.RayCastAllOptimized(x, s.rayDirs, mapMat, s.mapResolution, s.mapOriginX, s.mapOriginY, s.maxRange, distances)
		return distances
	}

	// Process noise covariance
	Q := mat.New(3, 3)
	Q.Eye()
	Q.MulC(0.01)

	// Measurement noise covariance
	R := mat.New(m, m)
	R.Eye()
	R.MulC(0.1)

	// Create EKF with numerical Jacobians
	s.ekf = ekalman.New(n, m, fFunc, hFunc, nil, nil, Q, R)

	// Set initial state
	s.ekf.SetState(s.poseVec)

	// Set initial covariance
	initialP := mat.New(3, 3)
	initialP.Eye()
	initialP.MulC(1.0)
	s.ekf.SetCovariance(initialP)
}

// getMapMatrix converts matTypes.Matrix to mat.Matrix for grid functions.
// Grid functions require mat.Matrix (slice-backed), so we need to extract it.
func (s *SLAM) getMapMatrix() mat.Matrix {
	switch v := s.mapGrid.(type) {
	case mat.Matrix:
		return v
	default:
		// Clone to mat.Matrix for grid functions
		rows := s.mapGrid.Rows()
		cols := s.mapGrid.Cols()
		mapMat := mat.New(rows, cols)
		for i := 0; i < rows; i++ {
			row := s.mapGrid.Row(i)
			rowVec := row.View().(vec.Vector)
			for j := 0; j < cols && j < rowVec.Len(); j++ {
				mapMat[i][j] = rowVec[j]
			}
		}
		return mapMat
	}
}

// poseToVector converts 3x3 pose matrix to [px, py, heading] vector.
func (s *SLAM) poseToVector(pose mat.Matrix3x3) vec.Vector {
	s.poseVec[0] = pose[0][2]                           // px
	s.poseVec[1] = pose[1][2]                           // py
	s.poseVec[2] = math32.Atan2(pose[1][0], pose[0][0]) // heading
	return s.poseVec
}

// vectorToPose converts [px, py, heading] vector to 3x3 pose matrix.
func (s *SLAM) vectorToPose(v vec.Vector) mat.Matrix3x3 {
	px := v[0]
	py := v[1]
	heading := v[2]
	cosH := math32.Cos(heading)
	sinH := math32.Sin(heading)
	return mat.Matrix3x3{
		{cosH, -sinH, px},
		{sinH, cosH, py},
		{0, 0, 1},
	}
}

// SetMaxRange sets the maximum sensor range in meters.
func (s *SLAM) SetMaxRange(maxRange float32) {
	if maxRange <= 0 {
		panic("slam: maxRange must be positive")
	}
	s.maxRange = maxRange
}

// SetPose sets the robot pose as a 3x3 transformation matrix.
func (s *SLAM) SetPose(pose mat.Matrix3x3) {
	s.pose = pose
	s.outputPose = pose
	s.poseToVector(pose)
	s.ekf.SetState(s.poseVec)
}

// GetPose returns the current robot pose as a 3x3 transformation matrix.
func (s *SLAM) GetPose() mat.Matrix3x3 {
	return s.outputPose
}

// SetMap sets the occupancy grid map.
func (s *SLAM) SetMap(mapGrid matTypes.Matrix) {
	if mapGrid == nil {
		panic("slam: mapGrid cannot be nil")
	}
	rows := mapGrid.Rows()
	cols := mapGrid.Cols()
	if rows == 0 || cols == 0 {
		panic("slam: mapGrid must be non-empty")
	}
	s.mapGrid = mapGrid
	s.mapRows = rows
	s.mapCols = cols
	// Reset log-odds map if mapping was enabled
	if s.logOddsMap != nil {
		s.logOddsMap = nil
		if s.enableMapping {
			s.initLogOddsMap()
		}
	}
}

// Map returns the current occupancy grid map.
func (s *SLAM) Map() matTypes.Matrix {
	return s.mapGrid
}

// initLogOddsMap initializes the log-odds map from the current occupancy grid.
func (s *SLAM) initLogOddsMap() {
	s.logOddsMap = mat.New(s.mapRows, s.mapCols)
	for i := 0; i < s.mapRows; i++ {
		row := s.mapGrid.Row(i)
		rowVec := row.View().(vec.Vector)
		for j := 0; j < s.mapCols && j < rowVec.Len(); j++ {
			p := rowVec[j]
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

// SetMappingEnabled enables or disables online map building.
func (s *SLAM) SetMappingEnabled(enabled bool) {
	s.enableMapping = enabled
	if enabled && s.logOddsMap == nil {
		s.initLogOddsMap()
	}
}

// Online returns whether online map building is enabled.
func (s *SLAM) Online() bool {
	return s.enableMapping
}

// SetOnline enables or disables online map building.
func (s *SLAM) SetOnline(enabled bool) {
	s.SetMappingEnabled(enabled)
}

// SetOptimizedRaycast enables or disables optimized ray casting.
func (s *SLAM) SetOptimizedRaycast(enabled bool) {
	s.useOptimizedRaycast = enabled
}

// extractInputMatrix extracts angles and distances from input matrix using direct interface methods.
func (s *SLAM) extractInputMatrix(input matTypes.Matrix) (angles, distances vec.Vector) {
	// Use interface methods directly - no type assertion needed!
	anglesRow := input.Row(0)
	distancesRow := input.Row(1)

	// Return views as vectors
	return anglesRow.View().(vec.Vector), distancesRow.View().(vec.Vector)
}

// UpdateMeasurement updates the SLAM filter with angle and distance measurements.
// Uses direct matrix interface methods to avoid allocations.
func (s *SLAM) UpdateMeasurement(angles, distances vec.Vector) {
	if len(angles) != len(distances) {
		panic("slam: angles and distances must have the same length")
	}
	if len(angles) != s.numRays {
		panic("slam: angles vector must have same length as rayAngles")
	}

	// Copy measurements to pre-allocated buffers
	copy(s.measuredDistances, distances)

	// Update pose vector from current pose matrix
	s.poseToVector(s.pose)

	// Update map if mapping is enabled
	if s.enableMapping && s.logOddsMap != nil {
		mapMat := s.getMapMatrix() // Get mat.Matrix for grid functions
		InverseSensorModelAll(
			mapMat, s.poseVec, angles, distances,
			s.mapResolution, s.mapOriginX, s.mapOriginY, s.maxRange,
			s.logOddsMap,
		)
	}

	// Update EKF with measurements
	s.ekf.UpdateMeasurement(distances)

	// Get updated pose from EKF
	updatedPoseVec := s.ekf.Output()
	s.pose = s.vectorToPose(updatedPoseVec)
	s.outputPose = s.pose

	// Compute expected distances
	mapMat := s.getMapMatrix() // Get mat.Matrix for grid functions
	if s.useOptimizedRaycast {
		grid.RayCastAllOptimized(updatedPoseVec, s.rayDirs, mapMat, s.mapResolution, s.mapOriginX, s.mapOriginY, s.maxRange, s.expectedDistances)
	} else {
		expected := grid.RayCastAll(updatedPoseVec, s.rayAngles, mapMat, s.mapResolution, s.mapOriginX, s.mapOriginY, s.maxRange)
		copy(s.expectedDistances, expected)
	}
}

// Reset resets the filter state to the map origin.
func (s *SLAM) Reset() {
	identityPose := mat.Matrix3x3{
		{1, 0, s.mapOriginX},
		{0, 1, s.mapOriginY},
		{0, 0, 1},
	}
	s.pose = identityPose
	s.outputPose = identityPose
	s.targetPose = identityPose

	s.poseVec[0] = s.mapOriginX
	s.poseVec[1] = s.mapOriginY
	s.poseVec[2] = 0.0

	s.ekf.SetState(s.poseVec)
	initialP := mat.New(3, 3)
	initialP.Eye()
	initialP.MulC(1.0)
	s.ekf.SetCovariance(initialP)
}

// Update implements the Filter interface.
// Input matrix format: 2 rows x numRays columns
//
//	Row 0: [angle1, angle2, ..., angleN] - ray angles in radians
//	Row 1: [distance1, distance2, ..., distanceN] - distances for each ray (meters)
//
// Output: 3x3 transformation matrix representing pose (translation + orientation)
func (s *SLAM) Update(timestep float32, input matTypes.Matrix) {
	if input == nil {
		return
	}

	// Store input matrix (use interface directly!)
	s.inputMatrix = input

	// Extract angles and distances using direct interface methods
	angles, distances := s.extractInputMatrix(input)

	// Update with measurements
	s.UpdateMeasurement(angles, distances)

	// Update output
	s.outputPose = s.pose
}

// Input returns the input matrix (2 rows: angles and distances).
func (s *SLAM) Input() matTypes.Matrix {
	return s.inputMatrix
}

// Output returns the estimated pose as a 3x3 transformation matrix.
func (s *SLAM) Output() mat.Matrix3x3 {
	return s.outputPose
}

// GetTarget returns the target pose as a 3x3 transformation matrix.
func (s *SLAM) GetTarget() mat.Matrix3x3 {
	return s.targetPose
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
