package grid

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// RayDirections stores pre-computed ray directions for efficiency.
type RayDirections struct {
	cosAngles vec.Vector // Pre-computed cosines of ray angles
	sinAngles vec.Vector // Pre-computed sines of ray angles
	numRays   int        // Number of rays
}

// NewRayDirections pre-computes ray directions for efficiency.
// This avoids repeated trigonometric computations in the hot path.
func NewRayDirections(rayAngles vec.Vector) *RayDirections {
	numRays := len(rayAngles)
	cosAngles := vec.New(numRays)
	sinAngles := vec.New(numRays)

	for i := 0; i < numRays; i++ {
		cosAngles[i] = math32.Cos(rayAngles[i])
		sinAngles[i] = math32.Sin(rayAngles[i])
	}

	return &RayDirections{
		cosAngles: cosAngles,
		sinAngles: sinAngles,
		numRays:   numRays,
	}
}

// RayCast casts a single ray against an occupancy grid map and returns the distance to the first obstacle.
// Returns the distance in meters, or maxRange if no obstacle is found.
//
// Parameters:
//   - pose: Robot pose [px, py, heading] in world coordinates
//   - rayAngle: Ray angle relative to robot heading (radians)
//   - mapGrid: Occupancy grid map (rows x cols)
//   - mapResolution: Grid cell size in meters
//   - mapOriginX: X coordinate of map origin in world coordinates (meters)
//   - mapOriginY: Y coordinate of map origin in world coordinates (meters)
//   - maxRange: Maximum sensor range in meters
//
// Returns distance to first obstacle in meters, or maxRange if no obstacle found.
func RayCast(
	pose vec.Vector,
	rayAngle float32,
	mapGrid mat.Matrix,
	mapResolution, mapOriginX, mapOriginY, maxRange float32,
) float32 {
	if len(pose) < 3 {
		return maxRange
	}

	px := pose[0]
	py := pose[1]
	heading := pose[2]

	// Convert world coordinates to grid coordinates
	gridX := (px - mapOriginX) / mapResolution
	gridY := (py - mapOriginY) / mapResolution

	// Check if starting position is within map bounds
	if gridX < 0 || gridX >= float32(len(mapGrid[0])) || gridY < 0 || gridY >= float32(len(mapGrid)) {
		return maxRange
	}

	// Compute ray direction (absolute angle)
	rayAngleAbs := heading + rayAngle
	dirX := math32.Cos(rayAngleAbs)
	dirY := math32.Sin(rayAngleAbs)

	// Cast ray step by step
	step := mapResolution / 2.0 // Half cell for accuracy
	distance := float32(0)

	maxRangeGrid := maxRange / mapResolution

	for distance < maxRangeGrid {
		currentX := gridX + distance*dirX
		currentY := gridY + distance*dirY

		// Check bounds
		if currentX < 0 || currentX >= float32(len(mapGrid[0])) ||
			currentY < 0 || currentY >= float32(len(mapGrid)) {
			return maxRange // Ray out of bounds
		}

		// Get occupancy at cell
		cellX := int(currentX)
		cellY := int(currentY)

		// Check bounds again for integer indices
		if cellX < 0 || cellX >= len(mapGrid[0]) || cellY < 0 || cellY >= len(mapGrid) {
			return maxRange
		}

		occupancy := mapGrid[cellY][cellX]

		// If occupied (threshold > 0.5), hit found
		if occupancy > 0.5 {
			// Return distance in meters
			return distance * mapResolution
		}

		distance += step / mapResolution // Step in grid units
	}

	// No hit found within max range
	return maxRange
}

// RayCastOptimized casts a single ray using pre-computed directions and Bresenham algorithm.
// This is the optimized version for embedded systems.
func RayCastOptimized(
	pose vec.Vector,
	rayIdx int,
	rayDirs *RayDirections,
	mapGrid mat.Matrix,
	mapResolution, mapOriginX, mapOriginY, maxRange float32,
) float32 {
	if len(pose) < 3 || rayIdx < 0 || rayIdx >= rayDirs.numRays {
		return maxRange
	}

	px := pose[0]
	py := pose[1]
	heading := pose[2]

	// Convert world coordinates to grid coordinates
	gridX := (px - mapOriginX) / mapResolution
	gridY := (py - mapOriginY) / mapResolution

	// Check if starting position is within map bounds
	mapRows := len(mapGrid)
	mapCols := len(mapGrid[0])

	startX := int(gridX)
	startY := int(gridY)

	if startX < 0 || startX >= mapCols || startY < 0 || startY >= mapRows {
		return maxRange
	}

	// Use pre-computed ray direction and apply robot heading
	rayCos := rayDirs.cosAngles[rayIdx]
	raySin := rayDirs.sinAngles[rayIdx]
	cosHeading := math32.Cos(heading)
	sinHeading := math32.Sin(heading)

	// Rotate ray direction by robot heading
	dirX := rayCos*cosHeading - raySin*sinHeading
	dirY := raySin*cosHeading + rayCos*sinHeading

	// Compute end point at max range using Bresenham
	maxRangeGrid := maxRange / mapResolution
	endGridX := gridX + dirX*maxRangeGrid
	endGridY := gridY + dirY*maxRangeGrid
	endX := int(endGridX)
	endY := int(endGridY)

	// Bresenham's line algorithm (optimized)
	dx := abs(endX - startX)
	dy := abs(endY - startY)
	sx := 1
	sy := 1

	if endX < startX {
		sx = -1
	}
	if endY < startY {
		sy = -1
	}

	err := dx - dy
	x := startX
	y := startY

	for {
		// Check bounds
		if x < 0 || x >= mapCols || y < 0 || y >= mapRows {
			return maxRange
		}

		// Check occupancy at cell (early termination)
		if mapGrid[y][x] > 0.5 {
			// Distance from start to current cell
			cellDistX := float32(x-startX) * mapResolution
			cellDistY := float32(y-startY) * mapResolution
			distance := math32.Sqrt(cellDistX*cellDistX + cellDistY*cellDistY)
			return distance
		}

		// Check if we've reached the end
		if x == endX && y == endY {
			break
		}

		e2 := 2 * err

		if e2 > -dy {
			err -= dy
			x += sx
		}

		if e2 < dx {
			err += dx
			y += sy
		}
	}

	// No obstacle found
	return maxRange
}

// RayCastAll casts all rays against the occupancy grid map.
// Returns a vector of expected distances for each ray.
func RayCastAll(
	pose vec.Vector,
	rayAngles vec.Vector,
	mapGrid mat.Matrix,
	mapResolution, mapOriginX, mapOriginY, maxRange float32,
) vec.Vector {
	numRays := len(rayAngles)
	distances := vec.New(numRays)

	for i := 0; i < numRays; i++ {
		distances[i] = RayCast(pose, rayAngles[i], mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange)
	}

	return distances
}

// RayCastAllOptimized casts all rays using pre-computed directions and optimized algorithm.
// This version is optimized for embedded systems with minimal allocations.
func RayCastAllOptimized(
	pose vec.Vector,
	rayDirs *RayDirections,
	mapGrid mat.Matrix,
	mapResolution, mapOriginX, mapOriginY, maxRange float32,
	distances vec.Vector, // Pre-allocated destination vector
) vec.Vector {
	if len(distances) < rayDirs.numRays {
		distances = vec.New(rayDirs.numRays)
	}

	for i := 0; i < rayDirs.numRays; i++ {
		distances[i] = RayCastOptimized(pose, i, rayDirs, mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange)
	}

	return distances
}

// RayProjection projects a ray of specified length and checks if it intersects an obstacle.
// Returns true if ray intersects an obstacle within the specified length, false otherwise.
//
// Parameters:
//   - startX, startY: Starting position in grid coordinates (integers)
//   - dirX, dirY: Ray direction vector (normalized)
//   - length: Maximum ray length in grid units
//   - mapGrid: Occupancy grid map (rows x cols)
//   - occupancyThreshold: Threshold for considering a cell occupied (default: 0.5)
//
// Returns true if intersection found, false otherwise.
func RayProjection(
	startX, startY int,
	dirX, dirY, length float32,
	mapGrid mat.Matrix,
	occupancyThreshold float32,
) bool {
	if occupancyThreshold <= 0 {
		occupancyThreshold = 0.5
	}

	mapRows := len(mapGrid)
	mapCols := len(mapGrid[0])

	if startX < 0 || startX >= mapCols || startY < 0 || startY >= mapRows {
		return false
	}

	// Compute end point
	endX := int(float32(startX) + dirX*length)
	endY := int(float32(startY) + dirY*length)

	// Bresenham's line algorithm
	dx := abs(endX - startX)
	dy := abs(endY - startY)
	sx := 1
	sy := 1

	if endX < startX {
		sx = -1
	}
	if endY < startY {
		sy = -1
	}

	err := dx - dy
	x := startX
	y := startY

	maxDistanceSq := length * length

	for {
		// Check bounds
		if x < 0 || x >= mapCols || y < 0 || y >= mapRows {
			return false
		}

		// Check distance (in grid units, squared)
		dxF := float32(x - startX)
		dyF := float32(y - startY)
		distSq := dxF*dxF + dyF*dyF
		if distSq > maxDistanceSq {
			return false
		}

		// Check occupancy at cell
		if mapGrid[y][x] > occupancyThreshold {
			return true // Intersection found
		}

		// Check if we've reached the end
		if x == endX && y == endY {
			break
		}

		e2 := 2 * err

		if e2 > -dy {
			err -= dy
			x += sx
		}

		if e2 < dx {
			err += dx
			y += sy
		}
	}

	// No intersection found
	return false
}

// GetOccupancyReadings gets occupancy readings for given angles using optimized raycasting.
// Returns a vector of occupancy values (1.0 = occupied, 0.0 = free) at each ray endpoint.
//
// Parameters:
//   - pose: Robot pose [px, py, heading] in world coordinates
//   - rayAngles: Vector of ray angles relative to robot heading (radians)
//   - mapGrid: Occupancy grid map (rows x cols)
//   - mapResolution: Grid cell size in meters
//   - mapOriginX: X coordinate of map origin (meters)
//   - mapOriginY: Y coordinate of map origin (meters)
//   - maxRange: Maximum sensor range (meters)
//   - rayDirs: Pre-computed ray directions (optional, will compute if nil)
//
// Returns vector of occupancy values at ray endpoints.
func GetOccupancyReadings(
	pose vec.Vector,
	rayAngles vec.Vector,
	mapGrid mat.Matrix,
	mapResolution, mapOriginX, mapOriginY, maxRange float32,
	rayDirs *RayDirections,
) vec.Vector {
	numRays := len(rayAngles)
	readings := vec.New(numRays)

	// Pre-compute directions if not provided
	if rayDirs == nil {
		rayDirs = NewRayDirections(rayAngles)
	}

	// Cast rays and get occupancy at endpoints
	for i := 0; i < numRays; i++ {
		distance := RayCastOptimized(pose, i, rayDirs, mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange)

		// If distance is less than maxRange, ray hit something
		if distance < maxRange {
			// Get occupancy at hit point
			px := pose[0]
			py := pose[1]
			heading := pose[2]

			rayCos := rayDirs.cosAngles[i]
			raySin := rayDirs.sinAngles[i]
			cosHeading := math32.Cos(heading)
			sinHeading := math32.Sin(heading)

			dirX := rayCos*cosHeading - raySin*sinHeading
			dirY := raySin*cosHeading + rayCos*sinHeading

			// Compute hit point in world coordinates
			hitX := px + dirX*distance
			hitY := py + dirY*distance

			// Convert to grid coordinates
			gridX := int((hitX - mapOriginX) / mapResolution)
			gridY := int((hitY - mapOriginY) / mapResolution)

			// Get occupancy at hit point
			if gridX >= 0 && gridX < len(mapGrid[0]) && gridY >= 0 && gridY < len(mapGrid) {
				readings[i] = mapGrid[gridY][gridX]
			} else {
				readings[i] = 0.0 // Out of bounds
			}
		} else {
			readings[i] = 0.0 // No hit (free space at max range)
		}
	}

	return readings
}

// abs returns absolute value of an integer.
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
