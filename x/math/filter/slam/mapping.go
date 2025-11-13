package slam

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

const (
	// Inverse sensor model parameters
	defaultPFree     = float32(0.3)  // Probability of free space along ray
	defaultPOccupied = float32(0.7)  // Probability of occupied at measurement
	defaultPPrior    = float32(0.5)  // Prior probability (unknown)
	defaultAlpha     = float32(1.0)  // Sensor model parameter (free space)
	defaultBeta      = float32(0.05) // Sensor model parameter (occupied)
)

// InverseSensorModel applies inverse sensor model to update occupancy grid.
// Uses log-odds representation for numerical stability and efficiency.
//
// Parameters:
//   - mapGrid: Occupancy grid map (rows x cols)
//   - pose: Robot pose [px, py, heading]
//   - rayAngle: Ray angle relative to robot heading (radians)
//   - measurement: Distance measurement (meters)
//   - mapResolution: Grid cell size in meters
//   - mapOriginX: Map origin X coordinate (meters)
//   - mapOriginY: Map origin Y coordinate (meters)
//   - maxRange: Maximum sensor range (meters)
//   - logOddsMap: Log-odds representation of map (can be nil, will create if needed)
func InverseSensorModel(
	mapGrid mat.Matrix,
	pose vec.Vector,
	rayAngle float32,
	measurement float32,
	mapResolution, mapOriginX, mapOriginY, maxRange float32,
	logOddsMap mat.Matrix,
) mat.Matrix {
	if len(pose) < 3 {
		return logOddsMap
	}

	px := pose[0]
	py := pose[1]
	heading := pose[2]

	// Convert world coordinates to grid coordinates
	gridX := (px - mapOriginX) / mapResolution
	gridY := (py - mapOriginY) / mapResolution

	mapRows := len(mapGrid)
	mapCols := len(mapGrid[0])

	startX := int(gridX)
	startY := int(gridY)

	// Check bounds
	if startX < 0 || startX >= mapCols || startY < 0 || startY >= mapRows {
		return logOddsMap
	}

	// Compute ray direction
	rayAngleAbs := heading + rayAngle
	dirX := math32.Cos(rayAngleAbs)
	dirY := math32.Sin(rayAngleAbs)

	// Initialize log-odds map if needed
	if logOddsMap == nil || len(logOddsMap) != mapRows || len(logOddsMap[0]) != mapCols {
		logOddsMap = mat.New(mapRows, mapCols)
		// Initialize from current map
		for i := 0; i < mapRows; i++ {
			for j := 0; j < mapCols; j++ {
				p := mapGrid[i][j]
				if p < 0.01 {
					p = 0.01 // Avoid log(0)
				}
				if p > 0.99 {
					p = 0.99 // Avoid log(inf)
				}
				logOddsMap[i][j] = math32.Log(p / (1.0 - p))
			}
		}
	}

	// Clamp measurement to max range
	if measurement > maxRange {
		measurement = maxRange
	}

	// Compute cells along ray using Bresenham
	maxRangeGrid := maxRange / mapResolution
	endGridX := gridX + dirX*maxRangeGrid
	endGridY := gridY + dirY*maxRangeGrid
	endX := int(endGridX)
	endY := int(endGridY)

	dx := absInt(endX - startX)
	dy := absInt(endY - startY)
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

	measurementGrid := measurement / mapResolution

	// Log-odds update values
	logOddsFree := math32.Log(defaultPFree / (1.0 - defaultPFree))
	logOddsOccupied := math32.Log(defaultPOccupied / (1.0 - defaultPOccupied))
	logOddsPrior := math32.Log(defaultPPrior / (1.0 - defaultPPrior))

	distanceGrid := float32(0)

	for {
		// Check bounds
		if x < 0 || x >= mapCols || y < 0 || y >= mapRows {
			break
		}

		// Check if we've reached measurement distance
		if distanceGrid >= measurementGrid {
			// At or beyond measurement - mark as occupied if at measurement point
			if distanceGrid <= measurementGrid+defaultBeta {
				logOddsMap[y][x] += logOddsOccupied - logOddsPrior
				// Clamp log-odds
				if logOddsMap[y][x] > 10.0 {
					logOddsMap[y][x] = 10.0
				}
				if logOddsMap[y][x] < -10.0 {
					logOddsMap[y][x] = -10.0
				}
			}
			break
		}

		// Free space along ray (before measurement point)
		logOddsMap[y][x] += logOddsFree - logOddsPrior

		// Clamp log-odds to avoid numerical issues
		if logOddsMap[y][x] > 10.0 {
			logOddsMap[y][x] = 10.0
		}
		if logOddsMap[y][x] < -10.0 {
			logOddsMap[y][x] = -10.0
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

		// Update distance (approximate: one cell step)
		distanceGrid += 1.0
	}

	return logOddsMap
}

// absInt returns absolute value of an integer.
func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// LogOddsToProbability converts log-odds to probability.
func LogOddsToProbability(logOdds float32) float32 {
	return 1.0 / (1.0 + math32.Exp(-logOdds))
}

// UpdateMapFromLogOdds updates occupancy grid from log-odds map.
func UpdateMapFromLogOdds(mapGrid mat.Matrix, logOddsMap mat.Matrix) {
	mapRows := len(mapGrid)
	mapCols := len(mapGrid[0])

	for i := 0; i < mapRows; i++ {
		for j := 0; j < mapCols; j++ {
			mapGrid[i][j] = LogOddsToProbability(logOddsMap[i][j])
		}
	}
}

// InverseSensorModelAll applies inverse sensor model to all rays.
// Updates map from all measurements.
func InverseSensorModelAll(
	mapGrid mat.Matrix,
	pose vec.Vector,
	rayAngles vec.Vector,
	measurements vec.Vector,
	mapResolution, mapOriginX, mapOriginY, maxRange float32,
	logOddsMap mat.Matrix,
) mat.Matrix {
	if len(rayAngles) != len(measurements) {
		return logOddsMap
	}

	numRays := len(rayAngles)

	// Initialize log-odds map if needed
	if logOddsMap == nil || len(logOddsMap) != len(mapGrid) || len(logOddsMap[0]) != len(mapGrid[0]) {
		logOddsMap = mat.New(len(mapGrid), len(mapGrid[0]))
		// Initialize from current map
		for i := 0; i < len(mapGrid); i++ {
			for j := 0; j < len(mapGrid[0]); j++ {
				p := mapGrid[i][j]
				if p < 0.01 {
					p = 0.01
				}
				if p > 0.99 {
					p = 0.99
				}
				logOddsMap[i][j] = math32.Log(p / (1.0 - p))
			}
		}
	}

	// Update map from each ray
	for i := 0; i < numRays; i++ {
		logOddsMap = InverseSensorModel(
			mapGrid, pose, rayAngles[i], measurements[i],
			mapResolution, mapOriginX, mapOriginY, maxRange,
			logOddsMap,
		)
	}

	// Update occupancy grid from log-odds
	UpdateMapFromLogOdds(mapGrid, logOddsMap)

	return logOddsMap
}
