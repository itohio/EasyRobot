# Simple SLAM Filter

This package implements a simple Simultaneous Localization And Mapping (SLAM) filter for robot localization using ray-based sensors (e.g., LiDAR, sonar) and a pre-given occupancy grid map.

## Overview

The SLAM filter localizes a robot by:
1. Casting rays from the estimated pose against an occupancy grid map
2. Computing expected distances for each ray
3. Comparing expected distances with measured distances
4. Updating the pose estimate using an Extended Kalman Filter (EKF)

## Features

- **Arbitrary number of rays**: Support any number of sensor rays
- **Pre-given map**: Occupancy grid map provided beforehand (can be updated online)
- **Online map building**: Configurable option to build/update map from measurements
- **EKF-based localization**: Uses Extended Kalman Filter for pose estimation
- **Optimized ray casting**: Pre-computed directions, Bresenham algorithm for embedded
- **Real-time capable**: Optimized for embedded systems with minimal allocations

## Usage Example

### Basic Usage

```go
package main

import (
    "math"
    "github.com/itohio/EasyRobot/x/math/filter/slam"
    "github.com/itohio/EasyRobot/x/math/mat"
    "github.com/itohio/EasyRobot/x/math/vec"
)

func main() {
    // Configure rays (360 rays, 1 degree spacing)
    numRays := 360
    rayAngles := vec.New(numRays)
    for i := 0; i < numRays; i++ {
        rayAngles[i] = float32(i) * math.Pi / 180.0
    }

    // Load occupancy grid map (100x100 cells, 0.1m resolution)
    mapGrid := mat.New(100, 100)
    // ... populate map from file or generate ...
    // Values: 0 = free, 1 = occupied, 0.5 = unknown
    
    mapResolution := float32(0.1)  // 10cm per cell
    mapOriginX := float32(0.0)
    mapOriginY := float32(0.0)

    // Create SLAM filter
    slamFilter := slam.New(rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY)

    // Set initial pose (optional, defaults to map origin)
    slamFilter.SetPose(vec.NewFrom(5.0, 5.0, 0.0))  // Start at (5m, 5m), facing east

    // Set maximum sensor range (optional, defaults to 10m)
    slamFilter.SetMaxRange(5.0)  // 5 meter range

    // Enable online map building (optional, defaults to false)
    slamFilter.SetMappingEnabled(true)

    // Use optimized ray casting (default: true, uses pre-computed directions)
    slamFilter.SetOptimizedRaycast(true)

    // Update with distance measurements
    distances := vec.New(numRays)
    // ... populate distances from sensor ...
    slamFilter.UpdateMeasurement(distances)
    
    // Map is automatically updated if mapping is enabled

    // Get estimated pose
    pose := slamFilter.GetOutput()
    px := pose[0]
    py := pose[1]
    heading := pose[2]

    // Get expected distances (for debugging)
    expected := slamFilter.GetExpectedDistances()

    // Get residuals (measured - expected, for debugging)
    residuals := slamFilter.GetResiduals()
}
```

### Creating a Map

```go
// Create a simple map with obstacles
mapGrid := mat.New(100, 100)

// Initialize all cells to free (0)
for i := range mapGrid {
    for j := range mapGrid[i] {
        mapGrid[i][j] = 0.0  // Free
    }
}

// Add some obstacles
// Wall along bottom edge
for j := 0; j < 100; j++ {
    mapGrid[0][j] = 1.0  // Occupied
}

// Wall along left edge
for i := 0; i < 100; i++ {
    mapGrid[i][0] = 1.0  // Occupied
}

// Obstacle in center
for i := 45; i < 55; i++ {
    for j := 45; j < 55; j++ {
        mapGrid[i][j] = 1.0  // Occupied
    }
}
```

### Using Filter Interface

```go
// SLAM filter implements the Filter interface
var filter filter.Filter = slamFilter

// Set measurement input
copy(filter.GetInput(), distances)

// Update with timestep
filter.Update(0.1)  // 0.1 second timestep

// Get estimated pose
pose := filter.GetOutput()
```

### Different Ray Configurations

```go
// 4 rays: cardinal directions
rayAngles := vec.NewFrom(0.0, math.Pi/2, math.Pi, 3*math.Pi/2)

// 8 rays: cardinal and diagonal
rayAngles := vec.New(8)
for i := 0; i < 8; i++ {
    rayAngles[i] = float32(i) * math.Pi / 4.0
}

// Custom ray configuration
rayAngles := vec.NewFrom(
    0.0,           // East
    math.Pi / 6,   // 30 degrees
    math.Pi / 4,   // 45 degrees
    math.Pi / 2,   // North
    // ... more angles
)
```

## API Reference

### New

Creates a new SLAM filter.

```go
func New(
    rayAngles vec.Vector,        // Ray angles in radians
    mapGrid mat.Matrix,          // Occupancy grid map
    mapResolution float32,       // Grid cell size in meters
    mapOriginX float32,          // Map origin X coordinate (meters)
    mapOriginY float32,          // Map origin Y coordinate (meters)
) *SLAM
```

### SetPose

Sets the initial robot pose.

```go
func (s *SLAM) SetPose(pose vec.Vector) *SLAM
// pose: [px, py, heading] in world coordinates
```

### SetMaxRange

Sets the maximum sensor range in meters.

```go
func (s *SLAM) SetMaxRange(maxRange float32) *SLAM
```

### UpdateMeasurement

Updates the filter with distance measurements.

```go
func (s *SLAM) UpdateMeasurement(distances vec.Vector) *SLAM
// distances: Vector of distance measurements for each ray (meters)
```

### GetPose

Returns the current pose estimate.

```go
func (s *SLAM) GetPose() vec.Vector
// Returns: [px, py, heading] in world coordinates
```

### GetExpectedDistances

Returns the expected distances computed from the current pose estimate.

```go
func (s *SLAM) GetExpectedDistances() vec.Vector
```

### GetResiduals

Returns the difference between measured and expected distances.

```go
func (s *SLAM) GetResiduals() vec.Vector
// Returns: measured - expected distances
```

## Map Format

The occupancy grid map is a matrix where:
- **0.0**: Free space
- **1.0**: Occupied (obstacle)
- **0.5**: Unknown
- **Any value > 0.5**: Treated as occupied
- **Any value ≤ 0.5**: Treated as free

## Ray Angles

Ray angles are specified in radians relative to the robot heading:
- **0 radians**: Ray in the direction the robot is facing
- **Positive angles**: Rotated counterclockwise
- **Negative angles**: Rotated clockwise

Example:
- `0.0`: Ray in direction of robot heading
- `math.Pi / 2`: Ray 90 degrees to the left
- `-math.Pi / 2`: Ray 90 degrees to the right

## Coordinate System

- **Map origin**: World coordinates of the bottom-left corner of the map
- **Grid coordinates**: Integer indices into the map matrix
- **World coordinates**: Real-world coordinates in meters

Conversion:
```go
gridX = (worldX - mapOriginX) / mapResolution
gridY = (worldY - mapOriginY) / mapResolution
```

## Performance Considerations

1. **Number of rays**: More rays provide better localization but increase computation
2. **Map size**: Larger maps require more ray casting time
3. **Map resolution**: Finer resolution (smaller cells) increases accuracy but computation
4. **Max range**: Longer range increases ray casting distance

## Optimizations for Embedded Systems

### 1. Ray Casting Optimization

**Pre-computed ray directions**: 
- Ray angles cos/sin values are pre-computed at initialization
- Avoids repeated trigonometric computations in hot path
- Reduces computation time by ~50%

**Bresenham algorithm**:
- Uses discrete grid traversal instead of floating-point steps
- More efficient for longer rays
- Guarantees all cells along ray are visited exactly once

**Minimal allocations**:
- Pre-allocated temporary vectors
- Reuses memory in hot path
- Reduces garbage collection

### 2. Online Map Building

**Log-odds representation**:
- Numerically stable occupancy updates
- Additive updates (faster than probability updates)
- Avoids saturation issues

**Inverse sensor model**:
- Updates only cells along ray path
- Efficient sparse updates
- Configurable via `SetMappingEnabled()`

### 3. Other Optimizations

**Cached ray directions**: Pre-computed at initialization
**Efficient coordinate conversion**: Inline calculations
**Early termination**: Stops ray casting when obstacle found
**Optimized Bresenham**: Integer-only operations for grid traversal

## Why Extended Kalman Filter (EKF)?

**Question**: Can we substitute EKF with standard KF?

**Answer**: **No, EKF is required** because:

1. **Measurement function is nonlinear**: Ray casting distance `d = h(px, py, heading, θ, M)` cannot be expressed as linear function `H * x`
2. **Pose-dependent**: Ray path depends on robot pose nonlinearly
3. **Map interaction**: Distance depends on which cells ray traverses

**KF assumes**: `z = H * x` (linear measurement)
**EKF handles**: `z = h(x)` (nonlinear measurement) by linearizing around current pose

See `KF_VS_EKF.md` for detailed explanation.

## Limitations

1. **2D only**: Currently only supports 2D maps and poses
2. **Single hypothesis**: EKF assumes single mode (no multi-hypothesis tracking)
3. **No loop closure**: Cannot detect revisited locations
4. **No dynamic obstacles**: Assumes static obstacles (but map can be updated online)

## Future Enhancements

- Online map building (full SLAM)
- 3D support
- Particle filter for multi-hypothesis tracking
- Loop closure detection
- Dynamic obstacle handling
- Map optimization

