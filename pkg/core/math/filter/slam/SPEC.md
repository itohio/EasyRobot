# Simple SLAM Filter Specification

## Overview

The Simple SLAM filter package provides localization for robots using ray-based sensors (e.g., LiDAR, sonar) and a pre-given occupancy grid map. It estimates robot pose (position and heading) by comparing expected ray cast distances against measured distances.

## Components

### SLAM Filter (`slam.go`)

**Purpose**: Robot localization using ray-based sensors and occupancy grid map

**Description**: Implements a simple SLAM filter that localizes a robot by:
1. Casting rays from estimated pose against occupancy grid map
2. Computing expected distances for each ray
3. Comparing expected distances with measured distances
4. Updating pose estimate using Extended Kalman Filter

**Mathematical Model**:

**State Vector**:
```
x = [px, py, heading]
   - px, py: robot position (meters)
   - heading: robot orientation (radians)
```

**Measurements**:
```
z = [d0, d1, d2, ..., dN]
   - di: distance measurement for ray i (meters)
   - N: number of rays (arbitrary)
```

**Map**:
```
M: Occupancy Grid Matrix (rows x cols)
   - M[i][j]: occupancy value at grid cell (i, j)
   - Typically: 0 = free, 1 = occupied, 0.5 = unknown
   - Resolution: grid cell size in meters
   - Origin: (originX, originY) - world coordinates of map origin
```

**Measurement Function**:
```
h_i(x) = rayCastDistance(x, θ_i, M)
   - For each ray i with angle θ_i
   - Cast ray from pose x against map M
   - Return distance to first obstacle
```

**Localization**:
- Uses Extended Kalman Filter (EKF) for pose estimation
- State transition: optional odometry input or static pose
- Measurement update: compares expected vs measured ray distances

**Type Definition**:
```go
type SLAM struct {
    // Ray configuration
    rayAngles vec.Vector      // Array of ray angles (radians)
    
    // Map
    mapGrid mat.Matrix        // Occupancy grid map
    mapResolution float32     // Grid cell size (meters)
    mapOriginX float32        // Map origin X coordinate (meters)
    mapOriginY float32        // Map origin Y coordinate (meters)
    
    // State
    pose vec.Vector           // Robot pose [px, py, heading]
    
    // Localization
    ekf *ekalman.EKF          // Extended Kalman Filter
    
    // Configuration
    maxRange float32          // Maximum sensor range (meters)
    
    // Filter interface
    Input vec.Vector          // Measured distances input
    Output vec.Vector         // Estimated pose output
    Target vec.Vector         // Target pose (optional)
    
    // Dimensions
    numRays int               // Number of rays
    mapRows int               // Map rows
    mapCols int               // Map columns
}
```

**Operations**:

1. **Initialization**:
   - `New(rayAngles vec.Vector, mapGrid mat.Matrix, mapResolution, mapOriginX, mapOriginY float32) *SLAM`: Create new SLAM filter

2. **Update**:
   - `Update(distances vec.Vector) *SLAM`: Update with distance measurements
   - `Update(timestep float32) filter.Filter`: Update with timestep (Filter interface)

3. **State Management**:
   - `Reset() *SLAM`: Reset filter state
   - `SetPose(pose vec.Vector) *SLAM`: Set initial pose
   - `GetPose() vec.Vector`: Get current pose estimate

4. **Filter Interface**:
   - `GetInput() vec.Vector`: Get measurement input
   - `GetOutput() vec.Vector`: Get pose estimate
   - `GetTarget() vec.Vector`: Get target pose

**Characteristics**:
- Supports arbitrary number of rays
- Works with pre-given occupancy grid map
- Uses EKF for pose estimation
- Efficient ray casting algorithm
- Real-time capable

## Ray Casting (`raycast.go`)

**Purpose**: Cast rays against occupancy grid map to compute expected distances

**Algorithm**:
1. Convert robot pose to grid coordinates
2. For each ray angle:
   a. Compute ray direction
   b. Traverse grid cells along ray
   c. Check occupancy at each cell
   d. Return distance to first occupied cell
   e. If no obstacle found within max range, return max range

**Functions**:
- `RayCast(pose vec.Vector, rayAngle float32, mapGrid mat.Matrix, mapResolution, mapOriginX, mapOriginY, maxRange float32) float32`: Cast single ray
- `RayCastAll(pose vec.Vector, rayAngles vec.Vector, mapGrid mat.Matrix, mapResolution, mapOriginX, mapOriginY, maxRange float32) vec.Vector`: Cast all rays

## Localization (`localization.go`)

**Purpose**: Estimate robot pose using ray-based measurements

**Algorithm**:
1. **Predict**: Update pose using state transition (optional odometry)
2. **Measure**: Cast rays to get expected distances
3. **Update**: Use EKF to update pose based on measurement residuals
4. **Refine**: Iterate if needed (optional)

**Implementation**:
- Uses `ekalman.EKF` for pose estimation
- Measurement function: ray casting
- Measurement Jacobian: numerical differentiation

## Usage Example

```go
// Configure rays (360 rays, 1 degree spacing)
numRays := 360
rayAngles := vec.New(numRays)
for i := 0; i < numRays; i++ {
    rayAngles[i] = float32(i) * math.Pi / 180.0
}

// Load occupancy grid map
mapGrid := mat.New(100, 100)
// ... populate map ...
mapResolution := float32(0.1)  // 10cm per cell
mapOriginX := float32(0.0)
mapOriginY := float32(0.0)

// Create SLAM filter
slam := slam.New(rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY)

// Set initial pose
slam.SetPose(vec.NewFrom(5.0, 5.0, 0.0))

// Update with distance measurements
distances := vec.New(numRays)
// ... populate from sensor ...
slam.Update(distances)

// Get estimated pose
pose := slam.GetOutput()
```

## Questions

1. Should we support online map building (full SLAM)?
2. Should we support 3D maps and poses?
3. Should we support particle filter as alternative to EKF?
4. How to handle dynamic obstacles in static map?
5. Should we support map updates during localization?
6. How to optimize for large maps?
7. Should we support multi-resolution maps?
8. How to handle ray casting at map boundaries?
9. Should we support probabilistic occupancy values?
10. How to validate map and ray configuration?

## Design Decisions

### Architecture

1. **Map Representation**:
   - Occupancy grid as matrix (simple and efficient)
   - Pre-given (not built online)
   - Static (no updates during localization)

2. **Localization Method**:
   - Extended Kalman Filter (handles nonlinear measurements)
   - Can be extended with particle filter later

3. **Ray Casting**:
   - Discrete grid traversal (Bresenham algorithm)
   - Configurable step size
   - Early termination on obstacle hit

### Performance

1. **Ray Casting**:
   - O(d) where d is distance in cells
   - Early termination optimizes common case
   - Can be parallelized for multiple rays

2. **EKF Update**:
   - O(n³) where n is state dimension (3 for 2D pose)
   - Numerical Jacobian: O(n*numRays) function evaluations
   - Pre-allocate matrices for efficiency

## Implementation Notes

### Current Implementation

- Simple localization only (not full SLAM)
- 2D maps and poses
- EKF-based localization
- Pre-given occupancy grid map

### Missing Features

- Online map building
- 3D support
- Particle filter
- Loop closure
- Map optimization
- Dynamic obstacles

