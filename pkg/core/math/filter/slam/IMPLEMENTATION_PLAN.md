# Simple SLAM Implementation Plan

## Overview

This document outlines the implementation plan for a simple Simultaneous Localization And Mapping (SLAM) filter that works with ray-based sensors (e.g., LiDAR, sonar).

## 2025-11-13 Maintenance Notes

- Align ray-casting unit tests with the intended map geometry by placing obstacles along the rays under test.
- Extend EKF measurement update internals with correctly sized temporary matrices so matrix multiplications remain dimensionally valid.
- Keep helper allocations reusable to preserve performance characteristics of prediction and update steps.

## Requirements

1. **Arbitrary number of rays**: Support any number of sensor rays
2. **Ray angles**: Array of ray angles provided at instantiation
3. **Distance measurements**: Vector of distances provided on each update
4. **Pre-given map**: Occupancy grid matrix provided beforehand (can be updated online)
5. **Localization**: Estimate robot pose (position and orientation) given the map
6. **Online map building**: Configurable option to build/update map from measurements
7. **Embedded optimization**: Optimized for embedded devices (pre-computed, minimal allocations)

## Architecture

### Components

1. **SLAM Filter** (`slam.go`): Main filter implementation
2. **Ray Casting** (`raycast.go`): Ray casting against occupancy grid
    - Consider using optimized shadowcasting algorithm
3. **Localization** (`localization.go`): Pose estimation using measurements

### Mathematical Model

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
```

**Map**:
```
M: Occupancy Grid Matrix (rows x cols)
   - M[i][j]: occupancy probability at grid cell (i, j)
   - Typically: 0 = free, 1 = occupied, 0.5 = unknown
   - Can be updated online using inverse sensor model (log-odds representation)
```

**Optimizations**:
- Pre-computed ray directions (cos/sin) for efficiency
- Bresenham algorithm for ray casting (embedded-friendly)
- Minimal allocations in hot path
- Log-odds representation for map building (numerical stability)

**Measurement Model**:
- For each ray angle `θ_i`:
  1. Cast ray from robot position `(px, py)` at angle `heading + θ_i`
  2. Compute expected distance `d_expected` by intersecting ray with map
  3. Compare with measured distance `d_measured`
  4. Use difference for localization update

## Implementation Details

### Type Definition

```go
type SLAM struct {
    // Ray configuration
    rayAngles vec.Vector        // Array of ray angles (radians) relative to robot heading
    rayDirs *RayDirections      // Pre-computed ray directions (cos/sin) for efficiency
    
    // Map
    mapGrid mat.Matrix          // Occupancy grid map (rows x cols)
    logOddsMap mat.Matrix       // Log-odds representation for map building (nil if disabled)
    mapResolution float32       // Grid cell size in meters
    mapOriginX float32          // X coordinate of map origin (meters)
    mapOriginY float32          // Y coordinate of map origin (meters)
    
    // State
    pose vec.Vector             // Robot pose [px, py, heading]
    
    // Localization filter
    ekf *ekalman.EKF            // Extended Kalman Filter for pose estimation
    // Note: EKF is required because measurement function (ray casting) is nonlinear in pose.
    // KF cannot be substituted because h(pose) cannot be expressed as H * pose.
    
    // Temporary storage
    expectedDistances vec.Vector // Expected distances for each ray
    measuredDistances vec.Vector // Measured distances (input)
    
    // Configuration
    maxRange float32            // Maximum sensor range (meters)
    enableMapping bool          // Enable online map building (configurable)
    useOptimizedRaycast bool    // Use optimized ray casting (default: true)
    
    // Filter interface
    Input vec.Vector            // Measured distances input
    Output vec.Vector           // Estimated pose output [px, py, heading]
    Target vec.Vector           // Target pose (optional)
    
    // Dimensions
    numRays int                 // Number of rays
    mapRows int                 // Map rows
    mapCols int                 // Map columns
}
```

### Operations

1. **Initialization**:
   - `New(rayAngles vec.Vector, mapGrid mat.Matrix, mapResolution, mapOriginX, mapOriginY float32) *SLAM`: Create new SLAM filter
   - Validate ray angles and map dimensions
   - Initialize pose estimate (can start at map origin or provided pose)

2. **Update**:
   - `Update(distances vec.Vector) *SLAM`: Update filter with distance measurements
   - Cast rays against map to get expected distances
   - Use difference between expected and measured distances for localization
   - Update pose estimate using EKF

3. **State Management**:
   - `Reset() *SLAM`: Reset filter state
   - `SetPose(pose vec.Vector) *SLAM`: Set initial pose
   - `GetPose() vec.Vector`: Get current pose estimate

4. **Filter Interface**:
   - `Update(timestep float32) filter.Filter`: Update with timestep
   - `GetInput() vec.Vector`: Get measurement input
   - `GetOutput() vec.Vector`: Get pose estimate
   - `GetTarget() vec.Vector`: Get target pose

### Ray Casting Algorithm

**Input**:
- Robot pose: `(px, py, heading)`
- Ray angle: `θ_i` (relative to robot heading)
- Map: `M` (occupancy grid)
- Map parameters: `resolution`, `originX`, `originY`
- Max range: `maxRange`

**Algorithm**:
```
1. Convert world coordinates to grid coordinates:
   gridX = (px - originX) / resolution
   gridY = (py - originY) / resolution

2. Compute ray direction:
   rayAngle = heading + θ_i
   dirX = cos(rayAngle)
   dirY = sin(rayAngle)

3. Cast ray step by step:
   distance = 0
   step = resolution / 2  // Half cell for accuracy
   
   while distance < maxRange:
       currentX = gridX + distance * dirX / resolution
       currentY = gridY + distance * dirY / resolution
       
       // Check bounds
       if currentX < 0 or currentX >= mapCols or currentY < 0 or currentY >= mapRows:
           return maxRange  // Ray out of bounds
       
       // Get occupancy at cell
       cellX = int(currentX)
       cellY = int(currentY)
       occupancy = M[cellY][cellX]
       
       // If occupied (threshold > 0.5), hit found
       if occupancy > 0.5:
           return distance
       
       distance += step

   return maxRange  // No hit found
```

**Optimizations**:
- Use Bresenham's line algorithm for discrete grid traversal
- Early termination when ray hits obstacle
- Adaptive step size for better accuracy

### Localization Using EKF

**State Transition**:
- Pose prediction (with odometry/control input):
  ```
  px_new = px + vx*dt
  py_new = py + vy*dt
  heading_new = heading + ω*dt
  ```
- Or static pose (no motion model):
  ```
  pose_new = pose  // Identity
  ```

**Measurement Function**:
- For each ray `i`:
  ```
  h_i(pose) = rayCastDistance(pose, rayAngles[i], mapGrid)
  ```
- Jacobian of measurement function:
  ```
  H[i][0] = ∂h_i/∂px  (numerical or analytical)
  H[i][1] = ∂h_i/∂py
  H[i][2] = ∂h_i/∂heading
  ```

**Update Process**:
1. Predict pose using state transition
2. Cast rays to get expected distances
3. Compute measurement Jacobian
4. Update pose using EKF update step
5. Refine pose estimate

## Implementation Steps

### Phase 1: Core Structure
1. ✅ Create SPEC.md (design document)
2. ✅ Create IMPLEMENTATION_PLAN.md (this document)
3. Create `slam.go` with basic structure
4. Define types and interfaces

### Phase 2: Ray Casting
1. Implement `raycast.go` with ray casting algorithm
2. Implement grid-to-world and world-to-grid coordinate conversion
3. Implement Bresenham line algorithm for grid traversal
4. Add tests for ray casting

### Phase 3: Localization
1. Integrate EKF for pose estimation
2. Implement measurement function using ray casting
3. Implement measurement Jacobian computation (numerical)
4. Add pose prediction with optional odometry input

### Phase 4: Filter Integration
1. Implement Filter interface methods
2. Connect input/output for distance measurements
3. Implement Update method
4. Add state management (Reset, SetPose, GetPose)

### Phase 5: Testing
1. Unit tests for ray casting
2. Unit tests for coordinate conversion
3. Integration tests with known maps
4. Test with different numbers of rays
5. Test with different map sizes

### Phase 6: Optimization
1. Optimize ray casting algorithm
2. Add caching for ray intersections
3. Optimize EKF update for real-time performance

## File Structure

```
pkg/core/math/filter/slam/
├── SPEC.md                  # Design specification
├── IMPLEMENTATION_PLAN.md   # This file
├── slam.go                  # Main SLAM filter
├── raycast.go               # Ray casting implementation
├── localization.go          # Localization using EKF
├── slam_test.go            # Tests
└── README.md               # Usage examples
```

## Usage Example

```go
// Ray configuration: 360 rays, 1 degree spacing
numRays := 360
rayAngles := vec.New(numRays)
for i := 0; i < numRays; i++ {
    rayAngles[i] = float32(i) * math.Pi / 180.0
}

// Load occupancy grid map (100x100 cells, 0.1m resolution)
mapGrid := mat.New(100, 100)
// ... populate map from file or generate ...
mapResolution := float32(0.1)  // 10cm per cell
mapOriginX := float32(0.0)
mapOriginY := float32(0.0)

// Create SLAM filter
slam := slam.New(rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY)

// Set initial pose
slam.SetPose(vec.NewFrom(5.0, 5.0, 0.0))  // Start at (5m, 5m), facing east

// Update with distance measurements
distances := vec.New(numRays)
// ... populate distances from sensor ...
slam.Update(distances)

// Get estimated pose
pose := slam.GetOutput()
px := pose[0]
py := pose[1]
heading := pose[2]
```

## Design Decisions

### Ray Casting

**Why Bresenham algorithm?**
- Efficient discrete grid traversal
- Guarantees all cells along line are visited
- O(d) complexity where d is distance in cells

**Why half-cell step size?**
- Balances accuracy vs performance
- Prevents missing thin obstacles
- Configurable for different use cases

### Localization Method

**Why EKF?**
- Handles nonlinear measurement function (ray casting)
- Maintains uncertainty estimate (covariance)
- Optimal for Gaussian noise assumption
- Can incorporate odometry for better prediction

**Alternative: Particle Filter**
- Could handle multi-modal distributions
- Better for global localization
- More computationally expensive
- Future enhancement

### Map Representation

**Why occupancy grid?**
- Simple and efficient
- Easy to update
- Works well with ray casting
- Common in robotics

**Matrix representation:**
- Uses existing `mat.Matrix` primitive
- Easy to visualize and manipulate
- Can be extended to support probability maps

## Known Limitations

1. **Static map**: Map is provided beforehand, not built online
2. **2D only**: Currently only supports 2D maps and poses
3. **No map building**: Only localization, not simultaneous mapping
4. **Single hypothesis**: EKF assumes single mode (no multi-hypothesis tracking)
5. **Deterministic ray casting**: No uncertainty in ray casting itself

## Future Enhancements

1. **Online mapping**: Build map while localizing
2. **3D SLAM**: Extend to 3D maps and poses
3. **Particle filter**: Alternative localization method
4. **Loop closure**: Detect revisited locations
5. **Map optimization**: Bundle adjustment for map consistency
6. **Multi-sensor fusion**: Combine LiDAR with other sensors
7. **Dynamic obstacles**: Handle moving objects in map
8. **Map compression**: Efficient storage for large maps

## Testing Strategy

1. **Unit Tests**:
   - Ray casting on known maps
   - Coordinate conversion
   - EKF update steps

2. **Integration Tests**:
   - Full SLAM loop with synthetic data
   - Localization accuracy on known trajectories
   - Performance with different numbers of rays

3. **Validation**:
   - Compare with ground truth poses
   - Test with different map sizes
   - Test with different sensor configurations

## Performance Targets

- **Update rate**: > 10 Hz for 360 rays
- **Memory**: < 10 MB for typical map (1000x1000 cells)
- **Accuracy**: < 0.1m position error, < 1° heading error (depending on sensor)

## Dependencies

- `pkg/core/math/mat`: Matrix operations
- `pkg/core/math/vec`: Vector operations
- `pkg/core/math/filter/ekalman`: Extended Kalman Filter for localization

