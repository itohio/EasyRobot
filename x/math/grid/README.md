# Grid Package

This package provides utilities for working with 2D grids and occupancy grids, including ray casting, matrix operations, and shape extraction.

## Overview

The `grid` package provides general-purpose grid operations that can be used by:
- SLAM algorithms
- Path planning
- Sensor fusion
- Map processing
- Any algorithm working with 2D occupancy grids

## Functions

### Ray Casting

#### NewRayDirections
Pre-computes ray directions (cos/sin) for efficiency.

```go
rayDirs := grid.NewRayDirections(rayAngles)
```

#### RayCast
Casts a single ray and returns distance to obstacle.

```go
distance := grid.RayCast(pose, rayAngle, mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange)
```

#### RayCastOptimized
Optimized ray casting using pre-computed directions and Bresenham algorithm.

```go
distance := grid.RayCastOptimized(pose, rayIdx, rayDirs, mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange)
```

#### RayCastAll
Casts all rays and returns distances.

```go
distances := grid.RayCastAll(pose, rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange)
```

#### RayCastAllOptimized
Optimized batch ray casting with pre-allocated vector.

```go
grid.RayCastAllOptimized(pose, rayDirs, mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange, distances)
```

#### RayProjection
Projects a ray of specified length and checks if it intersects an obstacle.

```go
intersects := grid.RayProjection(startX, startY, dirX, dirY, length, mapGrid, occupancyThreshold)
```

#### GetOccupancyReadings
Gets occupancy readings for given angles using optimized raycasting.

```go
readings := grid.GetOccupancyReadings(pose, rayAngles, mapGrid, mapResolution, mapOriginX, mapOriginY, maxRange, rayDirs)
```

### Matrix Operations

#### Mask
Masks one matrix with another (element-wise multiplication).

```go
result := grid.Mask(src, mask, dst)
```

#### ExtractRectangle
Extracts a rectangular region from a matrix.

```go
rect := grid.ExtractRectangle(src, x, y, width, height)
```

#### ExtractCircle
Extracts a circular region from a matrix.

```go
circle := grid.ExtractCircle(src, centerX, centerY, radius, fillValue)
```

#### ExtractEllipse
Extracts an elliptical region from a matrix.

```go
ellipse := grid.ExtractEllipse(src, centerX, centerY, a, b, angle, fillValue)
```

## Usage Examples

### Ray Casting

```go
import "github.com/itohio/EasyRobot/x/math/grid"

// Pre-compute ray directions
rayAngles := vec.NewFrom(0.0, math.Pi/2, math.Pi, 3*math.Pi/2)
rayDirs := grid.NewRayDirections(rayAngles)

// Cast single ray
pose := vec.NewFrom(5.0, 5.0, 0.0)
distance := grid.RayCast(pose, 0.0, mapGrid, 0.1, 0.0, 0.0, 10.0)

// Cast all rays (optimized)
distances := vec.New(4)
grid.RayCastAllOptimized(pose, rayDirs, mapGrid, 0.1, 0.0, 0.0, 10.0, distances)
```

### Matrix Extraction

```go
// Extract rectangle
rect := grid.ExtractRectangle(mapGrid, 10, 10, 50, 50)

// Extract circle
circle := grid.ExtractCircle(mapGrid, 50, 50, 10, 0.0)

// Extract ellipse
ellipse := grid.ExtractEllipse(mapGrid, 50, 50, 20, 10, math.Pi/4, 0.0)
```

### Matrix Masking

```go
// Create mask (1.0 = keep, 0.0 = zero out)
mask := mat.New(100, 100)
// ... populate mask ...

// Apply mask
result := grid.Mask(mapGrid, mask, nil)
```

## Performance

- **Pre-computed directions**: ~50% faster ray casting
- **Bresenham algorithm**: ~30% faster for longer rays
- **Minimal allocations**: Reuses memory in hot path
- **Early termination**: Stops when obstacle found

## Embedded Systems

The package is optimized for embedded systems:
- Pre-computed values where possible
- Integer-only operations (Bresenham)
- Minimal memory allocations
- Reusable temporary buffers

