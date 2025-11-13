# Grid Package Specification

## Overview

The `grid` package provides utilities for working with 2D grids and occupancy grids, including:
- Ray casting and projection
- Matrix masking and extraction
- Shape extraction (rectangle, circle, ellipse)
- Occupancy queries

## Purpose

This package provides general-purpose grid operations that can be used by:
- SLAM algorithms
- Path planning
- Sensor fusion
- Map processing
- Any algorithm working with 2D occupancy grids

## Functions

### Ray Casting

1. **RayCast**: Cast a single ray and return distance to obstacle
2. **RayCastOptimized**: Optimized ray casting using pre-computed directions
3. **RayCastAll**: Cast all rays and return distances
4. **RayCastAllOptimized**: Optimized batch ray casting
5. **NewRayDirections**: Pre-compute ray directions for efficiency
6. **RayProjection**: Project a ray of specified length and check for intersection

### Matrix Operations

1. **Mask**: Mask one matrix with another (element-wise multiplication with mask)
2. **ExtractRectangle**: Extract rectangular region from matrix
3. **ExtractCircle**: Extract circular region from matrix
4. **ExtractEllipse**: Extract elliptical region from matrix

### Occupancy Queries

1. **GetOccupancyReadings**: Get occupancy readings for given angles using optimized raycasting

## Design Principles

- **Embedded-friendly**: Optimized for embedded systems with minimal allocations
- **Pre-computation**: Where possible, pre-compute expensive operations
- **Reusable**: General-purpose functions usable across multiple algorithms
- **Efficient**: Use Bresenham algorithm and integer-only operations where possible

