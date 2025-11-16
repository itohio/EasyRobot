package lidar

import (
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
)

// Device represents a LiDAR device that provides distance and angle measurements.
// All LiDAR implementations should conform to this interface.
type Device interface {
	// Configure initializes the device. If init is true, performs full initialization.
	Configure(init bool) error

	// OnRead registers a callback that is invoked with a view of the internal 2xN matrix
	// each time a scan is completed. The matrix has:
	// - row 0: distances (mm)
	// - row 1: angles (deg)
	// The matrix columns equal the number of points in that scan.
	OnRead(fn func(matTypes.Matrix))

	// Read copies the latest completed scan into dst and returns number of valid points copied.
	// Expects dst to be a 2xK matrix; copies min(K, available) columns.
	// Returns the number of points copied.
	Read(dst matTypes.Matrix) int

	// GetMinAngle returns the minimum angle (in degrees) that this LiDAR can measure.
	GetMinAngle() float32

	// GetMaxAngle returns the maximum angle (in degrees) that this LiDAR can measure.
	GetMaxAngle() float32

	// GetPointCount returns the number of points in the current/latest scan.
	GetPointCount() int

	// Close stops the device and releases resources.
	Close()
}
