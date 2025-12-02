package gps

import "errors"

var (
	// ErrInsufficientPoints is returned when multilateration is called with fewer than 3 points
	ErrInsufficientPoints = errors.New("gps: at least 3 points required for multilateration")
	// ErrMismatchedLengths is returned when the number of points doesn't match the number of distances
	ErrMismatchedLengths = errors.New("gps: number of points must match number of distances")
)

