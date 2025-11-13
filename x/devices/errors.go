package devices

import "errors"

// Common device errors that are platform-agnostic.
// These replace machine.* errors to allow code to work on both TinyGo and standard Go.
var (
	// ErrInvalidInputPin is returned when an invalid pin number is provided.
	ErrInvalidInputPin = errors.New("invalid input pin")

	// ErrInvalidOutputPin is returned when an invalid output pin is provided.
	ErrInvalidOutputPin = errors.New("invalid output pin")

	// ErrInvalidState is returned when a device is in an invalid state for the operation.
	ErrInvalidState = errors.New("invalid state")

	// ErrTimeout is returned when an operation times out.
	ErrTimeout = errors.New("timeout")

	// ErrInvalidValue is returned when an invalid parameter value is provided.
	ErrInvalidValue = errors.New("invalid value")

	// ErrInvalidResponse is returned when a device returns an unexpected response.
	ErrInvalidResponse = errors.New("invalid response")

	// ErrInvalidSize is returned when a buffer size is invalid.
	ErrInvalidSize = errors.New("invalid size")

	// ErrNotSupported is returned when an operation is not supported on the current platform.
	ErrNotSupported = errors.New("operation not supported")
)
