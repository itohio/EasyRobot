package obtainer

import (
	"context"
	"fmt"
	"sync"

	"github.com/itohio/EasyRobot/x/math/colorscience"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// MeasurementCallback is called when a new measurement is available.
// The callback receives the measurement SPD and any error that occurred.
// If the callback returns an error, measurement listening may stop.
type MeasurementCallback func(spd colorscience.SPD, err error) error

// Obtainer represents a device that can obtain spectra directly
type Obtainer interface {
	// Connect establishes connection with the device
	Connect(ctx context.Context) error

	// Disconnect closes the connection
	Disconnect() error

	// Measure obtains a spectrum measurement from the device (PC-initiated).
	// This triggers a measurement programmatically and writes to the destination matrix.
	// The destination matrix must be 2 rows x NumWavelengths() columns:
	//   - Row 0: wavelengths (nm)
	//   - Row 1: spectral power distribution values
	// Use NumWavelengths() to determine the required matrix size.
	Measure(ctx context.Context, dst matTypes.Matrix) error

	// Start starts a background goroutine that waits for user-initiated measurements
	// (e.g., button presses) and calls the callback when measurements are available.
	// The listener runs until the context is cancelled or Stop is called.
	// Multiple measurements can be received while the listener is active.
	// For devices without button-initiated measurements, this may not be supported.
	Start(ctx context.Context, callback MeasurementCallback) error

	// Stop stops the background measurement listener.
	// This gracefully shuts down the listener goroutine.
	Stop(ctx context.Context) error

	// Wavelengths returns the wavelength vector for this device
	Wavelengths() vec.Vector

	// DeviceInfo returns device identification and version information
	DeviceInfo() DeviceInfo

	// NumWavelengths returns the number of wavelengths in the spectrum
	NumWavelengths() int
}

// DeviceInfo contains device identification and version information
type DeviceInfo struct {
	Name     string
	Model    string
	Serial   string
	Firmware string
	Build    string
}

// ObtainerFactory is a function that creates a new Obtainer from configuration
type ObtainerFactory func(ctx context.Context, config map[string]interface{}) (Obtainer, error)

var (
	registry     = make(map[string]ObtainerFactory)
	registryLock sync.RWMutex
)

// RegisterObtainer registers a device obtainer factory
func RegisterObtainer(deviceType string, factory ObtainerFactory) {
	registryLock.Lock()
	defer registryLock.Unlock()
	registry[deviceType] = factory
}

// NewObtainer creates a new obtainer for the specified device type
func NewObtainer(ctx context.Context, deviceType string, config map[string]interface{}) (Obtainer, error) {
	registryLock.RLock()
	factory, ok := registry[deviceType]
	registryLock.RUnlock()

	if !ok {
		return nil, fmt.Errorf("obtainer: unknown device type %q (available: %v)", deviceType, AvailableDevices())
	}

	return factory(ctx, config)
}

// AvailableDevices returns list of available device types
func AvailableDevices() []string {
	registryLock.RLock()
	defer registryLock.RUnlock()

	devices := make([]string, 0, len(registry))
	for deviceType := range registry {
		devices = append(devices, deviceType)
	}
	return devices
}
