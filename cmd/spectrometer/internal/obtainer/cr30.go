package obtainer

import (
	"context"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/cr30"
	"github.com/itohio/EasyRobot/x/math/colorscience"
	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

const (
	defaultCR30Baud = 19200
)

func init() {
	RegisterObtainer("cr30", NewCR30Obtainer)
}

// CR30Obtainer wraps the CR30 device to implement the Obtainer interface
type CR30Obtainer struct {
	device *cr30.Device
	serial devices.Serial
	info   DeviceInfo

	// Preallocated matrix for Start() listener (destination pattern)
	// Only used in Start() goroutine, not in Measure() which uses caller's destination
	listenerMatrix matTypes.Matrix // 2 rows: row0 = wavelengths, row1 = SPD values

	// Measurement listener state
	listenerCtx    context.Context
	listenerCancel context.CancelFunc
	listenerWg     sync.WaitGroup
	listenerMu     sync.Mutex
}

// NewCR30Obtainer creates a new CR30 obtainer from configuration
func NewCR30Obtainer(ctx context.Context, config map[string]interface{}) (Obtainer, error) {
	port, ok := config["port"].(string)
	if !ok || port == "" {
		return nil, fmt.Errorf("cr30 obtainer: port required in config")
	}

	baud := defaultCR30Baud
	if baudVal, ok := config["baud"]; ok {
		switch v := baudVal.(type) {
		case int:
			baud = v
		case int32:
			baud = int(v)
		case int64:
			baud = int(v)
		case float64:
			baud = int(v)
		}
	}

	// Create serial connection
	serialConfig := devices.DefaultSerialConfig()
	serialConfig.BaudRate = baud
	serial, err := devices.NewSerialWithConfig(port, serialConfig)
	if err != nil {
		return nil, fmt.Errorf("cr30 obtainer: failed to open serial port %q: %w", port, err)
	}

	// Create CR30 device
	device := cr30.New(serial)

	// Preallocate matrix for Start() listener (2 rows: wavelengths, SPD values)
	// NumWavelengths() doesn't require connection, returns constant 31
	numWl := device.NumWavelengths()
	listenerMatrix := mat.New(2, numWl)

	// Create obtainer
	obtainer := &CR30Obtainer{
		device:         device,
		serial:         serial,
		listenerMatrix: listenerMatrix,
	}

	// Connect and get device info
	if err := obtainer.Connect(ctx); err != nil {
		serial.Close()
		return nil, fmt.Errorf("cr30 obtainer: failed to connect: %w", err)
	}

	return obtainer, nil
}

// Connect establishes connection with the CR30 device
func (o *CR30Obtainer) Connect(ctx context.Context) error {
	if err := o.device.Connect(); err != nil {
		return fmt.Errorf("cr30 obtainer: connect failed: %w", err)
	}

	// Get device info from handshake
	info := o.device.DeviceInfo()
	o.info = DeviceInfo{
		Name:     info.Name,
		Model:    info.Model,
		Serial:   info.Serial,
		Firmware: info.Firmware,
		Build:    info.Build,
	}

	return nil
}

// Disconnect closes the connection
func (o *CR30Obtainer) Disconnect() error {
	// Stop listener if running
	_ = o.Stop(context.Background())

	// Disconnect device
	if o.device != nil {
		if err := o.device.Disconnect(); err != nil {
			return err
		}
	}
	if o.serial != nil {
		// Try to close via io.Closer interface
		if closer, ok := o.serial.(io.Closer); ok {
			return closer.Close()
		}
	}
	return nil
}

// Measure obtains a spectrum measurement from the CR30 device (PC-initiated).
// For averaging: use Start() for the first sample (button-initiated),
// then use Measure() for subsequent samples (PC-initiated).
// Writes to destination matrix (2 rows: wavelengths, values).
func (o *CR30Obtainer) Measure(ctx context.Context, dst matTypes.Matrix) error {
	// Validate destination matrix size
	if dst.Rows() != 2 || dst.Cols() != o.NumWavelengths() {
		return fmt.Errorf("cr30 obtainer: destination matrix must be 2 rows x %d columns, got %d rows x %d columns",
			o.NumWavelengths(), dst.Rows(), dst.Cols())
	}

	// Measure (triggers measurement and reads data directly into destination)
	if err := o.device.Measure(ctx, dst); err != nil {
		return fmt.Errorf("cr30 obtainer: measure failed: %w", err)
	}

	return nil
}

// Start starts a background goroutine that waits for button presses
// and calls the callback when measurements are available.
func (o *CR30Obtainer) Start(ctx context.Context, callback MeasurementCallback) error {
	o.listenerMu.Lock()
	defer o.listenerMu.Unlock()

	// Check if listener is already running
	if o.listenerCancel != nil {
		return fmt.Errorf("cr30 obtainer: measurement listener already running")
	}

	// Create cancellable context for the listener
	listenerCtx, cancel := context.WithCancel(ctx)
	o.listenerCtx = listenerCtx
	o.listenerCancel = cancel

	// Start listener goroutine
	o.listenerWg.Add(1)
	go func() {
		defer o.listenerWg.Done()

		for {
			// Check if we should stop
			select {
			case <-listenerCtx.Done():
				return
			default:
				// Continue
			}

			// Use preallocated matrix (destination pattern, reused for all measurements)
			dst := o.listenerMatrix

			// Wait for button press and read measurement
			// Use a timeout to periodically check context cancellation
			measureCtx, cancel := context.WithTimeout(listenerCtx, 30*time.Second)
			err := o.device.WaitMeasurement(measureCtx, dst)
			cancel()

			var spd colorscience.SPD
			if err == nil {
				// Extract wavelengths and values from matrix
				wavelengths := dst.Row(0).(vec.Vector)
				values := dst.Row(1).(vec.Vector)

				// Create SPD from wavelengths and values (allocation only happens here for callback)
				spd = colorscience.NewSPD(wavelengths, values)
			}

			// Call callback with measurement or error
			if callbackErr := callback(spd, err); callbackErr != nil {
				// Callback wants to stop listening
				return
			}

			// If there was an error (other than context cancellation), continue listening
			if err != nil && err != context.Canceled && err != context.DeadlineExceeded {
				// Continue listening despite error (callback handled it)
				continue
			}

			// Check if context was cancelled during measurement
			if listenerCtx.Err() != nil {
				return
			}
		}
	}()

	return nil
}

// Stop stops the background measurement listener gracefully.
func (o *CR30Obtainer) Stop(ctx context.Context) error {
	o.listenerMu.Lock()
	defer o.listenerMu.Unlock()

	if o.listenerCancel == nil {
		// Listener not running
		return nil
	}

	// Cancel listener context
	o.listenerCancel()

	// Wait for goroutine to finish (with timeout)
	done := make(chan struct{})
	go func() {
		o.listenerWg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// Listener stopped gracefully
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(5 * time.Second):
		return fmt.Errorf("cr30 obtainer: timeout waiting for listener to stop")
	}

	// Reset listener state
	o.listenerCancel = nil
	o.listenerCtx = nil

	return nil
}

// Wavelengths returns the wavelength vector for CR30 device.
// The caller should preallocate the destination vector and reuse it.
// For convenience, this method allocates and returns a new vector.
func (o *CR30Obtainer) Wavelengths() vec.Vector {
	wl := vec.New(o.NumWavelengths())
	o.device.Wavelengths(wl)
	return wl
}

// DeviceInfo returns device identification and version information
func (o *CR30Obtainer) DeviceInfo() DeviceInfo {
	return o.info
}

// NumWavelengths returns the number of wavelengths in the CR30 spectrum (31)
func (o *CR30Obtainer) NumWavelengths() int {
	return o.device.NumWavelengths()
}
