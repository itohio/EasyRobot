package array

import (
	"context"
	"sync"
	"time"

	devio "github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/tca9548a"
	"github.com/itohio/EasyRobot/x/devices/vl53l0x"
	mat "github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
)

// Device is an array-based LiDAR that uses TCA9548A I2C multiplexer and VL53L0X sensors.
// Each sensor is positioned at a fixed angle, and readings are assembled into a 2xN matrix.
type Device struct {
	router  *tca9548a.Router
	sensors []*vl53l0x.Device
	angles  []float32 // angle (deg) for each sensor

	ctx    context.Context
	cancel func()

	// preallocated buffers
	backing []float32  // length 2*numSensors
	mat2xN  mat.Matrix // 2 x numSensors view into backing

	numSensors int
	count      int // number of valid points in current scan

	onRead func(matTypes.Matrix)

	mu        sync.Mutex
	startOnce sync.Once
}

// Config holds configuration for the array-based LiDAR.
type Config struct {
	// Router is the TCA9548A router managing the I2C multiplexer.
	Router *tca9548a.Router

	// Angles is the angle (in degrees) for each sensor, in order.
	// Length must match the number of sensors.
	Angles []float32

	// SensorAddress is the I2C address for each VL53L0X sensor.
	// If 0, uses vl53l0x.DefaultAddress (0x29).
	// If length is less than numSensors, remaining sensors use DefaultAddress.
	SensorAddresses []uint8
}

// New creates a new array-based LiDAR device.
// The router must already be configured. The number of sensors is determined by the number of angles.
// Each sensor will be accessed via router channels 0, 1, 2, ... up to len(angles)-1.
func New(ctx context.Context, cfg Config) (*Device, error) {
	if cfg.Router == nil {
		return nil, devio.ErrInvalidState
	}
	if len(cfg.Angles) == 0 {
		return nil, devio.ErrInvalidState
	}
	if len(cfg.Angles) > 64 {
		return nil, devio.ErrInvalidState // TCA9548A supports 8 channels, but we allow multiple muxes
	}

	numSensors := len(cfg.Angles)
	cctx, cancel := context.WithCancel(ctx)

	// backing: [dist(0..num-1) | angle(0..num-1)]
	backing := make([]float32, 2*numSensors)
	m := mat.New(2, numSensors, backing...)

	// Create VL53L0X devices for each sensor
	sensors := make([]*vl53l0x.Device, numSensors)
	for i := 0; i < numSensors; i++ {
		channel, err := cfg.Router.Channel(uint8(i % 8)) // Support up to 8 sensors per mux
		if err != nil {
			cancel()
			return nil, err
		}

		addr := uint8(vl53l0x.DefaultAddress)
		if i < len(cfg.SensorAddresses) && cfg.SensorAddresses[i] != 0 {
			addr = cfg.SensorAddresses[i]
		}

		sensors[i] = vl53l0x.New(channel, addr)
	}

	d := &Device{
		router:     cfg.Router,
		sensors:    sensors,
		angles:     cfg.Angles,
		ctx:        cctx,
		cancel:     cancel,
		backing:    backing,
		mat2xN:     m,
		numSensors: numSensors,
	}
	return d, nil
}

// Configure initializes the router and all sensors. If init is true, performs full initialization.
func (d *Device) Configure(init bool) error {
	if err := d.router.Configure(init); err != nil {
		return err
	}

	for i, sensor := range d.sensors {
		if err := sensor.Configure(init); err != nil {
			return err
		}
		if init {
			// Start continuous measurement for each sensor
			if err := sensor.StartContinuous(50); err != nil { // 50ms period
				return err
			}
		}
		_ = i // sensor index, could be used for channel selection optimization
	}

	d.startOnce.Do(func() {
		go d.readLoop()
	})
	return nil
}

// Close stops the device and releases resources.
func (d *Device) Close() {
	if d.cancel != nil {
		d.cancel()
	}
	for _, sensor := range d.sensors {
		_ = sensor.StopContinuous()
	}
}

// OnRead registers a callback that is invoked with a view of the internal 2xN matrix
// each time a scan is completed.
func (d *Device) OnRead(fn func(matTypes.Matrix)) {
	d.mu.Lock()
	d.onRead = fn
	d.mu.Unlock()
}

// Read copies the latest completed scan into dst and returns number of valid points copied.
func (d *Device) Read(dst matTypes.Matrix) int {
	d.mu.Lock()
	defer d.mu.Unlock()

	k := d.count
	if k <= 0 {
		return 0
	}
	if k > dst.Cols() {
		k = dst.Cols()
	}

	// Copy two rows, k columns
	view := d.mat2xN.View().(mat.Matrix)
	dstm := dst.View().(mat.Matrix)
	copy(dstm[0][:k], view[0][:k])
	copy(dstm[1][:k], view[1][:k])
	return k
}

// GetMinAngle returns the minimum angle (in degrees) that this LiDAR can measure.
func (d *Device) GetMinAngle() float32 {
	if len(d.angles) == 0 {
		return 0
	}
	min := d.angles[0]
	for _, angle := range d.angles[1:] {
		if angle < min {
			min = angle
		}
	}
	return min
}

// GetMaxAngle returns the maximum angle (in degrees) that this LiDAR can measure.
func (d *Device) GetMaxAngle() float32 {
	if len(d.angles) == 0 {
		return 0
	}
	max := d.angles[0]
	for _, angle := range d.angles[1:] {
		if angle > max {
			max = angle
		}
	}
	return max
}

// GetPointCount returns the number of points in the current/latest scan.
func (d *Device) GetPointCount() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.count
}

func (d *Device) readLoop() {
	ticker := time.NewTicker(50 * time.Millisecond) // Read at ~20Hz
	defer ticker.Stop()

	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.readScan()
		}
	}
}

func (d *Device) readScan() {
	distRow := d.mat2xN[0][:]
	angRow := d.mat2xN[1][:]

	for i, sensor := range d.sensors {
		dist, err := sensor.ReadRangeContinuous()
		if err != nil {
			// Mark as invalid (0 distance)
			distRow[i] = 0
		} else {
			distRow[i] = float32(dist)
		}
		angRow[i] = d.angles[i]
	}

	d.mu.Lock()
	d.count = d.numSensors
	view := d.mat2xN.View().(mat.Matrix)
	view[0] = view[0][:d.count]
	view[1] = view[1][:d.count]
	cb := d.onRead
	d.mu.Unlock()

	if cb != nil {
		cb(view)
	}
}
