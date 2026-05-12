// Package tx provides a CRSF transmitter device that sends RC channel
// commands and receives telemetry from a flight controller or receiver.
package tx

import (
	"context"
	"io"
	"sync"
	"time"

	"github.com/itohio/EasyRobot/x/devices/crsf"
)

// Device acts as a CRSF transmitter/controller: it sends RC channel
// commands and reads telemetry responses from the connected receiver or FC.
type Device struct {
	dec    *crsf.Decoder
	enc    *crsf.Encoder
	ctx    context.Context
	cancel func()

	mu       sync.RWMutex
	battery  crsf.Battery
	gps      crsf.GPS
	attitude crsf.Attitude
	link     crsf.LinkStatistics
	mode     crsf.FlightMode

	onBattery    func(*crsf.Battery)
	onGPS        func(*crsf.GPS)
	onAttitude   func(*crsf.Attitude)
	onFlightMode func(*crsf.FlightMode)
	onLinkStats  func(*crsf.LinkStatistics)
	onFrame      func(*crsf.Frame)

	startOnce sync.Once
}

// New creates a transmitter device.
func New(ctx context.Context, r io.Reader, w io.Writer) *Device {
	cctx, cancel := context.WithCancel(ctx)
	return &Device{
		dec:    crsf.NewDecoder(r),
		enc:    crsf.NewEncoder(w),
		ctx:    cctx,
		cancel: cancel,
	}
}

// Close stops the read loop.
func (d *Device) Close() {
	if d.cancel != nil {
		d.cancel()
	}
}

func (d *Device) OnBattery(fn func(*crsf.Battery))           { d.onBattery = fn }
func (d *Device) OnGPS(fn func(*crsf.GPS))                   { d.onGPS = fn }
func (d *Device) OnAttitude(fn func(*crsf.Attitude))         { d.onAttitude = fn }
func (d *Device) OnFlightMode(fn func(*crsf.FlightMode))     { d.onFlightMode = fn }
func (d *Device) OnLinkStats(fn func(*crsf.LinkStatistics))  { d.onLinkStats = fn }

// OnFrame registers a catch-all for frame types not handled internally.
func (d *Device) OnFrame(fn func(*crsf.Frame)) { d.onFrame = fn }

// SendFrame writes an arbitrary frame to the receiver.
func (d *Device) SendFrame(f *crsf.Frame) error {
	return d.enc.WriteFrame(f)
}

// Start begins the background read loop that decodes incoming telemetry.
func (d *Device) Start() {
	d.startOnce.Do(func() {
		go d.readLoop()
	})
}

// SendChannels writes an RC channels frame to the receiver.
func (d *Device) SendChannels(ch *crsf.Channels) error {
	return d.enc.WriteChannels(ch)
}

// Battery returns a copy of the latest battery telemetry.
func (d *Device) Battery() crsf.Battery {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.battery
}

// GPS returns a copy of the latest GPS telemetry.
func (d *Device) GPS() crsf.GPS {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.gps
}

// Attitude returns a copy of the latest attitude.
func (d *Device) Attitude() crsf.Attitude {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.attitude
}

// LinkStats returns a copy of the latest link statistics.
func (d *Device) LinkStats() crsf.LinkStatistics {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.link
}

// FlightMode returns the latest flight mode string.
func (d *Device) FlightMode() crsf.FlightMode {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.mode
}

func (d *Device) readLoop() {
	for {
		select {
		case <-d.ctx.Done():
			return
		default:
		}

		f, err := d.dec.ReadFrame()
		if err != nil {
			if err == io.EOF {
				return
			}
			time.Sleep(time.Millisecond)
			continue
		}
		d.dispatch(f)
	}
}

func (d *Device) dispatch(f *crsf.Frame) {
	switch f.Type {
	case crsf.TypeBattery:
		d.handleBattery(f)
	case crsf.TypeGPS:
		d.handleGPS(f)
	case crsf.TypeAttitude:
		d.handleAttitude(f)
	case crsf.TypeFlightMode:
		d.handleFlightMode(f)
	case crsf.TypeLinkStats, crsf.TypeLinkStatsRX, crsf.TypeLinkStatsTX:
		d.handleLinkStats(f)
	default:
		if d.onFrame != nil {
			d.onFrame(f)
		}
	}
}

func (d *Device) handleBattery(f *crsf.Frame) {
	b, err := crsf.DecodeBattery(f)
	if err != nil {
		return
	}
	d.mu.Lock()
	d.battery = *b
	d.mu.Unlock()
	if d.onBattery != nil {
		d.onBattery(b)
	}
}

func (d *Device) handleGPS(f *crsf.Frame) {
	g, err := crsf.DecodeGPS(f)
	if err != nil {
		return
	}
	d.mu.Lock()
	d.gps = *g
	d.mu.Unlock()
	if d.onGPS != nil {
		d.onGPS(g)
	}
}

func (d *Device) handleAttitude(f *crsf.Frame) {
	a, err := crsf.DecodeAttitude(f)
	if err != nil {
		return
	}
	d.mu.Lock()
	d.attitude = *a
	d.mu.Unlock()
	if d.onAttitude != nil {
		d.onAttitude(a)
	}
}

func (d *Device) handleFlightMode(f *crsf.Frame) {
	fm, err := crsf.DecodeFlightMode(f)
	if err != nil {
		return
	}
	d.mu.Lock()
	d.mode = *fm
	d.mu.Unlock()
	if d.onFlightMode != nil {
		d.onFlightMode(fm)
	}
}

func (d *Device) handleLinkStats(f *crsf.Frame) {
	ls, err := crsf.DecodeLinkStats(f)
	if err != nil {
		return
	}
	d.mu.Lock()
	d.link = *ls
	d.mu.Unlock()
	if d.onLinkStats != nil {
		d.onLinkStats(ls)
	}
}
