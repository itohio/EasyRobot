// Package rx provides a CRSF receiver device that reads RC channel data
// from an ELRS/Crossfire receiver and sends telemetry back.
package rx

import (
	"context"
	"io"
	"sync"
	"time"

	"github.com/itohio/EasyRobot/x/devices/crsf"
)

const failsafeTimeout = time.Second

// Device reads RC commands from a CRSF receiver and optionally sends
// telemetry back. It manages its own read loop and provides thread-safe
// access to the latest channel/link state.
type Device struct {
	dec    *crsf.Decoder
	enc    *crsf.Encoder
	ctx    context.Context
	cancel func()

	mu            sync.RWMutex
	channels      crsf.Channels
	link          crsf.LinkStatistics
	lastChannelAt time.Time
	failsafe      bool

	Telemetry Scheduler

	onChannels  func(*crsf.Channels)
	onLinkStats func(*crsf.LinkStatistics)
	onFailsafe  func(bool)
	onFrame     func(*crsf.Frame)

	startOnce sync.Once
}

// New creates a receiver device. Pass nil for w if telemetry is not needed.
func New(ctx context.Context, r io.Reader, w io.Writer) *Device {
	cctx, cancel := context.WithCancel(ctx)
	d := &Device{
		dec:    crsf.NewDecoder(r),
		ctx:    cctx,
		cancel: cancel,
	}
	if w != nil {
		d.enc = crsf.NewEncoder(w)
	}
	return d
}

// Close stops the read loop.
func (d *Device) Close() {
	if d.cancel != nil {
		d.cancel()
	}
}

func (d *Device) OnChannels(fn func(*crsf.Channels))        { d.onChannels = fn }
func (d *Device) OnLinkStats(fn func(*crsf.LinkStatistics)) { d.onLinkStats = fn }

// OnFailsafe registers a callback for failsafe state changes.
// Called with true when channel frames stop arriving (>1s), false when they resume.
func (d *Device) OnFailsafe(fn func(active bool)) { d.onFailsafe = fn }

// OnFrame registers a catch-all callback for any frame type not handled internally.
// Use this to implement device discovery, parameter config, or direct commands
// without modifying the library.
func (d *Device) OnFrame(fn func(*crsf.Frame)) { d.onFrame = fn }

// Start begins the background read loop that decodes incoming frames.
func (d *Device) Start() {
	d.startOnce.Do(func() {
		go d.readLoop()
	})
}

// Channels returns a copy of the latest channel values.
func (d *Device) Channels() crsf.Channels {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.channels
}

// LinkStats returns a copy of the latest link statistics.
func (d *Device) LinkStats() crsf.LinkStatistics {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.link
}

// IsFailsafe reports whether the receiver has lost connection (no channel
// frames for >1 second).
func (d *Device) IsFailsafe() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.failsafe
}

// SendBattery writes battery telemetry to the transmitter.
func (d *Device) SendBattery(b *crsf.Battery) error {
	if d.enc == nil {
		return nil
	}
	return d.enc.WriteBattery(b)
}

// SendGPS writes GPS telemetry to the transmitter.
func (d *Device) SendGPS(g *crsf.GPS) error {
	if d.enc == nil {
		return nil
	}
	return d.enc.WriteGPS(g)
}

// SendAttitude writes attitude telemetry to the transmitter.
func (d *Device) SendAttitude(a *crsf.Attitude) error {
	if d.enc == nil {
		return nil
	}
	return d.enc.WriteAttitude(a)
}

// SendFlightMode writes flight mode string to the transmitter.
func (d *Device) SendFlightMode(fm *crsf.FlightMode) error {
	if d.enc == nil {
		return nil
	}
	return d.enc.WriteFlightMode(fm)
}

// SendFrame writes an arbitrary frame (for config responses, etc.).
func (d *Device) SendFrame(f *crsf.Frame) error {
	if d.enc == nil {
		return nil
	}
	return d.enc.WriteFrame(f)
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
			d.checkFailsafe()
			time.Sleep(time.Millisecond)
			continue
		}
		d.dispatch(f)
	}
}

func (d *Device) dispatch(f *crsf.Frame) {
	switch f.Type {
	case crsf.TypeChannelsPacked:
		d.handleChannels(f)
	case crsf.TypeLinkStats, crsf.TypeLinkStatsRX:
		d.handleLinkStats(f)
	default:
		if d.onFrame != nil {
			d.onFrame(f)
		}
	}
}

func (d *Device) handleChannels(f *crsf.Frame) {
	ch, err := crsf.DecodeChannels(f)
	if err != nil {
		return
	}
	d.mu.Lock()
	d.channels = *ch
	d.lastChannelAt = time.Now()
	wasFailsafe := d.failsafe
	d.failsafe = false
	d.mu.Unlock()

	if wasFailsafe && d.onFailsafe != nil {
		d.onFailsafe(false)
	}
	if d.onChannels != nil {
		d.onChannels(ch)
	}
	d.sendScheduledTelemetry()
}

func (d *Device) sendScheduledTelemetry() {
	if d.enc == nil {
		return
	}
	if tf := d.Telemetry.Next(); tf != nil {
		_ = d.enc.WriteFrame(tf)
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

func (d *Device) checkFailsafe() {
	d.mu.Lock()
	if d.lastChannelAt.IsZero() || d.failsafe {
		d.mu.Unlock()
		return
	}
	if time.Since(d.lastChannelAt) > failsafeTimeout {
		d.failsafe = true
		d.mu.Unlock()
		if d.onFailsafe != nil {
			d.onFailsafe(true)
		}
		return
	}
	d.mu.Unlock()
}
