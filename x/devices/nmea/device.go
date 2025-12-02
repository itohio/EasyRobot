package nmea

import (
	"context"
	"io"
	"sync"
	"time"

	"github.com/itohio/EasyRobot/x/devices"
)

// Device reads and parses NMEA sentences from a serial stream.
type Device struct {
	serial devices.Serial
	parser *Parser
	ctx    context.Context
	cancel func()

	mu   sync.RWMutex
	data *Data

	onUpdate func(*Data)

	startOnce sync.Once
}

// New creates a new NMEA GPS device.
func New(ctx context.Context, serial devices.Serial) *Device {
	cctx, cancel := context.WithCancel(ctx)
	return &Device{
		serial: serial,
		parser: NewParser(),
		ctx:    cctx,
		cancel: cancel,
		data:   &Data{},
	}
}

// Close stops the device and cancels the read loop.
func (d *Device) Close() {
	if d.cancel != nil {
		d.cancel()
	}
}

// OnUpdate registers a callback that is invoked whenever GPS data is updated.
func (d *Device) OnUpdate(fn func(*Data)) {
	d.mu.Lock()
	d.onUpdate = fn
	d.mu.Unlock()
}

// Data returns the current GPS data.
func (d *Device) Data() *Data {
	d.mu.RLock()
	defer d.mu.RUnlock()
	// Return a copy to avoid race conditions
	result := *d.data
	return &result
}

// Start begins reading and parsing NMEA sentences from the serial stream.
func (d *Device) Start() error {
	var err error
	d.startOnce.Do(func() {
		go d.readLoop()
	})
	return err
}

func (d *Device) readLoop() {
	tmp := make([]byte, 1024)
	for {
		select {
		case <-d.ctx.Done():
			return
		default:
		}

		n, err := d.serial.Read(tmp)
		if n > 0 {
			sentences, _, parseErr := d.parser.ParseBytes(tmp[:n])
			if parseErr == nil {
				for _, s := range sentences {
					d.processSentence(s)
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				return
			}
			// Continue on transient errors
			time.Sleep(10 * time.Millisecond)
		}
	}
}

func (d *Device) processSentence(s *Sentence) {
	var newData *Data
	var err error

	switch s.Type {
	case "GPGGA", "GNGGA":
		newData, err = ParseGGA(s)
	case "GPRMC", "GNRMC":
		newData, err = ParseRMC(s)
	default:
		// Unknown sentence type, ignore
		return
	}

	if err != nil || newData == nil {
		return
	}

	// Merge with existing data
	d.mu.Lock()
	MergeData(d.data, newData)
	cb := d.onUpdate
	dataCopy := *d.data
	d.mu.Unlock()

	// Invoke callback
	if cb != nil {
		cb(&dataCopy)
	}
}

