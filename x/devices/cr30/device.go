package cr30

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"strings"
	"time"

	"github.com/itohio/EasyRobot/x/devices"
)

const (
	defaultBaudRate = 19200
	packetSize      = 60

	// Command bytes
	cmdDeviceInfo = 0x0A
	cmdInit       = 0x17
	cmdCheck      = 0x13
	cmdQueryParam = 0x28
	cmdMeasure    = 0x01

	// Subcommand bytes for device info
	subcmdName     = 0x00
	subcmdSerial   = 0x01
	subcmdFirmware = 0x02
	subcmdBuild    = 0x03

	// Subcommand bytes for measurement
	subcmdMeasureHeader = 0x09
	subcmdChunk10       = 0x10
	subcmdChunk11       = 0x11
	subcmdChunk12       = 0x12
	subcmdChunk13       = 0x13

	// Default timeouts
	defaultTimeout      = 1 * time.Second
	defaultMeasureTimeout = 1.5 * time.Second
	defaultWaitTimeout  = 15 * time.Second
)

var (
	ErrNotConnected     = errors.New("cr30: device not connected")
	ErrTimeout          = errors.New("cr30: operation timed out")
	ErrInvalidResponse  = errors.New("cr30: invalid response")
	ErrIncompleteData   = errors.New("cr30: incomplete data")
)

// Info contains device identification and version information.
type Info struct {
	Name     string
	Model    string
	Serial   string
	Firmware string
	Build    string
}

// Header contains measurement header packet information.
type Header struct {
	Cmd     byte
	Subcmd  byte
	Param   byte
	Payload []byte
}

// Chunk contains information about a measurement data chunk.
type Chunk struct {
	Subcmd    byte
	Payload   []byte
	SPDBytes  []byte
	SPDFloats []float32
	Error     string
}

// Measurement contains a complete measurement result.
type Measurement struct {
	Spectrum []float32 // 31 spectral values (400-700nm, 10nm steps)
	Header   Header
	Chunks   []Chunk
	Raw      []byte // Raw SPD bytes (124 bytes = 31 floats)
}

// Device wraps a serial connection to a CR30 colorimeter.
type Device struct {
	serial devices.Serial
	builder *PacketBuilder
	parser  *PacketParser
	info    Info
	connected bool
	verbose   bool
}

// New creates a new CR30 device connection.
func New(serial devices.Serial) *Device {
	return &Device{
		serial:   serial,
		builder:  NewPacketBuilder(),
		parser:   NewPacketParser(),
		connected: false,
		verbose:   false,
	}
}

// SetVerbose enables or disables verbose logging.
func (d *Device) SetVerbose(v bool) {
	d.verbose = v
}

// Connect establishes connection with the device and performs handshake.
func (d *Device) Connect() error {
	if d.connected {
		return nil
	}

	// Flush any existing data
	d.flush()

	// Perform handshake
	if err := d.Handshake(); err != nil {
		return err
	}

	d.connected = true
	return nil
}

// Disconnect closes the connection.
func (d *Device) Disconnect() error {
	d.connected = false
	return nil
}

// Connected returns whether the device is connected.
func (d *Device) Connected() bool {
	return d.connected
}

// DeviceInfo returns the device information retrieved during handshake.
func (d *Device) DeviceInfo() Info {
	return d.info
}

// Handshake performs the full CR30 handshake and populates device info.
func (d *Device) Handshake() error {
	// Flush receive buffer
	d.flush()

	// Query device information
	aaCommands := []struct {
		subcmd byte
		parse  func([]byte)
	}{
		{subcmdName, d.parseName},
		{subcmdSerial, d.parseSerial},
		{subcmdFirmware, d.parseFirmware},
		{subcmdBuild, nil}, // Optional
	}

	for _, cmd := range aaCommands {
		response, err := d.sendRecv(startAA, cmdDeviceInfo, cmd.subcmd, 0x00, nil, defaultTimeout)
		if err != nil {
			if d.verbose {
				// Log but continue
			}
			continue
		}

		payload, err := d.parser.ExtractPayload(response)
		if err != nil {
			continue
		}

		if cmd.parse != nil {
			cmd.parse(payload)
		}
	}

	// Initialize device
	_, _ = d.sendRecv(startBB, cmdInit, 0x00, 0x00, nil, defaultTimeout)

	// Check command
	checkData := []byte("Check")
	checkPayload := make([]byte, 12)
	copy(checkPayload, checkData)
	_, _ = d.sendRecv(startBB, cmdCheck, 0x00, 0x00, checkPayload, defaultTimeout)

	// Query parameters
	for _, idx := range []byte{0x00, 0x01, 0x02, 0x03, 0xFF} {
		_, _ = d.sendRecv(startBB, cmdQueryParam, 0x00, idx, nil, defaultTimeout)
	}

	return nil
}

// Measure triggers a PC-initiated measurement and reads the data.
func (d *Device) Measure(ctx context.Context) (*Measurement, error) {
	if !d.connected {
		return nil, ErrNotConnected
	}

	// Send measurement trigger
	headerPacket, err := d.sendRecvCtx(ctx, startBB, cmdMeasure, 0x00, 0x00, nil)
	if err != nil {
		return nil, err
	}

	return d.readMeasurementCtx(ctx, headerPacket)
}

// WaitMeasurement waits for a button press and reads the measurement.
func (d *Device) WaitMeasurement(ctx context.Context) (*Measurement, error) {
	if !d.connected {
		return nil, ErrNotConnected
	}

	// Wait for button press (device sends measurement header)
	headerPacket, err := d.recvCtx(ctx)
	if err != nil {
		return nil, err
	}

	// Verify it's a measurement header
	packet, err := d.parser.ParsePacket(headerPacket)
	if err != nil {
		return nil, err
	}
	if packet.Subcmd() != subcmdMeasureHeader {
		return nil, ErrInvalidResponse
	}

	return d.readMeasurementCtx(ctx, headerPacket)
}

// readMeasurementCtx reads a complete measurement after receiving the header.
func (d *Device) readMeasurementCtx(ctx context.Context, headerPacket []byte) (*Measurement, error) {
	// Flush receive buffer
	d.flush()

	// Parse header
	packet, err := d.parser.ParsePacket(headerPacket)
	if err != nil {
		return nil, err
	}

	measurement := &Measurement{
		Header: Header{
			Cmd:     packet.Cmd(),
			Subcmd:  packet.Subcmd(),
			Param:   packet.Param(),
			Payload: make([]byte, len(packet.Payload())),
		},
		Chunks: make([]Chunk, 0),
	}
	copy(measurement.Header.Payload, packet.Payload())

	// Reset parser state for new measurement
	d.parser.ResetSPDCollection()

	// Read all data chunks
	chunkSubcmds := []byte{subcmdChunk10, subcmdChunk11, subcmdChunk12, subcmdChunk13}
	for _, subcmd := range chunkSubcmds {
		// Check context before each chunk
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		chunkPacket, err := d.sendRecvCtx(ctx, startBB, cmdMeasure, subcmd, 0x00, nil)
		if err != nil {
			if d.verbose {
				// Log but continue
			}
			measurement.Chunks = append(measurement.Chunks, Chunk{
				Subcmd: subcmd,
				Error:  err.Error(),
			})
			continue
		}

		chunkInfo, err := d.parser.ParseSPDChunk(chunkPacket)
		if err != nil {
			if d.verbose {
				// Log but continue
			}
			measurement.Chunks = append(measurement.Chunks, Chunk{
				Subcmd: subcmd,
				Error:  err.Error(),
			})
			continue
		}

		measurement.Chunks = append(measurement.Chunks, Chunk{
			Subcmd:    chunkInfo.Subcmd,
			Payload:   chunkInfo.Payload,
			SPDBytes:  chunkInfo.SPDBytes,
			SPDFloats: chunkInfo.SPDFloats,
		})
	}

	// Get accumulated SPD bytes
	spdBytes := d.parser.GetAccumulatedSPD()
	measurement.Raw = spdBytes

	// Parse SPD data (124 bytes = 31 floats)
	if len(spdBytes) >= 124 {
		measurement.Spectrum = parseSPDFloats(spdBytes[:124])
	} else {
		return nil, ErrIncompleteData
	}

	return measurement, nil
}

// parseSPDFloats parses 31 float32 values from SPD bytes.
func parseSPDFloats(data []byte) []float32 {
	if len(data) < 124 {
		return nil
	}
	floats := make([]float32, 31)
	for i := 0; i < 31; i++ {
		offset := i * 4
		bits := binary.LittleEndian.Uint32(data[offset : offset+4])
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}

// parseName parses device name and model from payload.
func (d *Device) parseName(payload []byte) {
	if len(payload) < 45 {
		return
	}
	d.info.Name = strings.TrimRight(string(payload[5:30]), "\x00")
	d.info.Model = strings.TrimRight(string(payload[35:45]), "\x00")
}

// parseSerial parses serial number from payload.
func (d *Device) parseSerial(payload []byte) {
	if len(payload) < 52 {
		return
	}
	part1 := strings.TrimRight(string(payload[16:30]), "\x00")
	part2 := strings.TrimRight(string(payload[45:52]), "\x00")
	if part2 != "" {
		d.info.Serial = part1 + " - " + part2
	} else {
		d.info.Serial = part1
	}
}

// parseFirmware parses firmware version and build from payload.
func (d *Device) parseFirmware(payload []byte) {
	if len(payload) < 30 {
		return
	}
	d.info.Build = strings.TrimRight(string(payload[1:20]), "\x00")
	d.info.Firmware = strings.TrimRight(string(payload[20:30]), "\x00")
}

// send sends a packet to the device.
func (d *Device) send(start, cmd, subcmd, param byte, data []byte) error {
	packet, err := d.builder.BuildPacket(start, cmd, subcmd, param, data)
	if err != nil {
		return err
	}

	if d.verbose {
		// Log packet (hex format)
	}

	_, err = d.serial.Write(packet)
	return err
}

// recv receives a packet from the device.
func (d *Device) recv(timeout time.Duration) ([]byte, error) {
	buffer := make([]byte, packetSize)
	deadline := time.Now().Add(timeout)

	// Read exactly 60 bytes
	read := 0
	for read < packetSize {
		if time.Now().After(deadline) {
			return nil, ErrTimeout
		}

		n, err := d.serial.Read(buffer[read:])
		if err != nil && err != io.EOF {
			return nil, err
		}
		if n == 0 {
			time.Sleep(10 * time.Millisecond)
			continue
		}
		read += n
	}

	// Validate packet
	if !d.parser.IsValidPacket(buffer) {
		return nil, ErrInvalidResponse
	}

	return buffer, nil
}

// recvCtx receives a packet from the device using context for cancellation.
func (d *Device) recvCtx(ctx context.Context) ([]byte, error) {
	buffer := make([]byte, packetSize)

	// Read exactly 60 bytes
	read := 0
	for read < packetSize {
		// Check context cancellation
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		n, err := d.serial.Read(buffer[read:])
		if err != nil && err != io.EOF {
			return nil, err
		}
		if n == 0 {
			// Use a select to check context while sleeping
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(10 * time.Millisecond):
				// Continue
			}
			continue
		}
		read += n
	}

	// Validate packet
	if !d.parser.IsValidPacket(buffer) {
		return nil, ErrInvalidResponse
	}

	return buffer, nil
}

// sendRecv sends a packet and waits for a response.
func (d *Device) sendRecv(start, cmd, subcmd, param byte, data []byte, timeout time.Duration) ([]byte, error) {
	if err := d.send(start, cmd, subcmd, param, data); err != nil {
		return nil, err
	}
	return d.recv(timeout)
}

// sendRecvCtx sends a packet and waits for a response using context for cancellation.
func (d *Device) sendRecvCtx(ctx context.Context, start, cmd, subcmd, param byte, data []byte) ([]byte, error) {
	if err := d.send(start, cmd, subcmd, param, data); err != nil {
		return nil, err
	}
	return d.recvCtx(ctx)
}

// flush discards any pending data in the receive buffer.
func (d *Device) flush() {
	buffer := make([]byte, 1024)
	for {
		n, _ := d.serial.Read(buffer)
		if n == 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
}

