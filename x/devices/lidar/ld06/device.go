package ld06

import (
	"context"
	"encoding/binary"
	"io"
	"math"
	"sync"
	"time"

	devio "github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/math/control/pid"
	mat "github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
)

// Device is a streaming LiDAR that fills a 2xN matrix:
// row 0: distance (mm), row 1: angle (deg)
type Device struct {
	ser    devio.Serial
	motor  devio.PWM
	ctx    context.Context
	cancel func()

	// preallocated buffers
	backing []float32  // length 2*maxSamples
	mat2xN  mat.Matrix // 2 x maxSamples view into backing

	maxSamples int
	count      int // number of valid columns in current scan

	onRead func(matTypes.Matrix)

	// parser state
	buf      []byte
	writeIdx int

	// rotation detection
	lastEndAngleDeg float64
	rotationStarted bool

	// motor / point-count control
	targetPoints int
	currentDuty  float32
	pid          pid.PID1D
	lastRotTime  time.Time
	calibrating  bool
	minDuty      float32
	maxDuty      float32
	minPoints    int
	maxPoints    int

	mu        sync.Mutex
	startOnce sync.Once
}

// New creates a new LD06 LiDAR device with preallocated storage for up to maxPoints samples per rotation.
// motor can be nil to disable PWM control (device will read data but not control motor speed).
// If targetPoints is 0 and motor is not nil, the device will auto-calibrate the point count by sweeping PWM during Configure.
// The internal read loop starts in Configure and stops when ctx is done. Motor control loops only start if motor is not nil.
func New(ctx context.Context, ser devio.Serial, motor devio.PWM, targetPoints, maxPoints int) *Device {
	if maxPoints <= 0 {
		maxPoints = 3600 // LD06 can have up to 4500 points/sec, ~450 points per rotation at 10Hz
	}
	cctx, cancel := context.WithCancel(ctx)

	// backing: [dist(0..max-1) | angle(0..max-1)]
	backing := make([]float32, 2*maxPoints)
	m := mat.New(2, maxPoints, backing...)

	d := &Device{
		ser:          ser,
		motor:        motor,
		ctx:          cctx,
		cancel:       cancel,
		backing:      backing,
		mat2xN:       m,
		maxSamples:   maxPoints,
		targetPoints: targetPoints,
		buf:          make([]byte, 0, 4096),
		minDuty:      0.1,
		maxDuty:      0.8,
		lastEndAngleDeg: -1,
	}
	return d
}

// Close stops the internal read loop.
func (d *Device) Close() {
	if d.cancel != nil {
		d.cancel()
	}
}

// OnRead registers a callback that is invoked with a view of the internal 2xN matrix
// each time a full rotation is assembled. The matrix columns equal the number of angles in that scan.
func (d *Device) OnRead(fn func(matTypes.Matrix)) {
	d.mu.Lock()
	d.onRead = fn
	d.mu.Unlock()
}

// Read copies the latest completed scan into dst and returns number of valid angles copied.
// Expects dst to be a 2xK matrix; copies min(K, available) columns.
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
// LD06 scans 360° continuously.
func (d *Device) GetMinAngle() float32 {
	return 0.0
}

// GetMaxAngle returns the maximum angle (in degrees) that this LiDAR can measure.
// LD06 scans 360° continuously.
func (d *Device) GetMaxAngle() float32 {
	return 360.0
}

// GetPointCount returns the number of points in the current/latest scan.
func (d *Device) GetPointCount() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.count
}

// Configure starts the internal read loop. If motor is not nil, also starts motor control loops.
// init is ignored for now, kept for consistency with other device drivers.
func (d *Device) Configure(_ bool) error {
	d.startOnce.Do(func() {
		if d.motor != nil {
			_ = d.motor.Set(0)
			d.currentDuty = 0
		}
		go d.readLoop()
		if d.motor != nil {
			if d.targetPoints == 0 {
				d.calibrating = true
				go d.calibrationLoop()
			} else {
				d.initPID()
				go d.controlLoop()
			}
		}
	})
	return nil
}

func (d *Device) readLoop() {
	tmp := make([]byte, 1024)
	for {
		select {
		case <-d.ctx.Done():
			return
		default:
		}
		n, err := d.ser.Read(tmp)
		if n > 0 {
			d.buf = append(d.buf, tmp[:n]...)
			for {
				consumed := d.consumeOnePacket()
				if consumed == 0 {
					break
				}
				copy(d.buf, d.buf[consumed:])
				d.buf = d.buf[:len(d.buf)-consumed]
			}
		}
		if err != nil {
			if err == io.EOF {
				return
			}
			// continue on transient errors
		}
	}
}

func (d *Device) consumeOnePacket() int {
	const header = 0x54
	i := 0
	for i < len(d.buf) && d.buf[i] != header {
		i++
	}
	if i > 0 {
		return i
	}
	if len(d.buf) < 2 {
		return 0
	}

	// Data length (byte 1)
	dataLen := int(d.buf[1])
	if dataLen == 0 || dataLen > 255 {
		return 1 // skip invalid header
	}

	// Packet length = 6 (header + len + speed + startAngle) + 3*dataLen + 2 (endAngle) + 2 (timestamp) + 1 (CRC)
	packetLen := 6 + 3*dataLen + 2 + 2 + 1
	if len(d.buf) < packetLen {
		return 0 // wait for more data
	}

	packet := d.buf[:packetLen]

	// Verify CRC8
	expectedCRC := packet[packetLen-1]
	if crc8(packet[:packetLen-1]) != expectedCRC {
		return 1 // bad CRC, skip header
	}

	// Parse packet
	// 2-3: radar speed (0.01 Hz units)
	radarSpeedCentiHz := binary.LittleEndian.Uint16(packet[2:4])
	_ = radarSpeedCentiHz // currently unused

	// 4-5: start angle (0.01° units, int16)
	startAngleRaw := int16(binary.LittleEndian.Uint16(packet[4:6]))
	startAngleDeg := float64(startAngleRaw) * 0.01

	// 6..(6+3n-1): data points
	// (6+3n)..(6+3n+1): end angle
	endAngleRaw := int16(binary.LittleEndian.Uint16(packet[6+3*dataLen : 6+3*dataLen+2]))
	endAngleDeg := float64(endAngleRaw) * 0.01

	// Normalize angles to [0, 360)
	startAngleDeg = normalizeAngle(startAngleDeg)
	endAngleDeg = normalizeAngle(endAngleDeg)

	// Check for rotation completion: if end angle < start angle (wraparound)
	// LD06 sends packets continuously; a full rotation is detected when
	// the end angle wraps around (e.g., end=10°, start=350°)
	rotationComplete := false
	if d.rotationStarted && d.lastEndAngleDeg >= 0 {
		// Check for wraparound: if end angle is significantly less than last end angle
		// and we're near 360°, or if start angle is much larger than end angle
		angleDiff := endAngleDeg - d.lastEndAngleDeg
		if angleDiff < -180 {
			// Wrapped around (e.g., lastEnd=350°, end=10°)
			rotationComplete = true
		} else if startAngleDeg > endAngleDeg+180 {
			// Start angle is much larger than end angle (wraparound in this packet)
			rotationComplete = true
		}
	} else {
		d.rotationStarted = true
	}

	// Process data points
	distRow := d.mat2xN[0][:]
	angRow := d.mat2xN[1][:]

	// Linear interpolation of angles between start and end
	angleSpan := endAngleDeg - startAngleDeg
	if angleSpan < -180 {
		angleSpan += 360 // handle wraparound
	}
	angleStep := angleSpan / float64(dataLen)

	for i := 0; i < dataLen; i++ {
		if d.writeIdx >= d.maxSamples {
			// overflow, reset rotation
			d.writeIdx = 0
			d.rotationStarted = false
			d.lastEndAngleDeg = -1
			return packetLen
		}

		// Distance (mm, uint16, little-endian)
		offset := 6 + 3*i
		distMm := uint16(packet[offset]) | (uint16(packet[offset+1]) << 8)
		_ = packet[offset+2] // intensity, ignored for now

		// Angle interpolation
		angle := startAngleDeg + angleStep*float64(i)
		angle = normalizeAngle(angle)

		distRow[d.writeIdx] = float32(distMm)
		angRow[d.writeIdx] = float32(angle)
		d.writeIdx++
	}

	d.lastEndAngleDeg = endAngleDeg

	// If rotation complete, emit scan
	if rotationComplete {
		d.emitScan()
	}

	return packetLen
}

func normalizeAngle(deg float64) float64 {
	deg = math.Mod(deg, 360.0)
	if deg < 0 {
		deg += 360.0
	}
	return deg
}

func (d *Device) emitScan() {
	d.mu.Lock()
	d.count = d.writeIdx
	now := time.Now()
	prev := d.lastRotTime
	d.lastRotTime = now
	// prepare view of size 2 x count (re-slice; zero-alloc)
	view := d.mat2xN.View().(mat.Matrix)
	view[0] = view[0][:d.count]
	view[1] = view[1][:d.count]
	cb := d.onRead
	points := d.count
	rotationDt := now.Sub(prev)
	calibrating := d.calibrating
	d.mu.Unlock()

	if cb != nil {
		cb(view)
	}

	if d.motor != nil && rotationDt > 0 {
		if calibrating {
			d.updateCalibration(points)
		} else {
			d.updatePID(points, float32(rotationDt.Seconds()))
		}
	}

	// reset rotation
	d.writeIdx = 0
	d.rotationStarted = false
	d.lastEndAngleDeg = -1
}

func (d *Device) initPID() {
	// Simple PID defaults for point-count control; can be tuned later.
	p := float32(0.001)
	i := float32(0.0005)
	dGain := float32(0.0001)
	d.pid = pid.New1D(p, i, dGain, 0.0, 1.0)
	d.pid.Target = float32(d.targetPoints)
	d.pid.Reset()
}

func (d *Device) updatePID(points int, dt float32) {
	if dt <= 0 || d.targetPoints <= 0 {
		return
	}
	d.pid.Input = float32(points)
	d.pid.Update(dt)
	duty := d.pid.Output
	if duty < 0 {
		duty = 0
	}
	if duty > 1 {
		duty = 1
	}
	if duty == d.currentDuty || d.motor == nil {
		return
	}
	_ = d.motor.Set(duty)
	d.currentDuty = duty
}

// calibrationLoop ramps duty from minDuty up to maxDuty to characterize point counts,
// then sets targetPoints to the average and enables PID control.
func (d *Device) calibrationLoop() {
	if d.motor == nil {
		return
	}
	// Initial slow ramp (~1s) to start receiving data.
	const steps = 20
	stepDur := 50 * time.Millisecond
	for i := 1; i <= steps; i++ {
		select {
		case <-d.ctx.Done():
			return
		default:
		}
		duty := d.minDuty * float32(i) / float32(steps)
		if duty > d.minDuty {
			duty = d.minDuty
		}
		_ = d.motor.Set(duty)
		d.currentDuty = duty
		time.Sleep(stepDur)
	}

	// Sweep up to maxDuty while rotations arrive; updateCalibration records min/max points.
	const sweepSteps = 30
	for i := 0; i <= sweepSteps; i++ {
		select {
		case <-d.ctx.Done():
			return
		default:
		}
		duty := d.minDuty + (d.maxDuty-d.minDuty)*float32(i)/float32(sweepSteps)
		_ = d.motor.Set(duty)
		d.currentDuty = duty
		time.Sleep(100 * time.Millisecond)
	}

	// Wait for potential last rotations at high duty.
	time.Sleep(300 * time.Millisecond)

	d.mu.Lock()
	minPts := d.minPoints
	maxPts := d.maxPoints
	d.calibrating = false
	if minPts > 0 && maxPts > 0 {
		d.targetPoints = (minPts + maxPts) / 2
	}
	d.mu.Unlock()

	if d.targetPoints > 0 {
		d.initPID()
		go d.controlLoop()
	}
}

func (d *Device) updateCalibration(points int) {
	if points <= 0 {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	// Track the highest count at low duty and lowest count at high duty.
	if d.minPoints == 0 || points > d.minPoints {
		d.minPoints = points
	}
	if d.maxPoints == 0 || points < d.maxPoints {
		d.maxPoints = points
	}
}

// controlLoop periodically nudges duty using PID in case rotations are sparse.
// Only runs when motor is not nil (started from Configure).
func (d *Device) controlLoop() {
	if d.motor == nil {
		return
	}
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.mu.Lock()
			points := d.count
			last := d.lastRotTime
			d.mu.Unlock()
			if points == 0 || last.IsZero() {
				continue
			}
			dt := float32(time.Since(last).Seconds())
			if dt <= 0 {
				continue
			}
			d.updatePID(points, dt)
		}
	}
}

