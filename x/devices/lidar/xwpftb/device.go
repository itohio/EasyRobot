package xwpftb

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
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
	buf            []byte
	slicesInRotate int
	writeIdx       int

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

// New creates a new XWPFTB LiDAR device with preallocated storage for up to maxPoints samples per rotation.
// motor can be nil to disable PWM control (device will read data but not control motor speed).
// If targetPoints is 0 and motor is not nil, the device will auto-calibrate the point count by sweeping PWM during Configure.
// The internal read loop starts in Configure and stops when ctx is done. Motor control loops only start if motor is not nil.
func New(ctx context.Context, ser devio.Serial, motor devio.PWM, targetPoints, maxPoints int) *Device {
	if maxPoints <= 0 {
		maxPoints = 2048
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
// XWPFTB scans 360° continuously.
func (d *Device) GetMinAngle() float32 {
	return 0.0
}

// GetMaxAngle returns the maximum angle (in degrees) that this LiDAR can measure.
// XWPFTB scans 360° continuously.
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
	readCount := 0
	for {
		select {
		case <-d.ctx.Done():
			slog.Info("XWPFTB readLoop stopping", "reason", "context cancelled")
			return
		default:
		}
		n, err := d.ser.Read(tmp)
		if n > 0 {
			readCount++
			hexDump := hex.EncodeToString(tmp[:n])
			slog.Debug("XWPFTB raw read",
				"bytes", n,
				"read_count", readCount,
				"buffer_size", len(d.buf),
				"hex", hexDump,
			)
			d.buf = append(d.buf, tmp[:n]...)
			slog.Info("XWPFTB buffer after append",
				"buffer_size", len(d.buf),
				"first_16_bytes_hex", hex.EncodeToString(d.buf[:min(16, len(d.buf))]),
			)
			for {
				consumed := d.consumeOneFrame()
				if consumed == 0 {
					break
				}
				slog.Debug("XWPFTB consumed frame",
					"consumed_bytes", consumed,
					"remaining_buffer_size", len(d.buf)-consumed,
				)
				copy(d.buf, d.buf[consumed:])
				d.buf = d.buf[:len(d.buf)-consumed]
			}
		}
		if err != nil {
			if err == io.EOF {
				slog.Warn("XWPFTB readLoop EOF", "read_count", readCount)
				return
			}
			slog.Warn("XWPFTB readLoop error", "err", err, "read_count", readCount)
			// continue on transient errors
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (d *Device) consumeOneFrame() int {
	const (
		headerByte1 = 0x55
		headerByte2 = 0xAA
		frameSize   = 60 // Fixed frame size per protocol
	)

	// Find the 2-byte header (0x55 0xAA)
	i := 0
	for i < len(d.buf)-1 {
		if d.buf[i] == headerByte1 && d.buf[i+1] == headerByte2 {
			break
		}
		i++
	}
	if i > 0 {
		slog.Debug("XWPFTB skipping bytes before header",
			"skipped_bytes", i,
			"skipped_hex", hex.EncodeToString(d.buf[:i]),
		)
		return i
	}
	if i >= len(d.buf)-1 {
		// Header not found or incomplete
		if len(d.buf) >= 2 {
			// Skip first byte and try again next time
			return 1
		}
		return 0
	}

	// Check if we have enough data for a complete frame
	if len(d.buf) < i+frameSize {
		slog.Debug("XWPFTB waiting for more data",
			"needed", i+frameSize,
			"have", len(d.buf),
		)
		return 0
	}

	frame := d.buf[i : i+frameSize]

	// Verify frame structure
	if frame[0] != headerByte1 || frame[1] != headerByte2 {
		slog.Warn("XWPFTB header mismatch after sync",
			"byte0", fmt.Sprintf("0x%02X", frame[0]),
			"byte1", fmt.Sprintf("0x%02X", frame[1]),
		)
		return i + 1
	}

	ftype := frame[2]
	dataLen := frame[3]

	if ftype != 0x23 {
		slog.Warn("XWPFTB unexpected frame type",
			"type", fmt.Sprintf("0x%02X", ftype),
			"expected", "0x23",
		)
		return i + 1
	}

	if dataLen != 0x10 {
		slog.Warn("XWPFTB unexpected data length",
			"data_len", dataLen,
			"expected", 0x10,
		)
		return i + 1
	}

	// Verify CRC (last 2 bytes)
	expectedCRC := binary.LittleEndian.Uint16(frame[frameSize-2:])
	calculatedCRC := crc16Cumulative(frame[:frameSize-2])
	if calculatedCRC != expectedCRC {
		slog.Warn("XWPFTB CRC mismatch",
			"expected", fmt.Sprintf("0x%04X", expectedCRC),
			"calculated", fmt.Sprintf("0x%04X", calculatedCRC),
			"frame_hex", hex.EncodeToString(frame),
		)
		return i + 1
	}

	slog.Debug("XWPFTB frame parsed",
		"type", fmt.Sprintf("0x%02X", ftype),
		"data_len", dataLen,
		"frame_hex", hex.EncodeToString(frame),
	)

	// Process the measurement data (bytes 2-59, excluding CRC)
	// frame[2] = TYPE, frame[3] = DATA_LENGTH, frame[4+] = measurement data
	d.consumeMeasurement(frame[2:])

	return i + frameSize
}

func (d *Device) consumeMeasurement(data []byte) {
	// Protocol format (after header 0x55 0xAA):
	// [0] = TYPE (0x23)
	// [1] = DATA_LENGTH (0x10)
	// [2-3] = speed_L, speed_H
	// [4-5] = start_angle_L, start_angle_H
	// [6-53] = 16 measurements, each 3 bytes: distance_L, distance_H, intensity
	// [54-55] = end_angle_L, end_angle_H
	// [56-57] = crc16_L, crc16_H (already verified)

	const (
		expectedDataLen = 0x10 // 16 measurements
		measurements    = 16
		bytesPerPoint   = 3
	)

	if len(data) < 56 {
		slog.Warn("XWPFTB measurement data too short",
			"data_len", len(data),
			"expected", 56,
		)
		return
	}

	// Extract speed (bytes 2-3, little-endian)
	speed := binary.LittleEndian.Uint16(data[2:4])

	// Extract start angle using the formula from README:
	// uint16_t start_angle = (((data[7] & 0x7F) << 8) + data[6]) - 0x2000;
	// In README, data[6] = start_angle_L, data[7] = start_angle_H
	// In our data array: data[4] = start_angle_L, data[5] = start_angle_H
	// So we use: data[4] (L) and data[5] (H) which corresponds to README's data[6] and data[7]
	startAngleRaw := uint16(data[4]) | (uint16(data[5]&0x7F) << 8)
	startAngleDeg := float64(int16(startAngleRaw)-0x2000) / 100.0 // Convert to degrees

	// Extract end angle (same formula, but at different position)
	// In README's data array, end_angle would be at data[56] and data[57]
	// In our data array (which starts at frame[2]), end_angle is at data[54] and data[55]
	endAngleRaw := uint16(data[54]) | (uint16(data[55]&0x7F) << 8)
	endAngleDeg := float64(int16(endAngleRaw)-0x2000) / 100.0

	slog.Debug("XWPFTB measurement header",
		"speed", speed,
		"start_angle_deg", startAngleDeg,
		"end_angle_deg", endAngleDeg,
	)

	distRow := d.mat2xN[0][:]
	angRow := d.mat2xN[1][:]

	pointsAdded := 0
	angleSpan := endAngleDeg - startAngleDeg
	if angleSpan < 0 {
		angleSpan += 360.0
	}
	stepDeg := angleSpan / float64(measurements)

	for i := 0; i < measurements; i++ {
		if d.writeIdx >= d.maxSamples {
			slog.Warn("XWPFTB write index overflow",
				"write_idx", d.writeIdx,
				"max_samples", d.maxSamples,
				"points_in_slice", measurements,
				"slices_in_rotate", d.slicesInRotate,
			)
			d.slicesInRotate = 0
			d.writeIdx = 0
			return
		}

		// Measurement data starts at byte 6 in our data array
		// In README's data array (starting from TYPE byte), measurements start at byte 8
		// Each measurement: distance_L, distance_H, intensity
		// README formula: distance = (((data[9+offset] & 0x3F) << 8) | data[8+offset]) * 0.1
		// where offset = i*3 for measurement i
		// In our data array: data[6+i*3] = distance_L, data[7+i*3] = distance_H, data[8+i*3] = intensity
		// This corresponds to README's data[8+i*3] and data[9+i*3]
		offset := 6 + i*bytesPerPoint
		if offset+bytesPerPoint > len(data) {
			slog.Warn("XWPFTB measurement point out of bounds",
				"point_index", i,
				"offset", offset,
				"data_len", len(data),
			)
			break
		}

		// Distance calculation matching README formula:
		// README: (((data[9+offset] & 0x3F) << 8) | data[8+offset])
		// Our data: data[6+i*3] = distance_L (README's data[8+i*3])
		//           data[7+i*3] = distance_H (README's data[9+i*3])
		distanceRaw := uint16(data[offset]) | ((uint16(data[offset+1]) & 0x3F) << 8)
		distMm := float64(distanceRaw) * 0.1 // Scale factor is 0.1mm per unit

		// Calculate angle for this point
		angle := startAngleDeg + stepDeg*float64(i)
		angle = math.Mod(angle, 360.0)
		if angle < 0 {
			angle += 360.0
		}

		distRow[d.writeIdx] = float32(distMm)
		angRow[d.writeIdx] = float32(angle)
		d.writeIdx++
		pointsAdded++
	}

	slog.Debug("XWPFTB measurement processed",
		"points_added", pointsAdded,
		"total_points", measurements,
		"write_idx", d.writeIdx,
		"start_angle_deg", startAngleDeg,
		"end_angle_deg", endAngleDeg,
		"step_deg", stepDeg,
	)

	d.slicesInRotate++
	if d.slicesInRotate >= 15 {
		// rotation complete
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

		slog.Info("XWPFTB rotation complete",
			"points", points,
			"slices", d.slicesInRotate,
			"rotation_dt_ms", rotationDt.Milliseconds(),
			"calibrating", calibrating,
		)

		if cb != nil {
			cb(view)
		} else {
			slog.Warn("XWPFTB no callback registered for rotation complete")
		}

		if d.motor != nil && rotationDt > 0 {
			if calibrating {
				d.updateCalibration(points)
			} else {
				d.updatePID(points, float32(rotationDt.Seconds()))
			}
		}

		// reset rotation
		d.slicesInRotate = 0
		d.writeIdx = 0
	} else {
		slog.Debug("XWPFTB slice added to rotation",
			"slices_in_rotate", d.slicesInRotate,
			"points_so_far", d.writeIdx,
			"slices_needed", 15,
		)
	}
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
