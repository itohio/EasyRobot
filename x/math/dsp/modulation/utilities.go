package modulation

import (
	"math"
)

// Sync frame configuration
const (
	// PreambleLength is the number of symbols in the sync preamble
	PreambleLength = 16
	// PreamblePattern is alternating 0/1 pattern for sync
	PreamblePattern = 0xAAAA // 10101010...

	// Signal detection thresholds
	MinSignalEnergy = 0.01 // Minimum energy to consider as signal
	LockThreshold   = 0.7  // Lock detection threshold (0-1)
)

// SignalDetector detects presence of signal in noise
type SignalDetector struct {
	// Energy-based detection
	energyWindow []float32
	energyIdx    int
	energySum    float32
	windowSize   int

	// Thresholds
	noiseFloor   float32
	signalThresh float32
	minSNR       float32 // Minimum SNR in dB required to lock (default 3dB)

	// State
	locked      bool
	lockCount   int
	unlockCount int
	sampleCount int // Total samples processed (for adaptation delay)
}

// NewSignalDetector creates a new signal detector
func NewSignalDetector(windowSize int, signalThresh float32) *SignalDetector {
	return &SignalDetector{
		energyWindow: make([]float32, windowSize),
		windowSize:   windowSize,
		signalThresh: signalThresh,
		noiseFloor:   0.001,
		minSNR:       3.0, // Require at least 3dB SNR to lock
	}
}

// SetMinSNR sets the minimum SNR in dB required to lock (default 3dB)
func (sd *SignalDetector) SetMinSNR(snrDb float32) {
	sd.minSNR = snrDb
}

// Update updates the detector with a new sample
func (sd *SignalDetector) Update(sample float32) {
	energy := sample * sample

	// Update sliding window
	oldEnergy := sd.energyWindow[sd.energyIdx]
	sd.energySum = sd.energySum - oldEnergy + energy
	sd.energyWindow[sd.energyIdx] = energy
	sd.energyIdx = (sd.energyIdx + 1) % sd.windowSize
	sd.sampleCount++

	// Update noise floor (exponential moving average when no signal)
	avgEnergy := sd.energySum / float32(sd.windowSize)
	if !sd.locked {
		// Update noise floor more aggressively when unlocked
		// Use a faster adaptation rate to quickly learn the noise level
		// Only update if energy is reasonable (not a huge spike)
		if avgEnergy < sd.noiseFloor*100 {
			// Faster adaptation: 95% old, 5% new (was 99%/1%)
			sd.noiseFloor = sd.noiseFloor*0.95 + avgEnergy*0.05
		}
	}

	// Signal detection with SNR requirement
	signalLevel := avgEnergy / (sd.noiseFloor + 0.0001)

	// Check if we're seeing a clean signal (no noise) vs noisy signal
	// If signalLevel is close to 1.0 AND noise floor has adapted, it means clean signal
	// If signalLevel is much higher than 1.0, it means signal is significantly above noise
	// Require noise floor to have adapted (above initial value) before treating as clean signal
	// Also require minimum samples processed to allow noise floor to adapt
	noiseFloorAdapted := sd.noiseFloor > 0.0005 // Has adapted from initial 0.001
	hasEnoughSamples := sd.sampleCount > sd.windowSize*2 // Need at least 2 windows of samples
	isCleanSignal := signalLevel < 1.5 && noiseFloorAdapted && hasEnoughSamples // Signal level close to 1.0 AND adapted = clean signal

	// Require both:
	// 1. Signal level above threshold (relative to noise floor)
	// 2. Minimum SNR (absolute requirement) - but only for noisy signals
	//    For clean signals, use absolute energy threshold instead
	minSNRRatio := float32(math.Pow(10, float64(sd.minSNR)/10)) // Convert dB to ratio (e.g., 4dB = 2.5x)
	meetsThreshold := signalLevel > sd.signalThresh

	var meetsSNR bool
	if isCleanSignal {
		// Clean signal (no noise) - use absolute energy threshold
		// Require energy to be above a reasonable threshold (0.01 = -20dB)
		meetsSNR = avgEnergy > 0.01
	} else {
		// Noisy signal - require SNR above noise
		meetsSNR = signalLevel > minSNRRatio
	}

	// Don't allow locking until noise floor has had time to adapt
	// This prevents locking onto noise at the start
	canLock := hasEnoughSamples

	if meetsThreshold && meetsSNR && canLock {
		sd.lockCount++
		sd.unlockCount = 0
		// Require fewer consecutive samples to lock (was windowSize/2, now windowSize/4)
		// This allows faster locking while still requiring consistent signal
		if sd.lockCount > sd.windowSize/4 {
			sd.locked = true
		}
	} else {
		sd.unlockCount++
		sd.lockCount = 0
		if sd.unlockCount > sd.windowSize {
			sd.locked = false
		}
	}
}

// IsLocked returns true if signal is detected
func (sd *SignalDetector) IsLocked() bool {
	return sd.locked
}

// GetSignalLevel returns current signal level (0-1)
func (sd *SignalDetector) GetSignalLevel() float32 {
	avgEnergy := sd.energySum / float32(sd.windowSize)
	signalLevel := avgEnergy / (sd.noiseFloor + 0.0001)
	if signalLevel > 1.0 {
		return 1.0
	}
	return signalLevel
}

// Reset resets the detector state
func (sd *SignalDetector) Reset() {
	for i := range sd.energyWindow {
		sd.energyWindow[i] = 0
	}
	sd.energySum = 0
	sd.energyIdx = 0
	sd.locked = false
	sd.lockCount = 0
	sd.unlockCount = 0
	sd.noiseFloor = 0.001
}

// CostasLoop implements a Costas loop for carrier phase recovery
// Used for PSK/QAM demodulation
type CostasLoop struct {
	// Loop filter parameters
	alpha float32 // Proportional gain
	beta  float32 // Integral gain

	// State
	phaseError float32
	phaseEst   float32
	freqEst    float32

	// Modulation order (2 for BPSK, 4 for QPSK, etc.)
	modulationOrder int
}

// NewCostasLoop creates a new Costas loop
func NewCostasLoop(modulationOrder int, alpha, beta float32) *CostasLoop {
	return &CostasLoop{
		modulationOrder: modulationOrder,
		alpha:           alpha,
		beta:            beta,
	}
}

// Update updates the phase estimate based on I/Q samples
func (cl *CostasLoop) Update(iVal, qVal float32) {
	// Phase error detection depends on modulation order
	var error float32
	if cl.modulationOrder == 2 {
		// BPSK: error = sign(I) * Q
		if iVal >= 0 {
			error = qVal
		} else {
			error = -qVal
		}
	} else if cl.modulationOrder == 4 {
		// QPSK: error = sign(I) * Q - sign(Q) * I
		var iSign, qSign float32
		if iVal >= 0 {
			iSign = 1
		} else {
			iSign = -1
		}
		if qVal >= 0 {
			qSign = 1
		} else {
			qSign = -1
		}
		error = iSign*qVal - qSign*iVal
	} else {
		// For higher orders, use atan2
		error = float32(math.Atan2(float64(qVal), float64(iVal)))
	}

	// Update frequency estimate (integral term)
	cl.freqEst += cl.beta * error

	// Update phase estimate (proportional + integral)
	cl.phaseEst += cl.alpha*error + cl.freqEst

	// Normalize phase
	if cl.phaseEst > math.Pi {
		cl.phaseEst -= 2 * math.Pi
	} else if cl.phaseEst < -math.Pi {
		cl.phaseEst += 2 * math.Pi
	}

	cl.phaseError = error
}

// GetPhase returns the current phase estimate
func (cl *CostasLoop) GetPhase() float32 {
	return cl.phaseEst
}

// Reset resets the Costas loop state
func (cl *CostasLoop) Reset() {
	cl.phaseError = 0
	cl.phaseEst = 0
	cl.freqEst = 0
}

// TimingRecovery implements symbol timing recovery using early-late gate
type TimingRecovery struct {
	samplesPerSymbol float32
	timingOffset     float32
	mu               float32 // Fractional timing offset

	// Early-late gate parameters
	earlyLateSpacing float32 // Spacing between early and late samples (typically 0.5)

	// State
	lastSymbol float32
}

// NewTimingRecovery creates a new timing recovery
func NewTimingRecovery(samplesPerSymbol float32, earlyLateSpacing float32) *TimingRecovery {
	return &TimingRecovery{
		samplesPerSymbol: samplesPerSymbol,
		earlyLateSpacing: earlyLateSpacing,
		mu:               0.5, // Start at middle of symbol
	}
}

// Update updates timing estimate based on early-late samples
func (tr *TimingRecovery) Update(early, onTime, late float32) {
	// Early-late gate algorithm
	error := float32(math.Abs(float64(early))) - float32(math.Abs(float64(late)))

	// Update timing offset
	tr.mu += 0.1 * error // Adjust step size as needed

	// Keep mu in [0, 1] range
	if tr.mu < 0 {
		tr.mu = 0
	} else if tr.mu > 1 {
		tr.mu = 1
	}

	tr.timingOffset = tr.mu * tr.samplesPerSymbol
}

// GetTimingOffset returns the current timing offset
func (tr *TimingRecovery) GetTimingOffset() float32 {
	return tr.timingOffset
}

// Reset resets the timing recovery state
func (tr *TimingRecovery) Reset() {
	tr.timingOffset = 0
	tr.mu = 0.5
	tr.lastSymbol = 0
}

