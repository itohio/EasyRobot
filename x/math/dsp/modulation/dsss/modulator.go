package dsss

import (
	"fmt"
	"math"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// MinimumSamplesPerChip is the minimum number of samples per chip recommended for proper DSSS operation
const MinimumSamplesPerChip = 4

// RecommendedSamplesPerChip is the recommended number of samples per chip for good DSSS performance
const RecommendedSamplesPerChip = 8

// CalculateMinimumSampleRate calculates the minimum sample rate required for DSSS modulation/demodulation
func CalculateMinimumSampleRate(symbolRate, spreadingFactor, carrierFreq float32, minSamplesPerChip int) float32 {
	if minSamplesPerChip <= 0 {
		minSamplesPerChip = MinimumSamplesPerChip
	}

	chipRate := symbolRate * spreadingFactor
	nyquistRate := 4 * carrierFreq
	chipSamplingRate := chipRate * float32(minSamplesPerChip)

	if nyquistRate > chipSamplingRate {
		return nyquistRate
	}
	return chipSamplingRate
}

// CalculateRecommendedSampleRate calculates the recommended sample rate for DSSS
func CalculateRecommendedSampleRate(symbolRate, spreadingFactor, carrierFreq float32) float32 {
	return CalculateMinimumSampleRate(symbolRate, spreadingFactor, carrierFreq, RecommendedSamplesPerChip)
}

// Bandwidth calculates the required bandwidth for DSSS modulation
//
// The bandwidth depends on the chip rate and the pulse shaping rolloff factor.
// For raised cosine pulse shaping, the bandwidth is:
//
//	bandwidth = chipRate * (1 + rolloffFactor)
//
// Where:
//
//	chipRate = symbolRate * spreadingFactor
//
// Parameters:
//   - symbolRate: symbol rate in symbols/sec (baud rate)
//   - spreadingFactor: number of chips per symbol (code length)
//   - rolloffFactor: raised cosine roll-off factor (0 to 1, typically 0.35-0.5)
//
// Returns:
//   - bandwidth in Hz
//
// Example:
//
//	symbolRate = 1000 symbols/sec
//	spreadingFactor = 7 chips/symbol
//	rolloffFactor = 0.5
//	chipRate = 1000 * 7 = 7000 chips/sec
//	bandwidth = 7000 * (1 + 0.5) = 10500 Hz
func Bandwidth(symbolRate, spreadingFactor, rolloffFactor float32) float32 {
	chipRate := symbolRate * spreadingFactor
	return chipRate * (1 + rolloffFactor)
}

// Modulator implements DSSS modulation from scratch following SPEC.md
// Uses raised cosine pulse shaping and reuses SineGenerator/CosineGenerator
type Modulator struct {
	// Parameters
	sampleRate      float32
	carrierFreq     float32
	symbolRate      float32
	spreadingFactor int // L: chips per symbol
	modulationOrder int // M: 2 for binary, 4 for 4-ary, etc.
	amplitude       float32

	// Derived parameters
	samplesPerChip   int // N_sc
	samplesPerSymbol int // L * N_sc
	chipDuration     float32
	bitsPerSymbol    int // k = log2(M)

	// Pulse shaping
	rolloffFactor float32 // α for raised cosine (0 to 1)

	// Spreading codes
	code      Code      // For binary DSSS
	codeTable CodeTable // For M-ary DSSS

	// State
	bitBuffer     buffer.Bits[byte]
	carrierGen    dsp.CosineGenerator // For carrier modulation
	sampleCounter int64               // Global sample counter for phase continuity

	// Raised cosine pulse cache (pre-computed for efficiency)
	pulseShape []float32 // Pre-computed pulse shape samples
}

// NewModulator creates a new DSSS modulator following SPEC.md
// Parameters:
//   - sampleRate: sampling frequency in Hz
//   - carrierFreq: carrier frequency in Hz (0 for baseband)
//   - symbolRate: symbol rate in symbols/sec
//   - rolloffFactor: raised cosine roll-off factor (0 to 1, typically 0.35-0.5)
//   - codes: spreading codes (1 for binary, M for M-ary)
func NewModulator(sampleRate, carrierFreq, symbolRate, rolloffFactor float32, codes ...Code) *Modulator {
	if len(codes) == 0 {
		panic("DSSS modulator: at least one code must be provided")
	}

	// Validate codes
	firstCodeLength := codes[0].length
	if firstCodeLength == 0 {
		panic("DSSS modulator: codes must not be empty")
	}

	for i, code := range codes {
		if code.length == 0 {
			panic(fmt.Sprintf("DSSS modulator: code %d must not be empty", i))
		}
		if code.length != firstCodeLength {
			panic(fmt.Sprintf("DSSS modulator: code %d has length %d, expected %d", i, code.length, firstCodeLength))
		}
	}

	// Determine modulation order
	numCodes := len(codes)
	var modulationOrder int
	if numCodes == 1 {
		modulationOrder = 2 // Binary DSSS
	} else {
		modulationOrder = numCodes
		if (modulationOrder & (modulationOrder - 1)) != 0 {
			panic(fmt.Sprintf("DSSS modulator: M-ary requires M to be a power of 2, got %d codes", modulationOrder))
		}
	}

	spreadingFactor := firstCodeLength
	chipRate := symbolRate * float32(spreadingFactor)
	samplesPerChip := int(sampleRate / chipRate)
	if samplesPerChip <= 0 {
		panic(fmt.Sprintf("DSSS modulator: samplesPerChip must be positive, got %d", samplesPerChip))
	}

	chipDuration := 1.0 / chipRate
	bitsPerSymbol := int(math.Log2(float64(modulationOrder)))

	// Initialize carrier generator (for upconversion)
	carrierGen := dsp.NewCosineGenerator(sampleRate, carrierFreq, 1.0, 0)

	// Pre-compute raised cosine pulse shape
	pulseShape := computeRaisedCosinePulse(samplesPerChip, chipDuration, sampleRate, rolloffFactor)

	mod := &Modulator{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		spreadingFactor:  spreadingFactor,
		modulationOrder:  modulationOrder,
		amplitude:        1.0,
		samplesPerChip:   samplesPerChip,
		samplesPerSymbol: spreadingFactor * samplesPerChip,
		chipDuration:     chipDuration,
		bitsPerSymbol:    bitsPerSymbol,
		rolloffFactor:    rolloffFactor,
		bitBuffer:        buffer.NewBits[byte](bitsPerSymbol),
		carrierGen:       carrierGen,
		sampleCounter:    0,
		pulseShape:       pulseShape,
	}

	// Set code or codeTable
	if numCodes == 1 {
		mod.code = codes[0]
	} else {
		mod.codeTable = make(CodeTable, modulationOrder)
		copy(mod.codeTable, codes)
	}

	return mod
}

// computeRaisedCosinePulse computes raised cosine pulse shape samples
// Following SPEC.md Component 3, Section B
func computeRaisedCosinePulse(samplesPerChip int, chipDuration, sampleRate, rolloffFactor float32) []float32 {
	pulse := make([]float32, samplesPerChip)
	dt := 1.0 / sampleRate
	tc := chipDuration

	for i := range samplesPerChip {
		// Time relative to pulse center
		t := float32(i)*dt - tc/2.0

		var value float32
		if math32.Abs(t) < 1e-9 {
			// At t=0: sinc(0) = 1
			value = 1.0
		} else {
			// Raised cosine: sinc(πt/T_c) × cos(παt/T_c) / (1 - (2αt/T_c)²)
			piT := math32.Pi * t / tc
			sinc := math32.Sin(piT) / piT

			denom := 1.0 - math32.Pow(2.0*rolloffFactor*t/tc, 2.0)
			if math32.Abs(denom) < 1e-6 {
				// Handle singularity at t = ±T_c/(2α)
				value = (rolloffFactor / 2.0) * math32.Sin(math32.Pi/(2.0*rolloffFactor))
			} else {
				cosTerm := math32.Cos(rolloffFactor * piT)
				value = sinc * cosTerm / denom
			}
		}

		pulse[i] = value
	}

	return pulse
}

// Reset resets the modulator state
func (m *Modulator) Reset() {
	m.bitBuffer.Clear()
	m.sampleCounter = 0
	m.carrierGen = dsp.NewCosineGenerator(m.sampleRate, m.carrierFreq, 1.0, 0)
}

// SetAmplitude sets the signal amplitude for modulation
func (m *Modulator) SetAmplitude(amplitude float32) {
	m.amplitude = amplitude
}

// Encode encodes bytes into amplitude vector
// Implements modulation.Modulator interface
func (m *Modulator) Encode(amplitudes vecTypes.Vector, data []byte) int {
	if amplitudes == nil {
		// Calculate expected length
		bitsNeeded := m.bitBuffer.GetWrittenBits() + len(data)*8
		symbolsNeeded := (bitsNeeded + m.bitsPerSymbol - 1) / m.bitsPerSymbol
		return symbolsNeeded * m.samplesPerSymbol
	}

	// Add bytes to bit buffer
	m.bitBuffer.ReadFromBytes(data)

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	samplesWritten := 0

	// Process symbols
	for m.bitBuffer.HasValue() && samplesWritten < len(ampData) {
		// Read symbol value
		symbolValue, ok := m.bitBuffer.Read()
		if !ok {
			break
		}
		symbolIndex := int(symbolValue)

		// Get spreading code for this symbol (Component 2: Chip Generator)
		var code Code
		if m.modulationOrder == 2 {
			// Binary DSSS: use single code, multiply by symbol value
			code = m.code
		} else {
			// M-ary DSSS: select code from table
			code = m.codeTable[symbolIndex]
		}

		// Generate symbol samples: chips × pulse shape × carrier
		for chipIdx := 0; chipIdx < m.spreadingFactor && samplesWritten < len(ampData); chipIdx++ {
			// Get bipolar chip value
			chipValue := code.GetBipolar(chipIdx)

			// For binary DSSS, multiply chip by symbol value (+1 or -1)
			if m.modulationOrder == 2 {
				if symbolIndex == 0 {
					chipValue = -chipValue // Bit 0 -> -code
				} else {
					chipValue = +chipValue // Bit 1 -> +code
				}
			}

			// Apply pulse shaping (Component 3: Pulse Shaper)
			for s := 0; s < m.samplesPerChip && samplesWritten < len(ampData); s++ {
				// Baseband sample: chip × pulse shape
				baseband := m.amplitude * chipValue * m.pulseShape[s]

				// Upconvert to passband (Component 4: Upconverter)
				if m.carrierFreq > 0 {
					// Use carrier generator for phase continuity
					// Generate one sample of carrier
					carrierSamples := make([]float32, 1)
					m.carrierGen.Generate(carrierSamples)
					ampData[samplesWritten] = baseband * carrierSamples[0]
				} else {
					// Baseband: no carrier
					ampData[samplesWritten] = baseband
				}

				samplesWritten++
				m.sampleCounter++
			}
		}
	}

	return samplesWritten
}
