package qam

import (
	"io"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/filter/agc"
	"github.com/itohio/EasyRobot/x/math/filter/pll"
)

var _ chain.Processor[byte, float32] = (*Demodulator)(nil)

// Demodulator converts IQ samples back into QAM symbols using PLL + AGC.
type Demodulator struct {
	cfg   Config
	table constellationTable

	fifo     []float32
	buffered int
	readBuf  []float32

	pllLoop *pll.Loop
	agc     *agc.Filter

	silenceCount int

	tmpSymbol [1]byte
}

// NewDemodulator builds a streaming demodulator with PLL and AGC enabled.
func NewDemodulator(cfg Config) (*Demodulator, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	symbolSamples := cfg.SamplesPerSymbol * 2
	// Use optimal PLL parameters based on target SNR (25dB) and preamble length
	targetSNR := float32(25.0) // Target 25dB SNR operation
	loopBw := cfg.LoopBandwidth
	if loopBw <= 0 {
		// Use optimal bandwidth - wider for acquisition if preamble exists
		if cfg.PreambleLength > 0 {
			loopBw = cfg.OptimalAcquisitionBandwidth(targetSNR)
		} else {
			loopBw = cfg.OptimalLoopBandwidth(targetSNR)
		}
	}
	damping := cfg.OptimalLoopDamping(targetSNR)
	pllLoop := pll.NewFromLoopBandwidth(loopBw, cfg.SymbolRate(), damping)
	agcFilter := agc.New(1.0, 0.001)
	agcFilter.SetSampleRate(cfg.SymbolRate())

	return &Demodulator{
		cfg:     cfg,
		table:   newConstellationTable(cfg),
		fifo:    make([]float32, symbolSamples*2),
		pllLoop: pllLoop,
		agc:     agcFilter,
	}, nil
}

// Process implements chain.Processor.
func (d *Demodulator) Process(dst buffer.Writer[byte], src buffer.Reader[float32]) int {
	if dst == nil {
		return d.estimateBytes(src)
	}
	bitsWriter, ok := dst.(buffer.BitsWriter)
	if !ok || bitsWriter == nil {
		return 0
	}
	if bitsWriter.BitsLength() != d.cfg.BitsPerSymbol {
		return 0
	}

	totalBytes := 0
	sourceDone := false
	symbolSamples := d.cfg.SamplesPerSymbol * 2

	for {
		for d.buffered < symbolSamples && !sourceDone {
			needed := symbolSamples - d.buffered
			if len(d.readBuf) < needed {
				d.readBuf = make([]float32, needed)
			}
			n, err := src.Read(d.readBuf[:needed])
			if n > 0 {
				copy(d.fifo[d.buffered:], d.readBuf[:n])
				d.buffered += n
			}
			if err == io.EOF {
				sourceDone = true
			}
			if err != nil && err != io.EOF {
				sourceDone = true
			}
		}

		if d.buffered < symbolSamples {
			break
		}

		sym := d.decodeSymbol(d.fifo[:symbolSamples])
		if sym >= 0 {
			d.tmpSymbol[0] = byte(sym)
			writtenBits, _ := bitsWriter.Write(d.tmpSymbol[:])
			if writtenBits == 0 {
				break
			}
			totalBytes += writtenBits / d.cfg.BitsPerSymbol
		}

		copy(d.fifo, d.fifo[symbolSamples:d.buffered])
		d.buffered -= symbolSamples
	}

	return totalBytes
}

func (d *Demodulator) decodeSymbol(samples []float32) int {
	if len(samples) == 0 {
		return -1
	}

	iAcc, qAcc := d.integrateSymbol(samples)
	if !d.hasSignal(iAcc, qAcc) {
		d.silenceCount++
		if d.silenceCount >= d.cfg.TransmissionGap {
			d.resetLoops()
		}
		return -1
	}
	d.silenceCount = 0

	iRot, qRot := d.pllLoop.Rotate(iAcc, qAcc)
	iNorm, qNorm := d.applyAGC(iRot, qRot)

	symbol := d.quantizeSymbol(iNorm, qNorm)

	ideal := d.table.points[symbol]
	phase := math32.Atan2(qNorm, iNorm)
	idealPhase := math32.Atan2(ideal.q, ideal.i)
	phaseError := wrapPi(phase - idealPhase)
	d.pllLoop.Update(phaseError)

	return symbol
}

func (d *Demodulator) integrateSymbol(samples []float32) (float32, float32) {
	var iAcc, qAcc float32
	env := d.table.envelope
	for i := range env {
		base := i * 2
		envVal := env[i]
		iAcc += samples[base] * envVal
		qAcc += samples[base+1] * envVal
	}
	if d.table.envelopeEnergy > 0 {
		inv := 1 / d.table.envelopeEnergy
		iAcc *= inv
		qAcc *= inv
	}
	return iAcc, qAcc
}

func (d *Demodulator) quantizeSymbol(iVal, qVal float32) int {
	iGray, _ := d.table.quantizeAxis(iVal)
	qGray, _ := d.table.quantizeAxis(qVal)
	iBinary := grayToBinary(iGray, d.table.axisBits)
	qBinary := grayToBinary(qGray, d.table.axisBits)
	return (iBinary << d.table.axisBits) | qBinary
}

func (d *Demodulator) applyAGC(i, q float32) (float32, float32) {
	mag := math32.Hypot(i, q)
	if mag <= 0 {
		return i, q
	}
	scaled := d.agc.Process(mag)
	gain := scaled / (mag + 1e-6)
	return i * gain, q * gain
}

func (d *Demodulator) estimateBytes(src buffer.Reader[float32]) int {
	if src == nil {
		return 0
	}
	length := src.Len()
	if length <= 0 {
		return 0
	}
	return d.cfg.MaxBytesForSamples(length)
}

// Reset clears the internal buffering and control loops.
func (d *Demodulator) Reset() {
	d.buffered = 0
	d.resetLoops()
}

func (d *Demodulator) resetLoops() {
	d.silenceCount = 0
	if d.pllLoop != nil {
		d.pllLoop.Reset()
	}
	if d.agc != nil {
		d.agc.Reset()
	}
}

func (d *Demodulator) hasSignal(i, q float32) bool {
	magSq := i*i + q*q
	minMag := d.table.avgSymbolPower * 0.02
	return magSq >= minMag
}
