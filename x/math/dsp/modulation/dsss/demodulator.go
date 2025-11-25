package dsss

import (
	"errors"
	"io"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/dsp"
)

var _ chain.Processor[byte, float32] = (*Demodulator)(nil)

type Demodulator struct {
	config           Config
	samplesPerSymbol int

	downconverter dsp.CosineGenerator

	sampleBuffer []float32
	readBuf      []float32
	buffered     int

	active           bool
	preambleDetected bool
	preambleProgress int
	silenceSymbols   int
}

func NewDemodulator(config Config) (*Demodulator, error) {
	if config.SpreadingFactor <= 0 {
		return nil, errors.New("dsss: spreading factor must be positive")
	}
	if config.SamplesPerChip == 0 {
		config.SamplesPerChip = config.CalculateSamplesPerChip(config.SampleRate)
	}
	if err := config.ValidateSampleRate(config.SampleRate); err != nil {
		return nil, err
	}

	return &Demodulator{
		config:           config,
		samplesPerSymbol: config.SpreadingFactor * config.SamplesPerChip,
		downconverter:    dsp.NewCosineGenerator(config.SampleRate, config.CarrierFreq, 1.0, 0.0),
	}, nil
}

func (d *Demodulator) Process(dst buffer.Writer[byte], src buffer.Reader[float32]) int {
	if dst == nil {
		return d.estimateBytes(src)
	}

	bitsWriter, ok := dst.(buffer.BitsWriter)
	if !ok || bitsWriter == nil {
		return 0
	}
	if bitsWriter.BitsLength() != d.config.BitsPerSymbol {
		return 0
	}

	d.ensureBuffers()
	writtenSymbols := 0

	for {
		needed := d.samplesPerSymbol - d.buffered
		if needed > 0 {
			if needed > len(d.readBuf) {
				d.readBuf = make([]float32, needed)
			}
			n, err := src.Read(d.readBuf[:needed])
			if n > 0 {
				copy(d.sampleBuffer[d.buffered:], d.readBuf[:n])
				d.buffered += n
			}
			if err == io.EOF && d.buffered < d.samplesPerSymbol {
				break
			}
			if d.buffered < d.samplesPerSymbol {
				continue
			}
		}

		writtenSymbols += d.processSymbol(bitsWriter, d.sampleBuffer[:d.samplesPerSymbol])
		d.buffered = 0
	}

	return writtenSymbols
}

func (d *Demodulator) ensureBuffers() {
	if len(d.sampleBuffer) < d.samplesPerSymbol {
		d.sampleBuffer = make([]float32, d.samplesPerSymbol)
	}
	if len(d.readBuf) < d.samplesPerSymbol {
		d.readBuf = make([]float32, d.samplesPerSymbol)
	}
}

func (d *Demodulator) processSymbol(dst buffer.BitsWriter, samples []float32) int {
	d.downconverter.GenerateWithOp(samples, false, func(val *float32, carrier float32) {
		*val *= carrier
	})

	energy := d.symbolEnergy(samples)
	if !d.active {
		if energy > d.config.EnergyThreshold {
			d.active = true
			d.preambleDetected = false
			d.preambleProgress = 0
		} else {
			return 0
		}
	}

	symbol := d.detectSymbol(samples)

	if !d.preambleDetected && d.config.PreambleLength > 0 {
		pattern := d.config.PreamblePattern
		if len(pattern) == 0 {
			pattern = []int{1, 0}
		}
		expected := pattern[d.preambleProgress%len(pattern)]
		if symbol == expected {
			d.preambleProgress++
			if d.preambleProgress >= d.config.PreambleLength {
				d.preambleDetected = true
				d.silenceSymbols = 0
			}
		} else {
			d.preambleProgress = 0
		}
		return 0
	}

	if energy < d.config.EnergyThreshold {
		d.silenceSymbols++
		if d.silenceSymbols >= d.config.TransmissionTimeout {
			d.active = false
			d.preambleDetected = false
			d.preambleProgress = 0
			d.silenceSymbols = 0
			return 0
		}
	} else {
		d.silenceSymbols = 0
	}

	bits, err := dst.Write([]byte{byte(symbol)})
	if err != nil {
		return 0
	}
	if d.config.BitsPerSymbol == 0 {
		return 0
	}
	return bits / d.config.BitsPerSymbol
}

func (d *Demodulator) detectSymbol(samples []float32) int {
	if d.config.ModulationOrder == 2 {
		score := d.codeCorrelation(samples, d.config.Codes[0])
		if score >= 0 {
			return 1
		}
		return 0
	}

	bestIdx := 0
	bestScore := float32(-math32.MaxFloat32)
	for idx, code := range d.config.Codes {
		score := d.codeCorrelation(samples, code)
		if score > bestScore {
			bestScore = score
			bestIdx = idx
		}
	}
	return bestIdx
}

func (d *Demodulator) codeCorrelation(samples []float32, code []int8) float32 {
	samplesPerChip := d.config.SamplesPerChip
	var corr float32
	for chip := 0; chip < d.config.SpreadingFactor; chip++ {
		start := chip * samplesPerChip
		end := start + samplesPerChip
		if end > len(samples) {
			end = len(samples)
		}
		var sum float32
		for i := start; i < end; i++ {
			sum += samples[i]
		}
		corr += sum * float32(code[chip])
	}
	return corr
}

func (d *Demodulator) symbolEnergy(samples []float32) float32 {
	if len(samples) == 0 {
		return 0
	}
	var energy float32
	for _, sample := range samples {
		energy += sample * sample
	}
	return energy / float32(len(samples))
}

func (d *Demodulator) estimateBytes(src buffer.Reader[float32]) int {
	totalSamples := src.Len()
	if d.samplesPerSymbol == 0 || totalSamples == 0 {
		return 0
	}
	symbols := totalSamples / d.samplesPerSymbol
	bits := symbols * d.config.BitsPerSymbol
	return bits / 8
}

func (d *Demodulator) Reset() {
	d.buffered = 0
	d.active = false
	d.preambleDetected = false
	d.preambleProgress = 0
	d.silenceSymbols = 0
	d.downconverter = dsp.NewCosineGenerator(d.config.SampleRate, d.config.CarrierFreq, 1.0, 0.0)
}
