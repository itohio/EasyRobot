package dsp

import (
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/x/math/filter"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// ModulatorEncoder interface for digital modulators
type ModulatorEncoder interface {
	// Encode encodes bytes into amplitude vector
	// Returns false when encoding is complete
	Encode(amplitudes vecTypes.Vector, data []byte) bool

	// Reset resets the encoder state
	Reset()
}

// ModulatorDecoder interface for digital demodulators
type ModulatorDecoder interface {
	// Decode decodes amplitude vector into bytes
	// Returns false when decoding is complete
	Decode(data []byte, amplitudes vecTypes.Vector) bool

	// Reset resets the decoder state
	Reset()

	// OnIsEndOfPacket sets a callback to check if packet is complete
	OnIsEndOfPacket(callback func([]byte) bool)
}

// IQEncoder handles IQ modulation for digital communication
type IQEncoder struct {
	sampleRate  float32
	carrierFreq float32
}

// NewIQEncoder creates a new IQ encoder
func NewIQEncoder(sampleRate, carrierFreq float32) *IQEncoder {
	return &IQEncoder{
		sampleRate:  sampleRate,
		carrierFreq: carrierFreq,
	}
}

// Modulate modulates baseband I/Q signals onto carrier frequency
// I and Q are the in-phase and quadrature components
// Returns the modulated signal
func (iq *IQEncoder) Modulate(I, Q vecTypes.Vector) vec.Vector {
	iVec := I.(vec.Vector)
	qVec := Q.(vec.Vector)
	iData := []float32(iVec)
	qData := []float32(qVec)

	if len(iData) != len(qData) {
		panic("IQEncoder.Modulate: I and Q vectors must have same length")
	}

	length := len(iData)
	result := vec.New(length)
	resultData := []float32(result)

	omega := 2 * math.Pi * iq.carrierFreq / iq.sampleRate

	for i := 0; i < length; i++ {
		carrierI := float32(math.Cos(float64(omega * float32(i))))
		carrierQ := float32(math.Sin(float64(omega * float32(i))))
		resultData[i] = iData[i]*carrierI - qData[i]*carrierQ
	}

	return result
}

// IQDecoder handles IQ demodulation for digital communication
type IQDecoder struct {
	sampleRate    float32
	carrierFreq   float32
	lowPassFilter filter.Processor[float32] // Low-pass filter for baseband filtering
}

// NewIQDecoder creates a new IQ decoder
func NewIQDecoder(sampleRate, carrierFreq float32) *IQDecoder {
	return &IQDecoder{
		sampleRate:  sampleRate,
		carrierFreq: carrierFreq,
	}
}

// SetFilter sets the filter for baseband filtering
// Recommended: FIR low-pass filter with cutoff = symbolRate/2
func (iq *IQDecoder) SetFilter(f filter.Processor[float32]) {
	iq.lowPassFilter = f
}

// Demodulate demodulates a modulated signal into I/Q components
// Returns I and Q components
func (iq *IQDecoder) Demodulate(signal vecTypes.Vector) (I, Q vec.Vector) {
	signalData := []float32(signal.(vec.Vector))
	length := len(signalData)

	I = vec.New(length)
	Q = vec.New(length)

	iData := []float32(I)
	qData := []float32(Q)

	omega := 2 * math.Pi * iq.carrierFreq / iq.sampleRate

	for i := 0; i < length; i++ {
		carrierI := float32(math.Cos(float64(omega * float32(i))))
		carrierQ := float32(math.Sin(float64(omega * float32(i))))
		iData[i] = signalData[i] * carrierI
		qData[i] = -signalData[i] * carrierQ
	}

	// Apply low-pass filtering if filter is set
	if iq.lowPassFilter != nil {
		for i := 0; i < length; i++ {
			iData[i] = iq.lowPassFilter.Process(iData[i])
			qData[i] = iq.lowPassFilter.Process(qData[i])
		}
	}

	return I, Q
}

// PSKDecoder handles Phase Shift Keying demodulation
type PSKDecoder struct {
	sampleRate      float32
	carrierFreq     float32
	symbolRate      float32
	modulationOrder int                       // 2 for BPSK, 4 for QPSK, etc.
	lowPassFilter   filter.Processor[float32] // Low-pass filter for baseband filtering
}

// NewPSKDecoder creates a new PSK decoder
func NewPSKDecoder(sampleRate, carrierFreq, symbolRate float32, modulationOrder int) *PSKDecoder {
	return &PSKDecoder{
		sampleRate:      sampleRate,
		carrierFreq:     carrierFreq,
		symbolRate:      symbolRate,
		modulationOrder: modulationOrder,
	}
}

// SetFilter sets the filter for baseband filtering
// Recommended: FIR low-pass filter with cutoff = symbolRate/2
func (psk *PSKDecoder) SetFilter(f filter.Processor[float32]) {
	psk.lowPassFilter = f
}

// Demodulate demodulates PSK signal into symbols
func (psk *PSKDecoder) Demodulate(signal vecTypes.Vector) vec.Vector {
	// First, IQ demodulate
	decoder := NewIQDecoder(psk.sampleRate, psk.carrierFreq)
	if psk.lowPassFilter != nil {
		decoder.SetFilter(psk.lowPassFilter)
	}
	I, Q := decoder.Demodulate(signal)

	// Sample at symbol rate
	symbolsPerSecond := int(psk.sampleRate / psk.symbolRate)
	iData := []float32(I)
	numSymbols := len(iData) / symbolsPerSecond
	symbols := vec.New(numSymbols)

	qData := []float32(Q)
	symbolData := []float32(symbols)

	for i := 0; i < numSymbols; i++ {
		sampleIdx := i * symbolsPerSecond
		if sampleIdx >= len(iData) {
			break
		}

		// For BPSK (2-PSK), decision based on I component sign
		// For QPSK (4-PSK), decision based on both I and Q quadrant
		if psk.modulationOrder == 2 {
			// BPSK
			if iData[sampleIdx] > 0 {
				symbolData[i] = 1.0
			} else {
				symbolData[i] = 0.0
			}
		} else if psk.modulationOrder == 4 {
			// QPSK - map to 0,1,2,3
			iBit := 0
			qBit := 0
			if iData[sampleIdx] > 0 {
				iBit = 1
			}
			if qData[sampleIdx] > 0 {
				qBit = 1
			}
			symbolData[i] = float32(iBit*2 + qBit)
		}
	}

	return symbols
}

// FSKDecoder handles Frequency Shift Keying demodulation
type FSKDecoder struct {
	sampleRate float32
	freqLow    float32 // Low frequency
	freqHigh   float32 // High frequency
	symbolRate float32
}

// NewFSKDecoder creates a new FSK decoder
func NewFSKDecoder(sampleRate, freqLow, freqHigh, symbolRate float32) *FSKDecoder {
	return &FSKDecoder{
		sampleRate: sampleRate,
		freqLow:    freqLow,
		freqHigh:   freqHigh,
		symbolRate: symbolRate,
	}
}

// Demodulate demodulates FSK signal into symbols
func (fsk *FSKDecoder) Demodulate(signal vecTypes.Vector) vec.Vector {
	signalData := []float32(signal.(vec.Vector))

	// FSK demodulation using frequency discrimination
	// This is a simplified implementation

	samplesPerSecond := int(fsk.sampleRate / fsk.symbolRate)
	numSymbols := len(signalData) / samplesPerSecond
	symbols := vec.New(numSymbols)
	symbolData := []float32(symbols)

	// Simple frequency discrimination
	for i := 0; i < numSymbols; i++ {
		startSample := i * samplesPerSecond
		endSample := (i + 1) * samplesPerSecond
		if endSample > len(signalData) {
			endSample = len(signalData)
		}

		// Compute correlation with both frequencies
		corrLow := fsk.correlateFrequency(signalData, fsk.freqLow, startSample, endSample)
		corrHigh := fsk.correlateFrequency(signalData, fsk.freqHigh, startSample, endSample)

		if corrHigh > corrLow {
			symbolData[i] = 1.0 // High frequency = 1
		} else {
			symbolData[i] = 0.0 // Low frequency = 0
		}
	}

	return symbols
}

// correlateFrequency computes correlation with a specific frequency
func (fsk *FSKDecoder) correlateFrequency(signalData []float32, freq float32, start, end int) float32 {
	sum := float32(0)

	omega := 2 * math.Pi * freq / fsk.sampleRate

	for i := start; i < end; i++ {
		phase := omega * float32(i)
		reference := float32(math.Cos(float64(phase)))
		sum += signalData[i] * reference
	}

	return sum / float32(end-start)
}

// QAM16Encoder handles 16-QAM modulation (4 bits per symbol)
type QAM16Encoder struct {
	sampleRate       float32
	carrierFreq      float32
	symbolRate       float32
	samplesPerSymbol int
	currentPhase     float32
	bitBuffer        []byte
	bitIndex         int
}

// NewQAM16Encoder creates a new 16-QAM encoder
func NewQAM16Encoder(sampleRate, carrierFreq, symbolRate float32) *QAM16Encoder {
	samplesPerSymbol := int(sampleRate / symbolRate)
	return &QAM16Encoder{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		samplesPerSymbol: samplesPerSymbol,
		bitBuffer:        make([]byte, 0, 1024),
	}
}

// Reset resets the encoder state
func (qam *QAM16Encoder) Reset() {
	qam.bitBuffer = qam.bitBuffer[:0]
	qam.bitIndex = 0
	qam.currentPhase = 0
}

// Encode encodes bytes into amplitude vector
// Returns false when encoding is complete
func (qam *QAM16Encoder) Encode(amplitudes vecTypes.Vector, data []byte) bool {
	// Convert bytes to bits
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			bit := (b >> uint(i)) & 1
			qam.bitBuffer = append(qam.bitBuffer, byte(bit))
		}
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	samplesWritten := 0

	// Process 4 bits at a time (16-QAM symbols)
	for len(qam.bitBuffer) >= 4 && samplesWritten < len(ampData) {
		// Extract 4 bits
		bits := qam.bitBuffer[:4]
		qam.bitBuffer = qam.bitBuffer[4:]

		// Map 4 bits to I/Q values (-3, -1, 1, 3)
		iVal, qVal := qam.bitsToIQ(bits)

		// Generate symbol samples
		for s := 0; s < qam.samplesPerSymbol && samplesWritten < len(ampData); s++ {
			phase := 2 * math.Pi * qam.carrierFreq * qam.currentPhase / qam.sampleRate
			carrierI := float32(math.Cos(float64(phase)))
			carrierQ := float32(math.Sin(float64(phase)))

			// Modulate: I*cos + Q*sin (standard QAM modulation)
			ampData[samplesWritten] = iVal*carrierI - qVal*carrierQ

			qam.currentPhase += 1.0
			samplesWritten++
		}
	}

	// Return true if we still have data to process
	return len(qam.bitBuffer) > 0 || len(data) > 0
}

// bitsToIQ maps 4 bits to I/Q values for 16-QAM
func (qam *QAM16Encoder) bitsToIQ(bits []byte) (float32, float32) {
	// 16-QAM constellation: 4 levels per dimension
	// Map bits [b3 b2 b1 b0] to I/Q coordinates
	iBits := (bits[0] << 1) | bits[1] // bits 0,1 for I
	qBits := (bits[2] << 1) | bits[3] // bits 2,3 for Q

	// Map 2-bit values to amplitude levels: 00->-3, 01->-1, 10->1, 11->3
	iVal := float32(iBits*2 - 3)
	qVal := float32(qBits*2 - 3)

	return iVal, qVal
}

// QAM16Decoder handles 16-QAM demodulation (4 bits per symbol)
type QAM16Decoder struct {
	sampleRate       float32
	carrierFreq      float32
	symbolRate       float32
	samplesPerSymbol int
	currentPhase     float32
	bitBuffer        []byte
	endOfPacket      func([]byte) bool
	lowPassFilter    filter.Processor[float32] // Low-pass filter for baseband filtering
}

// NewQAM16Decoder creates a new 16-QAM decoder
func NewQAM16Decoder(sampleRate, carrierFreq, symbolRate float32) *QAM16Decoder {
	samplesPerSymbol := int(sampleRate / symbolRate)
	return &QAM16Decoder{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		samplesPerSymbol: samplesPerSymbol,
		bitBuffer:        make([]byte, 0, 1024),
	}
}

// SetFilter sets the filter for baseband filtering
// Recommended: FIR low-pass filter with cutoff = symbolRate/2
func (qam *QAM16Decoder) SetFilter(f filter.Processor[float32]) {
	qam.lowPassFilter = f
}

// Reset resets the decoder state
func (qam *QAM16Decoder) Reset() {
	qam.bitBuffer = qam.bitBuffer[:0]
	qam.currentPhase = 0
}

// OnIsEndOfPacket sets a callback to check if packet is complete
func (qam *QAM16Decoder) OnIsEndOfPacket(callback func([]byte) bool) {
	qam.endOfPacket = callback
}

// Decode decodes amplitude vector into bytes
// Returns false when decoding is complete
func (qam *QAM16Decoder) Decode(data []byte, amplitudes vecTypes.Vector) bool {
	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	bytesWritten := 0

	// Process samples in symbol chunks
	for len(ampData) >= qam.samplesPerSymbol {
		// Extract one symbol worth of samples
		symbolSamples := ampData[:qam.samplesPerSymbol]
		ampData = ampData[qam.samplesPerSymbol:]

		// Demodulate symbol to get I/Q values
		iVal, qVal := qam.demodulateSymbol(symbolSamples)

		// Convert I/Q back to 4 bits
		bits := qam.iqToBits(iVal, qVal)
		qam.bitBuffer = append(qam.bitBuffer, bits...)

		// Convert bits to bytes when we have enough
		for len(qam.bitBuffer) >= 8 && bytesWritten < len(data) {
			byteVal := byte(0)
			for i := 0; i < 8; i++ {
				byteVal |= qam.bitBuffer[i] << uint(7-i)
			}
			data[bytesWritten] = byteVal
			bytesWritten++
			qam.bitBuffer = qam.bitBuffer[8:]

			// Check if packet is complete
			if qam.endOfPacket != nil && qam.endOfPacket(data[:bytesWritten]) {
				return false // Packet complete
			}
		}
	}

	// Return true if we still have data to process
	return len(ampData) > 0
}

// demodulateSymbol extracts I/Q values from symbol samples
func (qam *QAM16Decoder) demodulateSymbol(samples []float32) (float32, float32) {
	var iAcc, qAcc float32

	for i, sample := range samples {
		phase := 2 * math.Pi * qam.carrierFreq * (qam.currentPhase + float32(i)) / qam.sampleRate
		carrierI := float32(math.Cos(float64(phase)))
		carrierQ := float32(math.Sin(float64(phase)))

		// Demodulate: correlate with carrier
		iAcc += sample * carrierI
		qAcc += sample * (-carrierQ) // Note: Q component has negative sign in standard demod
	}

	qam.currentPhase += float32(len(samples))

	// Normalize by number of samples
	iAcc /= float32(len(samples))
	qAcc /= float32(len(samples))

	// Apply low-pass filtering if filter is set
	if qam.lowPassFilter != nil {
		iAcc = qam.lowPassFilter.Process(iAcc)
		qAcc = qam.lowPassFilter.Process(qAcc)
	}

	return iAcc, qAcc
}

// iqToBits converts I/Q values back to 4 bits for 16-QAM
func (qam *QAM16Decoder) iqToBits(iVal, qVal float32) []byte {
	// Quantize I/Q values back to constellation points
	iBits := qam.quantizeValue(iVal)
	qBits := qam.quantizeValue(qVal)

	// Return as [i1 i0 q1 q0]
	return []byte{
		byte((iBits >> 1) & 1), // i1
		byte(iBits & 1),        // i0
		byte((qBits >> 1) & 1), // q1
		byte(qBits & 1),        // q0
	}
}

// quantizeValue maps continuous value to nearest constellation point (0,1,2,3)
func (qam *QAM16Decoder) quantizeValue(val float32) int {
	// Constellation points: -3, -1, 1, 3
	if val < -2 {
		return 0 // -3
	} else if val < 0 {
		return 1 // -1
	} else if val < 2 {
		return 2 // 1
	} else {
		return 3 // 3
	}
}

// ASKDecoder handles Amplitude Shift Keying demodulation
type ASKDecoder struct {
	sampleRate    float32
	carrierFreq   float32
	symbolRate    float32
	lowPassFilter filter.Processor[float32] // Low-pass filter for envelope filtering
}

// NewASKDecoder creates a new ASK decoder
func NewASKDecoder(sampleRate, carrierFreq, symbolRate float32) *ASKDecoder {
	return &ASKDecoder{
		sampleRate:  sampleRate,
		carrierFreq: carrierFreq,
		symbolRate:  symbolRate,
	}
}

// SetFilter sets the filter for envelope filtering
// Recommended: FIR low-pass filter with cutoff = symbolRate/2
func (ask *ASKDecoder) SetFilter(f filter.Processor[float32]) {
	ask.lowPassFilter = f
}

// Demodulate demodulates ASK signal into symbols
func (ask *ASKDecoder) Demodulate(signal vecTypes.Vector) vec.Vector {
	// ASK demodulation - envelope detection

	// First, IQ demodulate to get envelope
	decoder := NewIQDecoder(ask.sampleRate, ask.carrierFreq)
	I, Q := decoder.Demodulate(signal)

	iData := []float32(I)
	qData := []float32(Q)

	// Compute envelope: sqrt(I^2 + Q^2)
	envelope := vec.New(len(iData))
	envData := []float32(envelope)

	for i := range envData {
		envData[i] = float32(math.Sqrt(float64(iData[i]*iData[i] + qData[i]*qData[i])))
	}

	// Low-pass filter the envelope if filter is set
	if ask.lowPassFilter != nil {
		for i := range envData {
			envData[i] = ask.lowPassFilter.Process(envData[i])
		}
	}

	// Sample at symbol rate
	envDataFiltered := []float32(envelope)
	samplesPerSymbol := int(ask.sampleRate / ask.symbolRate)
	numSymbols := len(envDataFiltered) / samplesPerSymbol
	symbols := vec.New(numSymbols)
	symbolData := []float32(symbols)

	// Decision threshold (simple midpoint)
	maxVal := float32(0)
	minVal := float32(1e9)
	for _, v := range envDataFiltered {
		if v > maxVal {
			maxVal = v
		}
		if v < minVal {
			minVal = v
		}
	}
	threshold := (maxVal + minVal) / 2

	for i := 0; i < numSymbols; i++ {
		sampleIdx := i * samplesPerSymbol
		if sampleIdx >= len(envDataFiltered) {
			break
		}

		if envDataFiltered[sampleIdx] > threshold {
			symbolData[i] = 1.0 // High amplitude = 1
		} else {
			symbolData[i] = 0.0 // Low amplitude = 0
		}
	}

	return symbols
}

// PSKEncoder handles Phase Shift Keying encoding
type PSKEncoder struct {
	sampleRate       float32
	carrierFreq      float32
	symbolRate       float32
	modulationOrder  int // 2 for BPSK, 4 for QPSK, 8 for 8-PSK, etc.
	samplesPerSymbol int
}

// NewPSKEncoder creates a new PSK encoder
func NewPSKEncoder(sampleRate, carrierFreq, symbolRate float32, modulationOrder int) *PSKEncoder {
	samplesPerSymbol := int(sampleRate / symbolRate)
	return &PSKEncoder{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		modulationOrder:  modulationOrder,
		samplesPerSymbol: samplesPerSymbol,
	}
}

// GenerateSignal generates a PSK modulated signal from data bytes
func (psk *PSKEncoder) GenerateSignal(data []byte, amplitude float32, length int) vec.Vector {
	result := vec.New(length)
	resultData := []float32(result)

	bitsPerSymbol := int(math.Log2(float64(psk.modulationOrder)))
	phaseStep := 2 * math.Pi / float32(psk.modulationOrder)

	bitIndex := 0
	sampleIndex := 0

	// Convert bytes to bits
	bitBuffer := make([]byte, 0, len(data)*8)
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			bit := (b >> uint(i)) & 1
			bitBuffer = append(bitBuffer, byte(bit))
		}
	}

	// Process bits in symbol groups
	for i := 0; i < len(bitBuffer)-bitsPerSymbol+1 && sampleIndex < length; i += bitsPerSymbol {
		// Collect bits for one symbol
		symbol := 0
		for j := 0; j < bitsPerSymbol && i+j < len(bitBuffer); j++ {
			symbol = (symbol << 1) | int(bitBuffer[i+j])
		}

		// Generate symbol samples
		phaseOffset := float32(symbol) * phaseStep
		for s := 0; s < psk.samplesPerSymbol && sampleIndex < length; s++ {
			phase := 2*math.Pi*psk.carrierFreq*float32(sampleIndex)/psk.sampleRate + phaseOffset
			resultData[sampleIndex] = amplitude * float32(math.Cos(float64(phase)))
			sampleIndex++
		}
		bitIndex++
	}

	return result
}

// FSKEncoder handles Frequency Shift Keying encoding
type FSKEncoder struct {
	sampleRate       float32
	frequencies      []float32 // Frequencies for each symbol (2, 4, 8, or 16 frequencies)
	symbolRate       float32
	samplesPerSymbol int
	modulationOrder  int // Number of frequencies (2, 4, 8, 16)
}

// NewFSKEncoder creates a new FSK encoder with 2 frequencies (binary FSK)
func NewFSKEncoder(sampleRate, freqLow, freqHigh, symbolRate float32) *FSKEncoder {
	return NewMFSKEncoder(sampleRate, symbolRate, freqLow, freqHigh)
}

// NewMFSKEncoder creates a new M-FSK encoder with multiple frequencies
// modulationOrder is determined by the number of frequencies provided
func NewMFSKEncoder(sampleRate, symbolRate float32, frequencies ...float32) *FSKEncoder {
	if len(frequencies) < 2 {
		panic("FSK encoder requires at least 2 frequencies")
	}
	// Validate modulation order is power of 2
	modOrder := len(frequencies)
	if modOrder != 2 && modOrder != 4 && modOrder != 8 && modOrder != 16 {
		panic(fmt.Sprintf("FSK modulation order must be 2, 4, 8, or 16, got %d", modOrder))
	}

	samplesPerSymbol := int(sampleRate / symbolRate)
	return &FSKEncoder{
		sampleRate:       sampleRate,
		frequencies:      frequencies,
		symbolRate:       symbolRate,
		samplesPerSymbol: samplesPerSymbol,
		modulationOrder:  modOrder,
	}
}

// GenerateSignal generates an FSK modulated signal from data bytes
func (fsk *FSKEncoder) GenerateSignal(data []byte, amplitude float32, length int) vec.Vector {
	result := vec.New(length)
	resultData := []float32(result)

	bitsPerSymbol := int(math.Log2(float64(fsk.modulationOrder)))

	// Convert bytes to bits
	bitBuffer := make([]byte, 0, len(data)*8)
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			bit := (b >> uint(i)) & 1
			bitBuffer = append(bitBuffer, byte(bit))
		}
	}

	sampleIndex := 0
	phase := float32(0)

	// Process bits in symbol groups
	for i := 0; i < len(bitBuffer)-bitsPerSymbol+1 && sampleIndex < length; i += bitsPerSymbol {
		// Collect bits for one symbol
		symbol := 0
		for j := 0; j < bitsPerSymbol && i+j < len(bitBuffer); j++ {
			symbol = (symbol << 1) | int(bitBuffer[i+j])
		}

		// Select frequency based on symbol value
		if symbol >= len(fsk.frequencies) {
			symbol = len(fsk.frequencies) - 1
		}
		freq := fsk.frequencies[symbol]

		// Generate symbol samples
		for s := 0; s < fsk.samplesPerSymbol && sampleIndex < length; s++ {
			phase += 2 * math.Pi * freq / fsk.sampleRate
			resultData[sampleIndex] = amplitude * float32(math.Cos(float64(phase)))
			sampleIndex++
		}
	}

	return result
}

// GenerateSignal generates a QAM modulated signal from data bytes
func (qam *QAM16Encoder) GenerateSignal(data []byte, amplitude float32, length int) vec.Vector {
	result := vec.New(length)
	resultData := []float32(result)

	// Convert bytes to bits
	bitBuffer := make([]byte, 0, len(data)*8)
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			bit := (b >> uint(i)) & 1
			bitBuffer = append(bitBuffer, byte(bit))
		}
	}

	sampleIndex := 0
	qam.currentPhase = 0

	// Process 4 bits at a time (16-QAM symbols)
	for i := 0; i < len(bitBuffer)-3 && sampleIndex < length; i += 4 {
		bits := bitBuffer[i : i+4]

		// Map 4 bits to I/Q values
		iVal, qVal := qam.bitsToIQ(bits)
		iVal *= amplitude / 3.0 // Normalize to amplitude
		qVal *= amplitude / 3.0

		// Generate symbol samples
		for s := 0; s < qam.samplesPerSymbol && sampleIndex < length; s++ {
			phase := 2 * math.Pi * qam.carrierFreq * qam.currentPhase / qam.sampleRate
			carrierI := float32(math.Cos(float64(phase)))
			carrierQ := float32(math.Sin(float64(phase)))

			// Modulate: I*cos - Q*sin (standard QAM modulation)
			resultData[sampleIndex] = iVal*carrierI - qVal*carrierQ

			qam.currentPhase += 1.0
			sampleIndex++
		}
	}

	return result
}

// GenerateIQSignal generates an IQ modulated signal from I and Q vectors
func (iq *IQEncoder) GenerateIQSignal(I, Q vecTypes.Vector) vec.Vector {
	return iq.Modulate(I, Q)
}
