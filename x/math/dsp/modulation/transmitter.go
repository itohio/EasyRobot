package modulation

// Stats provides statistical information about decoding errors.
type Stats struct {
	// Average error magnitude
	Average float32
	// Minimum error magnitude observed
	Min float32
	// Maximum error magnitude observed
	Max float32
	// Standard deviation of error magnitudes
	StdDev float32
	// Number of samples used for statistics
	SampleCount int
}

// QualityStats provides decoding quality metrics with error statistics.
type QualityStats struct {
	// SNR is the estimated Signal-to-Noise Ratio in dB.
	// Returns 0 if insufficient samples have been processed.
	SNR float32
	// SignalStrength is the current signal strength/power level.
	// This is measured from the raw received signal BEFORE any normalization
	// (e.g., before AGC). After normalization, signal strength would be constant
	// and not useful for diagnostics. Returns 0 if no signal has been detected.
	SignalStrength float32
	// PhaseError statistics about phase errors during decoding.
	// Phase error is the difference between the received phase and the ideal
	// constellation point phase. Returns zero values if insufficient data.
	PhaseError Stats
	// FrequencyError statistics about carrier frequency offset errors.
	// Frequency error is the estimated frequency offset from the expected carrier.
	// Returns zero values if insufficient data or frequency tracking not available.
	FrequencyError Stats
	// ConstellationError statistics about constellation point errors.
	// Constellation error is the Euclidean distance between the received symbol
	// and the nearest ideal constellation point. Returns zero values if insufficient data.
	ConstellationError Stats
}

// TransmissionStats provides detailed information about transmission detection
// and lock status.
type TransmissionStats struct {
	// TransmissionStart is the sample index where the current transmission
	// was first detected (where signal energy exceeded threshold).
	// Returns 0 if no transmission has been detected.
	TransmissionStart int
	// TransmissionEnd is the sample index where the current transmission
	// was last detected (where signal energy fell below threshold).
	// Returns 0 if no transmission has been detected.
	TransmissionEnd int
	// PacketStart is the sample index where the packet payload begins
	// (after preamble completion). This is the index where decoding of actual
	// data symbols starts.
	// Returns 0 if preamble has not been completed or no preamble is used.
	PacketStart int
	// PreambleStart is the sample index where the last preamble was detected.
	// Returns 0 if no preamble has been detected.
	PreambleStart int
	// Whether the demodulator is currently locked
	IsLocked bool
	// Sample index where lock was last acquired (0 if never locked)
	LockStart int
	// Sample index where lock was last lost (0 if never lost or currently locked)
	LockEnd int
	// Number of lock acquisitions since last reset
	LockCount int
	// Number of lock losses since last reset
	UnlockCount int
}

// TransmitterParams provides transmission parameters and signal characteristics.
type TransmitterParams struct {
	// BaudRate is the symbol rate (baud rate) in symbols per second.
	BaudRate float32
	// SampleRate is the sample rate in samples per second.
	SampleRate float32
	// SignalStrength is the signal strength in dB relative to a reference level.
	// For voltage signals: 20*log10(amplitude/reference)
	// For power signals: 10*log10(power/reference)
	// Returns 0 if signal strength cannot be determined.
	SignalStrength float32
	// Bandwidth is the occupied bandwidth in Hz.
	Bandwidth float32
	// CarrierFrequency is the carrier frequency in Hz.
	CarrierFrequency float32
}

// Receiver defines the interface for demodulators that provide signal quality
// and decoding diagnostics.
type Receiver interface {
	// QualityStats returns decoding quality metrics with error statistics.
	Quality() QualityStats

	// TransmissionStats returns detailed information about transmission detection
	// and lock status.
	Transmission() TransmissionStats

	// Params returns expected transmission parameters including baud rate, sample rate,
	// signal strength, bandwidth, and carrier frequency.
	Params() TransmitterParams
}

// Transmitter defines the interface for modulators that provide transmission
// parameters and signal characteristics.
type Transmitter interface {
	// Params returns transmission parameters including baud rate, sample rate,
	// signal strength, bandwidth, and carrier frequency.
	Params() TransmitterParams
}
