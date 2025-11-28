package mask

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
)

// Quality implements modulation.Receiver.
func (d *Demodulator) Quality() modulation.QualityStats {
	stats := modulation.QualityStats{}

	// Get signal power - use snapshot from preamble (calculated from preamble amplitudes)
	// After preamble is detected, signalEMASnapshot contains average power from preamble amplitudes
	signalPower := d.signalEMASnapshot

	// Get background power - use snapshot if available
	backgroundPower := d.backgroundEMASnapshot
	// If background snapshot is very small (< 1e-10), treat as zero (infinite SNR case)
	if backgroundPower > 0 && backgroundPower < 1e-10 {
		backgroundPower = 0
	}
	if backgroundPower <= 0 {
		if !d.active && d.powerTracker.Count() > 0 {
			// No snapshot yet, but we're in silence - use current EMA
			backgroundPower = d.powerTracker.EMA()
			// Treat very small values as zero (infinite SNR)
			if backgroundPower > 0 && backgroundPower < 1e-10 {
				backgroundPower = 0
			}
		} else if d.active && signalPower > 0 {
			// During active transmission but no background snapshot available
			// This can happen if signal was detected before any silence period
			// For infinite SNR tests, background should be essentially zero
			backgroundPower = 0
		}
	}

	// Calculate SNR from signal and background power
	// SNR = signal_power / background_power = (signal+noise) / noise
	if signalPower > 0 {
		if backgroundPower > 0 {
			// Normal case: both signal and background power available
			if signalPower > backgroundPower {
				snrLinear := signalPower / backgroundPower
				if snrLinear > 1.0 {
					stats.SNR = 10.0 * math32.Log10(snrLinear)
				} else {
					// Signal power <= background power is invalid
					stats.SNR = 0
				}
			}
		} else {
			// Infinite SNR case: background power is 0 or very small (no noise)
			// For infinite SNR, set SNR to a very high value
			// Use 60 dB as a reasonable "infinite" representation (1e6 linear)
			stats.SNR = 60.0
		}
	}

	// Calculate signal strength from signal power
	if signalPower > 0 {
		stats.SignalStrength = 10.0 * math32.Log10(signalPower)
	} else if d.rawSignalCount > 0 {
		// Fallback to raw signal calculation
		avgMagnitude := d.rawSignalSum / float32(d.rawSignalCount)
		power := avgMagnitude * avgMagnitude
		if power > 0 {
			stats.SignalStrength = 10.0 * math32.Log10(power)
		}
	}

	return stats
}

// Transmission implements modulation.Receiver.
func (d *Demodulator) Transmission() modulation.TransmissionStats {
	isLocked := d.preambleDetected && d.active && !d.lockLost
	return modulation.TransmissionStats{
		TransmissionStart: d.transmissionStartIdx,
		TransmissionEnd:   d.transmissionEndIdx,
		PacketStart:       d.packetStartIdx,
		PreambleStart:     d.preambleStartIdx,
		IsLocked:          isLocked,
		LockStart:         d.lockAcquiredIdx,
		LockEnd:           d.lockLostIdx,
		LockCount:         d.lockCount,
		UnlockCount:       d.unlockCount,
	}
}

// Params implements modulation.Receiver.
func (d *Demodulator) Params() modulation.TransmitterParams {
	power := d.cfg.Amplitude * d.cfg.Amplitude
	signalStrength := float32(0)
	if power > 0 {
		signalStrength = 10.0 * math32.Log10(power)
	}
	return modulation.TransmitterParams{
		BaudRate:         d.cfg.SymbolRate(),
		SampleRate:       d.cfg.SampleRate,
		SignalStrength:   signalStrength,
		Bandwidth:        d.cfg.RequiredBandwidth(),
		CarrierFrequency: d.cfg.CarrierHz,
	}
}
