I want you to read x\math\dsp\modulation\mpsk\SPEC.md, 
.cursorrules,
x\math\dsp, filter, and primitive\fp32,
x\buffer, and
x\chain Processor interface.
Read what is available in dsp package and some from primitive\fp32 package.

Modulator should:
- validate src implements BitsReader interface and BitsLength match with modulator settings
- Read bytes one symbol at a time. Assume either buffer.Bits, or buffer.BitRing type, so reading should be assumed to be per symbol. Read into array of size1. Mask into appropriate number of bits before using to fetch symbol code.
- Maintains state like phase, current symbol portion written...
- write to dst samples as float32. It should be convenient to write the number of samples it takes to encode each symbol
- if no more samples left in src - return number of samples written. continue modulating with subsequent call
- if dst is nil - calculate maximum number of samples needed to encode the source 
- if dst has not enough samples - continue when more space for samples are provided in the subsequent call. The pattern is to call Process in a loop giving portions of a stream of data.

Demodulator should do the same but in reverse 
- validate dst implements BitsWriter
- read as many samples as needed to maintain internal state like transmission detection, symbol tracking, signal decoding. Should not be excessive like thousands of symbols/etc. we need to do calculations on the buffer so keep it small enough to be useful
- when symbol correlation is detected write to dst.
- if no more samples are available - return number of decoded bytes written and continue demodulating with the subsequent call. calling pattern follows modulator pattern.
- if no more space left in dst - return number of decoded bytes written and continue with subsequent Process call.
- demodulator should not write bytes if there is no transmission detected - write bytes when transmission is detected and being decoded
- stop decoding once transmission is no longer detected (and wait for next transmission in the stream of samples)


in both cases reading should be as long as:
- either src is exhausted or dst is exhausted.
- return number of elements written to the dst.

Modulator/Demodulator might need a preamble(alternating symbols) to achieve sync. Demodulator ideally should not write preamble bytes to dst.

Just write a plan what needs to be implemented in order for the modulator->noise->demodulator roundtrip to work:
- modulate one byte infSNR
- surround transmission samples with 0.5s of silence
- demodulate and receive one byte that matches transmitted byte
- split the input buffer into three parts (call demodulator.Process three times with advancing the buffer - just instantiate new buffer passed as src with a different slice of the samples buffer)

other test cases
- modulate four bytes infSNR
- modulate 33 bytes infSNR

- 7 bytes but with noise added to the signal - SNR 20dB
- 7 bytes but with noise added to the signal - SNR 15dB
- 7 bytes with 10dB SNR (if theoretically possible given the modulation scheme and symbol size)
- 7 bytes with 5dB SNR (if theoretically possible given the modulation scheme and symbol size)

Use in tests:
- 96khz sample rate
- 20khz carrier
- 2, 4, 8 bits per symbol (for 7bytes with 15db SNR)

There should be methods to calculate theoretical bandwidth required for the transmission based on modulator parameters so that users can calculate sample rate and carrier frequency. And theoretical SNR Limit calculation. Also, symbol separation is useful to know - depends on modulation scheme though - should be abstract value with sound foundation.

Prefer spectrally optimal symbol envelope shapes when modulating.
Use generators from dsp.

Write x\math\dsp\modulation\mfsk\IMPLEMENTATION_PLAN.md using ABSOLUTE PATH!!!!
Reuse PLL from filer/pll package.
NOTE: Different M require different minimum SNR.