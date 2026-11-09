---
type: algorithm
status: partial
tags: [easyrobot, audio, dsp]
workstream: W7
module: "[[audio-io]]"
summary: "changes a tensor's sample rate before device playback when WAV rate ≠ device rate."
source: [x/marshaller/audio/tensor_convert.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "code reading; no tests; only called on device playback (marshaller.go:472), not run"
---
# Naive audio resampling (`matchSampleRate`)

**What it does:** changes a tensor's sample rate before device playback when WAV rate ≠ device rate.

**Where:** `matchSampleRate` x/marshaller/audio/tensor_convert.go:278-297; `downsampleTensor` :300-366 (take every Nth frame), `upsampleTensor` :369-431 (zero-order hold), `resampleTensorLinear` :434-483 (linear interpolation, FP32 only). Caller: `playbackSingleDevice` marshaller.go:472.

**Inputs → outputs:** tensor `[frames,ch]`, source/target rate (int Hz) → new tensor of `frames*target/source` frames.

## Completeness review

- Complete: no — `TODO: Implement proper resampling with anti-aliasing` (:284); linear path rejects integer dtypes (:477-479), so INT16 44.1 k → 48 k playback errors.
- Tested: none.
- Correct: no for audio quality — decimation without low-pass aliases everything above the new Nyquist; ZOH upsampling leaves images (sinc² droop + spectral copies). Functionally index math looks right (bounds-checked).
- Not used on the read side: WAV read refuses rate mismatch (unmarshaller.go:242-246), so no resampling exists for the receive chain — relevant to channel models (clock drift / sample-rate offset simulation) in [[audio-modem-cli]].

## TODOs

- [ ] Polyphase/windowed-sinc resampler with anti-alias filter (reuse [[filters]] FIR) — x/marshaller/audio/tensor_convert.go:284 #p2
- [ ] Support integer dtypes in linear/fractional path — tensor_convert.go:477-479 #p2
- [ ] Tests: tone through resampler, check frequency and alias level — tensor_convert.go:278 #p2

## Links

- Module: [[audio-io]]; [[dsp-core]]; [[filters]]
