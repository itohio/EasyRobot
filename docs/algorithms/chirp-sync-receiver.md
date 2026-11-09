---
type: algorithm
status: partial
tags: [easyrobot, audio, dsp, sync]
workstream: W7
module: "[[audio-modem-cli]]"
summary: "`gen-chirp` writes silence → reference tone (default 4400 Hz, 250 ms, ref channel) → delay → chirp…"
source: [cmd/audio/gen_chirp.go, cmd/audio/recv_chirp.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "offline run 2026-11-09: gen-chirp -d 1s --signal-channel 0 --chirp-type log; recv-chirp -c 2 --channel 0 --chirp-duration 1s --padding 500ms --chirp-type log → ref detected at samples 48000-60000 (exact); correlation WAV 95999 samples but 95996 of them clipped at ±2^31-1"
---
# Chirp sync receiver (`gen-chirp` / `recv-chirp`)

**What it does:** `gen-chirp` writes silence → reference tone (default 4400 Hz, 250 ms, ref channel) → delay → chirp (linear/log/pink-log, signal channel) → silence. `recv-chirp` finds the reference tone with a sliding Goertzel, cuts the expected chirp region, cross-correlates it with a locally generated chirp, and writes the correlation to a WAV (room/channel impulse-response style measurement).

**Where:** cmd/audio/gen_chirp.go:71-309; cmd/audio/recv_chirp.go — `runRecvChirp` :74-268, `detectRefSignal` :271-352 (Goertzel window = ref length, step ¼ window, then 1-sample refinement around the max), `extractChannel` :354, `concatenateTensors` :398. Correlation via `dsp.ChirpGenerator.Correlate` / `LogarithmicChirpGenerator.Correlate` ([[dsp-core]]).

**Inputs → outputs:** WAV (or capture device) `[frames,ch]` → correlation WAV `[2N-1, 1]` (FP32 tensor written as int32 PCM).

## Completeness review

- Complete: no.
  - Gen/recv defaults disagree: gen puts ref on ch 0 and chirp on ch 1 (gen_chirp.go:57-58), recv expects both on one `--channel` (recv_chirp.go:65,193); default pair fails ("insufficient audio data: need 299969 frames, have 204000" with channel 1). CHIRP_PLAN.md specified separate `--ref-channel`/`--signal-channel` for recv; not implemented.
  - pink-log correlation falls back to log chirp (:230-234).
  - Adaptive threshold is dead code: noise loop `for i := 0; i < noiseSamples-windowSize` with `noiseSamples = windowSize` never runs (:288-296), so threshold is always 0.01 when `--ref-amplitude 0`; and it is compared with Goertzel amplitudes of un-normalised int32 samples (see [[audio-io]] scaling), so it is meaningless.
  - Plan's chunked streaming collection is only implemented for devices; WAV is read whole.
- Tested: no unit tests. Offline run above.
- Correct: no for output — reference detection is exact on a clean synthetic file, but correlation output is un-normalised and saturates the WAV (95996/95999 samples at full scale; peak position unusable). Input samples are int32-scale, so correlation magnitudes are ~1e9×N.

## TODOs

- [ ] Normalise input to [-1,1] and correlation output (divide by energy) before writing — cmd/audio/recv_chirp.go:180-243 #p1 #ER-022
- [ ] Separate `--ref-channel`/`--signal-channel` (match gen defaults / CHIRP_PLAN) — recv_chirp.go:65,193 #p2
- [ ] Fix dead adaptive-threshold loop — recv_chirp.go:288-296 #p2
- [ ] Report peak lag / delay estimate instead of only a WAV — recv_chirp.go:245-267 #p3
- [ ] Pink-log chirp correlate — recv_chirp.go:230-234 #p3

## Links

- Module: [[audio-modem-cli]]; [[dsp-core]]; [[audio-io]]
