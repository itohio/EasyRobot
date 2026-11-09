---
type: algorithm
status: implemented
tags: [easyrobot, dsp, modulation, adaptive]
workstream: W7
module: "[[modulation]]"
summary: "ok 4"
source: [x/math/dsp/modulation/adaptive/processor.go]
tests: [x/math/dsp/modulation/adaptive/processor_test.go]
test_result: pass
complete: yes
correct: unknown
evidence: "go test -count=1 -v ./x/math/dsp/modulation/adaptive → ok, 4 pass (2026-11-09)"
---
# Adaptive tier selector (quality-driven processor switch)

**What it does:** a generic `chain.Processor` wrapper that holds tiers `{MinQuality, Factory}` sorted from high to low and delegates to the highest tier whose `MinQuality ≤ quality`. `SetQuality(q)` re-creates the delegate when the tier changes. It is meant for switching modulation order (e.g. 16-QAM → QPSK → FSK) as the channel changes.

**Where:** `adaptive/processor.go:32-54` `New`, `:56-62` `Process`, `:71-89` `SetQuality`/`setQuality`, `:91-98` `selectTier`. Claims are in `adaptive/SPEC.md` (untrusted).

**Inputs → outputs:** generic over `dsp.Sample` Dst/Src; quality is a caller-supplied float32 with no defined unit.

## Completeness review

- Complete: yes for the stated scope (routing and switching). No hysteresis. A tier switch drops the delegate's internal state (a new instance from `Factory()`), so a switch in the middle of a stream loses buffered samples. Nothing feeds quality automatically, and the only candidate source, `Receiver.Quality().SNR`, is broken in every demodulator ([[modem-quality-estimators]]). TX and RX switching is not coordinated: there is no in-band signalling of the tier.
- Tested: `SelectsInitialTier`, `UpgradeAndDowngrade`, `ResetDelegates`, `NilTierError` pass with stub processors; they check tier choice and delegation, not modem behaviour.
- Correct: unknown end to end (never exercised with real modems).

## TODOs

- [ ] Hysteresis or dwell time on tier changes — adaptive/processor.go:75-89 #p2
- [ ] In-band tier signalling (header field) so the receiver can follow the transmitter — adaptive/SPEC.md #p2
- [ ] Feed from a working SNR/EVM estimator — see [[modem-quality-estimators]] #p2

## Links

- Module: [[modulation]] · [[W7-filters-codecs-voice]]
