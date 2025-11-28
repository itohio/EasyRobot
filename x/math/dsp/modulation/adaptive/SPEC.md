# Adaptive Modulation Scheme — Specification (`x/math/dsp/modulation/adaptive/SPEC.md`)

## 0. Scope

This specification defines a generic streaming adaptive modulator-demodulator system. The system dynamically selects and switches between different modulation/demodulation strategies (tiers) based on measured channel quality, using a list of user-supplied factories and thresholds. The component can act as both a modulator (e.g., bytes → float32 samples) or demodulator (e.g., float32 samples → bytes), and is generic over source and destination types.

The design follows the Processor interface contract and delegates streaming logic (Process) to the currently active tier implementation.

---

## 1. Design Motivation

- Allow robust, bandwidth/throughput-optimal communications across varying channel conditions.
- Switch modulation/demodulation strategies in real-time _without external rewiring_.
- Encapsulate all per-tier state, re-initialization, and processing internally.
- Support both transmit and receive (modulation and demodulation) within one generic instance.
- Enable applications to react to channel measurement by adjusting tier up/down using `.SetQuality(...)`.

---

## 2. High-Level Concept

- The adaptive Processor is parameterized over `[Dst, Src]`.
- Users instantiate with one or more tier factory functions and corresponding "minimum acceptable quality" thresholds (e.g., SNR in dB or any application-chosen metric, _higher is better_).
- At all times, the delegate implements `Process(dst, src) int` and follows the contract (streaming, nil-dst sizing, pausing on full/empty).
- When `.SetQuality(q)` is called:
    - The adaptive processor checks if the new quality warrants _downgrading_ or _upgrading_ the tier.
    - If so, it re-initializes using the appropriate factory and resets internal state.
- On upgrade/downgrade, user may need to re-align, synchronize, or send a new preamble—this is the domain of per-tier implementation (see notes).

---

## 3. Interfaces & API

### 3.1 Core Interface

```go
type Processor[Dst, Src any] interface {
    // Process streams from src to dst, returns processed element count.
    Process(dst Dst, src Src) int
    // SetQuality(q float32) — see below
}
```

### 3.2 Adaptive Processor Construction

```go
type Factory[Dst, Src any] func() Processor[Dst, Src]

type TierConfig[Dst, Src any] struct {
    MinQuality float32         // Minimum required quality (inclusive)
    Factory    Factory[Dst, Src]
}

func NewAdaptiveProcessor[Dst, Src any](
    tierConfigs []TierConfig[Dst, Src], 
    initialQuality float32,   // usually best quality supported, e.g. math.MaxFloat32
) Processor[Dst, Src]
```

- The `tierConfigs` slice is ordered _high-to-low_: first is highest tier, last is fallback/lowest-quality tier.
- On initialization, the processor picks the highest tier that `initialQuality >= MinQuality`.

### 3.3 Quality Adjustment

```go
// SetQuality: Change current quality index and (if needed) reset modular/demodulator state.
SetQuality(q float32)
```

- If the new `q` drops below any currently active tier's MinQuality, downgrade (and reset).
- If `q` rises to meet a higher-quality tier, upgrade (and reset).
- If the tier does not change, no action is taken.

---

## 4. Streaming Logic

- `Process(dst, src)` simply forwards to the _currently selected tier's_ Process method.
- All state necessary for demodulation/modulation is encapsulated within the delegate.
- Upon tier switch (upgrade/downgrade), the selected factory is called to create a new delegate, and per-tier state is reset. (Sticky state in Adaptive wrapper is not allowed; this avoids state-sharing bugs.)

---

## 5. Usage Pattern & Quality Workflow

1. **Initial Setup**  
   User provides all desired modulation/demodulation tiers and factories, with required thresholds:
   ```go
   // For example, descending list:
   tiers := []TierConfig[Dst, Src]{
       {MinQuality: 30, Factory: New16QAMModem},
       {MinQuality: 18, Factory: NewQPSKModem},
       {MinQuality: 10, Factory: NewBPSKModem},
       {MinQuality: -1e9, Factory: NewFSKModem},
   }
   modem := NewAdaptiveProcessor(tiers, measuredQuality)
   ```

2. **Streaming**  
   Call `.Process(dst, src)` as usual. (Streaming chunk size, NIL-dst sizing, etc. are dictated by delegate.)

3. **Quality Measurement & Tier Switching**  
   After every relevant received buffer, user computes the _current channel quality_ (e.g., instantaneous SNR).
   - If quality drops: call `modem.SetQuality(snr_measured)`
   - The adaptive processor auto-switches tier as needed and resets internal state.
   - (Optional: Provide notification/callback hook.)

4. **Upgrade/Downgrade**  
   If channel quality _improves_, user should also call `SetQuality`, and, if desired, force transmitter and receiver to re-sync.
   - Application should manage link re-negotiation, synchronization, or preamble/re-frame as needed. Adaptive processor does not enforce protocol boundaries.

---

## 6. Notes

- **Generic Type Parameters**: All types are generic over destination and source (e.g., `[]float32, buffer.BitsReader`, `[]byte, []byte`, etc).
- **Preamble/Framing**: On a tier switch, per-tier modem may require a fresh preamble/padding for symbol/timing lock. Application should ensure that both ends coordinate switching, using a protocol marker or explicit out-of-band sync if needed.
- **Downgrade Policy**: Adaptive processor always prefers the _highest tier_ currently supportable given the measured quality.
- **Quality Metric**: Can be any monotonic float32 (e.g., SNR dB, BER estimate, etc), as long as thresholds are consistently defined across tiers.
- **Thread Safety**: Adaptive processor does **not** guarantee threading safety; concurrent calls to `Process` and `SetQuality` are user responsibility.
- **Variant use**: Can be applied to modulation (tx side), demodulation (rx side), or joint modem, depending on the Processor factories supplied.

---

## 7. Example Selector Algorithm

Pseudocode for tier selection:

```
IndexTier(q):
    for i, tier in enumerate(tierConfigs):
        if q >= tier.MinQuality:
            return i
    return len(tierConfigs) - 1  // Fallback
```

On `SetQuality(q)`:
- If tier index differs from current: create&reset new delegate.

---

## 8. References

- See `x/math/dsp/modulation/mfsk` for factory-tiers pattern and streaming Processor interface.
- See user plans in `x/chain`, `buffer`, and higher-level DSP frameworks.

---

