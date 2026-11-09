// Package qi8 provides generic int8 neural-network kernels in pure Go.
//
// They are the bit-exact reference for accelerated paths (ESP32-S3 PIE, K210):
// any accelerated kernel must produce identical bytes. See SPEC.md.
//
// Conventions:
//   - Activations are NHWC with batch 1, int8 asymmetric (scale, zero point
//     zp in [-128, 127]). Weights are int8 symmetric (zero point 0) with
//     per-output-channel scales. Bias is int32 per output channel, already in
//     accumulator scale (inScale*weightScale[o]).
//   - Conv weights are OHWI [OutC][KH][KW][InC]; depthwise weights are
//     [KH][KW][C] (channel multiplier 1); fully connected weights are [Out][In].
//   - acc = bias[o] + sum (x - zpIn) * w in int32; padded taps contribute 0
//     (pad with zpIn). The reduction length is limited to MaxTaps so the sum
//     cannot wrap, and every size is overflow-checked against MaxElements;
//     violations return ErrShape.
//   - Output y = clamp(roundHalfAwayFromZero(acc*Mult / 2^(31-Shift)) + zpOut,
//     actMin, actMax) computed exactly in int64 with a single rounding (see
//     Requant). TFLite's two-step SaturatingRoundingDoublingHighMul +
//     RoundingDivideByPOT can differ from this by 1 LSB on rare ties; a
//     TFLite-compatible mode is a possible later option.
//   - Fused activations are expressed by the clamp range (ActNone, ActReLU,
//     ActReLU6).
//   - Padding is TensorFlow SAME or VALID. Pools share scale and zero point
//     between input and output and only count valid cells.
//
// Kernels never allocate: callers provide dst slices and shape structs.
// Errors are package-level sentinels (ErrShape, ErrLength, ErrRequant,
// ErrActivation) returned unwrapped.
package qi8
