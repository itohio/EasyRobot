//go:build qi8generic || (!esp32s3 && !qi8pieemu)

package qi8

// Backend seam for the prepared layers (layer.go). The free functions always
// run the generic inner loops directly; prepared layers call the identifiers
// below, which a backend file may implement with target-specific kernels.
//
// This file is the generic backend: every function forwards to the same
// inner loop the free function uses. It builds on every target except
// esp32s3, where impl_esp32s3.go (PIE) replaces it; `-tags qi8generic`
// forces the generic backend on any target, esp32s3 included.
//
// A backend file must define every identifier in this file with the same
// signature. Its output must be byte-identical to the generic backend (see
// SPEC.md, "Prepared layers and backends"). Contract for the weighted ops:
//   - *ScratchLen(pl) returns the bytes of caller-provided Scratch the backend
//     needs for repacked/aligned weights (include any alignment slack; the
//     caller's slice need not be aligned), or 0.
//   - *Pack(scratch, ...) runs once in Prepare after validation, with
//     scratch == Scratch[:n] (nil when n == 0). It must not allocate.
//   - *Run(...) gets the validated plan, the original weights and the packed
//     scratch; dst and src lengths are already checked.
//   - weightOffset(scratch) is where the packed weights start in a scratch
//     of non-zero ScratchLen: W placed there is packed in place (see
//     InPlaceW); any other overlap of W with Scratch[weightOffset:n] is
//     rejected in Prepare.
//
// Plans are passed by pointer and must not be modified by the backend.

// backendName is reported by Backend.
const backendName = "generic"

// weightOffset is unused by the generic backend (it packs nothing).
func weightOffset([]int8) int { return 0 }

func conv2DScratchLen(*convPlan) int { return 0 }

func conv2DPack(_, _ []int8, _ []int32, _ []Requant, _ *convPlan) {}

func conv2DRun(dst, src, w, _ []int8, bias []int32, rq []Requant, pl *convPlan) {
	conv2DGeneric(dst, src, w, bias, rq, pl)
}

func depthwiseConv2DScratchLen(*convPlan) int { return 0 }

func depthwiseConv2DPack(_, _ []int8, _ []int32, _ []Requant, _ *convPlan) {}

func depthwiseConv2DRun(dst, src, w, _ []int8, bias []int32, rq []Requant, pl *convPlan) {
	depthwiseConv2DGeneric(dst, src, w, bias, rq, pl)
}

func fullyConnectedScratchLen(*fcPlan) int { return 0 }

func fullyConnectedPack(_, _ []int8, _ []int32, _ []Requant, _ *fcPlan) {}

func fullyConnectedRun(dst, src, w, _ []int8, bias []int32, rq []Requant, pl *fcPlan) {
	fullyConnectedGeneric(dst, src, w, bias, rq, pl)
}

func maxPool2DRun(dst, src []int8, pl *poolPlan) { maxPool2DGeneric(dst, src, pl) }

func avgPool2DRun(dst, src []int8, pl *poolPlan) { avgPool2DGeneric(dst, src, pl) }

func reluRun(dst, src []int8, zp int8) { reluGeneric(dst, src, zp) }

func relu6Run(dst, src []int8, zp, q6 int8) { relu6Generic(dst, src, zp, q6) }

func addRun(dst, a, b []int8, p *AddParams) { addGeneric(dst, a, b, p) }
