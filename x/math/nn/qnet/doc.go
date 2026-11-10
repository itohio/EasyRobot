// Package qnet is a tiny int8 network runner on top of the qi8 kernels, sized
// for microcontrollers (ESP32-S3, K210) as well as the host.
//
// A Graph is a []Layer of plain data (kind, producers, shapes, quantisation,
// geometry, fused clamp, rounding mode, and slices of int8 weights, int32 bias
// and qi8.Requant), so an accelerator backend can inspect and compile layers.
// Layers run in order; each reads the layers in its From (by default the
// previous one), so residual networks (Add) are expressed directly.
//
// PlanMemory plans a graph once: backends may claim chains of layers as
// segments, every other layer runs on the built-in CPU kernels (qi8 prepared
// layers, PIE on the ESP32-S3), and all activations and backend scratch are
// placed in one arena whose tensors are reused once their readers have run.
// A Runner prepares the layers over a caller-provided arena of that size and
// executes the plan without allocating.
//
// MobileNetV1 fills a caller-provided []Layer with a MobileNetV1-shaped graph;
// ParamCounts sizes the parameter backing arrays, BindParams slices them into
// the layers, and SyntheticParams fills them deterministically. Package
// qnet/tfl converts TFLite int8 models into graphs.
//
// Planning and NewRunner/Init allocate once; Run does not. Errors are package
// or qi8 sentinels returned unwrapped. The package imports only the standard
// library and qi8. See SPEC.md.
package qnet
