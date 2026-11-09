// Package qnet is a tiny sequential int8 network runner on top of the qi8
// kernels, sized for microcontrollers (ESP32-S3, K210) as well as the host.
//
// A Graph is a []Layer of plain data (kind, shapes, quantisation, geometry,
// fused clamp, and slices of int8 weights, int32 bias and qi8.Requant), so an
// accelerator backend can inspect and compile layers. A Runner plans the graph
// once into segments, each claimed by the first Backend that accepts it
// (Generic runs any layer with qi8, one per segment), and executes the plan
// over a caller-provided ping-pong arena without allocating.
//
// MobileNetV1 fills a caller-provided []Layer with a MobileNetV1-shaped graph;
// ParamCounts sizes the parameter backing arrays, BindParams slices them into
// the layers, and SyntheticParams fills them deterministically.
//
// Nothing here allocates except NewRunner (one Runner); errors are package or
// qi8 sentinels returned unwrapped. The package imports only the standard
// library and qi8. See SPEC.md.
package qnet
