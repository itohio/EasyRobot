// Package tfl converts int8 TensorFlow Lite models into qnet graphs.
//
// Load parses the flatbuffer with a small bounds-checked reader (no
// reflection, no generated code, TinyGo-friendly; weights alias the model
// bytes, which may live in flash) and maps each operator onto qnet layers
// that reproduce TFLite's reference kernels with qi8.RoundTFLite. Unsupported
// operators or tensor types fail with an *OpError naming the operator. See
// ../SPEC.md, "TFLite import".
package tfl
