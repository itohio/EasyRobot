package gocv

// Package gocv provides a tensor backend backed by gocv.Mat for vision-first
// workloads. The implementation focuses on image-friendly shapes
// ([height, width, channels]) and only supports a minimal subset of the
// tensor API required by EasyRobot's vision pipelines. Unsupported tensor
// operations panic with ErrUnsupported.
