package main

import (
	"time"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
)

// Detector finds objects in camera frames: the part another model (e.g. an
// SSD on the KPU) replaces, while the camera loop, the stream format and
// the page stay. Frames are little-endian RGB565 of the size the detector
// was built for.
type Detector interface {
	// Detect returns the objects in img, boxes in img's pixels, classes
	// indexing Labels; the result is valid until the next Detect.
	Detect(img []byte) ([]frame.Detection, error)
	// Labels names the classes.
	Labels() []string
	// Stages returns the last Detect's preprocessing and inference times.
	Stages() (prep, infer time.Duration)
	// Check returns FNV-1a hashes of the last Detect's model input and
	// first two outputs (zero where a model has none), logged so a host run
	// on the same image can be compared bit for bit.
	Check() [3]uint32
}

// yoloxObjects is YOLOX-nano on qnet's CPU kernels behind Detector: the
// frame is letterboxed into the 416×416 input, and the boxes are mapped
// back to frame pixels. Buffers are allocated once; Detect allocates
// nothing.
type yoloxObjects struct {
	lb          *letterbox
	d           *detector
	in          []int8
	dets        [frame.MaxDetections]frame.Detection
	prep, infer time.Duration
}

// newYOLOXObjects wraps d for w×h frames; in is the detector input buffer
// (416×416×3), passed in so the caller can allocate it before the model.
func newYOLOXObjects(d *detector, in []int8, w, h int) *yoloxObjects {
	return &yoloxObjects{lb: newLetterbox(w, h), d: d, in: in}
}

func (y *yoloxObjects) Detect(img []byte) ([]frame.Detection, error) {
	t0 := time.Now()
	y.lb.run(y.in, img)
	t1 := time.Now()
	boxes, err := y.d.detect(y.in)
	y.prep, y.infer = t1.Sub(t0), time.Since(t1)
	if err != nil {
		return nil, err
	}
	n := min(len(boxes), len(y.dets))
	for i, b := range boxes[:n] {
		y.dets[i] = y.lb.toImage(b)
	}
	return y.dets[:n], nil
}

func (y *yoloxObjects) Labels() []string { return labels[:] }

func (y *yoloxObjects) Stages() (prep, infer time.Duration) { return y.prep, y.infer }

func (y *yoloxObjects) Check() [3]uint32 {
	return [3]uint32{fnv1a(y.in), fnv1a(y.d.scores(y.in)), fnv1a(y.d.boxes(y.in))}
}
