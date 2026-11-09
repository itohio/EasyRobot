package qi8

// Prepared layers validate their parameters once (Prepare) and then run many
// times (Run) with only the dst/src length checks. They compute exactly what
// the corresponding free function computes, through the backend seam in
// impl_*.go (see Backend); the free functions remain the always-generic
// reference.
//
// Usage: fill the exported fields, call Prepare, then Run. Changing an
// exported field (params or the W/Bias/RQ/Scratch slice headers) after
// Prepare makes Run return ErrNotPrepared until Prepare is called again. The
// contents of W, Bias and RQ must not be modified after Prepare either: they
// are validated (RQ) or repacked (by a backend that uses Scratch) only there,
// and in-place edits are not detected. InShape and OutShape return the zero
// Shape unless the last Prepare succeeded.

// Op is a prepared layer.
type Op interface {
	// InShape returns the shape src must have (len(src) == InShape().Len()).
	InShape() Shape
	// OutShape returns the shape dst must have.
	OutShape() Shape
	// Run computes dst from src. It returns ErrNotPrepared or ErrLength, or
	// nil.
	Run(dst, src []int8) error
}

var (
	_ Op = (*Conv2DLayer)(nil)
	_ Op = (*DepthwiseConv2DLayer)(nil)
	_ Op = (*FCLayer)(nil)
	_ Op = (*MaxPool2DLayer)(nil)
	_ Op = (*AvgPool2DLayer)(nil)
	_ Op = (*ReLULayer)(nil)
	_ Op = (*ReLU6Layer)(nil)
)

// Backend returns the name of the backend the prepared layers run on:
// "generic", or e.g. "esp32s3-pie" when built with an accelerated backend.
func Backend() string { return backendName }

// sameSlice reports whether a and b are the same slice (data pointer and
// length).
func sameSlice[T any](a, b []T) bool {
	return len(a) == len(b) && (len(a) == 0 || &a[0] == &b[0])
}

// weightSnap is the snapshot a weighted layer takes in Prepare.
type weightSnap struct {
	w, scratch []int8
	packed     []int8 // scratch[:ScratchLen], nil when unused
	bias       []int32
	rq         []Requant
	ok         bool
}

func (s *weightSnap) same(w []int8, bias []int32, rq []Requant, scratch []int8) bool {
	return s.ok && sameSlice(w, s.w) && sameSlice(bias, s.bias) && sameSlice(rq, s.rq) && sameSlice(scratch, s.scratch)
}

// packScratch returns scratch[:n], nil when n is 0, or ErrLength when scratch
// is shorter than n.
func packScratch(scratch []int8, n int) ([]int8, error) {
	if n == 0 {
		return nil, nil
	}
	if len(scratch) < n {
		return nil, ErrLength
	}
	return scratch[:n:n], nil
}

func convIn(pl *convPlan) Shape  { return Shape{H: pl.inH, W: pl.inW, C: pl.ch} }
func convOut(pl *convPlan) Shape { return Shape{H: pl.outH, W: pl.outW, C: pl.outC} }

// Conv2DLayer is a prepared Conv2D. W, Bias and RQ are as for Conv2D.
type Conv2DLayer struct {
	P    Conv2DParams
	W    []int8
	Bias []int32
	RQ   []Requant
	// Scratch is optional backend storage for repacked weights, at least
	// ScratchLen() long. Ignored when ScratchLen() is 0 (generic backend).
	Scratch []int8

	prep Conv2DParams
	pl   convPlan
	s    weightSnap
}

// ScratchLen returns the Scratch length the backend needs for P (0 for the
// generic backend or invalid params). Call it before Prepare.
func (l *Conv2DLayer) ScratchLen() int {
	pl, err := l.P.plan()
	if err != nil {
		return 0
	}
	return conv2DScratchLen(&pl)
}

// Prepare validates the layer and caches the resolved geometry. It returns
// the error Conv2D would return for these parameters and slices, or
// ErrLength when Scratch is shorter than ScratchLen().
func (l *Conv2DLayer) Prepare() error {
	l.pl, l.s = convPlan{}, weightSnap{}
	pl, err := l.P.plan()
	if err != nil {
		return err
	}
	if len(l.W) != pl.wLen {
		return ErrLength
	}
	if err := pl.setQuant(l.Bias, l.RQ, l.P.InZP, l.P.OutZP, l.P.ActMin, l.P.ActMax); err != nil {
		return err
	}
	packed, err := packScratch(l.Scratch, conv2DScratchLen(&pl))
	if err != nil {
		return err
	}
	conv2DPack(packed, l.W, l.Bias, l.RQ, &pl)
	l.prep, l.pl = l.P, pl
	l.s = weightSnap{w: l.W, scratch: l.Scratch, packed: packed, bias: l.Bias, rq: l.RQ, ok: true}
	return nil
}

// InShape returns the input shape.
func (l *Conv2DLayer) InShape() Shape { return convIn(&l.pl) }

// OutShape returns the output shape.
func (l *Conv2DLayer) OutShape() Shape { return convOut(&l.pl) }

// Run computes the convolution of src into dst.
func (l *Conv2DLayer) Run(dst, src []int8) error {
	if !l.s.same(l.W, l.Bias, l.RQ, l.Scratch) || l.P != l.prep {
		return ErrNotPrepared
	}
	if len(src) != l.pl.inLen || len(dst) != l.pl.outLen {
		return ErrLength
	}
	conv2DRun(dst, src, l.s.w, l.s.packed, l.s.bias, l.s.rq, &l.pl)
	return nil
}

// DepthwiseConv2DLayer is a prepared DepthwiseConv2D. W, Bias and RQ are as
// for DepthwiseConv2D.
type DepthwiseConv2DLayer struct {
	P    DepthwiseConv2DParams
	W    []int8
	Bias []int32
	RQ   []Requant
	// Scratch is optional backend storage for repacked weights, at least
	// ScratchLen() long. Ignored when ScratchLen() is 0 (generic backend).
	Scratch []int8

	prep DepthwiseConv2DParams
	pl   convPlan
	s    weightSnap
}

// ScratchLen returns the Scratch length the backend needs for P (0 for the
// generic backend or invalid params). Call it before Prepare.
func (l *DepthwiseConv2DLayer) ScratchLen() int {
	pl, err := l.P.plan()
	if err != nil {
		return 0
	}
	return depthwiseConv2DScratchLen(&pl)
}

// Prepare validates the layer and caches the resolved geometry. It returns
// the error DepthwiseConv2D would return for these parameters and slices, or
// ErrLength when Scratch is shorter than ScratchLen().
func (l *DepthwiseConv2DLayer) Prepare() error {
	l.pl, l.s = convPlan{}, weightSnap{}
	pl, err := l.P.plan()
	if err != nil {
		return err
	}
	if len(l.W) != pl.wLen {
		return ErrLength
	}
	if err := pl.setQuant(l.Bias, l.RQ, l.P.InZP, l.P.OutZP, l.P.ActMin, l.P.ActMax); err != nil {
		return err
	}
	packed, err := packScratch(l.Scratch, depthwiseConv2DScratchLen(&pl))
	if err != nil {
		return err
	}
	depthwiseConv2DPack(packed, l.W, l.Bias, l.RQ, &pl)
	l.prep, l.pl = l.P, pl
	l.s = weightSnap{w: l.W, scratch: l.Scratch, packed: packed, bias: l.Bias, rq: l.RQ, ok: true}
	return nil
}

// InShape returns the input shape.
func (l *DepthwiseConv2DLayer) InShape() Shape { return convIn(&l.pl) }

// OutShape returns the output shape.
func (l *DepthwiseConv2DLayer) OutShape() Shape { return convOut(&l.pl) }

// Run computes the depthwise convolution of src into dst.
func (l *DepthwiseConv2DLayer) Run(dst, src []int8) error {
	if !l.s.same(l.W, l.Bias, l.RQ, l.Scratch) || l.P != l.prep {
		return ErrNotPrepared
	}
	if len(src) != l.pl.inLen || len(dst) != l.pl.outLen {
		return ErrLength
	}
	depthwiseConv2DRun(dst, src, l.s.w, l.s.packed, l.s.bias, l.s.rq, &l.pl)
	return nil
}

// FCLayer is a prepared FullyConnected. W, Bias and RQ are as for
// FullyConnected. Its shapes are {1, 1, In} and {1, 1, Out}.
type FCLayer struct {
	P    FCParams
	W    []int8
	Bias []int32
	RQ   []Requant
	// Scratch is optional backend storage for repacked weights, at least
	// ScratchLen() long. Ignored when ScratchLen() is 0 (generic backend).
	Scratch []int8

	prep FCParams
	pl   fcPlan
	s    weightSnap
}

// ScratchLen returns the Scratch length the backend needs for P (0 for the
// generic backend or invalid params). Call it before Prepare.
func (l *FCLayer) ScratchLen() int {
	pl, err := l.P.plan()
	if err != nil {
		return 0
	}
	return fullyConnectedScratchLen(&pl)
}

// Prepare validates the layer. It returns the error FullyConnected would
// return for these parameters and slices, or ErrLength when Scratch is
// shorter than ScratchLen().
func (l *FCLayer) Prepare() error {
	l.pl, l.s = fcPlan{}, weightSnap{}
	pl, err := l.P.plan()
	if err != nil {
		return err
	}
	if len(l.W) != pl.wLen {
		return ErrLength
	}
	if err := pl.setQuant(l.Bias, l.RQ, l.P); err != nil {
		return err
	}
	packed, err := packScratch(l.Scratch, fullyConnectedScratchLen(&pl))
	if err != nil {
		return err
	}
	fullyConnectedPack(packed, l.W, l.Bias, l.RQ, &pl)
	l.prep, l.pl = l.P, pl
	l.s = weightSnap{w: l.W, scratch: l.Scratch, packed: packed, bias: l.Bias, rq: l.RQ, ok: true}
	return nil
}

// InShape returns {1, 1, In} (zero unless prepared).
func (l *FCLayer) InShape() Shape {
	if l.pl.in == 0 {
		return Shape{}
	}
	return Shape{H: 1, W: 1, C: l.pl.in}
}

// OutShape returns {1, 1, Out} (zero unless prepared).
func (l *FCLayer) OutShape() Shape {
	if l.pl.out == 0 {
		return Shape{}
	}
	return Shape{H: 1, W: 1, C: l.pl.out}
}

// Run computes dst = W·(src-InZP) + Bias, requantised.
func (l *FCLayer) Run(dst, src []int8) error {
	if !l.s.same(l.W, l.Bias, l.RQ, l.Scratch) || l.P != l.prep {
		return ErrNotPrepared
	}
	if len(src) != l.pl.in || len(dst) != l.pl.out {
		return ErrLength
	}
	fullyConnectedRun(dst, src, l.s.w, l.s.packed, l.s.bias, l.s.rq, &l.pl)
	return nil
}

// poolLayer is the shared state of the pooling layers.
type poolLayer struct {
	prep Pool2DParams
	pl   poolPlan
	ok   bool
}

func (s *poolLayer) prepare(p Pool2DParams) error {
	*s = poolLayer{}
	pl, err := p.plan()
	if err != nil {
		return err
	}
	*s = poolLayer{prep: p, pl: pl, ok: true}
	return nil
}

func (s *poolLayer) inShape() Shape {
	return Shape{H: s.pl.inH, W: s.pl.inW, C: s.pl.ch}
}

func (s *poolLayer) outShape() Shape {
	return Shape{H: s.pl.outH, W: s.pl.outW, C: s.pl.ch}
}

func (s *poolLayer) check(p Pool2DParams, dst, src []int8) error {
	if !s.ok || p != s.prep {
		return ErrNotPrepared
	}
	if len(src) != s.pl.inLen || len(dst) != s.pl.outLen {
		return ErrLength
	}
	return nil
}

// MaxPool2DLayer is a prepared MaxPool2D.
type MaxPool2DLayer struct {
	P Pool2DParams

	s poolLayer
}

// Prepare validates P; it returns the error MaxPool2D would return.
func (l *MaxPool2DLayer) Prepare() error { return l.s.prepare(l.P) }

// InShape returns the input shape.
func (l *MaxPool2DLayer) InShape() Shape { return l.s.inShape() }

// OutShape returns the output shape.
func (l *MaxPool2DLayer) OutShape() Shape { return l.s.outShape() }

// Run computes the max pooling of src into dst.
func (l *MaxPool2DLayer) Run(dst, src []int8) error {
	if err := l.s.check(l.P, dst, src); err != nil {
		return err
	}
	maxPool2DRun(dst, src, &l.s.pl)
	return nil
}

// AvgPool2DLayer is a prepared AvgPool2D.
type AvgPool2DLayer struct {
	P Pool2DParams

	s poolLayer
}

// Prepare validates P; it returns the error AvgPool2D would return.
func (l *AvgPool2DLayer) Prepare() error { return l.s.prepare(l.P) }

// InShape returns the input shape.
func (l *AvgPool2DLayer) InShape() Shape { return l.s.inShape() }

// OutShape returns the output shape.
func (l *AvgPool2DLayer) OutShape() Shape { return l.s.outShape() }

// Run computes the average pooling of src into dst.
func (l *AvgPool2DLayer) Run(dst, src []int8) error {
	if err := l.s.check(l.P, dst, src); err != nil {
		return err
	}
	avgPool2DRun(dst, src, &l.s.pl)
	return nil
}

// validLen checks an element-wise layer length: 1 ≤ n ≤ MaxElements.
func validLen(n int) error {
	if n <= 0 || n > MaxElements {
		return ErrShape
	}
	return nil
}

// ReLULayer is a prepared ReLU over Len elements (shape {1, 1, Len}); dst
// and src may alias. Unlike ReLU, which accepts any equal lengths including
// 0, Len must be in [1, MaxElements] (ErrShape).
type ReLULayer struct {
	Len int
	ZP  int8

	prepLen int
	prepZP  int8
	ok      bool
}

// Prepare validates Len.
func (l *ReLULayer) Prepare() error {
	l.ok = false
	if err := validLen(l.Len); err != nil {
		return err
	}
	l.prepLen, l.prepZP, l.ok = l.Len, l.ZP, true
	return nil
}

// InShape returns {1, 1, Len} (zero unless prepared).
func (l *ReLULayer) InShape() Shape {
	if !l.ok {
		return Shape{}
	}
	return Shape{H: 1, W: 1, C: l.prepLen}
}

// OutShape returns InShape().
func (l *ReLULayer) OutShape() Shape { return l.InShape() }

// Run computes dst[i] = max(src[i], ZP).
func (l *ReLULayer) Run(dst, src []int8) error {
	if !l.ok || l.Len != l.prepLen || l.ZP != l.prepZP {
		return ErrNotPrepared
	}
	if len(src) != l.prepLen || len(dst) != l.prepLen {
		return ErrLength
	}
	reluRun(dst, src, l.prepZP)
	return nil
}

// ReLU6Layer is a prepared ReLU6 over Len elements (shape {1, 1, Len}); dst
// and src may alias. Len must be in [1, MaxElements] (ErrShape); Q6 < ZP is
// ErrActivation.
type ReLU6Layer struct {
	Len    int
	ZP, Q6 int8

	prepLen        int
	prepZP, prepQ6 int8
	ok             bool
}

// Prepare validates Len and the clamp range.
func (l *ReLU6Layer) Prepare() error {
	l.ok = false
	if err := validLen(l.Len); err != nil {
		return err
	}
	if l.Q6 < l.ZP {
		return ErrActivation
	}
	l.prepLen, l.prepZP, l.prepQ6, l.ok = l.Len, l.ZP, l.Q6, true
	return nil
}

// InShape returns {1, 1, Len} (zero unless prepared).
func (l *ReLU6Layer) InShape() Shape {
	if !l.ok {
		return Shape{}
	}
	return Shape{H: 1, W: 1, C: l.prepLen}
}

// OutShape returns InShape().
func (l *ReLU6Layer) OutShape() Shape { return l.InShape() }

// Run computes dst[i] = clamp(src[i], ZP, Q6).
func (l *ReLU6Layer) Run(dst, src []int8) error {
	if !l.ok || l.Len != l.prepLen || l.ZP != l.prepZP || l.Q6 != l.prepQ6 {
		return ErrNotPrepared
	}
	if len(src) != l.prepLen || len(dst) != l.prepLen {
		return ErrLength
	}
	relu6Run(dst, src, l.prepZP, l.prepQ6)
	return nil
}

// AddLayer is a prepared requantised element-wise addition over Len
// elements (shape {1, 1, Len}); see AddParams. It has two inputs, so it does
// not implement Op. dst may alias a or b. Len must be in [1, MaxElements]
// (ErrShape).
type AddLayer struct {
	Len int
	P   AddParams

	prepLen int
	prepP   AddParams
	ok      bool
}

// Prepare validates Len and P.
func (l *AddLayer) Prepare() error {
	l.ok = false
	if err := validLen(l.Len); err != nil {
		return err
	}
	if err := l.P.valid(); err != nil {
		return err
	}
	l.prepLen, l.prepP, l.ok = l.Len, l.P, true
	return nil
}

// InShape returns {1, 1, Len} (zero unless prepared); both inputs have it.
func (l *AddLayer) InShape() Shape {
	if !l.ok {
		return Shape{}
	}
	return Shape{H: 1, W: 1, C: l.prepLen}
}

// OutShape returns InShape().
func (l *AddLayer) OutShape() Shape { return l.InShape() }

// Run computes dst = a + b requantised (see AddParams).
func (l *AddLayer) Run(dst, a, b []int8) error {
	if !l.ok || l.Len != l.prepLen || l.P != l.prepP {
		return ErrNotPrepared
	}
	if len(dst) != l.prepLen || len(a) != l.prepLen || len(b) != l.prepLen {
		return ErrLength
	}
	addRun(dst, a, b, &l.prepP)
	return nil
}
