// Command nnbench benchmarks small int8 CNNs built as qnet graphs and run by
// the qnet Runner (qi8 prepared layers).
//
// The same source builds for the ESP32-S3 (TinyGo; default backend
// "esp32s3-pie", or the generic one with -tags qi8generic) and for the host
// (go run). Weights and input are synthetic and deterministic, so the output
// checksums must be identical across builds: that is the whole-network
// bit-exactness check between backends. See README.md.
package main

import (
	"errors"
	"fmt"
	"runtime"
	"time"
	"unsafe"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

const (
	minRuns     = 10
	minDuration = 2 * time.Second
	reportEvery = 5 * time.Second

	heapReserve = 32 << 10 // chip heap left outside the arena
	hostArena   = 4 << 20
)

// Layer kinds of the per-kind breakdown.
const (
	kConv = iota // Conv2D with a kernel larger than 1×1
	kDepthwise
	kPointwise // 1×1 Conv2D
	kFC
	kPool
	kAdd
	kOther
	nKinds
)

var kindNames = [nKinds]string{"conv", "depthwise", "pointwise", "fc", "pool", "add", "other"}

// model is a network ready to run: everything is allocated in newModel.
type model struct {
	g      *qnet.Graph
	r      *qnet.Runner
	kinds  []int // per plan segment
	input  []int8
	out    []int8
	adds   int
	addLen int // elements summed by the Adds per inference

	weightBytes  int // weights stored outside Scratch
	scratchBytes int // backend Scratch (holds the packed weights in place)
	paramBytes   int // bias + requant
	actBytes     int // input + planned activations
}

// arena is a bump allocator over one buffer allocated once at start-up and
// reset for every model, so that a model never depends on the (conservative)
// GC having freed the previous one. Allocations are zeroed and 16-byte
// aligned.
type arena struct {
	buf []int8
	off int
}

func newArena(n int) *arena {
	raw := make([]int8, n+15)
	off := int((16 - uintptr(unsafe.Pointer(&raw[0]))&15) & 15)
	return &arena{buf: raw[off : off+n : off+n]}
}

func (a *arena) reset() {
	clear(a.buf[:a.off])
	a.off = 0
}

// int8s returns n bytes and adds the bytes consumed (with alignment) to *acct.
func (a *arena) int8s(n int, acct *int) ([]int8, error) {
	start := (a.off + 15) &^ 15
	if n < 0 || start+n > len(a.buf) {
		return nil, errArena
	}
	*acct += start + n - a.off
	a.off = start + n
	return a.buf[start : start+n : start+n], nil
}

func (a *arena) int32s(n int, acct *int) ([]int32, error) {
	b, err := a.int8s(n*int(unsafe.Sizeof(int32(0))), acct)
	if err != nil || n == 0 {
		return nil, err
	}
	return unsafe.Slice((*int32)(unsafe.Pointer(&b[0])), n), nil
}

func (a *arena) requants(n int, acct *int) ([]qi8.Requant, error) {
	b, err := a.int8s(n*int(unsafe.Sizeof(qi8.Requant{})), acct)
	if err != nil || n == 0 {
		return nil, err
	}
	return unsafe.Slice((*qi8.Requant)(unsafe.Pointer(&b[0])), n), nil
}

var errArena = errors.New("nnbench: model does not fit the arena")

// layerKind maps a layer to its breakdown bucket.
func layerKind(l *qnet.Layer) int {
	switch l.Kind {
	case qnet.Conv2D:
		if l.KH == 1 && l.KW == 1 {
			return kPointwise
		}
		return kConv
	case qnet.DepthwiseConv2D:
		return kDepthwise
	case qnet.FullyConnected:
		return kFC
	case qnet.MaxPool2D, qnet.AvgPool2D:
		return kPool
	case qnet.Add:
		return kAdd
	}
	return kOther
}

// newModel plans g with qnet.PlanMemory and allocates from a (reset first)
// the runner arena (backend scratch + activations), the input and the
// params. Weights go to the start of their layer's scratch when the backend
// repacks them (PIE), so Prepare packs them in place and only one copy
// exists. Params are filled with qnet.SyntheticParams(wSeed), the input from
// inSeed; NewRunner prepares every layer.
func newModel(g *qnet.Graph, a *arena, wSeed, inSeed uint64) (*model, error) {
	a.reset()
	m := &model{g: g}
	plan, err := qnet.PlanMemory(g)
	if err != nil {
		return nil, err
	}
	var slack int
	ra, err := a.int8s(plan.Size(), &slack)
	if err != nil {
		return nil, err
	}
	m.scratchBytes, m.actBytes = plan.Persistent, plan.Activations
	for i := range g.Layers {
		l := &g.Layers[i]
		nw, nb, nr := l.ParamCounts()
		if nw == 0 {
			continue
		}
		if s := plan.ScratchSlice(ra, i); s != nil {
			l.Weights = s[:nw:nw]
		} else if l.Weights, err = a.int8s(nw, &m.weightBytes); err != nil {
			return nil, err
		}
		if l.Bias, err = a.int32s(nb, &m.paramBytes); err != nil {
			return nil, err
		}
		if l.Requant, err = a.requants(nr, &m.paramBytes); err != nil {
			return nil, err
		}
	}
	if err := qnet.SyntheticParams(g, wSeed); err != nil {
		return nil, err
	}
	if m.input, err = a.int8s(g.Layers[0].In.Len(), &m.actBytes); err != nil {
		return nil, err
	}
	fillInput(m.input, inSeed)
	if m.r, err = qnet.NewRunner(g, ra); err != nil {
		return nil, err
	}
	for _, s := range m.r.Plan().Segments {
		l := &g.Layers[s.First]
		m.kinds = append(m.kinds, layerKind(l))
		if l.Kind == qnet.Add {
			m.adds++
			m.addLen += l.Out.Len()
		}
	}
	return m, nil
}

// fillInput fills x with uniform int8 values from SplitMix64(seed).
func fillInput(x []int8, seed uint64) {
	s := seed
	for i := range x {
		s += 0x9e3779b97f4a7c15
		z := s
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		x[i] = int8(z ^ (z >> 31))
	}
}

// run executes the network once, adding each segment's ticks to kt.
func (m *model) run(kt *[nKinds]uint64) error {
	for k, kind := range m.kinds {
		t0 := stamp()
		if err := m.r.RunSegment(m.input, k); err != nil {
			return err
		}
		kt[kind] += ticksSince(t0)
	}
	return nil
}

// fnv1a returns the 32-bit FNV-1a hash of b, continuing from h.
func fnv1a(h uint32, b []int8) uint32 {
	for _, v := range b {
		h ^= uint32(uint8(v))
		h *= 16777619
	}
	return h
}

const fnvOffset = 2166136261

// result is everything reported for one model.
type result struct {
	spec    *spec
	err     error
	inShape qi8.Shape
	layers  int
	adds    int
	addLen  int // elements summed by the Adds per inference
	macs    int64
	weights int // model weight bytes (qnet budget)

	runs         int
	meanMs       float64
	minMs        float64
	cycles       uint64 // mean CCOUNT cycles per inference (0 off-chip)
	outFNV       uint32 // final output
	allFNV       uint32 // every layer output of the warm-up run
	kindTicks    [nKinds]uint64
	kindMACs     [nKinds]int64
	weightBytes  int
	scratchBytes int
	paramBytes   int
	actBytes     int
	heapSetup    uint64 // bytes allocated during setup (TotalAlloc delta)
	heapAlloc    uint64 // HeapAlloc after setup
	heapSys      uint64
	loopMallocs  uint64 // heap allocations during the timed loop
}

func measure(sp *spec, a *arena) (r result) {
	r.spec = sp
	var ms0, ms1, ms2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&ms0)
	g, err := sp.build()
	if err != nil {
		r.err = err
		return r
	}
	m, err := newModel(&g, a, sp.wSeed, sp.inSeed)
	if err != nil {
		r.err = err
		return r
	}
	r.inShape, r.layers, r.adds, r.addLen, r.macs, r.weights = g.Layers[0].In, len(g.Layers), m.adds, m.addLen, g.MACs(), g.WeightBytes()
	for i := range g.Layers {
		r.kindMACs[layerKind(&g.Layers[i])] += g.Budget(i).MACs
	}
	r.weightBytes, r.scratchBytes, r.paramBytes, r.actBytes = m.weightBytes, m.scratchBytes, m.paramBytes, m.actBytes
	runtime.ReadMemStats(&ms1)
	r.heapSetup = ms1.TotalAlloc - ms0.TotalAlloc
	r.heapAlloc, r.heapSys = ms1.HeapAlloc, ms1.HeapSys

	// Warm-up, hashing every segment (= layer) output in order.
	var kt [nKinds]uint64
	h := uint32(fnvOffset)
	for k := range m.kinds {
		if err := m.r.RunSegment(m.input, k); err != nil {
			r.err = fmt.Errorf("segment %d: %w", k, err)
			return r
		}
		h = fnv1a(h, m.r.SegmentOutput(m.input, k))
	}
	m.out = m.r.SegmentOutput(m.input, len(m.kinds)-1)
	r.allFNV = h
	r.outFNV = fnv1a(fnvOffset, m.out)

	runtime.ReadMemStats(&ms1)
	var total, minD time.Duration
	var cyc uint64
	n := 0
	for n < minRuns || total < minDuration {
		t0 := time.Now()
		c0 := ccount()
		if err := m.run(&kt); err != nil {
			r.err = err
			return r
		}
		cyc += uint64(ccount() - c0)
		d := time.Since(t0)
		total += d
		if n == 0 || d < minD {
			minD = d
		}
		n++
	}
	runtime.ReadMemStats(&ms2)
	r.loopMallocs = ms2.Mallocs - ms1.Mallocs
	if h := fnv1a(fnvOffset, m.out); h != r.outFNV {
		r.err = fmt.Errorf("output changed between runs: %08x != %08x", h, r.outFNV)
	}
	r.runs = n
	r.meanMs = float64(total) / float64(n) / float64(time.Millisecond)
	r.minMs = float64(minD) / float64(time.Millisecond)
	r.cycles = cyc / uint64(n)
	for k := range kt {
		r.kindTicks[k] = kt[k] / uint64(n)
	}
	return r
}

// cpuMHz derives the CPU clock from CCOUNT against time over about 1 s
// (busy wait). It returns 0 off-chip.
func cpuMHz() (mhz float64, cycles uint32, us int64) {
	if !device {
		return 0, 0, 0
	}
	t0 := time.Now()
	c0 := ccount()
	for time.Since(t0) < time.Second {
	}
	cycles = ccount() - c0
	us = time.Since(t0).Microseconds()
	return float64(cycles) / float64(us), cycles, us
}

func report(rs []result, mhz float64, cyc uint32, us int64, boot *runtime.MemStats, arenaLen int) {
	fmt.Printf("\n=== nnbench backend=%s GOARCH=%s compiler=%s ===\n", qi8.Backend(), runtime.GOARCH, runtime.Compiler)
	if device {
		fmt.Printf("cpu %.2f MHz (CCOUNT %d cycles in %d us)\n", mhz, cyc, us)
	} else {
		fmt.Printf("cpu n/a (no CCOUNT off the ESP32-S3: cycles report 0; layer breakdown in ns)\n")
	}
	fmt.Printf("heap at boot: HeapSys %d B, HeapAlloc %d B; arena %d B (allocated once, reused by every model)\n",
		boot.HeapSys, boot.HeapAlloc, arenaLen)
	fmt.Printf("%-17s %-8s %7s %9s %5s %9s %9s %11s %8s %8s %8s\n",
		"model", "input", "MMAC", "weightsB", "runs", "ms_mean", "ms_min", "cyc/inf", "cyc/MAC", "out_fnv", "all_fnv")
	for i := range rs {
		r := &rs[i]
		if r.err != nil {
			fmt.Printf("%-17s ERROR %v\n", r.spec.name, r.err)
			continue
		}
		var cpm float64
		if r.macs > 0 {
			cpm = float64(r.cycles) / float64(r.macs)
		}
		fmt.Printf("%-17s %-8s %7.2f %9d %5d %9.3f %9.3f %11d %8.3f %08x %08x\n",
			r.spec.name, fmt.Sprintf("%dx%dx%d", r.inShape.H, r.inShape.W, r.inShape.C),
			float64(r.macs)/1e6, r.weights, r.runs, r.meanMs, r.minMs, r.cycles, cpm, r.outFNV, r.allFNV)
	}
	for i := range rs {
		r := &rs[i]
		if r.err != nil {
			continue
		}
		fmt.Printf("\n%s: %s\n", r.spec.name, r.spec.note)
		fmt.Printf("  seeds: weights qnet.SyntheticParams(%d), input SplitMix64(%d); %d layers, %d adds\n", r.spec.wSeed, r.spec.inSeed, r.layers, r.adds)
		fmt.Printf("  arena use: weights %d B + scratch %d B (packed weights in place) + bias/requant %d B + activations %d B = %d B\n",
			r.weightBytes, r.scratchBytes, r.paramBytes, r.actBytes, r.weightBytes+r.scratchBytes+r.paramBytes+r.actBytes)
		fmt.Printf("  heap: setup allocated %d B outside the arena, HeapAlloc %d B after setup, mallocs in timed loop %d\n",
			r.heapSetup, r.heapAlloc, r.loopMallocs)
		var sum uint64
		for _, t := range r.kindTicks {
			sum += t
		}
		fmt.Printf("  %-10s %12s %6s %10s %9s\n", "kind", tickUnit+"/inf", "share", "MACs", tickUnit+"/MAC")
		for k, t := range r.kindTicks {
			if t == 0 && r.kindMACs[k] == 0 {
				continue
			}
			var share, per float64
			if sum > 0 {
				share = 100 * float64(t) / float64(sum)
			}
			if r.kindMACs[k] > 0 {
				per = float64(t) / float64(r.kindMACs[k])
			}
			fmt.Printf("  %-10s %12d %5.1f%% %10d %9.3f\n", kindNames[k], t, share, r.kindMACs[k], per)
		}
		if r.adds > 0 {
			fmt.Printf("  add: %d elements/inf, %.3f %s/element\n", r.addLen, float64(r.kindTicks[kAdd])/float64(r.addLen), tickUnit)
		}
	}
}

func main() {
	if device {
		time.Sleep(2 * time.Second) // let the USB console attach
	}
	mhz, cyc, us := cpuMHz()
	var boot runtime.MemStats
	runtime.ReadMemStats(&boot)
	// One arena for all models: on the chip, the free heap minus a reserve
	// for the small per-model structs and fmt, capped per platform
	// (arenaCap); on the host, plenty.
	n := hostArena
	if device {
		n = arenaCap(int(boot.HeapSys-boot.HeapAlloc) - heapReserve)
	}
	a := newArena(n)
	rs := make([]result, len(specs))
	for i := range specs {
		fmt.Printf("measuring %s ...\n", specs[i].name)
		rs[i] = measure(&specs[i], a)
	}
	for {
		report(rs, mhz, cyc, us, &boot, n)
		if !device {
			return
		}
		time.Sleep(reportEvery)
	}
}
