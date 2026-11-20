package main

import (
	"io"
	"runtime"
	"strconv"
	"time"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/vision/detect/nms"
)

// reporter writes the result lines into one reused buffer, so a report
// allocates nothing and the heap figures show the detector alone. The host
// build prints the same lines as the board: its hashes and detections are
// the reference.
type reporter struct {
	w   io.Writer
	buf []byte
	ms  runtime.MemStats
}

func (r *reporter) str(s string) *reporter { r.buf = append(r.buf, s...); return r }
func (r *reporter) i64(v int64) *reporter  { r.buf = strconv.AppendInt(r.buf, v, 10); return r }
func (r *reporter) u64(v uint64) *reporter { r.buf = strconv.AppendUint(r.buf, v, 10); return r }
func (r *reporter) hex(v uint64) *reporter { r.buf = strconv.AppendUint(r.buf, v, 16); return r }
func (r *reporter) us(d time.Duration) *reporter {
	return r.i64(d.Microseconds())
}

func (r *reporter) flush() {
	r.buf = append(r.buf, '\r', '\n')
	r.w.Write(r.buf)
	r.buf = r.buf[:0]
}

// heap appends the heap figures.
func (r *reporter) heap() *reporter {
	runtime.ReadMemStats(&r.ms)
	return r.str(" heap_inuse ").u64(r.ms.HeapInuse).str(" heap_sys ").u64(r.ms.HeapSys).
		str(" total_alloc ").u64(r.ms.TotalAlloc).str(" mallocs ").u64(r.ms.Mallocs)
}

// setup reports what newDetector built.
func (r *reporter) setup(d *detector, in []int8, took time.Duration) {
	g := &d.file.Graph
	up, out := d.be.BufferBytes()
	r.str("ssd: setup layers ").i64(int64(len(g.Layers))).str(" segments ").i64(int64(len(d.kpuSeg))).
		str(" kpu ").i64(int64(len(d.runners))).str(" anchors ").i64(int64(d.dec.Anchors())).
		str(" arena ").i64(int64(d.arena)).str(" upload ").i64(int64(up)).str(" output ").i64(int64(out)).
		str(" input_fnv ").hex(uint64(fnv1a(in))).str(" setup_ms ").i64(took.Milliseconds())
	r.heap().flush()
}

// segments reports the last frame segment by segment: its layers, whether
// the KPU ran it, its time and (when hashed) its output's hash.
func (r *reporter) segments(d *detector) {
	g := &d.file.Graph
	for i, s := range d.run.Plan().Segments {
		on := "cpu"
		if d.kpuSeg[i] >= 0 {
			on = "kpu"
		}
		last := &g.Layers[s.First+s.Count-1]
		r.str("ssd: seg ").i64(int64(i)).str(" layers ").i64(int64(s.First)).str("+").i64(int64(s.Count)).
			str(" ").str(on).str(" ").str(g.Layers[s.First].Kind.String()).
			str(" bytes ").i64(int64(last.Out.Len())).str(" us ").us(d.segT[i])
		if d.hash {
			r.str(" fnv ").hex(uint64(d.fnv[i]))
		}
		r.flush()
	}
}

// frame reports frame n: total and per-stage times, the detections and the
// heap figures.
func (r *reporter) frame(n int, d *detector, dets []nms.Box) {
	r.str("ssd: frame ").i64(int64(n)).str(" total_us ").us(d.total)
	for st, name := range stageNames {
		r.str(" ").str(name).str("_us ").us(d.stageT[st])
	}
	r.str(" dets ").i64(int64(len(dets)))
	r.heap().flush()
	for _, b := range dets {
		r.str("ssd: det anchor ").i64(int64(b.Index)).str(" class ").i64(int64(b.Class)).str(" score ")
		r.buf = strconv.AppendFloat(r.buf, float64(b.Score), 'f', 4, 32)
		for _, v := range [4]float32{b.X0, b.Y0, b.X1, b.Y1} {
			r.buf = append(r.buf, ' ')
			r.buf = strconv.AppendFloat(r.buf, float64(v), 'f', 4, 32)
		}
		r.flush()
	}
}

// tensor writes one segment's output as hex lines (64 bytes a line)
// between a header with its length and a footer with its hash, for
// comparing the board's values with a host's. Reshapes are skipped: the
// same bytes as their input.
func (r *reporter) tensor(d *detector, seg int, t []int8) {
	const hexDigits = "0123456789abcdef"
	if d.file.Graph.Layers[d.run.Plan().Segments[seg].First].Kind == qnet.Reshape {
		return
	}
	r.str("ssd: tensor ").i64(int64(seg)).str(" bytes ").i64(int64(len(t))).flush()
	for p := t; len(p) > 0; {
		n := min(len(p), 64)
		for _, v := range p[:n] {
			r.buf = append(r.buf, hexDigits[uint8(v)>>4], hexDigits[uint8(v)&15])
		}
		r.flush()
		p = p[n:]
	}
	r.str("ssd: tensor ").i64(int64(seg)).str(" fnv ").hex(uint64(fnv1a(t))).flush()
}

// scoreThreshold is the score a detection must reach to be reported.
const scoreThreshold = 0.5

// loop runs the detector on in: the first frame hashes every segment (and,
// with dump, writes every segment's tensor as it is produced), and the
// segments are reported; then frames are timed without hashing. runs ≤ 0
// repeats forever (on the board, so a monitor can attach any time).
func loop(w io.Writer, d *detector, in []int8, took time.Duration, ticksPerSecond uint64, runs int, dump bool) error {
	r := &reporter{w: w, buf: make([]byte, 0, 256)}
	tensor := func(seg int, out []int8) { r.tensor(d, seg, out) }
	for n := 1; runs <= 0 || n <= runs; n++ {
		if n == 1 || runs <= 0 && n%8 == 1 {
			r.setup(d, in, took)
		}
		d.hash, d.each = n == 1, nil
		if n == 1 && dump {
			d.each = tensor
		}
		dets, err := d.frame(in, scoreThreshold)
		if err != nil {
			return err
		}
		d.split(ticksPerSecond)
		if n <= 2 {
			r.segments(d) // frame 1 with hashes, frame 2 with clean times
		}
		r.frame(n, d, dets)
	}
	return nil
}
