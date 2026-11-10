// Package kpu runs Kendryte kmodel v3 files (nncase v0.1.0-rc5) on the
// K210's KPU, in pure Go.
//
// It ports the kmodel v3 runtime of kendryte-standalone-sdk V0.5.6
// (lib/drivers/kpu.c) with two changes, so that it needs neither the DMA
// controller nor the PLIC:
//   - every conv runs with int_en set and the runner polls calc_done,
//     instead of stepping layers from the KPU and DMA interrupts;
//   - a conv whose output goes to main memory (kmodel.FlagMainMemOut) is
//     read back from KPU memory at its destination address, in its
//     write-back layout, as dense CHW: the bytes the C runtime receives by
//     DMA from the KPU output FIFO.
//
// The model's input (CHW uint8, the first conv's input shape) is always
// uploaded by the CPU.
//
// The parser, memory layout and CPU layers are in package kmodel and run
// on any host; only the KPU itself (kpu_k210.go) is K210-specific.
package kpu

import (
	"errors"
	"fmt"

	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
)

var (
	// ErrNoKPU is returned off the K210.
	ErrNoKPU = errors.New("kpu: no KPU on this target")
	// ErrTimeout is returned when a conv does not finish within the timeout.
	ErrTimeout = errors.New("kpu: layer timed out")
	// ErrAlign is returned when the model data is not 128-byte aligned
	// below 4 GiB (the KPU fetches weights by 32-bit address).
	ErrAlign = errors.New("kpu: model data must be 128-byte aligned below 4 GiB")
	// ErrInput is returned for an input of the wrong size.
	ErrInput = errors.New("kpu: input size does not match the model")
)

// engine is the KPU: its 2 MiB memory and a conv layer run to completion.
type engine interface {
	ram() []byte
	conv(a *kmodel.Args) error
}

// Runner runs one model. It is not safe for concurrent use, and only one
// Runner may use the KPU at a time.
type Runner struct {
	m    *kmodel.Model
	main []byte
	base uint32 // address of m.Data[0] as the KPU sees it
	e    engine
}

func newRunner(m *kmodel.Model, main []byte, base uintptr, e engine) (*Runner, error) {
	if err := m.Check(); err != nil {
		return nil, err
	}
	if base%128 != 0 || uint64(base)+uint64(len(m.Data)) > 1<<32 {
		return nil, ErrAlign
	}
	if len(main) < int(m.MainMemUsage) {
		return nil, fmt.Errorf("kpu: main memory %d B, model needs %d B", len(main), m.MainMemUsage)
	}
	return &Runner{m: m, main: main, base: uint32(base), e: e}, nil
}

// InputSize is the input's size in bytes (first conv's w·h·c).
func (r *Runner) InputSize() int {
	c, _ := r.m.Layers[0].Conv()
	a := r.m.Args(c)
	w, h, ch := a.InShape()
	return w * h * ch
}

// Run runs the model on input (CHW uint8). Outputs are then in Output.
func (r *Runner) Run(input []byte) error {
	ram := r.e.ram()
	for i, l := range r.m.Layers {
		var err error
		switch l.Type {
		case kmodel.K210Conv:
			err = r.conv(l, i == 0, input, ram)
		case kmodel.Dequantize:
			q, _ := l.Quant()
			q.Dequantize(r.main)
		case kmodel.Quantize:
			q, _ := l.Quant()
			q.Quantize(r.main)
		case kmodel.GlobalAveragePool2D:
			g, _ := l.GAP()
			g.Run(r.main)
		case kmodel.K210AddPadding:
			p, _ := l.Padding()
			p.AddPadding(r.main, ram)
		case kmodel.K210RemovePadding:
			p, _ := l.Padding()
			p.RemovePadding(r.main)
		case kmodel.K210Upload:
			u, _ := l.Upload()
			err = u.Run(r.main, ram)
		default:
			err = kmodel.ErrUnsupported // unreachable after Check
		}
		if err != nil {
			return fmt.Errorf("kpu: layer %d (%v): %w", i, l.Type, err)
		}
	}
	return nil
}

func (r *Runner) conv(l kmodel.Layer, first bool, input, ram []byte) error {
	c, _ := l.Conv()
	a := r.m.Args(c)
	if first {
		w, h, ch := a.InShape()
		if len(input) != w*h*ch {
			return ErrInput
		}
		if err := a.InLayout().UploadTo(ram, a.SrcAddr(), w, h, ch, input); err != nil {
			return err
		}
	}
	a.SetWeights(r.base + c.WeightsOffset)
	a.SetBN(r.base + c.BNOffset)
	a.SetAct(r.base + c.ActOffset)
	a.SetIntEnable()
	a.SetSendDataOut(false)
	if err := r.e.conv(&a); err != nil {
		return err
	}
	if c.Flags&kmodel.FlagMainMemOut == 0 {
		return nil
	}
	w, h, ch := a.OutShape()
	return a.OutLayout().DownloadFrom(ram, a.DstAddr(), w, h, ch, r.main[c.MainMemOut:])
}

// Output returns output i (a slice of main memory, valid until the next Run).
func (r *Runner) Output(i int) []byte {
	o := r.m.Outputs[i]
	return r.main[o.Address : o.Address+o.Size]
}

// AlignedCopy returns a copy of b in a new 128-byte-aligned buffer, as New
// requires for the model data.
func AlignedCopy(b []byte) []byte {
	buf := make([]byte, len(b)+127)
	off := (128 - int(addrOf(buf)%128)) % 128
	out := buf[off : off+len(b) : off+len(b)]
	copy(out, b)
	return out
}
