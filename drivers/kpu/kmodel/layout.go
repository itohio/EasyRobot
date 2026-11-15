package kmodel

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

// Args is a KPU layer argument (kpu_layer_argument_t): the 12 u64 words
// written, in order, to the KPU's layer-argument FIFO.
type Args [12]uint64

// Args copies conv c's layer argument out of the file. The caller patches
// the weight, batch-norm and activation addresses before sending it.
func (m *Model) Args(c Conv) Args {
	var a Args
	b := m.Data[c.LayerOffset:]
	for i := range a {
		a[i] = binary.LittleEndian.Uint64(b[8*i:])
	}
	return a
}

// SrcAddr is the input's KPU memory line address (image_addr.image_src_addr).
func (a *Args) SrcAddr() uint32 { return uint32(a[1] & 0x7fff) }

// DstAddr is the output's KPU memory line address (image_addr.image_dst_addr).
func (a *Args) DstAddr() uint32 { return uint32(a[1] >> 32 & 0x7fff) }

// InShape returns the input width, height and channels (image_size,
// image_channel_num; the registers hold value − 1).
func (a *Args) InShape() (w, h, c int) {
	return int(a[3]&0x3ff) + 1, int(a[3]>>10&0x1ff) + 1, int(a[2]&0x3ff) + 1
}

// OutShape returns the output width, height and channels.
func (a *Args) OutShape() (w, h, c int) {
	return int(a[3]>>32&0x3ff) + 1, int(a[3]>>42&0x1ff) + 1, int(a[2]>>32&0x3ff) + 1
}

// SetIntEnable sets interrupt_enabe.int_en: the KPU raises calc_done when
// the layer finishes.
func (a *Args) SetIntEnable() { a[0] |= 1 }

// SetSendDataOut sets or clears dma_parameter.send_data_out (stream the
// output through the KPU output FIFO as well). Streaming needs the
// per-channel byte count W·H − 1 in 16 bits: on an output of more than
// 65,536 pixels per channel it returns ErrFormat.
func (a *Args) SetSendDataOut(on bool) error {
	if on {
		if w, h, _ := a.OutShape(); w*h > maxChannelBytes {
			return rangeErr("send_data_out on a per-channel output of", w*h)
		}
	}
	a[11] &^= 1
	if on {
		a[11] |= 1
	}
	return nil
}

// maxChannelBytes is the largest per-channel output dma_parameter's 16-bit
// channel_byte_num (value − 1) holds.
const maxChannelBytes = 1 << 16

// SetWeights sets kernel_load_cfg.para_start_addr.
func (a *Args) SetWeights(addr uint32) { a[5] = a[5]&0xffff_ffff | uint64(addr)<<32 }

// SetBN sets kernel_pool_type_cfg.bwsx_base_addr (batch-norm table).
func (a *Args) SetBN(addr uint32) { a[4] = a[4]&0xffff_ffff | uint64(addr)<<32 }

// SetAct sets kernel_calc_type_cfg.active_addr (activation table).
func (a *Args) SetAct(addr uint32) { a[7] = a[7]&0xffff_ffff | uint64(addr)<<32 }

// NewArgs returns a layer argument with the constant fields nncase rc5
// writes for every conv: load_para 1, dma_burst_size 15 (kernel_pool_type_cfg),
// load_coor 1 (kernel_load_cfg), load_act 1 (kernel_calc_type_cfg). The
// setters below fill the rest. Each checks its values against the field
// widths and the KPU's limits and returns an error, leaving the argument
// unchanged, for any value that does not fit: a
// masked value can read back as a different valid one (width 1025 as 1).
func NewArgs() Args {
	var a Args
	a[4] = 1<<10 | 15<<16
	a[5] = 1
	a[7] = 1 << 31
	return a
}

func setBits(w *uint64, lo, n uint, v uint64) {
	m := (uint64(1)<<n - 1) << lo
	*w = *w&^m | v<<lo&m
}

func rangeErr(field string, v any) error {
	return fmt.Errorf("%w: %s %v out of range", ErrFormat, field, v)
}

func signedFits(v int64, bits uint) bool { return v >= -1<<(bits-1) && v < 1<<(bits-1) }

func checkShape(w, h, c int) error {
	if w < 1 || w > maxWidth || h < 1 || h > maxHeight || c < 1 || c > maxChannels {
		return rangeErr("shape (w, h, c)", [3]int{w, h, c})
	}
	return nil
}

// SetSrcAddr sets the input's KPU memory line address (15 bits).
func (a *Args) SetSrcAddr(addr uint32) error {
	if addr > 0x7fff {
		return rangeErr("source line address", addr)
	}
	setBits(&a[1], 0, 15, uint64(addr))
	return nil
}

// SetDstAddr sets the output's KPU memory line address (15 bits).
func (a *Args) SetDstAddr(addr uint32) error {
	if addr > 0x7fff {
		return rangeErr("destination line address", addr)
	}
	setBits(&a[1], 32, 15, uint64(addr))
	return nil
}

// SetInShape sets the input width (1–1024), height (1–512) and channels
// (1–1024); the registers hold value − 1.
func (a *Args) SetInShape(w, h, c int) error {
	if err := checkShape(w, h, c); err != nil {
		return err
	}
	setBits(&a[3], 0, 10, uint64(w-1))
	setBits(&a[3], 10, 9, uint64(h-1))
	setBits(&a[2], 0, 10, uint64(c-1))
	return nil
}

// SetOutShape sets the output width, height and channels (limits as
// SetInShape) and the matching dma_parameter byte counts (W·H per channel,
// W·H·C in all). The 16-bit per-channel count only matters for
// send_data_out: above 65,536 pixels per channel it is refused with
// send_data_out on, and saturated at 65,535 with it off, where the KPU
// ignores it (a 300×220 output, 66,000 pixels, runs and matches the Go
// model on the UnitV).
func (a *Args) SetOutShape(w, h, c int) error {
	if err := checkShape(w, h, c); err != nil {
		return err
	}
	if w*h > maxChannelBytes && a[11]&1 != 0 {
		return rangeErr("per-channel output with send_data_out on", w*h)
	}
	setBits(&a[3], 32, 10, uint64(w-1))
	setBits(&a[3], 42, 9, uint64(h-1))
	setBits(&a[2], 32, 10, uint64(c-1))
	setBits(&a[11], 16, 16, uint64(min(w*h, maxChannelBytes)-1))
	setBits(&a[11], 32, 32, uint64(w*h*c-1))
	return nil
}

func checkLayout(l Layout) error {
	if (l.Group != 1 && l.Group != 2 && l.Group != 4) || l.RowLines < 1 || l.RowLines > 15 || l.ChanLines < 1 || l.ChanLines > 0x7fff {
		return rangeErr("layout", l)
	}
	return nil
}

// SetInLayout sets kernel_calc_type_cfg's coef_group (1, 2 or 4),
// row_switch_addr (1–15) and channel_switch_addr (1–32767).
func (a *Args) SetInLayout(l Layout) error {
	if err := checkLayout(l); err != nil {
		return err
	}
	setBits(&a[7], 28, 3, uint64(l.Group))
	setBits(&a[7], 16, 4, uint64(l.RowLines))
	setBits(&a[7], 0, 15, uint64(l.ChanLines))
	return nil
}

// SetOutLayout sets write_back_cfg's wb_group, wb_row_switch_addr and
// wb_channel_switch_addr (ranges as SetInLayout).
func (a *Args) SetOutLayout(l Layout) error {
	if err := checkLayout(l); err != nil {
		return err
	}
	setBits(&a[8], 20, 3, uint64(l.Group))
	setBits(&a[8], 16, 4, uint64(l.RowLines))
	setBits(&a[8], 0, 15, uint64(l.ChanLines))
	return nil
}

// SetKernel sets kernel_type (k = 3 → 1 for 3×3, k = 1 → 0 for 1×1) and
// interrupt_enabe.depth_wise_layer.
func (a *Args) SetKernel(k int, depthwise bool) error {
	if k != 1 && k != 3 {
		return rangeErr("kernel size", k)
	}
	var t, d uint64
	if k == 3 {
		t = 1
	}
	if depthwise {
		d = 1
	}
	setBits(&a[4], 0, 3, t)
	setBits(&a[0], 3, 1, d)
	return nil
}

// SetPool sets pool_type: 0 none, 1/2 2×2 max/average, 3/4 4×4 max/average,
// 5 left-top and 6 right-top 2×2 samplers (the types modelled by Emulate;
// nncase v0.2's K210 runtime also lists 7 left-top 4×4 and 8/9 2×2
// stride-1 mean/max). Pools work on regular convs only: the KPU does not
// pool in depthwise mode (observed on the K210).
func (a *Args) SetPool(t int) error {
	if t < 0 || t > 6 {
		return rangeErr("pool type", t)
	}
	setBits(&a[4], 4, 4, uint64(t))
	return nil
}

// SetPadValue sets pad_value, the uint8 read for out-of-image taps.
func (a *Args) SetPadValue(v uint8) { setBits(&a[4], 24, 8, uint64(v)) }

// SetConvValue sets conv_value: arg_x and arg_w (signed 24-bit) with their
// shifts shr_x and shr_w (0–15).
func (a *Args) SetConvValue(argX int64, shrX uint, argW int64, shrW uint) error {
	if !signedFits(argX, 24) || !signedFits(argW, 24) || shrX > 15 || shrW > 15 {
		return rangeErr("conv_value (arg_x, shr_x, arg_w, shr_w)", [4]int64{argX, int64(shrX), argW, int64(shrW)})
	}
	setBits(&a[9], 0, 4, uint64(shrW))
	setBits(&a[9], 4, 4, uint64(shrX))
	setBits(&a[9], 8, 24, uint64(argW))
	setBits(&a[9], 32, 24, uint64(argX))
	return nil
}

// SetArgAdd sets conv_value2.arg_add (signed 40-bit).
func (a *Args) SetArgAdd(v int64) error {
	if !signedFits(v, 40) {
		return rangeErr("arg_add", v)
	}
	setBits(&a[10], 0, 40, uint64(v))
	return nil
}

// SetLoad sets how the KPU loads the weights: paraSize bytes per load
// (kernel_load_cfg.para_size, < 2^17), loads loads (1–64; load_time =
// loads − 1) and chansPerLoad output channels per load (1–1024;
// image_channel_num.o_ch_num_coef).
func (a *Args) SetLoad(paraSize uint32, loads, chansPerLoad int) error {
	if paraSize >= 1<<17 || loads < 1 || loads > 64 || chansPerLoad < 1 || chansPerLoad > maxChannels {
		return rangeErr("load (para_size, loads, channels per load)", [3]int64{int64(paraSize), int64(loads), int64(chansPerLoad)})
	}
	setBits(&a[5], 15, 17, uint64(paraSize))
	setBits(&a[5], 1, 6, uint64(loads-1))
	setBits(&a[2], 48, 10, uint64(chansPerLoad-1))
	return nil
}

// Layout is how a CHW uint8 tensor sits in KPU memory, in 64-byte lines:
// Group channels share a line (each gets 64/Group bytes of it), a row takes
// RowLines lines, and a group of channels takes ChanLines lines. Byte
// (c, y, x) of a tensor at line address addr is at
//
//	addr·64 + (c/Group)·ChanLines·64 + (c%Group)·64/Group + y·RowLines·64 + x
type Layout struct{ Group, RowLines, ChanLines int }

// InLayout is the layer input's layout (kernel_calc_type_cfg: coef_group,
// row_switch_addr, channel_switch_addr).
func (a *Args) InLayout() Layout {
	return Layout{int(a[7] >> 28 & 7), int(a[7] >> 16 & 0xf), int(a[7] & 0x7fff)}
}

// OutLayout is the layer output's layout (write_back_cfg: wb_group,
// wb_row_switch_addr, wb_channel_switch_addr).
func (a *Args) OutLayout() Layout {
	return Layout{int(a[8] >> 20 & 7), int(a[8] >> 16 & 0xf), int(a[8] & 0x7fff)}
}

// UploadLayout is the layout kpu_upload_core writes for a w×h input: up to
// 16 wide, 4 channels per line; up to 32, 2; wider, one channel per row of
// ⌈w/64⌉ lines.
func UploadLayout(w, h int) Layout {
	switch {
	case w <= 16:
		return Layout{4, 1, h}
	case w <= 32:
		return Layout{2, 1, h}
	}
	rl := (w + 63) / 64
	return Layout{1, rl, rl * h}
}

// padLayout is where K210AddPadding puts a 1×1 tensor: the 4×4, 4-channel
// line layout of a 16-wide input (kpu_add_padding).
var padLayout = Layout{4, 1, 4}

func (l Layout) offset(addr uint32, c, y, x int) int {
	return int(addr)*64 + c/l.Group*l.ChanLines*64 + c%l.Group*(64/l.Group) + y*l.RowLines*64 + x
}

// check reports whether a w×h×c tensor at addr fits the KPU's limits, the
// layout and KPU memory. The extent is computed in uint64, so it cannot
// wrap on any host.
func (l Layout) check(addr uint32, w, h, c int) error {
	if w <= 0 || w > maxWidth || h <= 0 || h > maxHeight || c <= 0 || c > maxChannels {
		return ErrFormat
	}
	switch l.Group {
	case 1:
		if w > l.RowLines*64 {
			return ErrFormat
		}
	case 2, 4:
		if w > 64/l.Group {
			return ErrFormat
		}
	default:
		return ErrFormat
	}
	g := uint64(l.Group)
	last := uint64(addr)*64 + uint64(c-1)/g*uint64(l.ChanLines)*64 + uint64(c-1)%g*(64/g) +
		uint64(h-1)*uint64(l.RowLines)*64 + uint64(w-1)
	if last >= RAMSize {
		return ErrFormat
	}
	return nil
}

// UploadTo writes the CHW tensor src (w·h·c bytes) into KPU memory ram at
// line address addr in layout l.
func (l Layout) UploadTo(ram []byte, addr uint32, w, h, c int, src []byte) error {
	if err := l.check(addr, w, h, c); err != nil || len(ram) < RAMSize || len(src) < w*h*c {
		return ErrFormat
	}
	for ch := range c {
		for y := range h {
			o := l.offset(addr, ch, y, 0)
			copyRow(ram[o:o+w], src[(ch*h+y)*w:])
		}
	}
	return nil
}

// DownloadFrom reads a w×h×c tensor at line address addr in layout l out of
// KPU memory ram into dst as dense CHW (w·h·c bytes).
func (l Layout) DownloadFrom(ram []byte, addr uint32, w, h, c int, dst []byte) error {
	if err := l.check(addr, w, h, c); err != nil || len(ram) < RAMSize || len(dst) < w*h*c {
		return ErrFormat
	}
	for ch := range c {
		for y := range h {
			o := l.offset(addr, ch, y, 0)
			copyRow(dst[(ch*h+y)*w:(ch*h+y+1)*w], ram[o:o+w])
		}
	}
	return nil
}

// copyRow copies min(len(dst), len(src)) bytes, 8 at a time while both
// sides are 8-byte aligned. The K210's KPU memory is reached through an
// uncached alias, where every access is a bus transaction: one 64-bit
// access instead of eight byte accesses is the C runtime's aligned upload
// path (kpu_upload_core) and makes upload and read-back about 8× cheaper
// (observed on the K210).
func copyRow(dst, src []byte) {
	n := min(len(dst), len(src))
	i := 0
	if n >= 8 && uintptr(unsafe.Pointer(unsafe.SliceData(dst)))%8 == 0 && uintptr(unsafe.Pointer(unsafe.SliceData(src)))%8 == 0 {
		d := unsafe.Slice((*uint64)(unsafe.Pointer(unsafe.SliceData(dst))), n/8)
		s := unsafe.Slice((*uint64)(unsafe.Pointer(unsafe.SliceData(src))), n/8)
		copy64(d, s)
		i = n &^ 7
	}
	for ; i < n; i++ {
		dst[i] = src[i]
	}
}

// copy64 is a plain loop, not the copy builtin, so that every access is
// one 64-bit load or store (a memmove may fall back to byte accesses).
func copy64(d, s []uint64) {
	for j := range d {
		d[j] = s[j]
	}
}

func f32(u uint32) float32 { return math.Float32frombits(u) }
