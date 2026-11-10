package kmodel

import (
	"encoding/binary"
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
// output through the KPU output FIFO as well).
func (a *Args) SetSendDataOut(on bool) {
	a[11] &^= 1
	if on {
		a[11] |= 1
	}
}

// SetWeights sets kernel_load_cfg.para_start_addr.
func (a *Args) SetWeights(addr uint32) { a[5] = a[5]&0xffff_ffff | uint64(addr)<<32 }

// SetBN sets kernel_pool_type_cfg.bwsx_base_addr (batch-norm table).
func (a *Args) SetBN(addr uint32) { a[4] = a[4]&0xffff_ffff | uint64(addr)<<32 }

// SetAct sets kernel_calc_type_cfg.active_addr (activation table).
func (a *Args) SetAct(addr uint32) { a[7] = a[7]&0xffff_ffff | uint64(addr)<<32 }

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
// wrap on any host (review R-K210-K8b K1).
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
// (ER-K210-K8 evidence, K8c-1).
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
