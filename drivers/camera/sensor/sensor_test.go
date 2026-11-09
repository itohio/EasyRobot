package sensor

import (
	"errors"
	"testing"
)

// fakeBus answers like an SCCB bus: registers of present devices, 0xFF from
// absent ones. The OV2640 device only exposes its id in bank 1.
type fakeBus struct {
	regs8  map[uint8]map[uint8]uint8
	regs16 map[uint8]map[uint16]uint8
	ov2640 bool // device 0x30 is an OV2640 with banked registers
	bank   uint8
	fail   map[uint8]bool // devices whose transfers return an error
}

var errBus = errors.New("bus error")

func (b *fakeBus) ReadReg8(dev, reg uint8) (uint8, error) {
	if b.fail[dev] {
		return 0, errBus
	}
	if dev == addrOV2640 && b.ov2640 {
		if b.bank != 1 {
			return 0x00, nil
		}
		return map[uint8]uint8{regPID: 0x26, regVER: 0x42}[reg], nil
	}
	if r, ok := b.regs8[dev]; ok {
		return r[reg], nil
	}
	return 0xFF, nil
}

func (b *fakeBus) WriteReg8(dev, reg, val uint8) error {
	if b.fail[dev] {
		return errBus
	}
	if dev == addrOV2640 && reg == regBankSel {
		b.bank = val
	}
	return nil
}

func (b *fakeBus) ReadReg16(dev uint8, reg uint16) (uint8, error) {
	if b.fail[dev] {
		return 0, errBus
	}
	if r, ok := b.regs16[dev]; ok {
		return r[reg], nil
	}
	return 0xFF, nil
}

func (b *fakeBus) WriteReg16(dev uint8, reg uint16, val uint8) error { return nil }

func TestIdentify(t *testing.T) {
	tests := []struct {
		name    string
		bus     *fakeBus
		want    ID
		wantErr error
	}{
		{
			name: "OV7740",
			bus:  &fakeBus{regs8: map[uint8]map[uint8]uint8{0x21: {regPID: 0x77, regVER: 0x42}}},
			want: ID{Model: OV7740, Addr: 0x21, PID: 0x77, VER: 0x42},
		},
		{
			name: "OV2640 needs the bank select",
			bus:  &fakeBus{ov2640: true},
			want: ID{Model: OV2640, Addr: 0x30, PID: 0x26, VER: 0x42},
		},
		{
			name: "OV5640 16-bit id",
			bus:  &fakeBus{regs16: map[uint8]map[uint16]uint8{0x3C: {regPID16: 0x56, regVER16: 0x40}}},
			want: ID{Model: OV5640, Addr: 0x3C, PID: 0x56, VER: 0x40},
		},
		{
			name: "OV3660 16-bit id",
			bus:  &fakeBus{regs16: map[uint8]map[uint16]uint8{0x3C: {regPID16: 0x36, regVER16: 0x60}}},
			want: ID{Model: OV3660, Addr: 0x3C, PID: 0x36, VER: 0x60},
		},
		{
			name: "unknown part at 0x21 is reported, not hidden",
			bus:  &fakeBus{regs8: map[uint8]map[uint8]uint8{0x21: {regPID: 0x76, regVER: 0x73}}},
			want: ID{Model: Unknown, Addr: 0x21, PID: 0x76, VER: 0x73},
		},
		{
			name:    "nothing answers",
			bus:     &fakeBus{},
			wantErr: ErrNotFound,
		},
		{
			name:    "all-zero answers count as absent",
			bus:     &fakeBus{regs8: map[uint8]map[uint8]uint8{0x21: {}}, regs16: map[uint8]map[uint16]uint8{0x3C: {}}},
			wantErr: ErrNotFound,
		},
		{
			name: "bus error at one address moves on to the next",
			bus:  &fakeBus{ov2640: true, fail: map[uint8]bool{0x21: true}},
			want: ID{Model: OV2640, Addr: 0x30, PID: 0x26, VER: 0x42},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Identify(tt.bus)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("err = %v, want %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Fatalf("got %+v, want %+v", got, tt.want)
			}
		})
	}
}
