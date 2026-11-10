//go:build k210

package kpu

import (
	"device/kendryte"
	"device/riscv"
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
)

// aiIOBase is the KPU memory's uncached alias (AI_IO_BASE_ADDR; the cached
// one is 0x8060_0000). The C runtime writes the input there too.
const aiIOBase = 0x4060_0000

var aiRAM = unsafe.Slice((*byte)(unsafe.Pointer(uintptr(aiIOBase))), kmodel.RAMSize)

// convTimeout bounds one conv layer: 1 s at the UnitV's 390 MHz CPU clock.
const convTimeout = 390_000_000

type k210 struct{}

func (k210) ram() []byte { return aiRAM }

// conv sends the 12 layer-argument words (kpu_send_layer) and waits for
// calc_done.
func (k210) conv(a *kmodel.Args) error {
	k := kendryte.KPU
	k.INTERRUPT_CLEAR.Set(7)
	for _, w := range a {
		k.LAYER_ARGUMENT_FIFO.Set(w)
	}
	t0 := riscv.CYCLE.Get()
	for (k.INTERRUPT_RAW.Get()|k.INTERRUPT_STATUS.Get())&1 == 0 {
		if riscv.CYCLE.Get()-t0 > convTimeout {
			return ErrTimeout
		}
	}
	k.INTERRUPT_CLEAR.Set(7)
	return nil
}

// open powers the KPU and sets it up like kpu_run_kmodel: interrupts
// cleared, FIFO thresholds 10/1, 8-bit mode from the model flags. Only
// calc_done is unmasked; the PLIC line stays disabled and conv polls.
func open(eightBit bool) (engine, error) {
	if err := enablePLL1(); err != nil {
		return nil, err
	}
	kendryte.SYSCTL.CLK_EN_PERI.SetBits(kendryte.SYSCTL_CLK_EN_PERI_AI_CLK_EN)
	k := kendryte.KPU
	k.INTERRUPT_CLEAR.Set(7)
	k.FIFO_THRESHOLD.Set(10 | 1<<4)
	var m uint64
	if eightBit {
		m = 1
	}
	k.EIGHT_BIT_MODE.Set(m)
	k.INTERRUPT_MASK.Set(0b110)
	return k210{}, nil
}

const in0Hz = 26_000_000

// enablePLL1 starts PLL1, the KPU's clock, at 26 MHz·46/3 ≈ 398.7 MHz
// unless it already runs (powered, output on, not bypassed, locked). The
// sequence is sysctl_pll_set_freq's: output off, power off, set dividers,
// power on, pulse reset, wait for lock clearing slips, output on.
func enablePLL1() error {
	s := kendryte.SYSCTL
	const on = kendryte.SYSCTL_PLL1_PWRD | kendryte.SYSCTL_PLL1_OUT_EN
	if v := s.PLL1.Get(); v&on == on && v&kendryte.SYSCTL_PLL1_BYPASS == 0 && pll1Locked() {
		return nil
	}
	s.PLL1.ClearBits(kendryte.SYSCTL_PLL1_OUT_EN)
	s.PLL1.ClearBits(kendryte.SYSCTL_PLL1_PWRD)
	v := s.PLL1.Get() &^ (kendryte.SYSCTL_PLL1_CLKR_Msk | kendryte.SYSCTL_PLL1_CLKF_Msk |
		kendryte.SYSCTL_PLL1_CLKOD_Msk | kendryte.SYSCTL_PLL1_BWADJ_Msk | kendryte.SYSCTL_PLL1_BYPASS)
	v |= 0<<kendryte.SYSCTL_PLL1_CLKR_Pos | 45<<kendryte.SYSCTL_PLL1_CLKF_Pos |
		2<<kendryte.SYSCTL_PLL1_CLKOD_Pos | 45<<kendryte.SYSCTL_PLL1_BWADJ_Pos
	s.PLL1.Set(v)
	s.PLL1.SetBits(kendryte.SYSCTL_PLL1_PWRD)
	spin(1000)
	s.PLL1.ClearBits(kendryte.SYSCTL_PLL1_RESET)
	s.PLL1.SetBits(kendryte.SYSCTL_PLL1_RESET)
	spin(1000)
	s.PLL1.ClearBits(kendryte.SYSCTL_PLL1_RESET)
	t0 := riscv.CYCLE.Get()
	for !pll1Locked() {
		s.PLL_LOCK.SetBits(kendryte.SYSCTL_PLL_LOCK_PLL_SLIP_CLEAR1)
		if riscv.CYCLE.Get()-t0 > convTimeout {
			return ErrTimeout
		}
	}
	s.PLL1.SetBits(kendryte.SYSCTL_PLL1_OUT_EN)
	return nil
}

func pll1Locked() bool {
	return kendryte.SYSCTL.PLL_LOCK.Get()>>kendryte.SYSCTL_PLL_LOCK_PLL_LOCK1_Pos&1 != 0
}

func spin(cycles uintptr) {
	t0 := riscv.CYCLE.Get()
	for riscv.CYCLE.Get()-t0 < cycles {
	}
}

// ClockHz reports the KPU clock: PLL1 / (AI_GCLK threshold + 1), or 0 while
// PLL1 is off. K210 only.
func ClockHz() uint32 {
	v := kendryte.SYSCTL.PLL1.Get()
	if v&kendryte.SYSCTL_PLL1_PWRD == 0 || v&kendryte.SYSCTL_PLL1_OUT_EN == 0 {
		return 0
	}
	hz := uint64(in0Hz)
	if v&kendryte.SYSCTL_PLL1_BYPASS == 0 {
		r := uint64(v&kendryte.SYSCTL_PLL1_CLKR_Msk>>kendryte.SYSCTL_PLL1_CLKR_Pos) + 1
		f := uint64(v&kendryte.SYSCTL_PLL1_CLKF_Msk>>kendryte.SYSCTL_PLL1_CLKF_Pos) + 1
		od := uint64(v&kendryte.SYSCTL_PLL1_CLKOD_Msk>>kendryte.SYSCTL_PLL1_CLKOD_Pos) + 1
		hz = hz * f / (r * od)
	}
	th := uint64(kendryte.SYSCTL.GetCLK_TH0_AI_GCLK())
	return uint32(hz / (th + 1))
}
