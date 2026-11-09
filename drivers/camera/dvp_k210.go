//go:build k210

package camera

import (
	"device/kendryte"
	"time"
)

// K210 clock tree (reported, Kendryte standalone SDK sysctl.c): PLL0 is fed
// by the 26 MHz oscillator; ACLK (= CPU) is PLL0/(2<<div) when selected;
// APB1, which clocks the DVP's SCCB and XCLK dividers, is ACLK/(sel+1).
const in0Hz = 26_000_000

func pll0Hz() uint32 {
	v := kendryte.SYSCTL.PLL0.Get()
	if v&kendryte.SYSCTL_PLL0_BYPASS_Msk != 0 {
		return in0Hz
	}
	r := (v&kendryte.SYSCTL_PLL0_CLKR_Msk)>>kendryte.SYSCTL_PLL0_CLKR_Pos + 1
	f := (v&kendryte.SYSCTL_PLL0_CLKF_Msk)>>kendryte.SYSCTL_PLL0_CLKF_Pos + 1
	od := (v&kendryte.SYSCTL_PLL0_CLKOD_Msk)>>kendryte.SYSCTL_PLL0_CLKOD_Pos + 1
	return uint32(uint64(in0Hz) * uint64(f) / uint64(r*od))
}

func aclkHz() uint32 {
	sel := kendryte.SYSCTL.CLK_SEL0.Get()
	if sel&kendryte.SYSCTL_CLK_SEL0_ACLK_SEL == 0 {
		return in0Hz
	}
	div := (sel & kendryte.SYSCTL_CLK_SEL0_ACLK_DIVIDER_SEL_Msk) >> kendryte.SYSCTL_CLK_SEL0_ACLK_DIVIDER_SEL_Pos
	return pll0Hz() / (2 << div)
}

func apb1Hz() uint32 {
	sel := kendryte.SYSCTL.CLK_SEL0.Get()
	div := (sel&kendryte.SYSCTL_CLK_SEL0_APB1_CLK_SEL_Msk)>>kendryte.SYSCTL_CLK_SEL0_APB1_CLK_SEL_Pos + 1
	return aclkHz() / div
}

// Clocks are the K210 clock rates the DVP runs from (diagnostics).
type Clocks struct{ PLL0, ACLK, APB1, XCLK, SCCB uint32 }

// sleep waits at least d. It checks a deadline against time.Now because a
// bare time.Sleep on TinyGo 0.43/K210 can return early
// (docs/evidence/ER-K210-M1.md §4).
func sleep(d time.Duration) {
	end := time.Now().Add(d)
	for left := time.Until(end); left > 0; left = time.Until(end) {
		time.Sleep(left)
	}
}

// dvpInit enables and resets the DVP and sets its SCCB clock to about
// sccbHz. It returns the SCCB rate achieved.
func dvpInit(sccbHz uint32) uint32 {
	kendryte.SYSCTL.CLK_EN_CENT.SetBits(kendryte.SYSCTL_CLK_EN_CENT_APB1_CLK_EN)
	kendryte.SYSCTL.CLK_EN_PERI.SetBits(kendryte.SYSCTL_CLK_EN_PERI_DVP_CLK_EN)
	kendryte.SYSCTL.PERI_RESET.SetBits(kendryte.SYSCTL_PERI_RESET_DVP_RESET)
	sleep(10 * time.Microsecond)
	kendryte.SYSCTL.PERI_RESET.ClearBits(kendryte.SYSCTL_PERI_RESET_DVP_RESET)

	axi := kendryte.DVP.AXI.Get() &^ kendryte.DVP_AXI_GM_MLEN_Msk
	kendryte.DVP.AXI.Set(axi | kendryte.DVP_AXI_GM_MLEN_BYTE4)

	// SCL low and high counts in APB1 cycles (8 bits each).
	cnt := apb1Hz() / (2 * sccbHz)
	if cnt > 255 {
		cnt = 255
	}
	if cnt < 1 {
		cnt = 1
	}
	// The DVP's eight data inputs are the SPI0 D0–D7 pins; this routes them
	// to the DVP (SDK sysctl_set_spi0_dvp_data). Without it every pixel
	// reads 0 (observed, docs/evidence/ER-K210-M4.md).
	kendryte.SYSCTL.MISC.SetBits(kendryte.SYSCTL_MISC_SPI_DVP_DATA_ENABLE)
	cfg := kendryte.DVP.SCCB_CFG.Get() &^ (kendryte.DVP_SCCB_CFG_SCL_LCNT_Msk | kendryte.DVP_SCCB_CFG_SCL_HCNT_Msk)
	kendryte.DVP.SCCB_CFG.Set(cfg | cnt<<kendryte.DVP_SCCB_CFG_SCL_LCNT_Pos | cnt<<kendryte.DVP_SCCB_CFG_SCL_HCNT_Pos)
	return apb1Hz() / (2 * cnt)
}

// setXCLK starts the sensor clock at about hz and returns the rate achieved.
func setXCLK(hz uint32) uint32 {
	apb1 := apb1Hz()
	var div uint32
	if apb1 > 2*hz {
		div = (apb1+hz)/(2*hz) - 1 // round(apb1 / 2hz) - 1
	}
	if div > 255 {
		div = 255
	}
	cfg := kendryte.DVP.CMOS_CFG.Get() &^ kendryte.DVP_CMOS_CFG_CLK_DIV_Msk
	kendryte.DVP.CMOS_CFG.Set(cfg | div<<kendryte.DVP_CMOS_CFG_CLK_DIV_Pos | kendryte.DVP_CMOS_CFG_CLK_ENABLE)
	return apb1 / (2 * (div + 1))
}

// sensorPowerCycle drives the DVP's PWDN and RESET outputs: power down,
// power up, then a reset pulse (SDK dvp_reset; 200 ms steps). The sensor
// ends powered (PWDN low) and out of reset (RESET high).
func sensorPowerCycle() {
	kendryte.DVP.CMOS_CFG.SetBits(kendryte.DVP_CMOS_CFG_POWER_DOWN)
	sleep(200 * time.Millisecond)
	kendryte.DVP.CMOS_CFG.ClearBits(kendryte.DVP_CMOS_CFG_POWER_DOWN)
	sleep(200 * time.Millisecond)
	kendryte.DVP.CMOS_CFG.ClearBits(kendryte.DVP_CMOS_CFG_RESET)
	sleep(200 * time.Millisecond)
	kendryte.DVP.CMOS_CFG.SetBits(kendryte.DVP_CMOS_CFG_RESET)
	sleep(200 * time.Millisecond)
}

// SCCB_CTL bit 0 enables the write-data phase (SDK DVP_SCCB_WRITE_DATA_ENABLE);
// the controller generates the bus R/W bit itself.
const sccbWriteEnable = 1

func sccbTransfer() error {
	end := time.Now().Add(10 * time.Millisecond)
	for kendryte.DVP.STS.Get()&kendryte.DVP_STS_SCCB_EN != 0 {
		if time.Now().After(end) {
			return ErrTimeout
		}
	}
	kendryte.DVP.STS.Set(kendryte.DVP_STS_SCCB_EN | kendryte.DVP_STS_SCCB_EN_WE)
	for kendryte.DVP.STS.Get()&kendryte.DVP_STS_SCCB_EN != 0 {
		if time.Now().After(end) {
			return ErrTimeout
		}
	}
	return nil
}

func sccbByteNum(n uint32) {
	cfg := kendryte.DVP.SCCB_CFG.Get() &^ kendryte.DVP_SCCB_CFG_BYTE_NUM_Msk
	kendryte.DVP.SCCB_CFG.Set(cfg | n)
}

// sccbBus is the DVP's built-in SCCB master. Addresses are 7-bit; the
// controller takes the 8-bit write address.
type sccbBus struct{}

func (sccbBus) WriteReg8(dev, reg, val uint8) error {
	sccbByteNum(kendryte.DVP_SCCB_CFG_BYTE_NUM_NUM3)
	kendryte.DVP.SCCB_CTL.Set(sccbWriteEnable | uint32(dev)<<1 | uint32(reg)<<8 | uint32(val)<<16)
	return sccbTransfer()
}

func (sccbBus) ReadReg8(dev, reg uint8) (uint8, error) {
	sccbByteNum(kendryte.DVP_SCCB_CFG_BYTE_NUM_NUM2)
	kendryte.DVP.SCCB_CTL.Set(sccbWriteEnable | uint32(dev)<<1 | uint32(reg)<<8)
	if err := sccbTransfer(); err != nil {
		return 0, err
	}
	kendryte.DVP.SCCB_CTL.Set(uint32(dev) << 1)
	if err := sccbTransfer(); err != nil {
		return 0, err
	}
	return uint8(kendryte.DVP.SCCB_CFG.Get() >> kendryte.DVP_SCCB_CFG_RDATA_Pos), nil
}

func (sccbBus) WriteReg16(dev uint8, reg uint16, val uint8) error {
	sccbByteNum(kendryte.DVP_SCCB_CFG_BYTE_NUM_NUM4)
	kendryte.DVP.SCCB_CTL.Set(sccbWriteEnable | uint32(dev)<<1 | uint32(reg>>8)<<8 | uint32(reg&0xFF)<<16 | uint32(val)<<24)
	return sccbTransfer()
}

func (sccbBus) ReadReg16(dev uint8, reg uint16) (uint8, error) {
	sccbByteNum(kendryte.DVP_SCCB_CFG_BYTE_NUM_NUM3)
	kendryte.DVP.SCCB_CTL.Set(sccbWriteEnable | uint32(dev)<<1 | uint32(reg>>8)<<8 | uint32(reg&0xFF)<<16)
	if err := sccbTransfer(); err != nil {
		return 0, err
	}
	kendryte.DVP.SCCB_CTL.Set(uint32(dev) << 1)
	if err := sccbTransfer(); err != nil {
		return 0, err
	}
	return uint8(kendryte.DVP.SCCB_CFG.Get() >> kendryte.DVP_SCCB_CFG_RDATA_Pos), nil
}

// dvpConfigure sets burst mode, input format, frame size and outputs, with
// auto mode and interrupts off (Capture polls STS).
// format is DVP_DVP_CFG_FORMAT_RGB/YUV/Y; ai enables the planar AI output,
// display the RGB565 display output.
func dvpConfigure(format uint32, w, h uint16, ai, display bool) {
	cfg := kendryte.DVP.DVP_CFG.Get()
	cfg &^= kendryte.DVP_DVP_CFG_START_INT_ENABLE | kendryte.DVP_DVP_CFG_FINISH_INT_ENABLE |
		kendryte.DVP_DVP_CFG_AUTO_ENABLE | kendryte.DVP_DVP_CFG_AI_OUTPUT_ENABLE |
		kendryte.DVP_DVP_CFG_DISPLAY_OUTPUT_ENABLE | kendryte.DVP_DVP_CFG_FORMAT_Msk |
		kendryte.DVP_DVP_CFG_HREF_BURST_NUM_Msk | kendryte.DVP_DVP_CFG_LINE_NUM_Msk
	cfg |= kendryte.DVP_DVP_CFG_BURST_SIZE_4BEATS
	cfg |= format << kendryte.DVP_DVP_CFG_FORMAT_Pos
	// SDK dvp_set_image_size: bursts per line = width/8/4 with 4-beat bursts.
	cfg |= uint32(w/32) << kendryte.DVP_DVP_CFG_HREF_BURST_NUM_Pos
	cfg |= uint32(h) << kendryte.DVP_DVP_CFG_LINE_NUM_Pos
	if ai {
		cfg |= kendryte.DVP_DVP_CFG_AI_OUTPUT_ENABLE
	}
	if display {
		cfg |= kendryte.DVP_DVP_CFG_DISPLAY_OUTPUT_ENABLE
	}
	kendryte.DVP.DVP_CFG.Set(cfg)
}

// dvpCaptureOne waits for a frame start, starts conversion and waits for
// the frame to finish, within timeout.
func dvpCaptureOne(timeout time.Duration) error {
	const start, finish = kendryte.DVP_STS_FRAME_START, kendryte.DVP_STS_FRAME_FINISH
	clear := func(bits uint32) { kendryte.DVP.STS.Set(bits | bits<<1) } // each status bit's WE is the next bit
	end := time.Now().Add(timeout)
	clear(start | finish)
	for kendryte.DVP.STS.Get()&start == 0 {
		if time.Now().After(end) {
			return ErrTimeout
		}
	}
	kendryte.DVP.STS.Set(kendryte.DVP_STS_DVP_EN | kendryte.DVP_STS_DVP_EN_WE)
	// Clear both: a FINISH latched before conversion was enabled must not
	// end this capture early (review R-ER-K210-camera, hardware note).
	clear(start | finish)
	for kendryte.DVP.STS.Get()&finish == 0 {
		if time.Now().After(end) {
			return ErrTimeout
		}
	}
	clear(finish)
	return nil
}
