//go:build esp32s3

package camera

// ESP32-S3 LCD_CAM camera mode + one GDMA RX channel, polling only (no CPU
// interrupts). The register sequence follows espressif/esp32-camera v2.1.9
// target/esp32s3/ll_cam.c (ll_cam_config, ll_cam_dma_reset, ll_cam_start,
// ll_cam_set_pin); brought up in ER-ESP32CAM E3–E5 and M3–M5 (evidence
// outside the repo):
//   - frame start: poll the VSYNC pin and arm right after a VSYNC pulse;
//   - frame end: CAM_VS_EOF_EN, so VSYNC raises GDMA in_suc_eof (polled).
// LCD_CAM starts receiving only at the first VSYNC after CAM_START, so one
// capture costs about two frame periods.

import (
	"device/esp"
	"machine"
	"runtime/volatile"
	"time"
	"unsafe"
)

// GPIO matrix input signals (soc/esp32s3 gpio_sig_map.h).
const (
	sigCamData0   = 133
	sigCamPCLK    = 149
	sigCamHEnable = 150
	sigCamVSync   = 152
	periSelCam    = 5 // GDMA peripheral select: LCD_CAM (PERI_SEL also gates memory access, E3 §74)
)

const (
	nodeBytes = 4032 // GDMA descriptor buffer: ≤ 4095, multiple of 4
	maxNodes  = 128  // 516 KB: more than the S3's DMA-capable internal SRAM
	camClkHz  = 160_000_000

	// DMA-capable internal SRAM on the data bus (SOC_DMA_LOW … SOC_DMA_HIGH).
	dmaLow  = 0x3FC88000
	dmaHigh = 0x3FD00000

	intSucEOF    = 1 << 1
	intDscrErr   = 1 << 3
	intDscrEmpty = 1 << 4
)

// dmaDesc is a GDMA linked-list descriptor: dw0 = size[11:0] |
// length[23:12] | suc_eof[30] | owner[31]; buffer address; next descriptor.
type dmaDesc struct{ dw0, buf, next uint32 }

// gdmaIn is one GDMA RX channel's register block (IN_CONF0_CHn at
// DMA + n×0xC0; offsets as in device/esp's DMA_Type for channel 0).
type gdmaIn struct {
	conf0, conf1, intRaw, intSt, intEna, intClr volatile.Register32 // 0x00–0x14
	fifoStatus, pop, link, state, sucEOFDesc    volatile.Register32 // 0x18–0x28
	errEOFDesc, dscr, dscrBF0, dscrBF1, wight   volatile.Register32 // 0x2C–0x3C
	_, pri, periSel                             volatile.Register32 // 0x40–0x48
}

func gdmaChannel(n uint8) *gdmaIn {
	return (*gdmaIn)(unsafe.Add(unsafe.Pointer(&esp.DMA.IN_CONF0_CH0), uintptr(n)*0xC0))
}

func inSignal(p machine.Pin, sig uint32, invert bool) {
	p.Configure(machine.PinConfig{Mode: machine.PinInput})
	v := esp.GPIO_FUNC_IN_SEL_CFG_SEL | uint32(p)<<esp.GPIO_FUNC_IN_SEL_CFG_IN_SEL_Pos
	if invert {
		v |= esp.GPIO_FUNC_IN_SEL_CFG_IN_INV_SEL
	}
	(*volatile.Register32)(unsafe.Add(unsafe.Pointer(&esp.GPIO.FUNC0_IN_SEL_CFG), uintptr(sig)*4)).Set(v)
}

// startXCLK drives pins.XCLK with a 50 % LEDC PWM at hz.
func startXCLK(pins Pins, hz uint32) (uint8, error) {
	pwm := pins.XCLKTimer
	if pwm == nil {
		pwm = machine.PWM0
	}
	if err := pwm.Configure(machine.PWMConfig{Period: uint64(1_000_000_000 / hz)}); err != nil {
		return 0, err
	}
	ch, err := pwm.Channel(pins.XCLK)
	if err != nil {
		return 0, err
	}
	pwm.Set(ch, (pwm.Top()+1)/2)
	return ch, nil
}

// camSetup routes the data and sync pins into LCD_CAM and resets the GDMA
// channel (ll_cam_set_pin, ll_cam_config, ll_cam_dma_reset).
func camSetup(pins Pins, ch *gdmaIn) {
	for i, p := range pins.D {
		inSignal(p, sigCamData0+uint32(i), false)
	}
	inSignal(pins.PCLK, sigCamPCLK, false)
	inSignal(pins.VSYNC, sigCamVSync, true) // vsync_invert = true (cam_hal.c)
	inSignal(pins.HREF, sigCamHEnable, false)

	esp.SYSTEM.SetPERIP_CLK_EN1_LCD_CAM_CLK_EN(1)
	esp.SYSTEM.SetPERIP_RST_EN1_LCD_CAM_RST(1)
	esp.SYSTEM.SetPERIP_RST_EN1_LCD_CAM_RST(0)
	esp.SYSTEM.SetPERIP_CLK_EN1_DMA_CLK_EN(1)
	esp.SYSTEM.SetPERIP_RST_EN1_DMA_RST(0)
	esp.DMA.SetMISC_CONF_CLK_EN(1)

	cam := esp.LCD_CAM
	cam.CAM_CTRL.Set(0)
	cam.SetCAM_CTRL_CAM_CLKM_DIV_NUM(camClkHz / 20_000_000)
	cam.SetCAM_CTRL_CAM_CLK_SEL(3) // PLL 160 MHz
	cam.SetCAM_CTRL_CAM_VSYNC_FILTER_THRES(4)
	cam.SetCAM_CTRL_CAM_VS_EOF_EN(1)
	cam.CAM_CTRL1.Set(0)
	cam.SetCAM_CTRL1_CAM_REC_DATA_BYTELEN(nodeBytes - 1)
	cam.SetCAM_CTRL1_CAM_VSYNC_FILTER_EN(1)
	cam.CAM_RGB_YUV.Set(0)
	cam.SetCAM_CTRL_CAM_UPDATE(1)

	ch.intClr.Set(0xFFFFFFFF)
	ch.intEna.Set(0)
	ch.conf0.Set(0)
	ch.conf0.SetBits(esp.DMA_IN_CONF0_CH_IN_RST)
	ch.conf0.ClearBits(esp.DMA_IN_CONF0_CH_IN_RST)
	ch.conf0.SetBits(esp.DMA_IN_CONF0_CH_INDSCR_BURST_EN | esp.DMA_IN_CONF0_CH_IN_DATA_BURST_EN)
	ch.conf1.ClearBits(esp.DMA_IN_CONF1_CH_IN_CHECK_OWNER)
	ch.periSel.Set(periSelCam)
}

func waitPin(p machine.Pin, level bool, deadline time.Time) bool {
	for p.Get() != level {
		if time.Now().After(deadline) {
			return false
		}
	}
	return true
}

// vsyncLevel samples VSYNC for d and returns the level during the pulse
// (the less frequent level) and the number of pulses.
func vsyncLevel(p machine.Pin, d time.Duration) (active bool, pulses int) {
	end := time.Now().Add(d)
	hi, total, edges := 0, 0, 0
	prev := p.Get()
	for time.Now().Before(end) {
		v := p.Get()
		if v {
			hi++
		}
		if v != prev {
			edges++
			prev = v
		}
		total++
	}
	return hi*2 < total, edges / 2
}

// buildChain points the descriptors at buf[:n] in nodeBytes pieces and
// returns how many it used.
func buildChain(d *[maxNodes]dmaDesc, buf []byte, n int) int {
	used := (n + nodeBytes - 1) / nodeBytes
	base := uintptr(unsafe.Pointer(unsafe.SliceData(buf)))
	for i := 0; i < used; i++ {
		size := nodeBytes
		if i == used-1 {
			size = n - i*nodeBytes
		}
		next := uint32(0)
		if i+1 < used {
			next = uint32(uintptr(unsafe.Pointer(&d[i+1])))
		}
		volatile.StoreUint32(&d[i].dw0, 1<<31|uint32(size)) // owner = DMA; length 0
		volatile.StoreUint32(&d[i].buf, uint32(base+uintptr(i*nodeBytes)))
		volatile.StoreUint32(&d[i].next, next)
	}
	return used
}

// received sums the descriptor lengths up to the EOF descriptor (or all of
// them when there is none).
func received(d *[maxNodes]dmaDesc, used int, eofDesc uint32) int {
	total := 0
	for i := 0; i < used; i++ {
		total += int(volatile.LoadUint32(&d[i].dw0)>>12) & 0xFFF
		if eofDesc == uint32(uintptr(unsafe.Pointer(&d[i]))) {
			break
		}
	}
	return total
}

// captureOnce arms LCD_CAM and the GDMA channel right after a VSYNC pulse and
// waits for the frame's EOF. It returns the received length and whether the
// frame ended (in_suc_eof) before the descriptors ran out or the deadline.
func captureOnce(pins Pins, vsyncActive bool, ch *gdmaIn, d *[maxNodes]dmaDesc, used int, deadline time.Time) (int, bool) {
	if !waitPin(pins.VSYNC, vsyncActive, deadline) || !waitPin(pins.VSYNC, !vsyncActive, deadline) {
		return 0, false
	}
	cam := esp.LCD_CAM
	cam.LC_DMA_INT_CLR.Set(0xF)
	cam.SetCAM_CTRL1_CAM_START(0)
	cam.SetCAM_CTRL1_CAM_RESET(1)
	cam.SetCAM_CTRL1_CAM_RESET(0)
	cam.SetCAM_CTRL1_CAM_AFIFO_RESET(1)
	cam.SetCAM_CTRL1_CAM_AFIFO_RESET(0)
	ch.conf0.SetBits(esp.DMA_IN_CONF0_CH_IN_RST)
	ch.conf0.ClearBits(esp.DMA_IN_CONF0_CH_IN_RST)
	ch.intClr.Set(0xFFFFFFFF)
	cam.SetCAM_CTRL1_CAM_REC_DATA_BYTELEN(nodeBytes - 1)
	ch.link.Set(uint32(uintptr(unsafe.Pointer(&d[0])))&esp.DMA_IN_LINK_CH_INLINK_ADDR_Msk | esp.DMA_IN_LINK_CH_INLINK_START)
	cam.SetCAM_CTRL_CAM_UPDATE(1)
	cam.SetCAM_CTRL1_CAM_START(1)

	ok := false
	for time.Now().Before(deadline) {
		raw := ch.intRaw.Get()
		if raw&intSucEOF != 0 {
			ok = true
			break
		}
		if raw&(intDscrEmpty|intDscrErr) != 0 {
			break
		}
	}
	cam.SetCAM_CTRL1_CAM_START(0)
	ch.link.SetBits(esp.DMA_IN_LINK_CH_INLINK_STOP)
	eof := uint32(0)
	if ok {
		eof = ch.sucEOFDesc.Get()
	}
	return received(d, used, eof), ok
}

// sccbBus is the SCCB master on a machine.I2C. Its buffers live in the
// struct; machine.I2C on the ESP32-S3 still allocates per transfer (TinyGo
// 0.43), so SCCB access (controls, Identify) allocates. Capture doesn't.
type sccbBus struct {
	i2c *machine.I2C
	w   [3]byte
	r   [1]byte
}

func (b *sccbBus) read(dev uint8, n int) (uint8, error) {
	if err := b.i2c.Tx(uint16(dev), b.w[:n], nil); err != nil {
		return 0, err
	}
	err := b.i2c.Tx(uint16(dev), nil, b.r[:])
	return b.r[0], err
}

func (b *sccbBus) ReadReg8(dev, reg uint8) (uint8, error) {
	b.w[0] = reg
	return b.read(dev, 1)
}

func (b *sccbBus) WriteReg8(dev, reg, val uint8) error {
	b.w[0], b.w[1] = reg, val
	return b.i2c.Tx(uint16(dev), b.w[:2], nil)
}

func (b *sccbBus) ReadReg16(dev uint8, reg uint16) (uint8, error) {
	b.w[0], b.w[1] = uint8(reg>>8), uint8(reg)
	return b.read(dev, 2)
}

func (b *sccbBus) WriteReg16(dev uint8, reg uint16, val uint8) error {
	b.w[0], b.w[1], b.w[2] = uint8(reg>>8), uint8(reg), val
	return b.i2c.Tx(uint16(dev), b.w[:3], nil)
}
