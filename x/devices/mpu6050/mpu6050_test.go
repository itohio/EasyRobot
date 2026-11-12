package mpu6050

import (
	"errors"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/devices"
)

// fakeBus is an MPU6050 register file behind devices.I2C: a write of
// [reg, v...] stores v from reg on (auto-increment); a write of [reg] then a
// read returns bytes from reg on. It records every transaction.
type fakeBus struct {
	regs     [128]byte
	txs      []tx
	err      error
	failFrom int // with err set: fail from the failFrom-th transaction on (0: all)
}

type tx struct {
	addr uint16
	w    []byte
	nr   int
}

func (b *fakeBus) Tx(addr uint16, w, r []byte) error {
	b.txs = append(b.txs, tx{addr: addr, w: append([]byte(nil), w...), nr: len(r)})
	if b.err != nil && (b.failFrom == 0 || len(b.txs) >= b.failFrom) {
		return b.err
	}
	if len(w) == 0 {
		return errors.New("fakeBus: no register")
	}
	reg := int(w[0])
	copy(b.regs[reg:], w[1:])
	copy(r, b.regs[reg:])
	return nil
}

func (b *fakeBus) ReadRegister(addr uint8, r uint8, buf []byte) error {
	return b.Tx(uint16(addr), []byte{r}, buf)
}

func (b *fakeBus) WriteRegister(addr uint8, r uint8, buf []byte) error {
	return b.Tx(uint16(addr), append([]byte{r}, buf...), nil)
}

// noAllocBus answers reads with a fixed register file without allocating,
// for the allocation tests.
type noAllocBus struct{ regs [128]byte }

func (b *noAllocBus) Tx(_ uint16, w, r []byte) error {
	copy(r, b.regs[w[0]:])
	return nil
}
func (b *noAllocBus) ReadRegister(uint8, uint8, []byte) error  { return nil }
func (b *noAllocBus) WriteRegister(uint8, uint8, []byte) error { return nil }

func put16(regs []byte, reg int, v int16) {
	regs[reg] = byte(uint16(v) >> 8)
	regs[reg+1] = byte(v)
}

// Register addresses below are literals from the MPU-6000/6050 register map
// (rev 4.2), not the package's constants, so a wrong constant fails a test.

func TestProbe(t *testing.T) {
	b := &fakeBus{}
	b.regs[0x75] = 0x68 // WHO_AM_I
	d := New(b, 0)
	id, ok, err := d.Probe()
	if err != nil || !ok || id != 0x68 {
		t.Fatalf("MPU6050: id %#x ok %v err %v", id, ok, err)
	}
	if b.txs[0].addr != 0x68 || b.txs[0].w[0] != 0x75 {
		t.Fatalf("transaction %+v, want register 0x75 at address 0x68", b.txs[0])
	}
	b.regs[0x75] = 0x70 // an MPU-6500 answers 0x70
	if id, ok, err := d.Probe(); err != nil || ok || id != 0x70 {
		t.Fatalf("other chip: id %#x ok %v err %v", id, ok, err)
	}
	b.err = errors.New("nack")
	if id, ok, err := d.Probe(); err == nil || ok || id != 0 {
		t.Fatalf("bus error: id %#x ok %v err %v, want 0, false, the error", id, ok, err)
	}
}

// TestConfigure: the configuration lands in the registers: SMPLRT_DIV 0x19,
// CONFIG 0x1A DLPF_CFG[2:0], GYRO_CONFIG 0x1B FS_SEL[4:3], ACCEL_CONFIG 0x1C
// AFS_SEL[4:3], INT_PIN_CFG 0x37 (LATCH_INT_EN 0x20 | INT_RD_CLEAR 0x10),
// INT_ENABLE 0x38 DATA_RDY_EN bit 0, PWR_MGMT_1 0x6B CLKSEL[2:0]. Accel and
// gyro use different range codes so a swap shows.
func TestConfigure(t *testing.T) {
	b := &fakeBus{}
	b.regs[0x6B] = 0x40 // sleep after reset
	d := New(b, 0x69)
	cfg := Config{SampleRateDiv: 9, DLPF: DLPF44Hz, Accel: Accel8G, Gyro: Gyro500DPS, DataReadyInterrupt: true}
	if err := d.Configure(cfg); err != nil {
		t.Fatal(err)
	}
	want := map[int]byte{
		0x6B: 0x01, // awake, PLL with the X gyro as clock
		0x19: 9,
		0x1A: 3,
		0x1B: 1 << 3, // ±500 °/s
		0x1C: 2 << 3, // ±8 g
		0x37: 0x30,
		0x38: 0x01,
	}
	for reg, v := range want {
		if b.regs[reg] != v {
			t.Errorf("register %#x = %#x, want %#x", reg, b.regs[reg], v)
		}
	}
	for _, x := range b.txs {
		if x.addr != 0x69 {
			t.Fatalf("transaction to %#x, want 0x69", x.addr)
		}
	}
	if got := d.SampleRate(); got != 100 {
		t.Errorf("sample rate %v Hz, want 1 kHz/(1+9) = 100", got)
	}
	if d.Config() != cfg {
		t.Errorf("Config() %+v, want %+v", d.Config(), cfg)
	}
	if err := d.Configure(Config{SampleRateDiv: 7, DLPF: DLPFOff}); err != nil {
		t.Fatal(err)
	}
	if got := d.SampleRate(); got != 1000 {
		t.Errorf("DLPF off: %v Hz, want 8 kHz/(1+7) = 1000", got)
	}
	if b.regs[0x38] != 0 {
		t.Errorf("INT_ENABLE %#x, want 0 without the data-ready interrupt", b.regs[0x38])
	}
	for _, bad := range []Config{{DLPF: 7}, {Accel: 4}, {Gyro: 4}} {
		n := len(b.txs)
		if err := d.Configure(bad); !errors.Is(err, devices.ErrInvalidValue) || len(b.txs) != n {
			t.Errorf("%+v: err %v after %d writes, want devices.ErrInvalidValue before any", bad, err, len(b.txs)-n)
		}
	}
}

// TestBusErrorsPropagate: every operation returns the bus error; a
// Configure that fails part way records exactly the writes that landed.
func TestBusErrorsPropagate(t *testing.T) {
	nack := errors.New("nack")
	b := &fakeBus{}
	d := New(b, 0)
	b.err = nack
	var r Raw
	var s Sample
	if err := d.ReadRaw(&r); err != nack {
		t.Errorf("ReadRaw: %v", err)
	}
	if err := d.Read(&s); err != nack {
		t.Errorf("Read: %v", err)
	}
	if ok, err := d.DataReady(); err != nack || ok {
		t.Errorf("DataReady: %v %v, want false and the error", ok, err)
	}
	// A Configure that fails part way leaves the chip with the writes that
	// landed; Config() and the scaling must follow them (review S1: the gyro
	// and accel range writes landing before a failed INT write left 1 g
	// reading 8× low). Writes in order: PWR_MGMT_1, SMPLRT_DIV, CONFIG,
	// GYRO_CONFIG, ACCEL_CONFIG, INT_PIN_CFG, INT_ENABLE.
	target := Config{SampleRateDiv: 9, DLPF: DLPF44Hz, Gyro: Gyro2000DPS, Accel: Accel8G, DataReadyInterrupt: true}
	for k := 1; k <= 7; k++ { // fail the k-th of Configure's 7 writes
		b := &fakeBus{}
		d := New(b, 0)
		b.err, b.failFrom = nack, k
		if err := d.Configure(target); err != nack || len(b.txs) != k {
			t.Errorf("write %d failing: err %v after %d writes, want the error at once", k, err, len(b.txs))
		}
		want := Config{}
		if k > 2 {
			want.SampleRateDiv = target.SampleRateDiv
		}
		if k > 3 {
			want.DLPF = target.DLPF
		}
		if k > 4 {
			want.Gyro = target.Gyro
		}
		if k > 5 {
			want.Accel = target.Accel
		}
		if d.Config() != want {
			t.Errorf("write %d failing: Config() %+v, want what landed %+v", k, d.Config(), want)
		}
		// the scaling matches the range in the chip: 4096 counts read as 1 g
		// once ±8 g landed, as 0.25 g before
		b.err = nil
		put16(b.regs[:], 0x3B+4, 4096)
		var s Sample
		if err := d.Read(&s); err != nil {
			t.Fatal(err)
		}
		g := 9.80665 / 4.0
		if k > 5 {
			g = 9.80665
		}
		if math.Abs(float64(s.Accel[2])-g) > 1e-4 {
			t.Errorf("write %d failing: accel z %v m/s², want %v for the range in the chip", k, s.Accel[2], g)
		}
	}
	d = New(b, 0)
	// a ready flag read before the failure must not leak into a failed read
	b.err, b.failFrom = nil, 0
	b.regs[0x3A] = 0x01
	if ok, _ := d.DataReady(); !ok {
		t.Fatal("ready flag not read")
	}
	b.err = nack
	if ok, err := d.DataReady(); ok || err != nack {
		t.Errorf("DataReady after a ready read, then a bus error: %v %v", ok, err)
	}
}

// TestReadRaw: one 14-byte burst from ACCEL_XOUT_H (0x3B) holds accel, temp
// and gyro, big-endian two's complement.
func TestReadRaw(t *testing.T) {
	b := &fakeBus{}
	vals := []int16{-2, 16384, -32768, -521, 32767, 131, -131}
	for i, v := range vals {
		put16(b.regs[:], 0x3B+2*i, v)
	}
	d := New(b, 0)
	var r Raw
	if err := d.ReadRaw(&r); err != nil {
		t.Fatal(err)
	}
	if len(b.txs) != 1 || len(b.txs[0].w) != 1 || b.txs[0].w[0] != 0x3B || b.txs[0].nr != 14 {
		t.Fatalf("transactions %+v, want one write of 0x3B and a 14-byte read", b.txs)
	}
	want := Raw{Accel: [3]int16{-2, 16384, -32768}, Temp: -521, Gyro: [3]int16{32767, 131, -131}}
	if r != want {
		t.Fatalf("raw %+v, want %+v", r, want)
	}
}

// TestScale: conversion to SI units for every range (accel LSB/g 16384, 8192,
// 4096, 2048; gyro LSB/(°/s) 131, 65.5, 32.8, 16.4; temperature raw/340 +
// 36.53 °C).
func TestScale(t *testing.T) {
	const g = 9.80665
	for i, lsb := range []float64{16384, 8192, 4096, 2048} {
		s := Config{Accel: AccelRange(i)}.Scale(Raw{Accel: [3]int16{int16(lsb), -int16(lsb), int16(lsb / 2)}})
		for k, want := range []float64{g, -g, g / 2} {
			if math.Abs(float64(s.Accel[k])-want) > 1e-5 {
				t.Errorf("accel range %d axis %d: %v, want %v m/s²", i, k, s.Accel[k], want)
			}
		}
	}
	for i, lsb := range []float64{131, 65.5, 32.8, 16.4} {
		raw := int16(math.Round(lsb * 100)) // 100 °/s
		s := Config{Gyro: GyroRange(i)}.Scale(Raw{Gyro: [3]int16{raw, -raw, 0}})
		want := float64(raw) / lsb * math.Pi / 180
		if math.Abs(float64(s.Gyro[0])-want) > 1e-6 || math.Abs(float64(s.Gyro[1])+want) > 1e-6 || s.Gyro[2] != 0 {
			t.Errorf("gyro range %d: %v, want ±%v rad/s", i, s.Gyro, want)
		}
	}
	for _, c := range []struct {
		raw  int16
		want float64
	}{{340, 37.53}, {-340, 35.53}, {-12420, 0}} {
		if s := (Config{}).Scale(Raw{Temp: c.raw}); math.Abs(float64(s.TempC)-c.want) > 1e-3 {
			t.Errorf("temperature counts %d: %v °C, want %v", c.raw, s.TempC, c.want)
		}
	}
}

func TestReadScalesWithTheConfiguredRanges(t *testing.T) {
	b := &fakeBus{}
	put16(b.regs[:], 0x3B+4, 4096) // accel z: 1 g at ±8 g
	put16(b.regs[:], 0x43, 164)    // gyro x (GYRO_XOUT_H): 10 °/s at ±2000 °/s
	d := New(b, 0)
	if err := d.Configure(Config{Accel: Accel8G, Gyro: Gyro2000DPS}); err != nil {
		t.Fatal(err)
	}
	var s Sample
	if err := d.Read(&s); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(s.Accel[2])-9.80665) > 1e-4 || math.Abs(float64(s.Gyro[0])-10*math.Pi/180) > 1e-5 {
		t.Fatalf("sample %+v", s)
	}
}

func TestDataReady(t *testing.T) {
	b := &fakeBus{}
	d := New(b, 0)
	if ok, err := d.DataReady(); err != nil || ok || b.txs[0].w[0] != 0x3A {
		t.Fatalf("idle: %v %v", ok, err)
	}
	b.regs[0x3A] = 0x01 // INT_STATUS DATA_RDY_INT
	if ok, err := d.DataReady(); err != nil || !ok {
		t.Fatalf("ready: %v %v", ok, err)
	}
}

// TestNoAllocation: reads and register writes allocate nothing (caller
// buffers, the device's own 14-byte buffer).
func TestNoAllocation(t *testing.T) {
	d := New(&noAllocBus{}, 0)
	var r Raw
	var s Sample
	if n := testing.AllocsPerRun(100, func() {
		_ = d.ReadRaw(&r)
		_ = d.Read(&s)
		_, _ = d.DataReady()
		_, _, _ = d.Probe()
		_ = d.Configure(Config{})
	}); n != 0 {
		t.Fatalf("%v allocations per read", n)
	}
}
