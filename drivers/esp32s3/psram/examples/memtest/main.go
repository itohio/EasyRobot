//go:build esp32s3

// memtest: board check for package psram (ER-ESP32CAM-E6). Prints the PSRAM
// mode registers, then runs over the whole mapped region:
//  1. address-line test: walking-one offsets get distinct markers, read back
//  2. pattern test: word i = i·0x9E3779B9 ^ seed, written then verified, two seeds
//  3. DMA-style cache maintenance: WritebackInvalidate / Invalidate on an arena slice
package main

import (
	"time"
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/esp32s3/psram"
)

func hex(v uint32) string {
	const d = "0123456789abcdef"
	var b [10]byte
	b[0], b[1] = '0', 'x'
	for i := range 8 {
		b[9-i] = d[v>>(4*i)&15]
	}
	return string(b[:])
}

func main() {
	time.Sleep(2 * time.Second)
	r, m, err := psram.Init()
	println("psram: MR0", hex(uint32(m.MR0)), "MR1", hex(uint32(m.MR1)), "MR2", hex(uint32(m.MR2)),
		"MR3", hex(uint32(m.MR3)), "MR4", hex(uint32(m.MR4)), "MR8", hex(uint32(m.MR8)))
	println("psram: vendor", hex(uint32(m.VendorID())), "size", m.Size(), "good die", m.GoodDie(),
		"fixed latency", m.FixedLatency(), "read latency code", m.ReadLatency(), "3V", m.VCC3V(),
		"burst", m.BurstLength(), "hybrid wrap", m.HybridWrap())
	if err != nil {
		println("psram: init failed:", err.Error())
		for {
			time.Sleep(time.Second)
		}
	}
	println("psram: mapped", r.Size(), "bytes at", hex(uint32(r.Base())))
	words := unsafe.Slice((*uint32)(unsafe.Pointer(r.Base())), r.Size()/4)

	// 1. Address lines: offset 0 and every power-of-two word offset.
	words[0] = 0xA5A5A5A5
	for bit := 0; 1<<bit < len(words); bit++ {
		words[1<<bit] = 0x5A000000 | uint32(bit)
	}
	// Write back and drop the cached lines so the reads below come from the
	// PSRAM, not from the D-cache (otherwise the cache serves most of them).
	if err := psram.WritebackInvalidate(r.Bytes()); err != nil {
		println("psram: writeback/invalidate:", err.Error())
	}
	bad := 0
	if words[0] != 0xA5A5A5A5 {
		bad++
		println("psram: addr test word 0 =", hex(words[0]))
	}
	for bit := 0; 1<<bit < len(words); bit++ {
		if v := words[1<<bit]; v != 0x5A000000|uint32(bit) {
			bad++
			println("psram: addr line", bit, "read", hex(v))
		}
	}
	println("psram: address-line test errors", bad)

	// 2. Full pattern test, two seeds (the second inverts every bit position).
	for _, seed := range []uint32{0x00000000, 0xFFFFFFFF} {
		t0 := time.Now()
		for i := range words {
			words[i] = uint32(i)*0x9E3779B9 ^ seed
		}
		tw := time.Since(t0)
		if err := psram.WritebackInvalidate(r.Bytes()); err != nil {
			println("psram: writeback/invalidate:", err.Error())
		}
		errs, first := 0, -1
		for i := range words {
			if words[i] != uint32(i)*0x9E3779B9^seed {
				if first < 0 {
					first = i
				}
				errs++
			}
		}
		println("psram: pattern seed", hex(seed), "errors", errs, "first bad word", first,
			"write ms", tw.Milliseconds(), "total ms", time.Since(t0).Milliseconds())
	}

	// 3. Arena + cache maintenance calls.
	a := psram.NewArena(r)
	b, err := a.Alloc(300_000)
	if err != nil {
		println("psram: alloc:", err.Error())
	} else {
		for i := range b {
			b[i] = byte(i)
		}
		e1 := psram.WritebackInvalidate(b)
		ok := true
		for i := range b {
			if b[i] != byte(i) {
				ok = false
				break
			}
		}
		println("psram: arena slice", len(b), "cap", cap(b), "at", hex(uint32(uintptr(unsafe.Pointer(&b[0])))),
			"writeback+invalidate err", e1 == nil, "data intact", ok)
	}
	println("psram: done")
	for {
		time.Sleep(time.Second)
	}
}
