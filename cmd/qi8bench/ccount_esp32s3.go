//go:build esp32s3

package main

// #include <stdint.h>
// uint32_t qi8bench_ccount(void);
import "C"

// device is true on the ESP32-S3: wait for the USB console, loop forever.
const device = true

// tickUnit names the unit of stamp/ticksSince.
const tickUnit = "cyc"

// stamp returns the Xtensa CCOUNT cycle counter (wraps every 2^32 cycles).
func stamp() uint32 { return uint32(C.qi8bench_ccount()) }

// ticksSince returns the cycles elapsed since t0 (wrapping difference).
func ticksSince(t0 uint32) uint64 { return uint64(stamp() - t0) }
