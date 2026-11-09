//go:build esp32s3

package main

// #include <stdint.h>
// uint32_t nnbench_ccount(void);
import "C"

// device is true on the ESP32-S3: wait for the USB console, loop forever.
const device = true

// tickUnit names the unit of stamp/ticksSince.
const tickUnit = "cyc"

// ccount returns the Xtensa CCOUNT cycle counter (wraps every 2^32 cycles).
func ccount() uint32 { return uint32(C.nnbench_ccount()) }

// stamp returns the current tick (CCOUNT).
func stamp() uint32 { return ccount() }

// ticksSince returns the ticks elapsed since t0 (wrapping difference).
func ticksSince(t0 uint32) uint64 { return uint64(ccount() - t0) }

// arenaCap: no cap on the S3 (the whole free heap minus heapReserve).
func arenaCap(n int) int { return n }
