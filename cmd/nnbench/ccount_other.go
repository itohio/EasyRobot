//go:build !esp32s3 && !k210

package main

import "time"

// device is false on the host: print the report once and exit.
const device = false

// tickUnit names the unit of stamp/ticksSince (no cycle counter on the host).
const tickUnit = "ns"

// ccount has no cycle counter to read off the ESP32-S3: it reports 0 cycles.
func ccount() uint32 { return 0 }

var epoch = time.Now()

// stamp returns the current tick: monotonic nanoseconds (low 32 bits are
// enough for one layer, see ticksSince).
func stamp() uint32 { return uint32(time.Since(epoch)) }

// ticksSince returns the ticks elapsed since t0 (wrapping difference).
func ticksSince(t0 uint32) uint64 { return uint64(stamp() - t0) }

// arenaCap: no cap off-chip (the host uses hostArena).
func arenaCap(n int) int { return n }
