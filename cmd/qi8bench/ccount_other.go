//go:build !esp32s3

package main

import "time"

// device is false off the chip: print one report and exit.
const device = false

// tickUnit names the unit of stamp/ticksSince (no cycle counter here).
const tickUnit = "ns"

var epoch = time.Now()

// stamp returns monotonic nanoseconds (the low 32 bits suffice per kernel).
func stamp() uint32 { return uint32(time.Since(epoch)) }

// ticksSince returns the nanoseconds elapsed since t0 (wrapping difference).
func ticksSince(t0 uint32) uint64 { return uint64(stamp() - t0) }
