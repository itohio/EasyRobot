//go:build k210

package main

import "device/riscv"

// device is true on the K210: loop forever, the report repeats.
const device = true

// tickUnit names the unit of stamp/ticksSince.
const tickUnit = "cyc"

// ccount returns the low 32 bits of the RISC-V cycle counter (CPU clock).
func ccount() uint32 { return uint32(riscv.CYCLE.Get()) }

// stamp returns the current tick.
func stamp() uint32 { return ccount() }

// ticksSince returns the ticks elapsed since t0 (wrapping difference).
func ticksSince(t0 uint32) uint64 { return uint64(ccount() - t0) }

// arenaCap caps the arena: the K210 heap is ~6 MiB, and an arena of all of
// it minus heapReserve left too little for MobileNet's setup (out of memory,
// K7 attempt 1). 4 MiB is 12x what the largest model uses.
func arenaCap(n int) int { return min(n, 4<<20) }
