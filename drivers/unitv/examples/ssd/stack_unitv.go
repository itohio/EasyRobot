//go:build unitv

package main

import (
	"device/riscv"
	"unsafe"
)

// On the UnitV target TinyGo runs main on the system stack (scheduler
// "none"), and the collector scans the stack conservatively. RAM keeps its
// contents across a reset, so frames built over the previous session's
// stack contents can hold stale words that look like pointers into the new
// heap and keep garbage alive. After a long run and a few resets, the setup
// stopped fitting ("out of memory" before the setup line). clearStack zeroes
// the unused part of the stack once, before anything is allocated.

// The stack lies between _heap_end and _stack_top (drivers/unitv/target/unitv.ld).
//
//go:extern _heap_end
var stackBottom [0]byte

//go:noinline
func clearStack() {
	sp := riscv.AsmFull("mv {}, sp", nil)
	bottom := uintptr(unsafe.Pointer(&stackBottom))
	end := sp - 512 // stay well below this frame
	for p := bottom; p+8 <= end; p += 8 {
		*(*uint64)(unsafe.Pointer(p)) = 0
	}
}
