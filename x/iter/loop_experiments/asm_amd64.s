// +build amd64

#include "textflag.h"

// func asmOpLoop(dst, src *float32, n int, op func(float32) float32)
// Calls op(src[i]) for each element and stores result in dst[i]
// op is a function pointer that must be called from Go
TEXT ·asmOpLoop(SB), NOSPLIT, $32-32
	MOVQ dst+0(FP), DI    // DI = dst pointer
	MOVQ src+8(FP), SI   // SI = src pointer
	MOVQ n+16(FP), CX   // CX = n (loop counter)
	MOVQ op+24(FP), DX  // DX = op function pointer
	
	// Check if n == 0
	TESTQ CX, CX
	JZ    done
	
	// Save op function pointer on stack for calling
	MOVQ DX, op-8(SP)
	
loop:
	// Load src[i] into XMM0 (Go calling convention uses XMM0 for float32 args)
	MOVSS (SI), X0
	
	// Call op function: op(src[i])
	// Go function calling convention:
	// - Function pointer in DX (already loaded)
	// - First float32 argument in XMM0 (already set)
	// - Return value in XMM0
	// We need to save registers that might be clobbered
	MOVQ DI, saved_di-16(SP)
	MOVQ SI, saved_si-24(SP)
	MOVQ CX, saved_cx-32(SP)
	
	// Call the function via function pointer
	// In Go, function values are descriptors, so we need to call through the descriptor
	// The function pointer is actually a pointer to a funcval structure
	// We need to load the actual code pointer from the descriptor
	MOVQ (DX), AX        // Load code pointer from funcval
	CALL AX              // Call the function
	
	// Restore registers
	MOVQ saved_di-16(SP), DI
	MOVQ saved_si-24(SP), SI
	MOVQ saved_cx-32(SP), CX
	
	// Result is in XMM0, store to dst[i]
	MOVSS X0, (DI)
	
	// Advance pointers
	ADDQ $4, SI  // src += 4 bytes (float32 size)
	ADDQ $4, DI  // dst += 4 bytes (float32 size)
	
	// Decrement counter and loop
	DECQ CX
	JNZ  loop
	
done:
	RET

// func asmOpLoopUnrolled(dst, src *float32, n int, op func(float32) float32)
// Unrolled version processing 4 elements at a time, calling op for each
TEXT ·asmOpLoopUnrolled(SB), NOSPLIT, $32-32
	MOVQ dst+0(FP), DI    // DI = dst pointer
	MOVQ src+8(FP), SI   // SI = src pointer
	MOVQ n+16(FP), CX   // CX = n (loop counter)
	MOVQ op+24(FP), DX  // DX = op function pointer
	
	// Check if n == 0
	TESTQ CX, CX
	JZ    done
	
	// Save op function pointer on stack
	MOVQ DX, op-8(SP)
	
	// Calculate number of unrolled iterations (n / 4)
	MOVQ CX, AX
	SHRQ $2, AX  // AX = n / 4
	TESTQ AX, AX
	JZ    remainder  // If less than 4 elements, handle remainder
	
	// Unrolled loop: process 4 elements at a time
unrolled_loop:
	// Process 4 elements by calling op for each
	// Element 0
	MOVSS (SI), X0
	MOVQ DI, saved_di-16(SP)
	MOVQ SI, saved_si-24(SP)
	MOVQ AX, saved_ax-32(SP)
	MOVQ (DX), BX
	CALL BX
	MOVQ saved_ax-32(SP), AX
	MOVQ saved_si-24(SP), SI
	MOVQ saved_di-16(SP), DI
	MOVSS X0, (DI)
	
	// Element 1
	MOVSS 4(SI), X0
	MOVQ DI, saved_di-16(SP)
	MOVQ SI, saved_si-24(SP)
	MOVQ AX, saved_ax-32(SP)
	MOVQ (DX), BX
	CALL BX
	MOVQ saved_ax-32(SP), AX
	MOVQ saved_si-24(SP), SI
	MOVQ saved_di-16(SP), DI
	MOVSS X0, 4(DI)
	
	// Element 2
	MOVSS 8(SI), X0
	MOVQ DI, saved_di-16(SP)
	MOVQ SI, saved_si-24(SP)
	MOVQ AX, saved_ax-32(SP)
	MOVQ (DX), BX
	CALL BX
	MOVQ saved_ax-32(SP), AX
	MOVQ saved_si-24(SP), SI
	MOVQ saved_di-16(SP), DI
	MOVSS X0, 8(DI)
	
	// Element 3
	MOVSS 12(SI), X0
	MOVQ DI, saved_di-16(SP)
	MOVQ SI, saved_si-24(SP)
	MOVQ AX, saved_ax-32(SP)
	MOVQ (DX), BX
	CALL BX
	MOVQ saved_ax-32(SP), AX
	MOVQ saved_si-24(SP), SI
	MOVQ saved_di-16(SP), DI
	MOVSS X0, 12(DI)
	
	// Advance pointers by 16 bytes (4 * float32)
	ADDQ $16, SI
	ADDQ $16, DI
	
	// Decrement counter
	DECQ AX
	JNZ  unrolled_loop
	
	// Handle remainder (0-3 elements)
remainder:
	// Calculate remaining elements: n % 4
	MOVQ CX, AX
	ANDQ $3, AX
	TESTQ AX, AX
	JZ   done
	
remainder_loop:
	// Load src[i]
	MOVSS (SI), X0
	
	// Call op function
	MOVQ DI, saved_di-16(SP)
	MOVQ SI, saved_si-24(SP)
	MOVQ AX, saved_ax-32(SP)
	MOVQ (DX), BX
	CALL BX
	MOVQ saved_ax-32(SP), AX
	MOVQ saved_si-24(SP), SI
	MOVQ saved_di-16(SP), DI
	
	// Store result
	MOVSS X0, (DI)
	
	// Advance pointers
	ADDQ $4, SI
	ADDQ $4, DI
	
	// Decrement and loop
	DECQ AX
	JNZ  remainder_loop
	
done:
	RET

