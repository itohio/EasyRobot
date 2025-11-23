// +build amd64

#include "textflag.h"

// Constant 1.0 as float32
DATA one<>+0(SB)/4, $0x3f800000  // 1.0
GLOBL one<>(SB), RODATA, $4

// func asmOpDirect(dst, src *float32, n int)
// Computes dst[i] = src[i]*src[i] + 1.0 for i in [0, n)
// Direct implementation without function calls
TEXT ·asmOpDirect(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI    // DI = dst pointer
	MOVQ src+8(FP), SI   // SI = src pointer
	MOVQ n+16(FP), CX   // CX = n (loop counter)
	
	// Load constant 1.0 into XMM0
	MOVSS one<>(SB), X0  // 1.0 as float32
	
	// Check if n == 0
	TESTQ CX, CX
	JZ    done
	
loop:
	// Load src[i] into XMM1
	MOVSS (SI), X1
	
	// Compute src[i] * src[i] into XMM1
	MULSS X1, X1
	
	// Add 1.0: XMM1 = XMM1 + 1.0
	ADDSS X0, X1
	
	// Store result to dst[i]
	MOVSS X1, (DI)
	
	// Advance pointers
	ADDQ $4, SI  // src += 4 bytes (float32 size)
	ADDQ $4, DI  // dst += 4 bytes (float32 size)
	
	// Decrement counter and loop
	DECQ CX
	JNZ  loop
	
done:
	RET

// func asmOpUnrolled(dst, src *float32, n int)
// Unrolled version processing 4 elements at a time
TEXT ·asmOpUnrolled(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI    // DI = dst pointer
	MOVQ src+8(FP), SI   // SI = src pointer
	MOVQ n+16(FP), CX   // CX = n (loop counter)
	
	// Load constant 1.0 into XMM0 (broadcast to all 4 lanes)
	MOVSS one<>(SB), X0  // 1.0 as float32
	SHUFPS $0x00, X0, X0  // Broadcast to all 4 lanes
	
	// Check if n == 0
	TESTQ CX, CX
	JZ    done
	
	// Calculate number of unrolled iterations (n / 4)
	MOVQ CX, AX
	SHRQ $2, AX  // AX = n / 4
	TESTQ AX, AX
	JZ    remainder  // If less than 4 elements, handle remainder
	
	// Unrolled loop: process 4 elements at a time
unrolled_loop:
	// Load 4 elements from src
	MOVUPS (SI), X1
	
	// Compute src[i] * src[i] for all 4 elements
	MULPS X1, X1
	
	// Add 1.0 to all elements
	ADDPS X0, X1
	
	// Store 4 results to dst
	MOVUPS X1, (DI)
	
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
	
	// Load single 1.0 for scalar operations
	MOVSS one<>(SB), X0
	
remainder_loop:
	// Load src[i]
	MOVSS (SI), X1
	
	// Compute src[i] * src[i]
	MULSS X1, X1
	
	// Add 1.0
	ADDSS X0, X1
	
	// Store result
	MOVSS X1, (DI)
	
	// Advance pointers
	ADDQ $4, SI
	ADDQ $4, DI
	
	// Decrement and loop
	DECQ AX
	JNZ  remainder_loop
	
done:
	RET

