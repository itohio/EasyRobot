// +build amd64

#include "textflag.h"

// func asmMatMul(C, A, B *float32, m, k, n int)
// Matrix multiplication: C = A * B
// A is m×k, B is k×n, C is m×n
// Uses optimized inner loop with SSE
TEXT ·asmMatMul(SB), NOSPLIT, $0-48
	MOVQ C+0(FP), DI    // DI = C pointer
	MOVQ A+8(FP), SI   // SI = A pointer
	MOVQ B+16(FP), DX  // DX = B pointer
	MOVQ m+24(FP), R8  // R8 = m
	MOVQ k+32(FP), R9  // R9 = k
	MOVQ n+40(FP), R10 // R10 = n
	
	// Check if any dimension is 0
	TESTQ R8, R8
	JZ    done
	TESTQ R9, R9
	JZ    done
	TESTQ R10, R10
	JZ    done
	
	// Outer loop: i = 0 to m-1
	XORQ R11, R11      // R11 = i = 0
i_loop:
	// Calculate base index for row i in A and C
	MOVQ R11, AX
	IMULQ R9, AX       // AX = i * k (base in A)
	MOVQ AX, R12       // R12 = i * k (save for A)
	
	MOVQ R11, AX
	IMULQ R10, AX      // AX = i * n (base in C)
	MOVQ AX, R13       // R13 = i * n (save for C)
	
	// Initialize row of C to zero
	MOVQ R10, CX       // CX = n
	XORQ BX, BX        // BX = j = 0
init_loop:
	XORPS X0, X0       // X0 = 0.0
	MOVQ R13, AX       // AX = i * n
	ADDQ BX, AX        // AX = i * n + j
	MOVSS X0, (DI)(AX*4)  // C[i*n + j] = 0
	INCQ BX
	DECQ CX
	JNZ  init_loop
	
	// Middle loop: l = 0 to k-1
	XORQ R14, R14      // R14 = l = 0
l_loop:
	// Load A[i][l]
	MOVQ R12, AX       // AX = i * k
	ADDQ R14, AX       // AX = i * k + l
	MOVSS (SI)(AX*4), X0  // X0 = A[i*k + l]
	
	// Broadcast A[i][l] to all 4 lanes for SIMD
	SHUFPS $0x00, X0, X0
	
	// Inner loop: j = 0 to n-1
	// Process 4 elements at a time if possible
	MOVQ R10, CX       // CX = n
	SHRQ $2, CX        // CX = n / 4
	TESTQ CX, CX
	JZ   j_remainder
	
	// Calculate base index for B: l * n
	MOVQ R14, AX       // AX = l
	IMULQ R10, AX      // AX = l * n (base in B)
	
	MOVQ R13, BX       // BX = i * n (base in C)
	XORQ R15, R15      // R15 = j counter = 0
	
j_loop_4:
	// Calculate B index: l*n + j
	MOVQ R14, AX       // AX = l
	IMULQ R10, AX      // AX = l * n
	ADDQ R15, AX       // AX = l * n + j
		
	// Calculate C index: i*n + j
	MOVQ R13, BX       // BX = i * n
	ADDQ R15, BX       // BX = i * n + j
		
	// Load 4 elements from B
	MOVUPS (DX)(AX*4), X1  // X1 = B[l*n + j:j+4]
	
	// Multiply: X1 = A[i][l] * B[l][j:j+4]
	MULPS X0, X1
	
	// Load 4 elements from C
	MOVUPS (DI)(BX*4), X2  // X2 = C[i*n + j:j+4]
	
	// Add: C[i][j:j+4] += A[i][l] * B[l][j:j+4]
	ADDPS X1, X2
	
	// Store back to C
	MOVUPS X2, (DI)(BX*4)
	
	// Advance j by 4
	ADDQ $4, R15       // j += 4
	
	// Check if more iterations
	DECQ CX
	JNZ  j_loop_4
	
	// Handle remainder (0-3 elements)
j_remainder:
	MOVQ R10, CX       // CX = n
	ANDQ $3, CX        // CX = n % 4
	TESTQ CX, CX
	JZ   l_loop_end
	
	// Calculate starting j for remainder
	MOVQ R10, AX       // AX = n
	ANDQ $0xFFFFFFFC, AX  // AX = n & ~3 (round down to multiple of 4)
	MOVQ AX, R15       // R15 = j (starting position)
	
remainder_loop:
	// Calculate indices
	MOVQ R14, AX       // AX = l
	IMULQ R10, AX      // AX = l * n
	ADDQ R15, AX       // AX = l * n + j
	
	MOVQ R13, BX       // BX = i * n
	ADDQ R15, BX       // BX = i * n + j
	
	// Load A[i][l] (scalar) - reload since X0 was used for SIMD
	MOVQ R12, AX       // AX = i * k
	ADDQ R14, AX       // AX = i * k + l
	MOVSS (SI)(AX*4), X0  // X0 = A[i*k + l]
	
	// Recalculate B index
	MOVQ R14, AX       // AX = l
	IMULQ R10, AX      // AX = l * n
	ADDQ R15, AX       // AX = l * n + j
	
	// Load B[l][j]
	MOVSS (DX)(AX*4), X1  // X1 = B[l*n + j]
	
	// Multiply: X1 = A[i][l] * B[l][j]
	MULSS X0, X1
	
	// Load C[i][j]
	MOVSS (DI)(BX*4), X2  // X2 = C[i*n + j]
	
	// Add: C[i][j] += A[i][l] * B[l][j]
	ADDSS X1, X2
	
	// Store back to C
	MOVSS X2, (DI)(BX*4)
	
	INCQ R15           // j++
	DECQ CX
	JNZ  remainder_loop
	
l_loop_end:
	// Next l
	INCQ R14           // l++
	CMPQ R14, R9       // l < k?
	JL   l_loop
	
	// Next i
	INCQ R11           // i++
	CMPQ R11, R8       // i < m?
	JL   i_loop
	
done:
	RET
