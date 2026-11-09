//go:build esp32s3

#include <stdint.h>

// nnbench_ccount reads the Xtensa CCOUNT special register.
uint32_t nnbench_ccount(void) {
    uint32_t c;
    __asm__ volatile ("rsr.ccount %0" : "=r"(c));
    return c;
}
