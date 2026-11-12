//go:build esp32s3

#include <stdint.h>

// qi8bench_ccount reads the Xtensa CCOUNT special register.
uint32_t qi8bench_ccount(void) {
    uint32_t c;
    __asm__ volatile ("rsr.ccount %0" : "=r"(c));
    return c;
}
