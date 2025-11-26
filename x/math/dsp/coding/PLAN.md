# Coding Packages Refactoring Plan

## Goal
Unify all coding subpackages to use consistent naming:
- Struct names: `Encoder` / `Decoder` (not `NRZIEncoder`, `HDB3Decoder`, etc.)
- Constructor names: `NewEncoder()` / `NewDecoder()` (not `NewNRZIEncoder()`, etc.)
- Package names: Match directory name (not `ecc`)

## Packages to Refactor

### ✅ Already Unified
- [x] `manchester` - Already uses `Encoder`/`Decoder`, `NewEncoder()`/`NewDecoder()`

### ⏳ To Do
- [x] `gray` - Package `ecc` → `gray`, `GrayEncoder` → `Encoder`, `NewGrayEncoder()` → `NewEncoder()`
- [x] `nrzi` - Package `ecc` → `nrzi`, `NRZIEncoder` → `Encoder`, `NewNRZIEncoder()` → `NewEncoder()`
- [x] `four5b` - Package `ecc` → `four5b`, `FourB5BEncoder` → `Encoder`, `NewFourB5BEncoder()` → `NewEncoder()`
- [x] `eight10b` - Package `ecc` → `eight10b`, `EightB10BEncoder` → `Encoder`, `NewEightB10BEncoder()` → `NewEncoder()`
- [x] `lfsr` - Package `ecc` → `lfsr`, `ScramblerEncoder` → `Encoder`, `NewScramblerEncoder()` → `NewEncoder()`
- [x] `rll` - Package `ecc` → `rll`, `RLLEncoder` → `Encoder`, `NewRLLEncoder()` → `NewEncoder()`
- [x] `miller` - Package `ecc` → `miller`, `MillerEncoder` → `Encoder`, `NewMillerEncoder()` → `NewEncoder()`
- [x] `ami` - Package `ecc` → `ami`, `AMIEncoder` → `Encoder`, `NewAMIEncoder()` → `NewEncoder()`
- [x] `hdb3` - Package `ecc` → `hdb3`, `HDB3Encoder` → `Encoder`, `NewHDB3Encoder()` → `NewEncoder()`
- [x] `b8zs` - Package `ecc` → `b8zs`, `B8ZSEncoder` → `Encoder`, `NewB8ZSEncoder()` → `NewEncoder()`
- [x] `biphase` - Package `ecc` → `biphase`, `BiphaseLEncoder` → `Encoder`, `NewBiphaseLEncoder()` → `NewEncoder()`
- [x] `ppm` - Package `ecc` → `ppm`, `PPMEncoder` → `Encoder`, `NewPPMEncoder()` → `NewEncoder()`
- [x] `diff_manchester` - Moved to separate package, refactored to `Encoder`/`Decoder`

### 🔍 Special Cases
- [ ] `manchester/diff_manchester_*.go` - Check if this should be moved to separate `diff_manchester` package or kept in `manchester` with different constructors

## Progress Log

### Started: [Current Time]
- Created PLAN.md
- Identified all packages to refactor
- Manchester package already unified (good reference)

### Completed: All Bit-Level Encoding Packages
- ✅ Moved `diff_manchester` to separate package
- ✅ Refactored all bit-level encoding packages:
  - `gray`, `nrzi`, `four5b`, `eight10b`, `lfsr`, `rll`, `miller`, `ami`, `hdb3`, `b8zs`, `biphase`, `ppm`, `diff_manchester`
- ✅ All packages now use:
  - Package name matching directory name (not `ecc`)
  - Struct names: `Encoder` / `Decoder` (not `*Encoder`, `*Decoder`)
  - Constructor names: `NewEncoder()` / `NewDecoder()` (not `New*Encoder()`, etc.)
  - Additional constructors preserved with `With*` suffix (e.g., `NewEncoderWithBits()`, `NewEncoderWithParams()`)
- ✅ All packages build successfully
- ✅ Removed duplicate `diff_manchester` files from `manchester` package

### Note
- `crc`, `hamming`, and `rs` packages still use `package ecc` - these are error correction codes, not bit-level encodings, so they may be intentionally left as-is or refactored separately

