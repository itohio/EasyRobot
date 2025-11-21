# Plan: Core Marshaller/Unmarshaller API Fixes

## Overview
This document identifies API inconsistencies, constructor naming issues, and interface design problems across all marshaller subpackages.

## Issues Identified

### 1. Constructor Naming Inconsistencies

**Problem:**
- `v4l` uses `New()` instead of `NewUnmarshaller()` (inconsistent with other packages)
- `text` uses `NewMarshaller()` (correct, but only marshals)
- `gob`, `json`, `yaml`, `proto`, `gocv` use `NewMarshaller()` / `NewUnmarshaller()` (consistent)
- `tflite` uses `NewUnmarshaller()` (correct, but only unmarshals)
- `graph` uses `NewMarshaller()` / `NewUnmarshaller()` (consistent)

**Recommendation:**
- Keep `v4l.New()` as per SPEC.md (simpler is better when only one direction)
- text is meant only for marshalling, thus be simpler with text.New instead of text.NewMarshaller
- Verify all constructors return concrete types (not interfaces)

**Status:** ✅ Verified - all constructors return concrete types

### 2. API Design Issues

#### 2.1 WithRelease Option Confusion ✅ FIXED
**Location:** `x/marshaller/gocv/marshaller.go:306`, `file_writer.go:80`, `display_writer.go:566`

**Status:** ✅ Fixed

**What was fixed:**
- Changed `autoRelease` field to `ReleaseAfterProcessing` in `gocv/config.go`
- Updated all references in `gocv/marshaller.go`, `gocv/file_writer.go`, `gocv/display_writer.go`
- Fixed release logic to only check `cfg.ReleaseAfterProcessing` (removed `numWriters == 1` condition)
- Updated `gocv/options.go` to use `ReleaseAfterProcessing` consistently

**Changes made:**
- All `cfg.autoRelease` references changed to `cfg.ReleaseAfterProcessing`
- Release logic now: `if cfg.ReleaseAfterProcessing { ... }` (removed incorrect writer count check)

#### 2.2 Context Handling ✅ FIXED
**Location:** `x/marshaller/gocv/marshaller.go:164-173`, `cmd/display/main.go`, `cmd/display/destination/display.go`

**Status:** ✅ Fixed

**What was fixed:**
- Removed context.Value anti-pattern from `gocv/marshaller.go`
- Removed context.Value usage from `cmd/display/main.go` and `cmd/display/destination/display.go`
- Added `WithCancel()` option to pass cancel function explicitly
- Added `WithOnClose()` callback option to handle window close events properly
- Simplified context handling - destinations derive cancellable contexts from parent
- Removed unused `parentCancel` field and context value extraction code

**Changes made:**
- Context is now passed directly without storing cancel function in values
- Destinations accept cancel function via `CancelSetter` interface
- Added `WithCancel(context.CancelFunc)` option for explicit cancel function passing
- Added `WithOnClose(CloseCallback)` option that receives actual `*cv.Window` and remaining window count
- Removed all `context.WithValue(ctx, "cancel", cancel)` calls

### 3. Missing SPEC.md Files

**Missing:**
- `x/marshaller/gob/SPEC.md` - No specification for Gob marshaller
- `x/marshaller/json/SPEC.md` - No specification for JSON marshaller
- `x/marshaller/yaml/SPEC.md` - No specification for YAML marshaller
- `x/marshaller/proto/SPEC.md` - Has `README.md` but no `SPEC.md`
- `x/marshaller/tflite/SPEC.md` - No specification for TFLite unmarshaller

**Existing:**
- ✅ `x/marshaller/SPEC.md` - General marshaller spec
- ✅ `x/marshaller/types/SPEC.md` - Common types spec
- ✅ `x/marshaller/gocv/SPEC.md` - GoCV marshaller spec
- ✅ `x/marshaller/v4l/SPEC.md` - V4L unmarshaller spec
- ✅ `x/marshaller/graph/SPEC.md` - Graph marshaller spec
- ✅ `x/marshaller/storage/SPEC.md` - Storage spec

### 4. Interface Compliance Issues

**Problem:**
- Some marshallers may not properly implement all interface methods
- Error handling inconsistencies across marshallers

**Recommendation:**
- Add interface conformance tests for all marshallers
- Standardize error wrapping using `types.NewError()`

### 5. Option Pattern Inconsistencies

**Problem:**
- Different marshallers handle options differently
- Some marshallers don't support all common options (e.g., `WithContext`, `WithRelease`)

**Recommendation:**
- All marshallers must accept `WithRelease()` option (even if ignored for non-sink marshallers)
- All marshallers should support `WithContext()` for cancellation
- Document which options are meaningful for which marshallers in their SPEC.md

## Action Items

1. ✅ **Fix `autoRelease` → `ReleaseAfterProcessing` naming** - COMPLETED
   - ✅ Updated `gocv/config.go` to use `ReleaseAfterProcessing` field
   - ✅ Updated all references in `gocv/marshaller.go`, `file_writer.go`, `display_writer.go`
   - ✅ Using `cfg.ReleaseAfterProcessing` consistently

2. ✅ **Fix context handling in `gocv/marshaller.go`** - COMPLETED
   - ✅ Removed context.Value anti-pattern
   - ✅ Added `WithCancel()` option for explicit cancel function passing
   - ✅ Added `WithOnClose()` callback option for window close handling
   - ✅ Simplified context handling - no cancel function extraction needed
   - ✅ Callback receives actual `*cv.Window` for proper decision making

3. **Create missing SPEC.md files**
   - `gob/SPEC.md` - Document Gob marshaller usage, binary format notes
   - `json/SPEC.md` - Document JSON marshaller, graph support, options
   - `yaml/SPEC.md` - Document YAML marshaller, graph support, options
   - `proto/SPEC.md` - Convert `proto/README.md` or create new spec
   - `tflite/SPEC.md` - Document TFLite model loading, inference, options

4. **Add interface conformance tests**
   - Test all marshallers implement `Marshaller` interface correctly
   - Test all unmarshallers implement `Unmarshaller` interface correctly
   - Test error wrapping consistency

5. **Document option support matrix**
   - Create table in `x/marshaller/SPEC.md` showing which options are supported by which marshallers
   - Document option semantics for each marshaller type

## Priority

- **High:** Fix `autoRelease` naming and context handling bugs
- **Medium:** Create missing SPEC.md files
- **Low:** Interface conformance tests and option matrix documentation
