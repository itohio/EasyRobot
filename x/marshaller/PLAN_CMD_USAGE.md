# Plan: Command-Line Tool Usage Issues

## Overview
This document identifies issues in `cmd/display`, `cmd/calib_mono`, and `cmd/calib_stereo` related to marshaller/unmarshaller usage, resource management, and API design.

## Issues Identified

### 1. cmd/display Issues

#### 1.1 Inconsistent Release Pattern ✅ FIXED
**Location:** `cmd/display/main.go:199-203`

**Status:** ✅ Fixed

**What was fixed:**
- ✅ Removed manual `t.Release()` calls from main loop
- ✅ Display destination now uses `WithRelease()` option
- ✅ Destinations handle release after consumption

**Fix Required:**
- ✅ Use `WithRelease()` option on destination marshallers - COMPLETED
- ✅ Remove manual `Release()` calls - COMPLETED
- ✅ Let marshallers handle release via `Releaser` interface - COMPLETED

#### 1.2 Context Value Pattern for Cancel Function ✅ FIXED
**Location:** `cmd/display/main.go:80`, `cmd/display/destination/display.go:59`

**Status:** ✅ Fixed

**What was fixed:**
- ✅ Removed `context.WithValue(ctx, "cancel", cancel)` anti-pattern
- ✅ Added `CancelSetter` interface for explicit cancel function passing
- ✅ Destinations now accept cancel function via `SetCancelFunc()` method
- ✅ Display destination uses `WithCancel()` and `WithOnClose()` options

**Fix Required:**
- ✅ Refactor to pass cancel function explicitly - COMPLETED
- ✅ Use proper context cancellation - COMPLETED

#### 1.3 Source/Destination Factory Design
**Location:** `cmd/display/source/factory.go`, `cmd/display/destination/factory.go`

**Problem:**
- Factory pattern with global flags - makes testing difficult
- Hard to create sources/destinations programmatically without flags

**Recommendation:**
- Provide both flag-based and programmatic creation
- Separate flag parsing from object creation
- Add builder pattern for programmatic usage

### 2. cmd/calib_mono Issues

#### 2.1 Manual Resource Management
**Location:** `cmd/calib_mono/main.go:196-226`

**Problem:**
- Manual `mat.Close()` calls throughout code
- Error paths may skip cleanup
- Not using `WithRelease()` option

**Example:**
```go
mat, err := tensorToMat(frame.Tensors[0])
if err != nil {
    continue  // Potential leak
}
defer mat.Close()
```

**Fix Required:**
- Use `WithRelease()` option on source unmarshaller
- Ensure proper cleanup on error paths
- Document resource cleanup patterns

#### 2.2 Calibration File Format Support
**Location:** `cmd/calib_mono/main.go:143`

**Problem:**
- Supports `json`, `yaml`, `gocv` formats
- But calibration saving/loading logic may not use marshallers consistently

**Fix Required:**
- Use marshallers for all calibration file I/O
- Ensure consistent format support
- Add format detection

### 3. cmd/calib_stereo Issues

#### 3.1 Custom Source Implementation
**Location:** `cmd/calib_stereo/main.go:177-306`

**Problem:**
- Custom source wrappers (`customCameraSource`, `customVideoSource`, `customImageSource`)
- Duplicates functionality from `cmd/display/source`
- Not reusing shared source implementation

**Status:** ⚠️ Remaining - Medium priority (code duplication)

**Fix Required:**
- Refactor to use `cmd/display/source` package
- Remove custom source implementations
- Share source creation logic

#### 3.2 Manual Mat Management ✅ FIXED
**Location:** `cmd/calib_stereo/main.go:347-400`, `475-524`

**Status:** ✅ Fixed

**What was fixed:**
- ✅ Changed all manual `mat.Close()` calls to `defer mat.Close()` immediately after creation
- ✅ All error paths now properly clean up Mat objects via defer
- ✅ Display destinations use `WithRelease()` option (via destination.NewAllDestinations())
- ✅ Removed manual cleanup at end of loops - using defer throughout

**Fix Required:**
- ✅ Simplify cleanup with proper defer usage - COMPLETED
- ✅ Use helper functions for common operations - COMPLETED (proper defer usage)

**Note:** Sources are producers and should NOT use `WithRelease()` - destinations (sinks) handle release after consumption.

#### 3.3 Frame Combining Logic
**Location:** `cmd/calib_stereo/main.go:537-567`

**Problem:**
- `combineFramesSideBySide` creates new Mat but caller must close
- Error handling incomplete
- Resource ownership unclear

**Fix Required:**
- Clarify resource ownership
- Add proper error handling
- Document cleanup requirements

### 4. Common Issues Across All Commands

#### 4.1 Event Handling
**Problem:**
- No clear API for handling keyboard/mouse events
- Event handling scattered across marshaller and cmd code
- Hard to implement custom event handlers

**Recommendation:**
- Use `types.KeyEvent` and `types.MouseEvent` interfaces consistently
- Provide examples of event handling
- Document event handling patterns

#### 4.2 Window Management
**Problem:**
- No clear API for window control (resize, move, close)
- Window management tied to display marshaller
- Hard to implement custom window behavior

**Recommendation:**
- Use `types.DisplayWindow` interface consistently
- Provide window management examples
- Document window lifecycle

#### 4.3 Configuration Marshalling
**Problem:**
- Calibration files use different formats
- No standard way to marshal/unmarshal configurations
- Hard to persist and restore settings

**Recommendation:**
- Use marshallers for all configuration I/O
- Provide examples of configuration marshalling
- Standardize configuration file formats

### 5. API Design Issues

#### 5.1 Source/Destination Interface
**Location:** `cmd/display/source/interface.go`, `cmd/display/destination/interface.go`

**Problem:**
- `Source` and `Destination` interfaces are in `cmd/display` package
- Should be in `marshaller/types` for broader reuse
- Makes it hard to share between commands

**Fix Required:**
- Move interfaces to `marshaller/types` if appropriate
- Or document why they're in `cmd/display`
- Ensure consistent API across commands

#### 5.2 Flag Registration Pattern
**Problem:**
- Global flag registration via `RegisterAllFlags()`
- Makes testing difficult
- Hard to create isolated command instances

**Recommendation:**
- Provide flag set parameter option
- Allow programmatic flag definition
- Add flag validation helpers

## Action Items

1. ✅ **Fix Resource Management in cmd/display** - **COMPLETED**
   - ✅ Use `WithRelease()` option consistently - COMPLETED (display sink uses WithRelease())
   - ✅ Remove manual `Release()` calls - COMPLETED (removed from main loop)
   - ✅ Fix context cancellation pattern - COMPLETED (WithCancel/WithOnClose added)

2. **Refactor cmd/calib_stereo**
   - Remove custom source implementations
   - Use `cmd/display/source` package
   - Simplify resource management

3. **Standardize Resource Cleanup**
   - Use `WithRelease()` option on all sources
   - Document cleanup patterns
   - Add examples

4. **Improve Event Handling**
   - Use shared event interfaces consistently
   - Provide event handling examples
   - Document event patterns

5. **Improve Configuration Marshalling**
   - Use marshallers for all config I/O
   - Standardize formats
   - Add examples

6. **Refactor Source/Destination Interfaces**
   - Consider moving to `marshaller/types`
   - Document interface design
   - Ensure consistency

## Priority

- ✅ **High:** Fix resource management and cleanup issues - **COMPLETED**
- **Medium:** Refactor calib_stereo to reuse shared code (remove custom source implementations)
- **Medium:** Improve event handling and window management documentation
- **Low:** Interface refactoring and configuration marshalling improvements
