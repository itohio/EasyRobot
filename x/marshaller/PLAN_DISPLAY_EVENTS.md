# Plan: Display and Event Handling System

## Overview
This document identifies issues with the display system, window management, keyboard/mouse event handling, and related APIs in the marshaller subsystem.

## Issues Identified

### 1. Event Loop Architecture

#### 1.1 Global Event Loop
**Location:** `gocv/display_writer.go:81-91`, `226-300`

**Problem:**
- Global event loop with `sync.Once` initialization
- Single OS-thread-locked goroutine for all GUI operations
- Hard to test, hard to shut down cleanly
- No way to have multiple event loops

**Code Structure:**
```go
var (
    eventLoopOnce   sync.Once
    guiCommands     chan GUICommand
    windowsMap      map[string]*windowInfo
    eventLoopCtx    context.Context
    eventLoopCancel context.CancelFunc
    eventLoopWg     sync.WaitGroup
)
```

**Issues:**
- Global state makes testing difficult
- Cannot have multiple independent display systems
- Shutdown order dependencies

**Recommendation:**
- Consider making event loop instance-based rather than global
- Or document why global is necessary
- Add proper shutdown mechanism

#### 1.2 Event Loop Thread Locking
**Location:** `gocv/display_writer.go:230`

**Problem:**
- Event loop goroutine locks to OS thread: `runtime.LockOSThread()`
- Must unlock before exit or thread leaks
- No guarantee unlock happens if goroutine panics

**Fix Required:**
- Ensure `defer runtime.UnlockOSThread()` always executes
- Add panic recovery to ensure cleanup
- Document thread locking requirements

### 2. Event Handling API

#### 2.1 Legacy vs New Event Interfaces
**Location:** `gocv/display_writer.go:62-67`, `types/display.go`

**Problem:**
- Dual event handler support: legacy `func(int) bool` and new `types.KeyEvent`
- Adapter code bridges between formats
- Inconsistent usage across codebase

**Current Code:**
```go
// Legacy handlers (for backward compatibility)
OnKey   func(int) bool
OnMouse func(event, x, y, flags int) bool
// New shared interface handlers
KeyHandler   func(types.KeyEvent) bool
MouseHandler func(types.MouseEvent) bool
```

**Recommendation:**
- Deprecate legacy handlers
- Use only `types.KeyEvent` and `types.MouseEvent`
- Remove adapter code once legacy removed

#### 2.2 Event Delivery Mechanism
**Location:** `gocv/display_writer.go:330-346`

**Problem:**
- Mouse events sent via `windowEvents` channel
- Channel may fill and events dropped silently
- No backpressure mechanism
- Events may be processed out of order

**Fix Required:**
- Add event buffering or backpressure
- Or use callback mechanism instead of channel
- Document event delivery guarantees

#### 2.3 Keyboard Event Processing
**Location:** `gocv/display_writer.go:400-450` (estimated)

**Problem:**
- Keyboard events processed via `WaitKey()` polling
- Polling interval may cause lag
- No async event mechanism
- Limited to single key at a time

**Recommendation:**
- Document event processing limitations
- Consider async event mechanisms if needed
- Add examples of keyboard event handling

### 3. Window Management

#### 3.1 Window Lifecycle
**Location:** `gocv/display_writer.go:302-367`

**Problem:**
- Window creation is asynchronous via command channel
- No guarantee window is ready before first frame
- Ready channel mechanism (`dw.ready`) may race

**Current Flow:**
1. Send `cmdCreateWindow` command
2. Wait for result channel (with timeout)
3. Close ready channel when window created
4. Wait for ready channel in WriteFrame

**Issues:**
- Race condition possible if WriteFrame called before ready
- Timeout handling incomplete
- Error handling unclear

**Fix Required:**
- Document window creation lifecycle
- Ensure proper synchronization
- Add error handling examples

#### 3.2 Window Close Detection
**Location:** `cmd/display/destination/display.go:77-98`

**Problem:**
- Polls window state every 100ms to detect close
- Inefficient polling mechanism
- May miss close event if window closes between polls

**Current Code:**
```go
ticker := time.NewTicker(100 * time.Millisecond)
for {
    select {
    case <-ticker.C:
        if window == nil || !window.IsOpen() {
            // Window closed
        }
    }
}
```

**Recommendation:**
- Use event-based close detection if available
- Or document polling limitation
- Consider callback mechanism

#### 3.3 Window Configuration
**Problem:**
- Window title, size, position configuration scattered
- No unified window configuration API
- Hard to persist/restore window settings

**Recommendation:**
- Use `types.DisplayOptions` consistently
- Add window configuration examples
- Consider window settings persistence

### 4. Display Sink API

#### 4.1 DisplaySink Interface
**Location:** `gocv/sink.go`

**Problem:**
- `DisplaySink` provides high-level API but implementation details leak through
- Event loop management hidden from users
- Hard to understand lifecycle

**Recommendation:**
- Document DisplaySink lifecycle clearly
- Add usage examples
- Explain event loop interaction

#### 4.2 Multiple Window Support
**Problem:**
- Multiple windows supported but lifecycle unclear
- Window ID generation may collide
- No window management API

**Recommendation:**
- Document multi-window usage
- Add window management examples
- Consider window registry or manager

### 5. Integration with cmd/display

#### 5.1 Destination Interface ✅ FIXED
**Location:** `cmd/display/destination/display.go`

**Status:** ✅ Fixed

**What was fixed:**
- ✅ Removed `context.WithValue` anti-pattern
- ✅ Added `CancelSetter` interface for explicit cancel function passing
- ✅ Destinations now accept cancel function via `SetCancelFunc()` method
- ✅ Display destination uses `WithCancel()` and `WithOnClose()` options

**Fix Required:**
- Clarify destination lifecycle (documentation needed)
- ✅ Fix context cancellation pattern - COMPLETED
- Document event loop interaction (documentation needed)

#### 5.2 Display Disable Flag
**Location:** `cmd/display/destination/display.go:48-52`

**Problem:**
- `--no-display` flag disables display but destination still created
- Inefficient to create display infrastructure when disabled

**Recommendation:**
- Don't create display destination if `--no-display` set
- Or document why destination is created anyway
- Consider lazy initialization

### 6. Event Handling Examples

#### 6.1 Missing Examples
**Problem:**
- No examples of keyboard event handling
- No examples of mouse event handling
- No examples of window event handling
- Hard to understand how to implement custom handlers

**Recommendation:**
- Add examples to `gocv/SPEC.md`
- Add examples to `cmd/display/README.md`
- Show common event handling patterns

#### 6.2 Event Handler Registration
**Problem:**
- Event handler registration via options not well documented
- Hard to register handlers after window creation
- No way to unregister handlers

**Recommendation:**
- Document handler registration patterns
- Add handler management API if needed
- Show examples of dynamic handler registration

## Action Items

1. **Fix Event Loop Architecture**
   - Document global event loop design decision
   - Add proper shutdown mechanism
   - Ensure thread unlock on exit

2. **Standardize Event Interfaces**
   - Deprecate legacy event handlers
   - Use only `types.KeyEvent` and `types.MouseEvent`
   - Remove adapter code

3. **Improve Window Management**
   - Document window lifecycle clearly
   - Fix window creation synchronization
   - Add window management examples

4. **Add Event Handling Examples**
   - Add examples to SPEC.md files
   - Show common patterns
   - Document event delivery guarantees

5. **Improve Display Sink API**
   - Document lifecycle clearly
   - Add usage examples
   - Explain event loop interaction

6. **Fix cmd/display Integration** ✅ MOSTLY COMPLETED
   - ✅ Fix context cancellation pattern - COMPLETED (WithCancel/WithOnClose added)
   - Clarify destination lifecycle (documentation needed)
   - Document event loop interaction (documentation needed)

## Priority

- **High:** Fix event loop shutdown and thread unlocking
- **High:** Standardize event interfaces and remove legacy code
- **Medium:** Improve window management and event delivery
- **Medium:** Add examples and documentation
- **Low:** Refactor event loop architecture (if needed)
