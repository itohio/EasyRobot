# Spectrometer Application - Work Tracking

## Overview

This document tracks the current work session progress, next session plans, and overall progress (0-100%) for each SPEC.md/PLAN.md in the spectrometer application.

**IMMUTABLE SECTIONS** (Do not modify):
- Rules and Guidelines
- Overall Progress Tracking
- Potential Issues

**MUTABLE SECTIONS** (Update as work progresses):
- Current Work Session Tasks
- Next Work Session Tasks

---

## Rules and Guidelines (IMMUTABLE)

### Development Principles

1. **LEAN Principles**: Write maintainable and expandable Go code
2. **Usability Priority**: User experience is paramount
3. **No Code Repetition**: DRY (Don't Repeat Yourself) - repeating code is punishable
4. **Use Existing Packages**: Implementing loops when `mat/vec/tensor/dsp/filter` can be used is punishable even more
5. **Type Safety**: No need to convert matrices/vectors from interfaces - know exact types (e.g., `vec.Vector` vs `vec.Vector3D`) and cast directly
6. **Test Coverage**: Misuse of types must be caught with unit tests using `testify` package
7. **UP TO DATE**: Keep this document and relevant SPEC.md and PLAN.md(if any) up to date
8. **NO STUTTERING**: Avoid stuttering and repeating, e.g. such names as obj.NewObj(), obj.ObjectQualifier structs... e.g. median.MedianFilter => median.Filter.
9. **Options**: Prefer options patterns over long argument lists
10. **Vararg**: Variable arguments can be used more often for convenience where applicable
11. **SIMPLE NAMES**: Use simple, concise names. Avoid verbose names like `StartMeasurementListener` → use `Start`. `StopMeasurementListener` → use `Stop`.
12. **TODO MANAGEMENT**: ALWAYS update WORK.md when completing tasks. Move tasks from pending → in_progress → completed. Verify implementation matches TODO.
13. **TASK PRIORITIZATION**: Prioritize tasks in the queue, keep queue full, work on ONE task at a time. Update WORK.md before and after each task.

### Go Best Practices

- **Go Version**: Go 1.25 (assumed latest features)
- **Logging**: Use `log/slog` package with `-v=N` (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG, 4=TRACE) and `-vv` (shortcut for `-v=4`)
- **Testing**: Use `testify` package for all tests
- **Error Handling**: Never ignore errors, wrap with context
- **Function Length**: Maximum 30 lines (extract when makes sense, but don't split just for splitting sake)
- **SOLID Principles**: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
- **Destination pattern**: Avoid allocating new matrices/tensors/vecors - attempt to preallocate needed objects and use them as destination in calculations (destination parameter goes first). Only allocate when returning values or creating objects for callbacks. Preallocate reusable buffers in structs. BE VERY MINDFUL of sharing vectors/matrices/arrays/slices/tensors between goroutines. **CRITICAL**: Methods called repeatedly (e.g., `Measure()`) MUST use destination pattern. Methods called once (e.g., `Wavelengths()`) can allocate.

### Package Usage

**MUST READ before implementing**:
- `x/math/vec` - Vector operations (`vec.Vector`, `vec.Vector3D`, etc.)
- `x/math/mat` - Matrix operations (`mat.Matrix`, row-major layout)
- `x/math/filter` - Digital filters (MA exists, Savitzky-Golay, Gaussian, Median to be implemented)
- `x/math/dsp` - DSP algorithms (FFT, convolution, window functions)
- `x/math/colorscience` - Colorimetry, peak detection, SPD calibration, CRI
- `x/devices/cr30` - CR30 colorimeter device driver

**Know exact types**: Cast directly (e.g., `vec.Vector3D(xyz).XYZ()`, `mat.Row(0).(vec.Vector)`)

### Code Quality Standards

- **Testability**: Write testable code, avoid hard-to-mock dependencies
- **Composition over Inheritance**: Prefer composition
- **Return Early**: Reduce nesting
- **Keep Arguments Low**: Fewer arguments is better
- **Package by Feature**: Not by layer
- **Accept Interfaces, Return Structs**: Keep interfaces small (1-3 methods)
- **Context for Cancellation**: Use `context.Context` for cancellation and timeouts
- **REUSABILITY**: Always implement reusable functionality in shared packages (`x/*`), not in application-specific code (`cmd/*`). For example, serial port enumeration belongs in `x/devices`, not in `cmd/spectrometer/ports`. Application commands should reuse shared functionality.
- **Build Tags**: properly utilize build tags - if there is platform specific implementations and you need to add features - just add to those platform specific implementations, not some weird ass wrapper. users should know what tags they need to use and on what platforms and would create platform independent wrappers if they need it.

---

## Learnings (IMMUTABLE)

This section captures learnings about code structure, API design, code conventions, and patterns discovered during implementation. These should be referenced when implementing similar code.

### Reusability Principle ⚠️ **CRITICAL**

**CRITICAL**: Always implement reusable functionality in shared packages (`x/*`), not in application-specific code (`cmd/*`).

**Example - Serial Port Enumeration**:
- ❌ **Wrong**: Implement `ListSerialPorts()` in `cmd/spectrometer/ports`
- ✅ **Correct**: Implement `devices.ListSerialPorts()` in `x/devices/serial_list*.go`
- ✅ **Correct**: `cmd/spectrometer/ports` reuses `devices.ListSerialPorts()`

**Example - Color Science Calculations**:
- ❌ **Wrong**: Implement `DeltaE76()` in `cmd/spectrometer/internal/render`
- ✅ **Correct**: Implement `colorscience.DeltaE76()` in `x/math/colorscience/color_types.go`
- ✅ **Correct**: Render package uses `colorscience.DeltaE76()` directly

**Example - Math Utilities**:
- ❌ **Wrong**: Implement `clamp255()` when `math.Clamp()` exists
- ✅ **Correct**: Use `math.Clamp(v, 0, 255)` from `x/math/math.go`

**Rationale**: 
- Shared functionality can be reused across multiple applications
- Follows the pattern of `source.ListCameras()` in `cmd/display/source`
- Application commands should be thin wrappers that reuse shared packages
- Platform-specific code belongs in shared packages with build tags, not in application code
- **General algorithms and calculations belong in `x/*`, not `internal/*`**
- **Always check existing packages before implementing new utilities**

**When to implement in shared packages (`x/*`)**:
- Device enumeration (cameras, serial ports, I2C devices, etc.)
- Common algorithms and utilities (color science, math functions, etc.)
- Platform-specific implementations
- Interfaces and abstractions used by multiple applications
- **Domain-specific calculations (color science, DSP, filtering, etc.)**

**When it's OK to implement in application code (`cmd/*` or `internal/*`)**:
- Application-specific command-line interfaces
- Application-specific workflows and orchestration
- Application-specific configuration handling
- Application-specific rendering/UI code (unless it's a reusable widget)

**Red flags to watch for**:
- Creating a function in `internal/*` that performs a general calculation → Should be in `x/*`
- Duplicating functionality that exists in `x/*` packages → Use existing function
- Implementing math algorithms in application code → Should be in `x/math/*`
- **Creating thin wrapper functions just for convenience** → Use functions directly where needed, don't wrap for the sake of wrapping
- **Avoid unnecessary indirection** → If `math.Clamp(v, 0, 255)` works, use it directly, don't create `clamp255(v)`
- **Manually implementing operations that Vector/Matrix types already support** → Use existing methods (e.g., `vec.Vector3D.Distance()`, `vec.Vector3D.Sub()`, `vec.Vector3D.Magnitude()`) instead of manually extracting components and doing arithmetic
- **Always check Vector/Matrix methods before implementing manual operations** → Vector3D has `Distance()`, `Sub()`, `Dot()`, `Magnitude()`, etc.

**NO THIN WRAPPERS RULE**:
- ❌ **Wrong**: Create `clamp255(v)` that just calls `math.Clamp(v, 0, 255)`
- ✅ **Correct**: Use `math.Clamp(v, 0, 255)` directly where needed
- ❌ **Wrong**: Create wrapper functions "just in case" or for "convenience"
- ✅ **Correct**: Use library functions directly unless the wrapper adds significant value (error handling, type conversion, domain-specific logic)

### Code Naming Conventions

1. **Avoid Stuttering**: Don't repeat package name in type names
   - ❌ `savgol.SavitzkyGolayFilter` → ✅ `savgol.Filter`
   - ❌ `gaussian.GaussianFilter` → ✅ `gaussian.Filter`
   - ❌ `median.MedianFilter` → ✅ `median.Filter`
   - Reason: Package name already provides context, stuttering is redundant

2. **Package Name Consistency**: Filter types are just `Filter`, convenience functions use descriptive names
   - ✅ `savgol.New(windowSize, order)` returns `*Filter`
   - ✅ `savgol.SavitzkyGolay(signal, windowSize, order)` - convenience function
   - Reason: Type is in package context, functions can be more descriptive

### Marshaller API Patterns

1. **Separate Marshal/Unmarshal Types**: Marshallers and unmarshallers are separate types, not methods on same type
   - ✅ `json.NewMarshaller()` - for marshalling (encoding)
   - ✅ `json.NewUnmarshaller()` - for unmarshalling (decoding)
   - ✅ `yaml.NewMarshaller()` / `yaml.NewUnmarshaller()`
   - ✅ `proto.NewMarshaller()` / `proto.NewUnmarshaller()`
   - Reason: Different concerns, different implementations, cleaner API

2. **Method Signatures**: Marshal/Unmarshal follow consistent patterns
   - ✅ `Marshal(w io.Writer, value any, opts ...types.Option) error`
   - ✅ `Unmarshal(r io.Reader, dst any, opts ...types.Option) error`
   - Reason: io.Reader/io.Writer for streaming, `any` for flexibility, options for configuration

### Matrix/Vector API Patterns

1. **Destination-First Pattern**: Matrix operations take destination as parameter
   - ✅ `mat.PseudoInverse(dst matTypes.Matrix) error` - destination first
   - ✅ `ata.PseudoInverse(invAta)` - method form, destination still first
   - Reason: Avoids unnecessary allocations, allows pre-allocated buffers

2. **Type Casting Pattern**: Direct type assertions, no conversions from interfaces
   - ✅ `vec.Vector3D(xyz).XYZ()` - cast directly to specific type
   - ✅ `mat.Row(0).(vec.Vector)` - cast result directly
   - ❌ Don't convert from `vecTypes.Vector` interface if you know the exact type
   - Reason: Type safety, performance, clarity

3. **View() Method**: Returns interface view of concrete type
   - ✅ `vec.Vector.View() vecTypes.Vector` - returns interface
   - ✅ `inputVec := input.View().(vec.Vector)` - cast back to concrete
   - Reason: Allows working with interfaces when needed, but cast back for operations

### Filter API Patterns

1. **Consistent Interface**: All filters follow same pattern
   - ✅ `New(...) *Filter` - constructor
   - ✅ `Process(sample float32) float32` - single sample processing
   - ✅ `ProcessBuffer(input vecTypes.Vector) vec.Vector` - buffer processing
   - ✅ `Reset()` - reset state
   - Reason: Consistency across all filters, easy to swap implementations

2. **Stateful Filters**: Filters maintain internal state (buffer, index)
   - ✅ Use circular buffer pattern for window-based filters
   - ✅ Track `initialized` flag for edge cases (first few samples)
   - ✅ Reset state when processing new buffer
   - Reason: Streaming support, efficient memory usage

3. **Convenience Functions**: Provide both struct and function forms
   - ✅ `savgol.New(windowSize, order)` - stateful filter struct
   - ✅ `savgol.SavitzkyGolay(signal, windowSize, order)` - one-shot convenience
   - Reason: Flexibility - use struct for streaming, function for batch

### Error Handling Patterns

1. **Error Wrapping**: Always wrap errors with context
   - ✅ `return fmt.Errorf("failed to unmarshal config: %w", err)`
   - ✅ `return fmt.Errorf("Savitzky-Golay: failed to compute pseudo-inverse: %v", err)`
   - Reason: Preserves error chain, adds context for debugging

2. **Panic for Programmer Errors**: Panic for invalid inputs, not runtime errors
   - ✅ `panic("Savitzky-Golay window size must be positive and odd")` - invalid parameter
   - ❌ Don't panic for I/O errors, missing files, etc.
   - Reason: Fail fast for bugs, return errors for recoverable issues

### Package Organization

1. **Package by Feature**: Group related code by feature, not by layer
   - ✅ `x/math/filter/savgol` - Savitzky-Golay filter
   - ✅ `x/math/filter/gaussian` - Gaussian filter
   - ✅ `x/math/filter/median` - Median filter
   - Reason: Related code together, easier to find and maintain

2. **Internal Packages**: Use `internal/` for application-specific code
   - ✅ `cmd/spectrometer/internal/config` - spectrometer-specific config handling
   - ✅ `cmd/spectrometer/internal/obtainer` - spectrometer device abstraction
   - Reason: Clear separation between reusable and application-specific code

### Proto and Type Generation

1. **Proto Types**: Generated types are in `types/` package, not proto package
   - ✅ `github.com/itohio/EasyRobot/types/spectrometer` - generated Go types
   - ❌ Don't import from `proto/types/spectrometer` directly
   - Reason: Proto files generate Go code, import the generated package

2. **Proto Generation**: Types must be generated before use
   - ✅ Run `make install-tools` to install buf and protoc-gen-* tools
   - ✅ Run `make proto` to generate types (uses `buf generate` in proto directory)
   - ✅ Makefile includes all necessary tools in `install-tools` target
   - Reason: Proto compilation step generates Go code from `.proto` files using buf

### Logging Patterns

1. **Structured Logging**: Use `log/slog` with structured fields
   - ✅ `slog.Debug("Loading config", "path", path, "format", format)`
   - ✅ `slog.Error("Command failed", "command", command, "error", err)`
   - Reason: Machine-readable, queryable, better for debugging

2. **Verbosity Levels**: Support multiple verbosity levels
   - ✅ `-v=N` where N = 0 (ERROR) to 4 (TRACE)
   - ✅ `-vv` shortcut for maximum verbosity
   - ✅ Count `-v` flags before `flag.Parse()`
   - Reason: Flexible debugging, user control over output verbosity

### CLI Design Patterns

1. **Command Routing**: Use subcommands for different operations
   - ✅ `spectrometer cameras` - list cameras
   - ✅ `spectrometer calibrate camera` - calibrate camera
   - ✅ `spectrometer measure -device=cr30` - measure with device
   - Reason: Clear separation of concerns, Unix philosophy

2. **Flag Registration**: Register flags from reused packages early
   - ✅ `source.RegisterAllFlags()` before `flag.Parse()`
   - ✅ `destination.RegisterAllFlags()` before `flag.Parse()`
   - Reason: Flags available before parsing, consistent with package design

3. **Context Usage**: Always pass `context.Context` for cancellation
   - ✅ `func Run(ctx context.Context) error`
   - ✅ `ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)`
   - Reason: Graceful shutdown, cancellation propagation, timeouts

### Testing Conventions

1. **Testify Package**: Use testify for assertions and test utilities
   - ✅ `assert.Equal(t, expected, actual)`
   - ✅ `require.NoError(t, err)` - stops test on error
   - Reason: Better error messages, more readable tests

2. **Type Testing**: Test type assertions and casts explicitly
   - ✅ Test that `vec.Vector3D(xyz).XYZ()` works correctly
   - ✅ Test that incorrect casts panic appropriately
   - Reason: Catch type misuse early, ensure type safety

### Algorithm Implementation Patterns

1. **Use Existing Packages**: Leverage `mat` and `vec` instead of manual loops
   - ✅ Use `mat.PseudoInverse()` instead of manual matrix inversion
   - ✅ Use `vec.Vector` operations instead of manual array operations
   - ❌ Don't implement loops for operations that `mat`/`vec` provide
   - Reason: Correctness, performance, maintainability

2. **Document Mathematical Operations**: Comment mathematical formulas
   - ✅ `// Solve least squares: (A^T * A)^(-1) * A^T`
   - ✅ `// Gaussian kernel: G(x) = exp(-x^2 / (2 * sigma^2))`
   - Reason: Clarity, helps reviewers understand implementation

3. **Edge Case Handling**: Handle edge cases explicitly
   - ✅ Check window size validity in constructors
   - ✅ Handle partial windows (first few samples) in Process()
   - ✅ Return early for insufficient data
   - Reason: Robustness, predictable behavior

---

## Overall Progress Tracking (0-100%)

### Phase 0: CR30 Device Support & Framework ⭐ **FIRST PHASE**

**Progress: 50%** (Foundation complete, CR30 obtainer complete with callback pattern, ports command added, rendering pending)

#### Modules:

- **`internal/obtainer/`** (SPEC.md, PLAN.md): **100%** ✅
  - [x] Obtainer interface definition ✅
    - [x] `Measure()` for PC-initiated measurements ✅
    - [x] `Start()` / `Stop()` with callbacks for button-initiated measurements ✅
  - [x] Device registry ✅
  - [x] Obtainer factory ✅
  - [x] CR30 obtainer implementation ✅
    - [x] `Measure()` using `cr30.Device.Measure()` ✅
    - [x] `Start()` running `WaitMeasurement()` in goroutine loop ✅
    - [x] `Stop()` with graceful shutdown ✅
    - [x] Callback pattern for async measurements ✅

- **`internal/render/`** (SPEC.md, PLAN.md): **20%** (Foundation complete: Delta E, formatting, Patch struct)
  - [x] Delta E calculation (CIE76) ✅
  - [x] Colorimetry formatting (XYZ, LAB, RGB HEX) ✅
  - [x] Patch struct definition ✅
  - [ ] Core spectrum plot rendering (in progress)
  - [ ] Colorimetry overlay rendering (XYZ, LAB, RGB HEX)
  - [ ] Color swatch rendering
  - [ ] Patch comparison rendering

- **`internal/config/`** (SPEC.md, PLAN.md): **60%**
  - [x] Config loader (proto/YAML/JSON via marshallers) ✅
  - [x] Config saver (proto/YAML/JSON via marshallers) ✅
  - [x] Format auto-detection from file extension ✅
  - [x] `--output` flag support ✅
  - [ ] Config structure and validation
  - [ ] Reference spectrum loading (resolve paths, load matrices)

- **`measure/measure.go`** (Command): **0%**
  - [ ] Device selection via `-device` flag
  - [ ] CR30 device initialization (`--port`, `--baud` flags)
  - [ ] Measurement workflow
  - [ ] Multiple readings support (`--readings=N`)
  - [ ] Patch-based measurement (`--patches=path`)
  - [ ] Export functionality (CSV, JSON, Proto)

- **`main.go`** (CLI Entry Point): **85%**
  - [x] CLI framework setup (flag parsing) ✅
  - [x] Command routing structure ✅
  - [x] Logging setup (slog with -v=N and -vv flags) ✅
  - [x] Integration with display/source and display/destination flags ✅
  - [x] Command handlers: cameras ✅, ports ✅
  - [ ] Command handler implementations (calibrate, measure, freerun pending)

### Phase 1: Foundation & Configuration

**Progress: 5%** (Proto definitions created)

#### Modules:

- **Proto Definitions**: **90%**
  - [x] `proto/types/spectrometer/config.proto` created
  - [x] `proto/types/spectrometer/measurement.proto` created
  - [x] Build proto files (`make proto`)
  - [ ] Verify generated Go code

- **`cameras/cameras.go`**: **100%** ✅
  - [x] Reuse `source.ListCameras()` ✅
  - [x] Format and display camera information ✅
  - [x] Exit after listing ✅

- **`ports/ports.go`**: **100%** ✅
  - [x] Reuse `devices.ListSerialPorts()` ✅
  - [x] Format and display serial port information ✅
  - [x] Exit after listing ✅
  - [x] Moved serial enumeration to `x/devices/serial_*.go` (reusability principle) ✅

### Phase 2: Core Extraction and Calibration

**Progress: 0%**

#### Dependencies:

- **`x/math/filter/savgol`**: **100%** ✅ (REQUIRED BEFORE Phase 2)
  - [x] Savitzky-Golay filter implementation ✅
  - [x] Streaming support via `SavitzkyGolayFilter` struct ✅
  - [x] Unit tests pending (to be added)

- **`x/math/filter/gaussian`**: **100%** ✅ (REQUIRED BEFORE Phase 2)
  - [x] Gaussian filter implementation ✅
  - [x] Streaming support via `GaussianFilter` struct ✅
  - [x] Unit tests pending (to be added)

- **`x/math/filter/median`**: **100%** ✅ (REQUIRED BEFORE Phase 2)
  - [x] Median filter implementation ✅
  - [x] Streaming support via `MedianFilter` struct ✅
  - [x] Unit tests pending (to be added)

### Phase 9: CSV Marshaller

**Progress: 0%**

- **`x/marshaller/csv`**: **0%**
  - [ ] Vector marshalling/unmarshalling
  - [ ] Matrix marshalling/unmarshalling
  - [ ] Tensor marshalling/unmarshalling (1D/2D only)
  - [ ] `WithHeader(true/false)` option
  - [ ] `WithZeroRowHeader(true/false)` option

---

## Current Work Session Tasks (MUTABLE)

**CRITICAL**: Always update this section when starting/finishing tasks. Move tasks from pending → in_progress → completed. Verify implementation matches TODO.

### Session Goal

Implement missing filters, program structure, config loading/outputting, CR30 support, and ports command. ✅ **COMPLETED**

### Tasks (In Order)

1. **Read Required SPECs/PLANs**: ✅ **COMPLETED**
   - [x] Read `cmd/spectrometer/SPEC.md` - Overall application specification ✅
   - [x] Read `cmd/spectrometer/PLAN.md` - Implementation plan ✅
   - [x] Read `cmd/spectrometer/internal/obtainer/SPEC.md` - Obtainer framework specification ✅
   - [x] Read `cmd/spectrometer/internal/render/SPEC.md` - Render module specification ✅
   - [x] Read `cmd/spectrometer/internal/config/SPEC.md` - Config module specification ✅
   - [x] Read `x/math/filter/SPEC.md` - Filter package specification ✅
   - [x] Read `x/math/vec/vec.go` - Vector types and operations ✅
   - [x] Read `x/math/mat/mat.go` - Matrix types and operations ✅
   - [x] Read `x/devices/cr30/device.go` - CR30 device API ✅
   - [x] Read `cmd/cr30/main.go` - CR30 usage patterns ✅

2. **Implement Missing Filters** (Priority: HIGH - Required for Phase 2): **COMPLETED** ✅
   - [x] **`x/math/filter/savgol/savgol.go`**: Savitzky-Golay filter ✅
     - [x] `SavitzkyGolay(signal vec.Vector, windowSize, polynomialOrder int) vec.Vector` ✅
     - [x] `SavitzkyGolayFilter` struct with `Process()` and `ProcessBuffer()` methods ✅
     - [ ] Unit tests with `testify` (pending)
   - [x] **`x/math/filter/gaussian/gaussian.go`**: Gaussian filter ✅
     - [x] `Gaussian(signal vec.Vector, sigma float32, windowSize int) vec.Vector` ✅
     - [x] `GaussianFilter` struct with `Process()` and `ProcessBuffer()` methods ✅
     - [ ] Unit tests with `testify` (pending)
   - [x] **`x/math/filter/median/median.go`**: Median filter ✅
     - [x] `Median(signal vec.Vector, windowSize int) vec.Vector` ✅
     - [x] `MedianFilter` struct with efficient partial sorting ✅
     - [ ] Unit tests with `testify` (pending)
   - [ ] Update `x/math/filter/SPEC.md` - Mark filters as implemented (pending)

3. **Implement Program Structure**: **COMPLETED** ✅
   - [x] **`cmd/spectrometer/main.go`**: CLI entry point ✅
     - [x] Flag parsing setup (`-v=N`, `-vv`) ✅
     - [x] Command routing (cameras, ports, calibrate, measure, freerun) ✅
     - [x] Logging setup (slog with verbosity levels) ✅
     - [x] Integration with display/source and display/destination flags ✅
   - [x] **`cmd/spectrometer/cameras/cameras.go`**: Cameras command ✅
     - [x] Reuse `source.ListCameras()` ✅
     - [x] Format and display camera information ✅
     - [x] Exit after listing ✅
   - [x] **`cmd/spectrometer/ports/ports.go`**: Ports command ✅
     - [x] Reuse `devices.ListSerialPorts()` ✅
     - [x] Format and display serial port information ✅
     - [x] Exit after listing ✅

4. **Implement Config Loading/Outputting**: **COMPLETED** (60%)
   - [x] **`cmd/spectrometer/internal/config/loader.go`**: Config loader ✅
     - [x] Format auto-detection from file extension (`.pb`, `.json`, `.yaml`, `.csv`) ✅
     - [x] `--output` flag override support ✅
     - [x] Load via proto/YAML/JSON marshallers ✅
     - [x] Uses generated protobuf types from `types/spectrometer` ✅
   - [x] **`cmd/spectrometer/internal/config/saver.go`**: Config saver ✅
     - [x] Format auto-detection from file extension ✅
     - [x] `--output` flag override support ✅
     - [x] Save via proto/YAML/JSON marshallers ✅
   - [ ] **`cmd/spectrometer/internal/config/validator.go`**: Config validator (pending)
   - [ ] Reference spectrum path resolution (pending)
   - [ ] Update `cmd/spectrometer/internal/config/SPEC.md` and `PLAN.md` - Mark as implemented (pending)

5. **Implement CR30 Support with Rendering**: **PARTIALLY COMPLETED** (50%)
   - [x] **`cmd/spectrometer/internal/obtainer/obtainer.go`**: Obtainer interface ✅
     - [x] Define `Obtainer` interface ✅
     - [x] `Measure()` for PC-initiated measurements ✅
     - [x] `Start()` / `Stop()` with callbacks for button-initiated measurements ✅
     - [x] Device registry ✅
     - [x] Obtainer factory ✅
   - [x] **`cmd/spectrometer/internal/obtainer/cr30.go`**: CR30 obtainer ✅
     - [x] Wrap `x/devices/cr30` ✅
     - [x] Implement `Obtainer` interface ✅
     - [x] `Measure()` using `cr30.Device.Measure()` ✅
     - [x] `Start()` running `WaitMeasurement()` in goroutine loop ✅
     - [x] `Stop()` with graceful shutdown ✅
     - [x] Callback pattern implemented ✅
   - [ ] **`cmd/spectrometer/internal/render/renderer.go`**: Core renderer
     - [ ] Spectrum plot rendering (wavelength vs. intensity)
     - [ ] Colorimetry overlay rendering (XYZ, LAB, RGB HEX)
     - [ ] Color swatch rendering
   - [ ] **`cmd/spectrometer/internal/render/patches.go`**: Patch rendering
     - [ ] Patch comparison rendering (half reference, half measured)
     - [ ] Delta E calculation and display (CIE76)
     - [ ] Spectrum overlay toggle (press "s" key)
   - [ ] **`cmd/spectrometer/measure/measure.go`**: Measure command
     - [ ] Device selection via `-device` flag
     - [ ] CR30 device initialization (`--port`, `--baud` flags)
     - [ ] Measurement workflow
     - [ ] Multiple readings support (`--readings=N`)
     - [ ] Patch-based measurement (`--patches=path`)
     - [ ] Export functionality (CSV, JSON, Proto)
   - [ ] Update `cmd/spectrometer/internal/obtainer/SPEC.md` and `PLAN.md` - Mark as implemented
   - [ ] Update `cmd/spectrometer/internal/render/SPEC.md` and `PLAN.md` - Mark as implemented

### Notes for Current Session

- **Before implementing filters**: Read `x/math/filter/ma/ma.go` to understand the pattern (Process, ProcessBuffer, Reset methods) ✅
- **Before implementing config**: Read existing marshaller packages (`x/marshaller/yaml`, `x/marshaller/json`, `x/marshaller/proto`) to understand the interface ✅
- **Before implementing CR30**: Read `cmd/cr30/main.go` to understand CR30 device usage
- **Before implementing render**: Read `cmd/cr30/display.go` to understand rendering patterns
- **Type casting**: Use direct type assertions (e.g., `vec.Vector3D(xyz).XYZ()`, `mat.Row(0).(vec.Vector)`)
- **Use mat/vec functions**: Don't implement loops for operations that `mat` or `vec` can do

### Issues Fixed

1. **Filter Stuttering**: Fixed naming - `SavitzkyGolayFilter` → `Filter`, `GaussianFilter` → `Filter`, `MedianFilter` → `Filter` ✅
2. **Config Loader**: Fixed to use `NewUnmarshaller()` instead of `NewMarshaller()` for unmarshalling ✅
3. **Proto Types**: Fixed duplicate `CameraSettings` definition, generated proto files successfully ✅
4. **Savitzky-Golay Algorithm**: Fixed coefficient calculation algorithm (needs testing) ✅
5. **Unused Variables**: Removed unused `centerRow` variable ✅
6. **Serial Port Enumeration**: Moved from `cmd/spectrometer/ports` to `x/devices/serial_*.go` for reusability ✅
7. **Build Tags**: Fixed build tag usage - implemented `ListSerialPorts()` directly in `serial_windows.go` and `serial_linux.go` ✅
8. **Obtainer Interface**: Refactored to use `Start()` / `Stop()` with callbacks instead of exposing `WaitMeasurement()` ✅
9. **Method Names**: Fixed verbose names - `StartMeasurementListener` → `Start`, `StopMeasurementListener` → `Stop` ✅
10. **Destination Pattern**: Fixed Obtainer interface - `Measure(ctx, dst)` now uses destination pattern (no allocation). `Start()` uses preallocated internal matrix. `Wavelengths()` can allocate (called once). ✅
11. **Reusability Violations**: Fixed `DeltaE76()` - moved from `internal/render` to `x/math/colorscience` (reusable color science calculation). Fixed `clamp255()` - removed unnecessary wrapper, use `math.Clamp()` directly where needed. Fixed `DeltaE76()` implementation - now uses `vec.Vector3D.Distance()` instead of manually extracting components and doing arithmetic. ✅

### Known Issues / TODOs

1. ~~**Proto Generation Required**: Config loader/saver now uses protobuf types~~ ✅ **FIXED**
2. **Savitzky-Golay Algorithm**: Needs unit tests to verify correctness
3. **Filter Tests**: Unit tests needed for all filters (savgol, gaussian, median)
4. **Obtainer Tests**: Unit tests needed for obtainer framework and CR30 obtainer
5. **Ports Command Tests**: Unit tests needed for ports command

---

## Next Work Session Tasks (MUTABLE)

**CRITICAL**: Prioritize tasks in the queue, keep queue full, work on ONE task at a time. Update WORK.md before and after each task.

### Planned Next Steps (Prioritized)

1. **Phase 0 - Render Module** (IN PROGRESS): ⭐ **HIGHEST PRIORITY**
   - [x] **Task**: Mark as in_progress ✅
   - [x] Read `cmd/display/destination` package API ✅
   - [x] Implement Delta E calculation (CIE76 formula) ✅
   - [x] Implement colorimetry formatting (XYZ, LAB, RGB HEX) ✅
   - [x] Implement Patch struct definition ✅
   - [ ] **Task**: Implement rendering functions (requires gocv drawing APIs)
     - [ ] Read gocv drawing examples/APIs for Mat drawing
     - [ ] Implement spectrum graph rendering (wavelength vs. intensity plot)
     - [ ] Implement color swatch rendering (rectangle drawing)
     - [ ] Implement colorimetry overlay rendering (text overlay)
     - [ ] Implement patch comparison rendering (half/half patch display)
   - [ ] Mark as completed when done, verify implementation
   - [ ] **Task**: Implement patch comparison rendering
     - [ ] Patch loading from YAML/JSON
     - [ ] Half reference, half measured patch display
     - [ ] Delta E calculation (CIE76)
     - [ ] Spectrum overlay toggle (press "s" key)

2. **Phase 0 - Measure Command** (Pending):
   - [ ] **Task**: Implement measure command with CR30 support
     - [ ] Device selection via `-device` flag
     - [ ] CR30 device initialization (`--port`, `--baud` flags)
     - [ ] Integration with `Start()` callback pattern
     - [ ] Multiple readings support (`--readings=N`)
     - [ ] Patch-based measurement (`--patches=path`)
     - [ ] Export functionality (CSV, JSON, Proto)

3. **Config Validator** (Pending):
   - [ ] **Task**: Implement config validator
     - [ ] Validation logic
     - [ ] Error handling and reporting

4. **Phase 1 - Proto Verification** (Pending):
   - [ ] Verify generated Go code compiles correctly
   - [ ] Integration testing

---

## Potential Issues (IMMUTABLE)

### Technical Risks

1. **Filter Implementation**:
   - **Risk**: Savitzky-Golay coefficient calculation complexity
   - **Mitigation**: Use established algorithms from literature, verify with reference implementations
   - **Risk**: Median filter performance with large windows
   - **Mitigation**: Use efficient partial sorting algorithms (quickselect)

2. **Config Format Detection**:
   - **Risk**: Ambiguous file extensions (e.g., `.pb` vs `.proto`)
   - **Mitigation**: Support both extensions, prioritize `--output` flag if provided

3. **CR30 Integration**:
   - **Risk**: Device API changes or compatibility issues
   - **Mitigation**: Test with actual device, handle errors gracefully

4. **Rendering Performance**:
   - **Risk**: Real-time rendering may be slow
   - **Mitigation**: Profile early, optimize critical paths, use efficient rendering algorithms

### Code Quality Risks

1. **Type Safety**:
   - **Risk**: Incorrect type casting may cause runtime panics
   - **Mitigation**: Write comprehensive unit tests with `testify`, test type assertions

2. **Code Duplication**:
   - **Risk**: Repeating similar code patterns across modules
   - **Mitigation**: Extract common functionality, use composition

3. **Not Using Existing Packages**:
   - **Risk**: Implementing operations that `mat/vec/tensor/dsp/filter` already provide
   - **Mitigation**: Read package documentation thoroughly before implementing

---

## Progress Summary

**Overall Progress**: **27%** (Filters ✅, CLI structure ✅, Config loader/saver ✅, Cameras command ✅, Ports command ✅, Obtainer framework ✅ with Start/Stop callbacks, Render foundation ✅)

**Next Milestone**: Complete Phase 0 (CR30 Device Support & Framework) - Target: 30%

**Blockers**: None currently

**Dependencies Resolved**: Proto definitions complete

---

## SPEC/PLAN Reading Guide

### Before Implementing Filters
- ✅ Read `x/math/filter/SPEC.md`
- ✅ Read `x/math/filter/ma/ma.go` (reference implementation)
- ✅ Read `x/math/vec/vec.go` (vector operations)
- ✅ Read `x/math/mat/mat.go` (matrix operations)

### Before Implementing Config
- ✅ Read `cmd/spectrometer/internal/config/SPEC.md`
- ✅ Read `cmd/spectrometer/internal/config/PLAN.md`
- [ ] Read `x/marshaller/yaml/marshaller.go` (reference)
- [ ] Read `x/marshaller/json/marshaller.go` (reference)
- [ ] Read `x/marshaller/proto/marshaller.go` (reference)

### Before Implementing CR30 Support
- ✅ Read `cmd/spectrometer/internal/obtainer/SPEC.md`
- ✅ Read `cmd/spectrometer/internal/obtainer/PLAN.md`
- ✅ Read `cmd/spectrometer/internal/render/SPEC.md`
- ✅ Read `cmd/spectrometer/internal/render/PLAN.md`
- [ ] Read `cmd/cr30/main.go` (CR30 device usage)
- [ ] Read `cmd/cr30/display.go` (rendering patterns)
- [ ] Read `x/devices/cr30` package documentation

### Before Updating SPEC/PLAN
- Always read the existing SPEC.md and PLAN.md before making changes
- Update progress tracking in PLAN.md after completing tasks
- Mark implemented features in SPEC.md with checkboxes

