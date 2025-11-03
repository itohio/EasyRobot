# Learn Package Specification

## Overview

This document specifies the design and implementation plan for optimization and learning algorithms within the math package.

**Target Platform**: Embedded and low-power devices
- Primary focus on ARM Cortex-M, ESP32, Raspberry Pi Zero
- Float32 precision throughout for memory efficiency
- GPU acceleration out of scope
- Emphasis on CPU cache-friendly algorithms
- Minimize heap allocations during hot paths

## Components

### 1. Newton's Method for Unconstrained Optimization

**Purpose**: Find local maxima/minima of a multivariate function using Newton's method.

**Algorithm**: 
- Uses gradient and Hessian information for quadratic convergence
- Iterative update: x_{k+1} = x_k - H^{-1}(x_k) * ∇f(x_k)
- Where H is the Hessian matrix and ∇f is the gradient

**Use Cases**:
- Optimization problems with known derivatives
- Smooth functions with well-behaved Hessians
- Robotics: inverse kinematics, control optimization

**Key Features**:
- Callback-based API for user-defined objective function
- Supports gradient and Hessian computation via finite differences
- Configurable convergence tolerance
- Maximum iteration limits
- Line search for robust convergence

**API Design**:

```go
// Objective function callback
// Returns function value at point x
type ObjectiveFunc func(x vec.Vector) float32

// Gradient callback (optional, can use finite differences)
// Computes gradient at point x, stores in grad
type GradientFunc func(x vec.Vector, grad vec.Vector)

// Hessian callback (optional, can use finite differences)
// Computes Hessian matrix at point x, stores in H
type HessianFunc func(x vec.Vector, H mat.Matrix)

// NewtonOptimizer is a stateful optimizer that reuses buffers
type NewtonOptimizer struct {
    // Configuration
    maxIterations   int
    tolerance       float32
    lineSearch      bool
    initialStepSize float32
    
    // Reusable buffers
    dimension       int
    current         vec.Vector
    gradient        vec.Vector
    hessian         mat.Matrix
    searchDir       vec.Vector
    workspace       vec.Vector  // For line search
}

// NewNewtonOptimizer creates a new optimizer with pre-allocated buffers
func NewNewtonOptimizer(dimension int) *NewtonOptimizer

// Configure sets optimization parameters
func (n *NewtonOptimizer) Configure(opts NewtonOptions)

// Optimize finds local maximum starting from start
func (n *NewtonOptimizer) Optimize(obj ObjectiveFunc, start vec.Vector, grad GradientFunc, hessian HessianFunc) (vec.Vector, float32, error)

// Options for Newton's method
type NewtonOptions struct {
    MaxIterations   int
    Tolerance       float32
    LineSearch      bool
    InitialStepSize float32
}
```

**Implementation Details**:
- Fallback to finite differences if gradients not provided
- Positive definite check for Hessian (modify if needed)
- Backtracking line search for stability
- Convergence criteria: ||∇f|| < tolerance

**Complexity**:
- O(n²) per iteration (Hessian computation)
- O(n³) for Hessian inversion
- Quadratic convergence rate near optimum

**Memory Considerations**:
- Hessian matrix: n² float32 values = 4n² bytes
- Practical limit: n ≤ 20-30 for 8KB-32KB RAM devices
- Consider quasi-Newton methods for n > 30

**Edge Cases**:
- Singular/ill-conditioned Hessian (more likely with float32)
- Functions without global optima
- Poor initial guess
- Non-convex functions

### 2. Fibonacci Search for 1D Optimization

**Purpose**: Find maximum of a unimodal function on a bounded interval using Fibonacci sequence.

**Algorithm**:
- Binary search-like method using Fibonacci numbers for efficient interval reduction
- Golden ratio based interval division
- No derivatives required
- Suitable for expensive function evaluations

**Use Cases**:
- 1D line search (used as component of higher-dimensional methods)
- Black-box optimization
- When function derivatives are unavailable or expensive
- Discrete optimization with bounded domains

**Key Features**:
- Convergence in O(log n) function evaluations
- Guaranteed optimal number of evaluations for given accuracy
- No derivative information needed
- Bounded domain search

**API Design**:

```go
// 1D objective function
type Objective1D func(x float32) float32

// FibonacciSearcher is a stateful searcher that caches Fibonacci sequence
type FibonacciSearcher struct {
    // Configuration
    maxEvaluations int
    tolerance      float32
    
    // Cached Fibonacci sequence
    fibSequence    []float32
    maxSize        int  // Maximum sequence length allocated
}

// NewFibonacciSearcher creates a new searcher with pre-allocated buffers
func NewFibonacciSearcher(maxSize int) *FibonacciSearcher

// Configure sets search parameters
func (f *FibonacciSearcher) Configure(opts FibonacciOptions)

// Search finds maximum on interval [a, b]
func (f *FibonacciSearcher) Search(obj Objective1D, a, b float32) (float32, float32, error)

// Options for Fibonacci search
type FibonacciOptions struct {
    N                int  // number of function evaluations
    Tolerance        float32
}
```

**Implementation Details**:
- Generate Fibonacci sequence up to F_n
- Interval reduction: [a, b] → smaller interval maintaining unimodal property
- Use golden ratio: φ = (1 + √5)/2 ≈ 1.618
- Adaptive interval bounds based on function evaluation

**Complexity**:
- O(log n) function evaluations for n iterations
- O(n) space for Fibonacci sequence
- Optimal for given number of evaluations

**Memory Considerations**:
- Fibonacci sequence: n float32 values = 4n bytes
- Practical limit: n ≤ 50 for sequence cache
- Beyond 50, regenerate as needed

**Limitations**:
- Only for unimodal functions
- 1D only
- Requires bounded domain

### 3. Polynomial Fitting

**Purpose**: Fit a polynomial of degree N-1 to a set of data points using least squares.

**Algorithm**:
- Least squares minimization: min ||Ax - b||²
- Where A is Vandermonde matrix, b is data values
- Solve normal equations: (A^T A) x = A^T b
- For polynomial: p(x) = c₀ + c₁x + c₂x² + ... + cₙ₋₁xⁿ⁻¹

**Use Cases**:
- Curve fitting to experimental data
- Approximation of unknown functions
- Smoothing noisy measurements
- Robotics: trajectory fitting, calibration

**Key Features**:
- Multiple data points (can be more than polynomial degree)
- Overdetermined systems (least squares solution)
- Underdetermined systems (minimum norm solution)
- Optional regularization (ridge regression)

**API Design**:

```go
// PolynomialFitter is a stateful fitter that reuses buffers
type PolynomialFitter struct {
    // Configuration
    degree         int
    regularization float32
    
    // Reusable buffers
    maxDegree      int
    vandermonde    mat.Matrix  // Vandermonde matrix
    normalEq       mat.Matrix  // A^T A matrix
    rhs            vec.Vector  // Right-hand side: A^T b
    workspace      vec.Vector  // Temporary workspace
}

// NewPolynomialFitter creates a new fitter with pre-allocated buffers
func NewPolynomialFitter(maxDegree int) *PolynomialFitter

// Configure sets fitting parameters
func (p *PolynomialFitter) Configure(opts PolynomialFitOptions)

// Fit polynomial of degree N-1 to data points
// x: input values, y: output values
// Returns: coefficients [c₀, c₁, c₂, ..., cₙ₋₁]
func (p *PolynomialFitter) Fit(x, y vec.Vector) (vec.Vector, error)

// Options for polynomial fitting
type PolynomialFitOptions struct {
    Degree         int
    Regularization float32  // Ridge regularization strength
}

// PolynomialEvaluator evaluates polynomials efficiently
type PolynomialEvaluator struct {
    // Cached values
    coefficients vec.Vector
    maxDegree    int
}

// NewPolynomialEvaluator creates an evaluator for given coefficients
func NewPolynomialEvaluator(coefficients vec.Vector) *PolynomialEvaluator

// Evaluate polynomial at point x
func (p *PolynomialEvaluator) Evaluate(x float32) float32

// Evaluate polynomial at multiple points
func (p *PolynomialEvaluator) EvaluateVec(x vec.Vector, y vec.Vector)
```

**Implementation Details**:
- Build Vandermonde matrix: V[i][j] = x[i]^j
- Solve using QR decomposition or normal equations
- Handle ill-conditioned cases with regularization
- Support for weighted least squares (optional)

**Complexity**:
- O(mn²) for Vandermonde matrix construction (m points, n degree)
- O(n³) for solving normal equations
- O(mn) for evaluation at m points

**Memory Considerations**:
- Vandermonde matrix: m×n float32 = 4mn bytes
- Normal equations: n² float32 = 4n² bytes
- Practical limits: 
  - Degree ≤ 10-15 for typical use
  - Data points ≤ 100-200 depending on RAM
  - Total memory: ~4mn + 4n² bytes

**Edge Cases**:
- Overfitting (degree too high; more problematic with float32 precision)
- Underfitting (degree too low)
- Ill-conditioned Vandermonde matrix (common with float32)
- Repeated x values

## Design Principles

### Callback-Based API

All optimization methods use callback functions for flexibility:
- No hardcoded function types
- User supplies objective function
- Optional derivative computation
- Minimal allocations

### Float32 Precision

Mandatory `float32` precision for embedded compatibility:
- Half the memory footprint of `float64`
- Faster on ARM Cortex-M (no FP64 hardware)
- Reduced cache pressure
- Acceptable precision for robotics applications
- No float64 option available

### Error Handling

Explicit error returns for:
- Convergence failures
- Singular matrices
- Invalid inputs
- Maximum iterations exceeded

### Configuration via Options

Options pattern for configurability:
```go
type Options struct {
    // Common options
}

func Algorithm(obj Func, opts Options) (Result, error)
```

### Stateful Optimization Objects

All optimization methods use stateful objects to minimize allocations:

**Benefits for Embedded Systems**:
- Pre-allocated buffers eliminate heap allocations during execution
- Cached intermediate computations avoid repeated work
- Reusable across multiple optimization runs
- Reduced GC pressure critical for real-time robotics
- Predictable memory usage enables stack-based allocation
- Lower power consumption from reduced cache misses

**Pattern**:
```go
// 1. Create object with required dimensions
optimizer := NewOptimizer(dimension)

// 2. Configure parameters
optimizer.Configure(opts)

// 3. Run optimization multiple times
result1, err1 := optimizer.Optimize(obj, start1)
result2, err2 := optimizer.Optimize(obj, start2)  // No allocations!

// 4. Reuse same object for different problems
optimizer.Configure(newOpts)
result3, err3 := optimizer.Optimize(obj2, start3)
```

## Testing Strategy

### Unit Tests

1. **Newton's Method**
   - Known functions (quadratic, Rosenbrock)
   - Convergence verification
   - Gradient/Hessian finite difference accuracy
   - Line search behavior

2. **Fibonacci Search**
   - Unimodal function maximum finding
   - Interval boundary preservation
   - Optimal evaluation count
   - Convergence rate

3. **Polynomial Fitting**
   - Exact fit (N points, degree N-1)
   - Overdetermined system (more points than degree)
   - Noise robustness
   - Degree selection

### Benchmarks

- Function evaluation counts
- Convergence rate
- Memory usage
- Time complexity verification

### Edge Case Tests

- Singular matrices
- Non-convex functions
- Noisy data
- Extreme input values
- Empty/singleton inputs

## Example Usage

### Newton's Method

```go
// Define objective function
obj := func(x vec.Vector) float32 {
    // Rosenbrock function for testing
    return -(100*sqr(x[1]-sqr(x[0])) + sqr(1-x[0]))
}

// Optional: provide gradient
grad := func(x vec.Vector, grad vec.Vector) {
    grad[0] = -(-400*x[0]*(x[1]-sqr(x[0])) - 2*(1-x[0]))
    grad[1] = -(200*(x[1]-sqr(x[0])))
}

// Create optimizer with pre-allocated buffers
optimizer := NewNewtonOptimizer(2)
opts := NewtonOptions{
    MaxIterations: 100,
    Tolerance: 1e-6,
    LineSearch: true,
}
optimizer.Configure(opts)

// Optimize
start := vec.Vector{-1.2, 1.0}
xopt, fopt, err := optimizer.Optimize(obj, start, grad, nil)

// Reuse optimizer for subsequent runs - no allocations!
xopt2, fopt2, err2 := optimizer.Optimize(obj, start2, grad, nil)
```

### Fibonacci Search

```go
// Define 1D function
obj := func(x float32) float32 {
    return -(x - 2)*(x - 2) + 4  // Parabola, max at x=2
}

// Create searcher with pre-allocated Fibonacci sequence
searcher := NewFibonacciSearcher(100)
opts := FibonacciOptions{
    N: 20,
    Tolerance: 1e-6,
}
searcher.Configure(opts)

// Search on interval [0, 5]
xmax, fmax, err := searcher.Search(obj, 0, 5)

// Reuse searcher for subsequent runs
xmax2, fmax2, err2 := searcher.Search(obj, -5, 0)
```

### Polynomial Fitting

```go
// Generate noisy data
x := vec.Vector{0, 1, 2, 3, 4, 5}
y := vec.Vector{1, 4.2, 9.1, 15.8, 25.1, 36.3}  // Noisy x²

// Create fitter with pre-allocated buffers
fitter := NewPolynomialFitter(10)  // Max degree 10
fitter.Configure(PolynomialFitOptions{
    Degree: 2,
})

// Fit quadratic polynomial
coeffs, err := fitter.Fit(x, y)

// Create evaluator for efficient evaluation
evaluator := NewPolynomialEvaluator(coeffs)

// Evaluate at new points
result := evaluator.Evaluate(2.5)

// Evaluate at multiple points efficiently
xNew := vec.Vector{2.5, 3.5, 4.5}
yNew := vec.New(3)
evaluator.EvaluateVec(xNew, yNew)
```

## Dependencies

- `vec.Vector`: Vector operations
- `mat.Matrix`: Matrix operations (for Newton's method, polynomial fitting)
- `github.com/chewxy/math32`: Math functions (sqrt, exp, etc.)

## Performance Considerations for Embedded Systems

- **Memory**: Minimize allocations in hot paths; pre-allocate all buffers
- **Cache**: Keep working set small; sequential memory access patterns
- **Computation**: Cache computed values (e.g., Fibonacci sequence); avoid redundant calculations
- **In-place**: Use in-place operations where possible to reduce memory traffic
- **Power**: Minimize data movement; exploit temporal locality
- **Determinism**: Fixed memory footprint enables schedulability analysis
- **No SIMD**: Assume scalar operations only (no NEON/SIMD dependencies)
- **Profile**: Benchmark on target hardware (ARM Cortex-M, ESP32)

## Future Enhancements

1. **Quasi-Newton methods**: BFGS, L-BFGS for Hessian-free optimization (memory-limited for large n)
2. **Simulated annealing**: Global optimization for non-convex problems
3. **Genetic algorithms**: Population-based optimization (limited population size)
4. **Automatic differentiation**: Compute gradients automatically via dual numbers
5. **Sparse optimization**: Handling large-scale problems with structured sparsity
6. **Approximate methods**: Trade accuracy for reduced computation
7. **Constraint handling**: Lagrange multipliers, penalty methods
8. **Fixed-point arithmetic**: For ultra-low-power devices without FPU
9. **TinyGo compatibility**: Ensure compatibility with TinyGo compiler
10. **Lookup tables**: Pre-computed tables for common functions on severely constrained devices

**Out of Scope**:
- GPU acceleration (no GPU on target platforms)
- Multi-threading/parallelization (complexity vs. benefit)
- Large-scale distributed optimization

## Questions to Resolve

1. Maximum practical dimension for Newton's method on embedded devices?
2. How to handle gradient/Hessian computation efficiently with limited memory?
3. Should we provide automatic convergence detection vs. user-specified iterations?
4. How to balance between robustness and performance for real-time systems?
5. Memory budgets: What's acceptable for pre-allocated buffers?
6. How to handle discontinuous or non-smooth functions efficiently?
7. Should we limit polynomial degree to prevent memory exhaustion?
8. How to integrate with automatic differentiation frameworks on embedded?
9. Fallback strategies when matrices become ill-conditioned on float32?
10. Practical Fibonacci sequence size limits for 1D search?

