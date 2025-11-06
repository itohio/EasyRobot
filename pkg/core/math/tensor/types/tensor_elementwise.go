package types

// TensorElementWise defines element-wise operations on tensors.
// This interface contains operations that apply functions element-by-element.
type TensorElementWise interface {
	// Binary operations (in-place)
	// Add adds another tensor element-wise in-place: t = t + other.
	// Panics if shapes don't match. Returns self for method chaining.
	Add(other Tensor) Tensor

	// Sub subtracts another tensor element-wise in-place: t = t - other.
	// Panics if shapes don't match. Returns self for method chaining.
	Sub(other Tensor) Tensor

	// Mul multiplies another tensor element-wise in-place: t = t * other.
	// Panics if shapes don't match. Returns self for method chaining.
	Mul(other Tensor) Tensor

	// Multiply is an alias for Mul (matches TensorFlow naming: tf.multiply).
	Multiply(other Tensor) Tensor

	// Div divides by another tensor element-wise in-place: t = t / other.
	// Panics if shapes don't match or on division by zero. Returns self for method chaining.
	Div(other Tensor) Tensor

	// Divide is an alias for Div (matches TensorFlow naming: tf.divide).
	Divide(other Tensor) Tensor

	// Subtract is an alias for Sub (matches TensorFlow naming: tf.subtract).
	Subtract(other Tensor) Tensor

	// Scale multiplies the tensor by a scalar in-place: t = scalar * t.
	// Returns self for method chaining.
	Scale(scalar float64) Tensor

	// ScalarMul is an alias for Scale (matches TensorFlow naming: tf.scalar_mul).
	ScalarMul(scalar float64) Tensor

	// Scalar operations (in-place)
	// AddScalar adds a scalar value to all elements in-place: t[i] = t[i] + scalar.
	// Returns self for method chaining.
	AddScalar(scalar float64) Tensor

	// SubScalar subtracts a scalar value from all elements in-place: t[i] = t[i] - scalar.
	// Returns self for method chaining.
	SubScalar(scalar float64) Tensor

	// MulScalar multiplies all elements by a scalar in-place: t[i] = t[i] * scalar.
	// Returns self for method chaining.
	MulScalar(scalar float64) Tensor

	// DivScalar divides all elements by a scalar in-place: t[i] = t[i] / scalar.
	// Returns self for method chaining.
	DivScalar(scalar float64) Tensor

	// Unary operations (in-place)
	// Square computes element-wise square in-place: t[i] = t[i]^2.
	// Returns self for method chaining.
	Square(dst Tensor) Tensor

	// Sqrt computes element-wise square root in-place: t[i] = sqrt(t[i]).
	// Panics on negative values. Returns self for method chaining.
	Sqrt(dst Tensor) Tensor

	// Exp computes element-wise exponential in-place: t[i] = exp(t[i]).
	// Returns self for method chaining.
	Exp(dst Tensor) Tensor

	// Log computes element-wise natural logarithm in-place: t[i] = log(t[i]).
	// Panics on non-positive values. Returns self for method chaining.
	Log(dst Tensor) Tensor

	// Pow computes element-wise power in-place: t[i] = t[i]^power.
	// Returns self for method chaining.
	Pow(dst Tensor, power float64) Tensor

	// Abs computes element-wise absolute value in-place: t[i] = |t[i]|.
	// Returns self for method chaining.
	Abs(dst Tensor) Tensor

	// Sign computes element-wise sign in-place: t[i] = sign(t[i]) (-1, 0, or 1).
	// Returns self for method chaining.
	Sign(dst Tensor) Tensor

	// Cos computes element-wise cosine in-place: t[i] = cos(t[i]).
	// Returns self for method chaining.
	Cos(dst Tensor) Tensor

	// Sin computes element-wise sine in-place: t[i] = sin(t[i]).
	// Returns self for method chaining.
	Sin(dst Tensor) Tensor

	// Negative computes element-wise negation in-place: t[i] = -t[i].
	// Returns self for method chaining.
	Negative(dst Tensor) Tensor

	// Element-wise Operations (Non-Mutating)
	// AddTo computes result = t + other and stores it in dst.
	// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
	// Panics if shapes don't match. Returns the destination tensor.
	AddTo(other Tensor, dst Tensor) Tensor

	// SubTo computes result = t - other and stores it in dst.
	// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
	// Panics if shapes don't match. Returns the destination tensor.
	SubTo(other Tensor, dst Tensor) Tensor

	// MulTo computes result = t * other (element-wise) and stores it in dst.
	// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
	// Panics if shapes don't match. Returns the destination tensor.
	MulTo(other Tensor, dst Tensor) Tensor

	// DivTo computes result = t / other (element-wise) and stores it in dst.
	// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
	// Panics if shapes don't match. Returns the destination tensor.
	DivTo(other Tensor, dst Tensor) Tensor

	// Scalar operations (destination-based)
	// ScaleTo multiplies the tensor by a scalar and stores result in dst.
	// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
	// Returns the destination tensor.
	ScaleTo(dst Tensor, scalar float64) Tensor

	// AddScalarTo adds a scalar value to all elements and stores result in dst.
	// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
	// Returns the destination tensor.
	AddScalarTo(dst Tensor, scalar float64) Tensor

	// SubScalarTo subtracts a scalar value from all elements and stores result in dst.
	// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
	// Returns the destination tensor.
	SubScalarTo(dst Tensor, scalar float64) Tensor

	// MulScalarTo multiplies all elements by a scalar and stores result in dst.
	// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
	// Returns the destination tensor.
	MulScalarTo(dst Tensor, scalar float64) Tensor

	// DivScalarTo divides all elements by a scalar and stores result in dst.
	// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
	// Returns the destination tensor.
	DivScalarTo(dst Tensor, scalar float64) Tensor

	// Comparison Operations
	// Comparison operations return new tensors with 1.0 where condition is true, 0.0 otherwise (matching TensorFlow behavior).

	// Equal returns a tensor with 1.0 where t == other, 0.0 otherwise.
	// Panics if shapes don't match. Returns a new tensor.
	Equal(other Tensor) Tensor

	// GreaterThan returns a tensor with 1.0 where t > other, 0.0 otherwise.
	// Panics if shapes don't match. Returns a new tensor.
	GreaterThan(other Tensor) Tensor

	// Greater is an alias for GreaterThan (matches TensorFlow naming).
	// Returns a tensor with 1.0 where t > other, 0.0 otherwise.
	Greater(other Tensor) Tensor

	// Less returns a tensor with 1.0 where t < other, 0.0 otherwise.
	// Panics if shapes don't match. Returns a new tensor.
	Less(other Tensor) Tensor

	// NotEqual returns a tensor with 1.0 where t != other, 0.0 otherwise (matches tf.not_equal).
	// Panics if shapes don't match. Returns a new tensor.
	NotEqual(other Tensor) Tensor

	// GreaterEqual returns a tensor with 1.0 where t >= other, 0.0 otherwise (matches tf.greater_equal).
	// Panics if shapes don't match. Returns a new tensor.
	GreaterEqual(other Tensor) Tensor

	// LessEqual returns a tensor with 1.0 where t <= other, 0.0 otherwise (matches tf.less_equal).
	// Panics if shapes don't match. Returns a new tensor.
	LessEqual(other Tensor) Tensor

	// Unary operations (destination-based, explicit To variants)
	// These are aliases for the unary operations above, provided for consistency with *To pattern.
	// SquareTo computes element-wise square and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	SquareTo(dst Tensor) Tensor

	// SqrtTo computes element-wise square root and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	SqrtTo(dst Tensor) Tensor

	// ExpTo computes element-wise exponential and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	ExpTo(dst Tensor) Tensor

	// LogTo computes element-wise natural logarithm and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	LogTo(dst Tensor) Tensor

	// PowTo computes element-wise power and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	PowTo(dst Tensor, power float64) Tensor

	// AbsTo computes element-wise absolute value and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	AbsTo(dst Tensor) Tensor

	// SignTo computes element-wise sign and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	SignTo(dst Tensor) Tensor

	// CosTo computes element-wise cosine and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	CosTo(dst Tensor) Tensor

	// SinTo computes element-wise sine and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	SinTo(dst Tensor) Tensor

	// NegativeTo computes element-wise negation and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	NegativeTo(dst Tensor) Tensor

	// Conditional Operations
	// Where performs element-wise selection: result[i] = condition[i] ? a[i] : b[i].
	// All tensors must have the same shape. Returns a new tensor.
	// Panics if shapes don't match.
	Where(condition, a, b Tensor) Tensor

	// WhereTo performs element-wise selection and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	WhereTo(dst Tensor, condition, a, b Tensor) Tensor
}

