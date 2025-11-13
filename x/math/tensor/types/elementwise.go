package types

// ElementWise defines element-wise operations on tensors.
// This interface contains operations that apply functions element-by-element.
type ElementWise interface {
	// Binary operations (destination-based)
	// Add performs element-wise addition: dst = t + other (matches tf.add).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if shapes don't match.
	Add(dst Tensor, other Tensor) Tensor

	// Subtract performs element-wise subtraction: dst = t - other (matches tf.subtract).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if shapes don't match.
	Subtract(dst Tensor, other Tensor) Tensor

	// Multiply performs element-wise multiplication: dst = t * other (matches tf.multiply).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if shapes don't match.
	Multiply(dst Tensor, other Tensor) Tensor

	// Divide performs element-wise division: dst = t / other (matches tf.divide).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if shapes don't match or on division by zero.
	Divide(dst Tensor, other Tensor) Tensor

	// ScalarMul multiplies the tensor by a scalar: dst = scalar * t (matches tf.scalar_mul).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	ScalarMul(dst Tensor, scalar float64) Tensor

	// Scalar operations (destination-based)
	// AddScalar adds a scalar value to all elements: dst[i] = t[i] + scalar.
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	AddScalar(dst Tensor, scalar float64) Tensor

	// SubScalar subtracts a scalar value from all elements: dst[i] = t[i] - scalar.
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	SubScalar(dst Tensor, scalar float64) Tensor

	// MulScalar multiplies all elements by a scalar: dst[i] = t[i] * scalar.
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	MulScalar(dst Tensor, scalar float64) Tensor

	// DivScalar divides all elements by a scalar: dst[i] = t[i] / scalar.
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	DivScalar(dst Tensor, scalar float64) Tensor

	// Unary operations (destination-based)
	// Square computes element-wise square: dst[i] = t[i]^2 (matches tf.square).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	Square(dst Tensor) Tensor

	// Sqrt computes element-wise square root: dst[i] = sqrt(t[i]) (matches tf.sqrt).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	// Panics on negative values.
	Sqrt(dst Tensor) Tensor

	// Exp computes element-wise exponential: dst[i] = exp(t[i]) (matches tf.exp).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	Exp(dst Tensor) Tensor

	// Log computes element-wise natural logarithm: dst[i] = log(t[i]) (matches tf.log).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	// Panics on non-positive values.
	Log(dst Tensor) Tensor

	// Pow computes element-wise power: dst[i] = t[i]^power (matches tf.pow).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	Pow(dst Tensor, power float64) Tensor

	// Abs computes element-wise absolute value: dst[i] = |t[i]| (matches tf.abs).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	Abs(dst Tensor) Tensor

	// Sign computes element-wise sign: dst[i] = sign(t[i]) (-1, 0, or 1) (matches tf.sign).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	Sign(dst Tensor) Tensor

	// Cos computes element-wise cosine: dst[i] = cos(t[i]) (matches tf.cos).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	Cos(dst Tensor) Tensor

	// Sin computes element-wise sine: dst[i] = sin(t[i]) (matches tf.sin).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	Sin(dst Tensor) Tensor

	// Negative computes element-wise negation: dst[i] = -t[i] (matches tf.negative).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	Negative(dst Tensor) Tensor

	// Comparison Operations
	// Comparison operations return tensors with 1.0 where condition is true, 0.0 otherwise (matching TensorFlow behavior).
	// All operations accept a dst parameter. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.

	// Equal returns a tensor with 1.0 where t == other, 0.0 otherwise.
	// Panics if shapes don't match.
	Equal(dst Tensor, other Tensor) Tensor

	// Greater returns a tensor with 1.0 where t > other, 0.0 otherwise (matches tf.greater).
	// Panics if shapes don't match.
	Greater(dst Tensor, other Tensor) Tensor

	// Less returns a tensor with 1.0 where t < other, 0.0 otherwise.
	// Panics if shapes don't match.
	Less(dst Tensor, other Tensor) Tensor

	// NotEqual returns a tensor with 1.0 where t != other, 0.0 otherwise (matches tf.not_equal).
	// Panics if shapes don't match.
	NotEqual(dst Tensor, other Tensor) Tensor

	// GreaterEqual returns a tensor with 1.0 where t >= other, 0.0 otherwise (matches tf.greater_equal).
	// Panics if shapes don't match.
	GreaterEqual(dst Tensor, other Tensor) Tensor

	// LessEqual returns a tensor with 1.0 where t <= other, 0.0 otherwise (matches tf.less_equal).
	// Panics if shapes don't match.
	LessEqual(dst Tensor, other Tensor) Tensor

	// Comparison operations (tensor-scalar, destination-based)
	// EqualScalar returns a tensor with 1.0 where t == scalar, 0.0 otherwise (matches tf.equal with scalar).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	EqualScalar(dst Tensor, scalar float64) Tensor

	// NotEqualScalar returns a tensor with 1.0 where t != scalar, 0.0 otherwise (matches tf.not_equal with scalar).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	NotEqualScalar(dst Tensor, scalar float64) Tensor

	// GreaterScalar returns a tensor with 1.0 where t > scalar, 0.0 otherwise (matches tf.greater with scalar).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	GreaterScalar(dst Tensor, scalar float64) Tensor

	// LessScalar returns a tensor with 1.0 where t < scalar, 0.0 otherwise (matches tf.less with scalar).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	LessScalar(dst Tensor, scalar float64) Tensor

	// GreaterEqualScalar returns a tensor with 1.0 where t >= scalar, 0.0 otherwise (matches tf.greater_equal with scalar).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	GreaterEqualScalar(dst Tensor, scalar float64) Tensor

	// LessEqualScalar returns a tensor with 1.0 where t <= scalar, 0.0 otherwise (matches tf.less_equal with scalar).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	LessEqualScalar(dst Tensor, scalar float64) Tensor

	// Conditional Operations
	// Where performs element-wise selection: dst[i] = condition[i] ? a[i] : b[i] (matches tf.where).
	// All tensors must have the same shape.
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if shapes don't match.
	Where(dst Tensor, condition, a, b Tensor) Tensor
}
