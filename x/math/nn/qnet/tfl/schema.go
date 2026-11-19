package tfl

// TFLite schema (tensorflow/lite/schema/schema.fbs): field slots, enum
// values and builtin operator codes used by the loader.

// Table field slots.
const (
	modelVersion       = 0
	modelOperatorCodes = 1
	modelSubgraphs     = 2
	modelBuffers       = 4

	opcodeDeprecatedBuiltin = 0
	opcodeCustom            = 1
	opcodeVersion           = 2
	opcodeBuiltin           = 3

	subgraphTensors   = 0
	subgraphInputs    = 1
	subgraphOutputs   = 2
	subgraphOperators = 3

	tensorShape        = 0
	tensorType         = 1
	tensorBuffer       = 2
	tensorName         = 3
	tensorQuantization = 4
	tensorSparsity     = 6

	quantScale     = 2
	quantZeroPoint = 3
	quantDetails   = 4 // union type
	quantDimension = 6

	operatorOpcodeIndex = 0
	operatorInputs      = 1
	operatorOutputs     = 2
	operatorOptionsType = 3
	operatorOptions     = 4
	operatorCustomOpts  = 5

	bufferData   = 0
	bufferOffset = 1
)

// TensorType values.
const (
	typeFloat32 = 0
	typeInt32   = 2
	typeInt64   = 4
	typeUInt8   = 3
	typeInt8    = 9
)

// Builtin operator codes.
const (
	opAdd             = 0
	opAveragePool2D   = 1
	opConcatenation   = 2
	opConv2D          = 3
	opDepthwiseConv2D = 4
	opDequantize      = 6
	opFullyConnected  = 9
	opLogistic        = 14
	opMaxPool2D       = 17
	opMul             = 18
	opRelu            = 19
	opReluN1To1       = 20
	opRelu6           = 21
	opReshape         = 22
	opResizeBilinear  = 23
	opSoftmax         = 25
	opPad             = 34
	opTranspose       = 39
	opMean            = 40
	opSub             = 41
	opPadV2           = 60
	opLeakyRelu       = 98
	opResizeNearest   = 97
	opQuantize        = 114
)

// BuiltinOptions union members (only those the loader reads).
const (
	optConv2D          = 1
	optDepthwiseConv2D = 2
	optPool2D          = 5
	optFullyConnected  = 8
	optSoftmax         = 9
	optAdd             = 11
	optReshape         = 17
	optConcatenation   = 10
	optMul             = 21
	optReducer         = 27
	optResizeNearest   = 74
	optResizeBilinear  = 15
	optLeakyRelu       = 75
)

// Option table slots.
const (
	convPadding = 0
	convStrideW = 1
	convStrideH = 2
	convAct     = 3
	convDilW    = 4
	convDilH    = 5

	dwPadding    = 0
	dwStrideW    = 1
	dwStrideH    = 2
	dwMultiplier = 3
	dwAct        = 4
	dwDilW       = 5
	dwDilH       = 6

	poolPadding = 0
	poolStrideW = 1
	poolStrideH = 2
	poolFilterW = 3
	poolFilterH = 4
	poolAct     = 5

	fcAct         = 0
	fcWeightsFmt  = 1
	fcKeepNumDims = 2

	softmaxBeta = 0

	addAct = 0

	concatAxis = 0
	concatAct  = 1

	mulAct = 0

	reducerKeepDims = 0

	resizeAlignCorners = 0
	resizeHalfPixel    = 1

	// ResizeBilinearOptions slots 0 and 1 are the deprecated new_height and
	// new_width.
	resizeBilinearAlignCorners = 2
	resizeBilinearHalfPixel    = 3

	leakyReluAlpha = 0
)

// Padding and fused activation enum values.
const (
	padSame  = 0
	padValid = 1

	actNone      = 0
	actRelu      = 1
	actReluN1To1 = 2
	actRelu6     = 3
)

// opName returns the TFLite name of a builtin operator for error messages.
func opName(code int32) string {
	switch code {
	case opAdd:
		return "ADD"
	case opAveragePool2D:
		return "AVERAGE_POOL_2D"
	case opConcatenation:
		return "CONCATENATION"
	case opConv2D:
		return "CONV_2D"
	case opDepthwiseConv2D:
		return "DEPTHWISE_CONV_2D"
	case opDequantize:
		return "DEQUANTIZE"
	case opFullyConnected:
		return "FULLY_CONNECTED"
	case opLogistic:
		return "LOGISTIC"
	case opMaxPool2D:
		return "MAX_POOL_2D"
	case opMul:
		return "MUL"
	case opRelu:
		return "RELU"
	case opReluN1To1:
		return "RELU_N1_TO_1"
	case opRelu6:
		return "RELU6"
	case opReshape:
		return "RESHAPE"
	case opSoftmax:
		return "SOFTMAX"
	case opPad:
		return "PAD"
	case opPadV2:
		return "PADV2"
	case opResizeBilinear:
		return "RESIZE_BILINEAR"
	case opLeakyRelu:
		return "LEAKY_RELU"
	case opMean:
		return "MEAN"
	case opSub:
		return "SUB"
	case opResizeNearest:
		return "RESIZE_NEAREST_NEIGHBOR"
	case opQuantize:
		return "QUANTIZE"
	case opTranspose:
		return "TRANSPOSE"
	}
	return ""
}
