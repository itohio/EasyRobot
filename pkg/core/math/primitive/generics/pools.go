package generics

import helpers "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

const MAX_DIMS = helpers.MAX_DIMS

type (
	Numeric          = helpers.Numeric
	ClampableToInt8  = helpers.ClampableToInt8
	ClampableToInt16 = helpers.ClampableToInt16
	ClampableToInt32 = helpers.ClampableToInt32
	ClampableToInt64 = helpers.ClampableToInt64
	Pool[T any]      = helpers.Pool[T]
	WorkerPool       = helpers.WorkerPool
	WorkerCallback   = helpers.WorkerCallback
	ChunkSizer       = helpers.ChunkSizer
	WorkerPoolOption = helpers.WorkerPoolOption
)

var (
	ErrPoolClosed             = helpers.ErrPoolClosed
	ErrWorkerCallbackNil      = helpers.ErrWorkerCallbackNil
	ErrPoolAlreadyInitialized = helpers.ErrPoolAlreadyInitialized
	ErrPoolNotInitialized     = helpers.ErrPoolNotInitialized
)

func ClampToInt8Value[U ClampableToInt8](v U) int8 {
	return helpers.ClampToInt8Value(v)
}

func ClampToInt16Value[U ClampableToInt16](v U) int16 {
	return helpers.ClampToInt16Value(v)
}

func ClampToInt32Value[U ClampableToInt32](v U) int32 {
	return helpers.ClampToInt32Value(v)
}

func ClampToInt64Value[U ClampableToInt64](v U) int64 {
	return helpers.ClampToInt64Value(v)
}

func ComputeStridesRank(shape []int) int {
	return helpers.ComputeStridesRank(shape)
}

func ComputeStrides(dst []int, shape []int) []int {
	return helpers.ComputeStrides(dst, shape)
}

func SizeFromShape(shape []int) int {
	return helpers.SizeFromShape(shape)
}

func EnsureStrides(dst []int, strides []int, shape []int) []int {
	return helpers.EnsureStrides(dst, strides, shape)
}

func IsContiguous(strides []int, shape []int) bool {
	return helpers.IsContiguous(strides, shape)
}

func AdvanceOffsets(shape []int, indices []int, offsets []int, stridesDst, stridesSrc []int) bool {
	return helpers.AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc)
}

func AdvanceOffsets3(shape []int, indices []int, offsets []int, stridesDst, stridesA, stridesB []int) bool {
	return helpers.AdvanceOffsets3(shape, indices, offsets, stridesDst, stridesA, stridesB)
}

func AdvanceOffsets4(shape []int, indices []int, offsets []int, stridesDst, stridesCond, stridesA, stridesB []int) bool {
	return helpers.AdvanceOffsets4(shape, indices, offsets, stridesDst, stridesCond, stridesA, stridesB)
}

func IterateOffsets(shape []int, stridesDst, stridesSrc []int, callback func(offsets []int)) {
	helpers.IterateOffsets(shape, stridesDst, stridesSrc, callback)
}

func IterateOffsetsWithIndices(shape []int, stridesDst, stridesSrc []int, callback func(indices []int, offsets []int)) {
	helpers.IterateOffsetsWithIndices(shape, stridesDst, stridesSrc, callback)
}

func ComputeStrideOffset(indices []int, strides []int) int {
	return helpers.ComputeStrideOffset(indices, strides)
}

func WithWorkers(workers int) WorkerPoolOption {
	return helpers.WithWorkers(workers)
}

func WithChunkSizer(sizer ChunkSizer) WorkerPoolOption {
	return helpers.WithChunkSizer(sizer)
}

func WithTargetChunkSize(size int) WorkerPoolOption {
	return helpers.WithTargetChunkSize(size)
}
