package tensor

type TensorSparseElement struct {
	Pos   []int
	Value float32
}

type TensorSparse struct {
	Dim  []int
	Data []TensorSparseElement
}
