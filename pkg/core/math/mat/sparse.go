package mat

type MatrixSparseTriplet struct {
	Row, Col int
	Value    float32
}

type MatrixSparse struct {
	Rows, Cols int
	Data       []MatrixSparseTriplet
}
