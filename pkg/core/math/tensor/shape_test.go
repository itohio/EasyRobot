package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShapeIterator(t *testing.T) {
	t.Run("no fixed dimensions - iterate all", func(t *testing.T) {
		shape := NewShape(2, 3)

		expected := [][]int{
			{0, 0}, {0, 1}, {0, 2},
			{1, 0}, {1, 1}, {1, 2},
		}

		var got [][]int
		for indices := range shape.Iterator() {
			got = append(got, indices)
		}

		assert.Equal(t, expected, got)
	})

	t.Run("fix first dimension", func(t *testing.T) {
		shape := NewShape(2, 3, 4)

		// Should iterate over dimensions 1 and 2 (12 combinations)
		expectedCount := 3 * 4
		count := 0

		for indices := range shape.Iterator(0, 1) {
			assert.Equal(t, 3, len(indices)) // All dimensions
			assert.Equal(t, 1, indices[0])   // Fixed dimension 0
			count++
		}

		assert.Equal(t, expectedCount, count)

		// Verify first and last indices
		var first, last []int
		for indices := range shape.Iterator(0, 1) {
			if first == nil {
				first = indices
			}
			last = indices
		}
		assert.Equal(t, []int{1, 0, 0}, first) // Fixed dim 0 = 1, remaining dims start at 0
		assert.Equal(t, []int{1, 2, 3}, last)  // Last combination
	})

	t.Run("fix middle dimension", func(t *testing.T) {
		shape := NewShape(2, 3, 4)

		// Should iterate over dimensions 0 and 2 (8 combinations)
		expectedCount := 2 * 4
		count := 0

		for indices := range shape.Iterator(1, 2) {
			assert.Equal(t, 3, len(indices)) // All dimensions
			assert.Equal(t, 2, indices[1])   // Fixed dimension 1
			count++
		}

		assert.Equal(t, expectedCount, count)

		// Verify first indices
		var first []int
		for indices := range shape.Iterator(1, 2) {
			if first == nil {
				first = indices
			}
			break
		}
		assert.Equal(t, []int{0, 2, 0}, first) // Fixed dim 1 = 2, remaining dims start at 0
	})

	t.Run("fix multiple dimensions", func(t *testing.T) {
		shape := NewShape(2, 3, 4, 5)

		// Should iterate over dimensions 1 and 3 (15 combinations)
		expectedCount := 3 * 5
		count := 0

		for indices := range shape.Iterator(0, 1, 2, 3) {
			assert.Equal(t, 4, len(indices)) // All dimensions
			assert.Equal(t, 1, indices[0])   // Fixed dimension 0
			assert.Equal(t, 3, indices[2])   // Fixed dimension 2
			count++
		}

		assert.Equal(t, expectedCount, count)

		// Verify first indices
		var first []int
		for indices := range shape.Iterator(0, 1, 2, 3) {
			if first == nil {
				first = indices
			}
			break
		}
		assert.Equal(t, []int{1, 0, 3, 0}, first)
	})

	t.Run("fix all dimensions", func(t *testing.T) {
		shape := NewShape(2, 3, 4)

		// Should iterate once (single combination)
		count := 0
		var indices []int

		for idx := range shape.Iterator(0, 1, 1, 2, 2, 3) {
			indices = idx
			count++
		}

		assert.Equal(t, 1, count)
		assert.Equal(t, []int{1, 2, 3}, indices)
	})

	t.Run("empty shape", func(t *testing.T) {
		shape := NewShape()

		count := 0
		var indices []int
		for idx := range shape.Iterator() {
			indices = idx
			count++
		}

		assert.Equal(t, 1, count) // Should yield once with empty indices
		assert.Equal(t, []int{}, indices)
	})

	t.Run("single dimension", func(t *testing.T) {
		shape := NewShape(3)

		expected := [][]int{{0}, {1}, {2}}
		var got [][]int

		for indices := range shape.Iterator() {
			got = append(got, indices)
		}

		assert.Equal(t, expected, got)
	})

	t.Run("row-major order", func(t *testing.T) {
		shape := NewShape(2, 3)

		// Row-major: last dimension changes fastest
		expected := [][]int{
			{0, 0}, {0, 1}, {0, 2}, // First row
			{1, 0}, {1, 1}, {1, 2}, // Second row
		}

		var got [][]int
		for indices := range shape.Iterator() {
			got = append(got, indices)
		}

		assert.Equal(t, expected, got)
	})

	t.Run("indices usable with At and SetAt", func(t *testing.T) {
		shape := NewShape(2, 3)
		tensor := FromFloat32(shape, []float32{1, 2, 3, 4, 5, 6})

		// Set values using iterator indices
		value := float32(10.0)
		for indices := range shape.Iterator() {
			tensor.SetAt(indices, value)
			value++
		}

		// Read values using iterator indices
		value = float32(10.0)
		for indices := range shape.Iterator() {
			got := tensor.At(indices...)
			assert.Equal(t, value, got)
			value++
		}
	})

	t.Run("fix dimension and use with At", func(t *testing.T) {
		shape := NewShape(2, 3, 4)
		data := make([]float32, shape.Size())
		for i := range data {
			data[i] = float32(i)
		}
		tensor := FromFloat32(shape, data)

		// Fix first dimension at 1, iterate over remaining
		expectedCount := 3 * 4
		count := 0

		for indices := range shape.Iterator(0, 1) {
			// indices should be usable directly with At
			_ = tensor.At(indices...)
			assert.Equal(t, 1, indices[0]) // Dimension 0 should be fixed
			count++
		}

		assert.Equal(t, expectedCount, count)
	})
}

func TestShapeIteratorPanics(t *testing.T) {
	t.Run("odd number of arguments", func(t *testing.T) {
		shape := NewShape(2, 3)
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Contains(t, r.(string), "even number")
		}()
		shape.Iterator(0, 1, 2) // Odd number of arguments
	})

	t.Run("invalid dimension index", func(t *testing.T) {
		shape := NewShape(2, 3)
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Contains(t, r.(string), "out of range")
		}()
		shape.Iterator(5, 0) // Dimension 5 doesn't exist
	})

	t.Run("invalid fixed value", func(t *testing.T) {
		shape := NewShape(2, 3)
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Contains(t, r.(string), "out of range")
		}()
		shape.Iterator(0, 5) // Value 5 out of range for dimension 0 (size 2)
	})

	t.Run("negative dimension index", func(t *testing.T) {
		shape := NewShape(2, 3)
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Contains(t, r.(string), "out of range")
		}()
		shape.Iterator(-1, 0)
	})

	t.Run("negative fixed value", func(t *testing.T) {
		shape := NewShape(2, 3)
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Contains(t, r.(string), "out of range")
		}()
		shape.Iterator(0, -1)
	})

	t.Run("duplicate axis", func(t *testing.T) {
		shape := NewShape(2, 3, 4)
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Contains(t, r.(string), "duplicate axis")
		}()
		shape.Iterator(0, 1, 0, 2) // Duplicate axis 0
	})
}
