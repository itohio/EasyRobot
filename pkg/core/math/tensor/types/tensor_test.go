package types

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMust(t *testing.T) {
	t.Run("no error", func(t *testing.T) {
		value := "test"
		result := Must(value, nil)
		assert.Equal(t, value, result)
	})

	t.Run("with error - panics", func(t *testing.T) {
		err := errors.New("test error")
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Equal(t, err, r)
		}()
		Must(nil, err)
	})

	t.Run("returns value when no error", func(t *testing.T) {
		tests := []struct {
			name  string
			value any
		}{
			{
				name:  "string",
				value: "test",
			},
			{
				name:  "int",
				value: 42,
			},
			{
				name:  "float",
				value: 3.14,
			},
			{
				name:  "slice",
				value: []int{1, 2, 3},
			},
			{
				name:  "nil value",
				value: nil,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := Must(tt.value, nil)
				assert.Equal(t, tt.value, result)
			})
		}
	})

	t.Run("different error types", func(t *testing.T) {
		tests := []struct {
			name string
			err  error
		}{
			{
				name: "simple error",
				err:  errors.New("simple error"),
			},
			{
				name: "wrapped error",
				err:  errors.New("wrapped: " + errors.New("inner").Error()),
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				defer func() {
					r := recover()
					assert.NotNil(t, r)
					assert.Equal(t, tt.err, r)
				}()
				Must(nil, tt.err)
			})
		}
	})
}
