package peg

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// mustCompile is a test helper that compiles a pattern and fails the test if compilation fails.
func mustCompile(t *testing.T, pattern string) *Program {
	t.Helper()
	prog, err := Compile(pattern)
	require.NoError(t, err)
	return prog
}

