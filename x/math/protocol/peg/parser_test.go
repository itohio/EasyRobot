package peg

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestParserErrors tests parser error handling
func TestParserErrors(t *testing.T) {
	require := require.New(t)
	tests := []struct {
		name    string
		pattern string
		wantErr bool
	}{
		{
			name:    "invalid hex literal odd digits",
			pattern: "^AA5$",
			wantErr: true,
		},
		{
			name:    "invalid hex byte",
			pattern: "^AAG$",
			wantErr: true,
		},
		{
			name:    "missing closing paren",
			pattern: "^(AA",
			wantErr: true,
		},
		{
			name:    "unexpected character",
			pattern: "^AA@$",
			wantErr: true,
		},
		{
			name:    "unknown field token",
			pattern: "^%X$",
			wantErr: true,
		},
		{
			name:    "invalid fixed string size",
			pattern: "^%abc$",
			wantErr: true,
		},
		{
			name:    "unexpected token after end",
			pattern: "^AA$BB",
			wantErr: true,
		},
		{
			name:    "valid pattern",
			pattern: "^AA%u$",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Compile(tt.pattern)
			if tt.wantErr {
				require.Error(err)
			} else {
				require.NoError(err)
			}
		})
	}
}

// TestParserEdgeCases tests parser edge cases
func TestParserEdgeCases(t *testing.T) {
	require := require.New(t)
	tests := []struct {
		name    string
		pattern string
		data    []byte
	}{
		{
			name:    "single child sequence",
			pattern: "^AA$",
			data:    []byte{0xAA},
		},
		{
			name:    "single branch group",
			pattern: "^(AA)$",
			data:    []byte{0xAA},
		},
		{
			name:    "end anchor with max length",
			pattern: "^AA$128",
			data:    []byte{0xAA},
		},
		{
			name:    "wildcard without number",
			pattern: "^AA*%u$",
			data:    []byte{0xAA, 0x01, 0x42},
		},
		{
			name:    "field with count",
			pattern: "^%3u$",
			data:    []byte{0x01, 0x02, 0x03},
		},
		{
			name:    "fixed string",
			pattern: "^%5s$",
			data:    []byte{'a', 'b', 0x00, 0x00, 0x00},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prog, err := Compile(tt.pattern)
			require.NoError(err)
			state := NewDefaultState()
			for _, b := range tt.data {
				state.AppendPacket(b)
			}
			_, err = prog.Decide(state)
			require.NoError(err)
		})
	}
}

// TestAllFieldTypes tests all field type tokens (parser recognition)
func TestAllFieldTypes(t *testing.T) {
	require := require.New(t)
	tests := []struct {
		name    string
		pattern string
		data    []byte
	}{
		{"int8", "^%i$", []byte{0x80}},
		{"int16 LE", "^%ii$", []byte{0x34, 0x12}},
		{"int16 BE", "^%II$", []byte{0x12, 0x34}},
		{"int32 LE", "^%iiii$", []byte{0x78, 0x56, 0x34, 0x12}},
		{"int32 BE", "^%IIII$", []byte{0x12, 0x34, 0x56, 0x78}},
		{"int64 LE", "^%iiiiii$", []byte{0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00}},
		{"int64 BE", "^%IIIIII$", []byte{0x00, 0x00, 0x00, 0x00, 0x12, 0x34, 0x56, 0x78}},
		{"uint32 LE", "^%uuuu$", []byte{0x78, 0x56, 0x34, 0x12}},
		{"uint32 BE", "^%UUUU$", []byte{0x12, 0x34, 0x56, 0x78}},
		{"float32", "^%f$", []byte{0x00, 0x00, 0x80, 0x3F}},
		{"float64", "^%F$", []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F}},
		{"CRC8", "^%c$", []byte{0x42}},
		{"CRC16", "^%cc$", []byte{0x34, 0x12}},
		{"length L", "^%L$", []byte{0x05}},
		{"length ll", "^%ll$", []byte{0x05, 0x00}},
		{"length LL", "^%LL$", []byte{0x00, 0x05}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prog, err := Compile(tt.pattern)
			require.NoError(err)
			state := NewDefaultState()
			for _, b := range tt.data {
				state.AppendPacket(b)
			}
			_, err = prog.Decide(state)
			require.NoError(err)
		})
	}
}

// TestParserNumberErrors tests parseNumber error cases
func TestParserNumberErrors(t *testing.T) {
	require := require.New(t)
	// This is tested indirectly through patterns, but let's add explicit test
	_, err := Compile("^AA$abc") // invalid number after $
	require.Error(err)
}

// TestParserConsumeChar tests consumeChar
func TestParserConsumeChar(t *testing.T) {
	require := require.New(t)
	// Test that ^ anchor is consumed
	prog, err := Compile("^AA$")
	require.NoError(err)
	require.NotNil(prog)
}

// TestParserPeekEOF tests peek at EOF
func TestParserPeekEOF(t *testing.T) {
	require := require.New(t)
	prog, err := Compile("AA")
	require.NoError(err)
	require.NotNil(prog)
}
