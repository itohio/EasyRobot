package main

import (
	"fmt"
	"unicode"

	"github.com/itohio/EasyRobot/x/math/protocol/peg"
)

// Printer prints packets with decoded values
type Printer struct {
	crcVal    *CRCValidator
	precision int
}

// NewPrinter creates a new packet printer
func NewPrinter(crcVal *CRCValidator, precision int) *Printer {
	return &Printer{
		crcVal:    crcVal,
		precision: precision,
	}
}

// PrintHexLine prints a line of hex bytes with optional text view
// This is the shared function used by both capture mode and packet printing
func (p *Printer) PrintHexLine(data []byte, bytesPerLine int, showText bool) {
	if bytesPerLine <= 0 {
		bytesPerLine = 64
	}

	// Print hex bytes (no spaces between hex chars)
	for _, b := range data {
		fmt.Printf("%02X", b)
	}

	// If text view is enabled, add printable text
	if showText {
		// Add spacing between hex and text (like hex viewers)
		fmt.Print("  ")
		// Print printable characters, replace non-printable with '.'
		for _, b := range data {
			if unicode.IsPrint(rune(b)) && b >= 32 && b < 127 {
				fmt.Printf("%c", b)
			} else {
				fmt.Print(".")
			}
		}
	}
	fmt.Println()
}

// PrintHexCompact prints bytes as compact hex (no spaces) with configurable bytes per line
// Uses PrintHexLine internally for consistency
func (p *Printer) PrintHexCompact(data []byte, bytesPerLine int, showText bool) {
	if bytesPerLine <= 0 {
		bytesPerLine = 64
	}
	for i := 0; i < len(data); i += bytesPerLine {
		end := i + bytesPerLine
		if end > len(data) {
			end = len(data)
		}
		p.PrintHexLine(data[i:end], bytesPerLine, showText)
	}
}

// PrintPacket prints packet with decoded values
func (p *Printer) PrintPacket(state peg.State, bytesPerLine int, isAlternative bool, strictCRC bool, showText bool) {
	packet := state.Packet()
	if len(packet) == 0 {
		return
	}

	crcValid, crcMsg := p.crcVal.Validate(state)

	if isAlternative {
		fmt.Println("--- ALTERNATIVE PACKET ---")
	}

	// Use PrintHexCompact which respects bytes-per-line and text view
	p.PrintHexCompact(packet, bytesPerLine, showText)
	p.printDecodedValues(state)

	if p.crcVal.HasCRC(state) {
		p.printCRCStatus(crcValid, crcMsg, strictCRC)
	}

	if isAlternative {
		fmt.Println("--- END ALTERNATIVE PACKET ---")
	}
}

func (p *Printer) printDecodedValues(state peg.State) {
	fields := state.Fields()
	packet := state.Packet()

	if len(fields) == 0 {
		return
	}

	fmt.Println("Decoded values:")
	for _, field := range fields {
		formatted := p.formatValue(field.Value)
		typeStr := p.formatFieldType(field.Type)
		fmt.Printf("  offset[%d]=%s (%s)", field.Offset, formatted, typeStr)
		if field.Name != "" {
			fmt.Printf(" [%s]", field.Name)
		}
		fmt.Println()
	}

	// Show declared length if set
	if state.DeclaredLength() > 0 {
		fmt.Printf("  Declared length: %d bytes\n", state.DeclaredLength())
	}
	if state.MaxLength() > 0 {
		fmt.Printf("  Max length: %d bytes\n", state.MaxLength())
	}
	if len(packet) != state.DeclaredLength() && state.DeclaredLength() > 0 {
		fmt.Printf("  Actual length: %d bytes\n", len(packet))
	}
}

func (p *Printer) printCRCStatus(crcValid bool, crcMsg string, strictCRC bool) {
	if crcValid {
		fmt.Println("CRC: VALID")
	} else {
		if strictCRC {
			fmt.Printf("CRC: INVALID - %s (DISCARDED)\n", crcMsg)
		} else {
			fmt.Printf("CRC: WARNING - %s\n", crcMsg)
		}
	}
}

func (p *Printer) formatValue(val interface{}) string {
	switch v := val.(type) {
	case float32:
		return fmt.Sprintf("%.*f", p.precision, float64(v))
	case float64:
		return fmt.Sprintf("%.*f", p.precision, v)
	default:
		return fmt.Sprintf("%v", val)
	}
}

func (p *Printer) formatFieldType(ft peg.FieldType) string {
	switch ft {
	case peg.FieldUint8:
		return "uint8"
	case peg.FieldInt8:
		return "int8"
	case peg.FieldUint16LE:
		return "uint16(LE)"
	case peg.FieldUint16BE:
		return "uint16(BE)"
	case peg.FieldInt16LE:
		return "int16(LE)"
	case peg.FieldInt16BE:
		return "int16(BE)"
	case peg.FieldUint32LE:
		return "uint32(LE)"
	case peg.FieldUint32BE:
		return "uint32(BE)"
	case peg.FieldInt32LE:
		return "int32(LE)"
	case peg.FieldInt32BE:
		return "int32(BE)"
	case peg.FieldUint64LE:
		return "uint64(LE)"
	case peg.FieldUint64BE:
		return "uint64(BE)"
	case peg.FieldInt64LE:
		return "int64(LE)"
	case peg.FieldInt64BE:
		return "int64(BE)"
	case peg.FieldFloat32:
		return "float32"
	case peg.FieldFloat64:
		return "float64"
	case peg.FieldVarintU:
		return "varint(uint)"
	case peg.FieldVarintS:
		return "varint(int)"
	case peg.FieldStringPascal:
		return "string(pascal)"
	case peg.FieldStringC:
		return "string(null-terminated)"
	case peg.FieldStringFixed:
		return "string(fixed)"
	default:
		return "unknown"
	}
}
