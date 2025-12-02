package nmea

import (
	"bytes"
	"errors"
	"strconv"
	"strings"
	"time"
)

var (
	// ErrInvalidSentence is returned when a sentence is invalid or cannot be parsed
	ErrInvalidSentence = errors.New("nmea: invalid sentence")
	// ErrChecksumMismatch is returned when sentence checksum doesn't match
	ErrChecksumMismatch = errors.New("nmea: checksum mismatch")
)

// Sentence represents a parsed NMEA sentence.
type Sentence struct {
	Type      string   // Sentence type (e.g., "GPGGA", "GPRMC")
	Fields    []string // Fields after splitting by comma
	Checksum  byte     // Calculated checksum
	Raw       string   // Raw sentence string
	Timestamp time.Time // When the sentence was parsed
}

// ParseSentence parses a single NMEA sentence string.
// Expected format: $TYPE,field1,field2,...*checksum\r\n
func ParseSentence(line string) (*Sentence, error) {
	line = strings.TrimSpace(line)
	if len(line) == 0 {
		return nil, ErrInvalidSentence
	}

	// Remove trailing \r\n if present
	line = strings.TrimRight(line, "\r\n")

	// Must start with $
	if line[0] != '$' {
		return nil, ErrInvalidSentence
	}

	// Find checksum separator
	checksumIdx := strings.LastIndex(line, "*")
	if checksumIdx < 0 {
		return nil, ErrInvalidSentence
	}

	// Extract checksum
	checksumStr := line[checksumIdx+1:]
	if len(checksumStr) != 2 {
		return nil, ErrInvalidSentence
	}

	checksum, err := strconv.ParseUint(checksumStr, 16, 8)
	if err != nil {
		return nil, ErrInvalidSentence
	}

	// Extract sentence without $ and checksum
	sentence := line[1:checksumIdx]

	// Verify checksum
	calculatedChecksum := calculateChecksum(sentence)
	if byte(checksum) != calculatedChecksum {
		return nil, ErrChecksumMismatch
	}

	// Split into type and fields
	parts := strings.Split(sentence, ",")
	if len(parts) == 0 {
		return nil, ErrInvalidSentence
	}

	return &Sentence{
		Type:      parts[0],
		Fields:    parts[1:],
		Checksum:  byte(checksum),
		Raw:       line,
		Timestamp: time.Now(),
	}, nil
}

// calculateChecksum calculates the NMEA checksum (XOR of all bytes).
func calculateChecksum(sentence string) byte {
	var checksum byte
	for i := range sentence {
		checksum ^= sentence[i]
	}
	return checksum
}

// Parser parses NMEA sentences from a byte stream.
type Parser struct {
	buf []byte
}

// NewParser creates a new NMEA parser.
func NewParser() *Parser {
	return &Parser{
		buf: make([]byte, 0, 1024),
	}
}

// ParseBytes processes incoming bytes and returns any complete sentences found.
// Returns a slice of parsed sentences and the number of bytes consumed.
func (p *Parser) ParseBytes(data []byte) ([]*Sentence, int, error) {
	p.buf = append(p.buf, data...)

	var sentences []*Sentence
	consumed := 0

	for {
		// Find start of sentence ($)
		startIdx := bytes.IndexByte(p.buf[consumed:], '$')
		if startIdx < 0 {
			// No more sentences, keep remaining data
			break
		}
		startIdx += consumed

		// Find end of sentence (\r\n or \n)
		endIdx := -1
		for i := startIdx + 1; i < len(p.buf); i++ {
			if p.buf[i] == '\n' {
				endIdx = i + 1
				break
			}
		}

		if endIdx < 0 {
			// Incomplete sentence, wait for more data
			break
		}

		// Extract sentence
		sentenceStr := string(p.buf[startIdx:endIdx])
		sentence, err := ParseSentence(sentenceStr)
		if err != nil {
			// Skip invalid sentence
			consumed = endIdx
			continue
		}

		sentences = append(sentences, sentence)
		consumed = endIdx
	}

	// Remove consumed bytes from buffer
	if consumed > 0 {
		copy(p.buf, p.buf[consumed:])
		p.buf = p.buf[:len(p.buf)-consumed]
	}

	return sentences, consumed, nil
}

// Reset clears the parser buffer.
func (p *Parser) Reset() {
	p.buf = p.buf[:0]
}

