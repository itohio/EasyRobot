package peg

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

var typeTokens = []string{
	"i", "L", "u", "ii", "II", "ll", "LL", "uu", "UU",
	"iiii", "IIII", "uuuu", "UUUU", "iiiiii", "IIIIII",
	"uuuuuu", "UUUUUU", "c", "cc", "f", "F", "v", "V", "s", "S",
}

func isIdentStart(ch rune) bool {
	return unicode.IsLetter(ch) || ch == '_'
}

func isIdentPart(ch rune) bool {
	return unicode.IsLetter(ch) || unicode.IsDigit(ch) || ch == '_'
}

// Compile converts pattern string to Program.
func Compile(pattern string) (*Program, error) {
	p := &parser{
		input: pattern,
	}

	var anchorStart bool
	var anchorEnd bool
	var maxLen int

	if p.consumeChar('^') {
		anchorStart = true
	}

	root, err := p.parseSequence()
	if err != nil {
		return nil, err
	}

	if p.peek() == '$' {
		anchorEnd = true
		p.pos++
		if unicode.IsDigit(p.peek()) {
			val, err := p.parseNumber()
			if err != nil {
				return nil, err
			}
			maxLen = val
		}
	}

	if !p.eof() {
		return nil, fmt.Errorf("unexpected token %c at position %d", p.peek(), p.pos)
	}

	return newProgram(pattern, root, anchorStart, anchorEnd, maxLen)
}

type parser struct {
	input    string
	pos      int
	fieldIdx int
}

func (p *parser) parseSequence() (node, error) {
	children := make([]node, 0)

	for !p.eof() {
		switch ch := p.peek(); ch {
		case ')', '|', '$':
			goto DONE
		default:
			elem, err := p.parseElement()
			if err != nil {
				return nil, err
			}
			children = append(children, elem)
		}
	}

DONE:
	if len(children) == 1 {
		return children[0], nil
	}
	return &sequenceNode{children: children}, nil
}

func (p *parser) parseElement() (node, error) {
	switch ch := p.peek(); {
	case ch == '(':
		return p.parseGroup()
	case ch == '*':
		return p.parseWildcard()
	case ch == '%':
		return p.parseField()
	case ch == '@':
		return p.parseCondition()
	case ch == '#':
		return p.parseOffset()
	case ch == 'l' || ch == 'L':
		return p.parseImplicitField()
	case ch == ';':
		p.pos++
		return p.parseElement()
	case isHexDigit(ch):
		return p.parseLiteral()
	case isTypeLetter(ch):
		// Type tokens can appear without % prefix (e.g., after groups)
		// Check if this looks like a type token
		saved := p.pos
		token := p.parseTypeToken()
		if token != "" {
			// Found a type token, parse it as a field
			specs, err := p.fieldSpec(token, 1)
			if err != nil {
				p.pos = saved
				return nil, err
			}
			return &fieldNode{spec: specs[0]}, nil
		}
		p.pos = saved
		return nil, fmt.Errorf("unexpected character %c at %d", ch, p.pos)
	default:
		return nil, fmt.Errorf("unexpected character %c at %d", ch, p.pos)
	}
}

func (p *parser) parseGroup() (node, error) {
	p.pos++ // skip '('
	lookahead := p.peek() == '#'
	branches := make([]node, 0)
	for {
		seq, err := p.parseSequence()
		if err != nil {
			return nil, err
		}
		branches = append(branches, seq)

		if p.peek() == '|' {
			p.pos++
			continue
		}
		break
	}
	if p.peek() != ')' {
		return nil, fmt.Errorf("missing closing ) at %d", p.pos)
	}
	p.pos++

	// If only one branch, return it as a sequence (not a choice)
	// This enables backtracking within groups
	if len(branches) == 1 {
		if lookahead {
			return &lookaheadNode{child: branches[0]}, nil
		}
		return branches[0], nil
	}
	if lookahead {
		return &lookaheadNode{child: &choiceNode{branches: branches}}, nil
	}
	return &choiceNode{branches: branches}, nil
}

func (p *parser) parseLiteral() (node, error) {
	start := p.pos
	for isHexDigit(p.peek()) {
		p.pos++
	}
	token := p.input[start:p.pos]
	if len(token)%2 != 0 {
		return nil, fmt.Errorf("literal requires even digits at %d", start)
	}
	bytes := make([]byte, len(token)/2)
	for i := 0; i < len(token); i += 2 {
		value, err := strconv.ParseUint(token[i:i+2], 16, 8)
		if err != nil {
			return nil, fmt.Errorf("invalid hex byte %s", token[i:i+2])
		}
		bytes[i/2] = byte(value)
	}
	return &literalNode{value: bytes}, nil
}

func (p *parser) parseWildcard() (node, error) {
	p.pos++ // skip *

	// Check for skip until pattern: *?pattern?
	if p.peek() == '?' {
		return p.parseSkipUntil()
	}

	// If no number follows, use -1 to mean "until MaxLength"
	count := -1
	if unicode.IsDigit(p.peek()) {
		num, err := p.parseNumber()
		if err != nil {
			return nil, err
		}
		count = num
	}
	return &wildcardNode{count: count}, nil
}

func (p *parser) parseSkipUntil() (node, error) {
	p.pos++ // skip ?
	// Parse pattern until next ?
	start := p.pos
	for !p.eof() && p.peek() != '?' {
		p.pos++
	}
	if p.eof() {
		return nil, fmt.Errorf("missing closing ? for skip until pattern at %d", start)
	}
	patternStr := p.input[start:p.pos]
	p.pos++ // skip closing ?

	// Parse pattern as hex bytes
	if len(patternStr)%2 != 0 {
		return nil, fmt.Errorf("skip until pattern requires even hex digits at %d", start)
	}
	pattern := make([]byte, len(patternStr)/2)
	for i := 0; i < len(patternStr); i += 2 {
		value, err := strconv.ParseUint(patternStr[i:i+2], 16, 8)
		if err != nil {
			return nil, fmt.Errorf("invalid hex byte in skip until pattern: %s", patternStr[i:i+2])
		}
		pattern[i/2] = byte(value)
	}
	return &skipUntilNode{pattern: pattern}, nil
}

func (p *parser) parseOffset() (node, error) {
	p.pos++ // skip #
	offset := 0
	relative := false

	if p.peek() == '+' || p.peek() == '-' {
		relative = true
		negative := p.peek() == '-'
		p.pos++ // skip + or -
		if !unicode.IsDigit(p.peek()) {
			return nil, fmt.Errorf("expected number after %c at %d", p.input[p.pos-1], p.pos-1)
		}
		num, err := p.parseNumber()
		if err != nil {
			return nil, err
		}
		if negative {
			offset = -num
		} else {
			offset = num
		}
	} else if unicode.IsDigit(p.peek()) {
		num, err := p.parseNumber()
		if err != nil {
			return nil, err
		}
		offset = num
	} else if p.peek() == '(' {
		// TODO: support #(expr) - for now just error
		return nil, fmt.Errorf("offset expressions #(expr) not yet implemented at %d", p.pos)
	} else {
		return nil, fmt.Errorf("expected number, +, -, or ( after # at %d", p.pos)
	}
	return &offsetNode{offset: offset, relative: relative}, nil
}

func (p *parser) parseCondition() (node, error) {
	p.pos++ // skip @
	if p.peek() != '(' {
		return nil, fmt.Errorf("expected ( after @ at %d", p.pos)
	}
	expr, err := p.parseExpressionBlock()
	if err != nil {
		return nil, err
	}
	return p.buildExpressionNode("", expr, true)
}

func (p *parser) parseField() (node, error) {
	p.pos++ // skip %

	useStar := false
	if p.peek() == '*' {
		useStar = true
		p.pos++
	}

	name, hasName := p.parseOptionalName()

	// Expression field
	if p.peek() == '(' {
		expr, err := p.parseExpressionBlock()
		if err != nil {
			return nil, err
		}
		return p.buildExpressionNode(name, expr, false)
	}

	if !useStar && p.peek() == '*' {
		useStar = true
		p.pos++
	}

	count, hasCount, stride, hasStride := p.parseArrayPrefix()

	// Structs and array of structs
	if p.peek() == '{' {
		if hasCount || useStar {
			num := 0
			if hasCount {
				num = count
			}
			return p.parseArrayOfStructs(num, useStar, name)
		}
		return p.parseStruct(name)
	}

	if useStar && !hasCount {
		token := p.parseTypeToken()
		if token == "" {
			return nil, fmt.Errorf("missing field type at %d", p.pos)
		}
		return p.parseArrayWithStar(token, name)
	}

	token := ""
	if !hasCount {
		token = p.parseTypeToken()
		if token == "" {
			return nil, fmt.Errorf("missing field type at %d", p.pos)
		}
	}

	if hasCount {
		if hasStride {
			if token == "" {
				token = p.parseTypeToken()
				if token == "" {
					return nil, fmt.Errorf("missing field type at %d", p.pos)
				}
			}
			return p.parseArrayWithStride(name, count, stride, token)
		}
		if token == "" {
			token = p.parseTypeToken()
			if token == "" {
				return nil, fmt.Errorf("missing field type at %d", p.pos)
			}
		}
		specs, err := p.fieldSpec(token, count)
		if err != nil {
			return nil, err
		}
		nodes := make([]node, len(specs))
		for i, s := range specs {
			if hasName {
				s.Name = fmt.Sprintf("%s_%d", name, i)
			}
			nodes[i] = &fieldNode{spec: s}
		}
		if len(nodes) == 1 {
			return nodes[0], nil
		}
		return &sequenceNode{children: nodes}, nil
	}

	specs, err := p.fieldSpec(token, 1)
	if err != nil {
		return nil, err
	}
	if hasName {
		specs[0].Name = name
	}
	return &fieldNode{spec: specs[0]}, nil
}

func (p *parser) parseOptionalName() (string, bool) {
	start := p.pos
	if !isIdentStart(p.peek()) {
		return "", false
	}
	for isIdentPart(p.peek()) {
		p.pos++
	}
	if p.peek() != ':' {
		p.pos = start
		return "", false
	}
	name := p.input[start:p.pos]
	p.pos++
	return name, true
}

func (p *parser) parseExpressionBlock() (string, error) {
	if p.peek() != '(' {
		return "", fmt.Errorf("expected ( at %d", p.pos)
	}
	p.pos++ // skip '('
	depth := 1
	start := p.pos
	for !p.eof() {
		ch := p.peek()
		if ch == '(' {
			depth++
		} else if ch == ')' {
			depth--
			if depth == 0 {
				expr := p.input[start:p.pos]
				p.pos++
				return expr, nil
			}
		}
		p.pos++
	}
	return "", fmt.Errorf("missing closing ) for expression starting at %d", start-1)
}

func (p *parser) parseArrayPrefix() (int, bool, int, bool) {
	saved := p.pos
	if !unicode.IsDigit(p.peek()) {
		return 0, false, 0, false
	}
	num, err := p.parseNumber()
	if err != nil {
		p.pos = saved
		return 0, false, 0, false
	}
	count := num
	hasStride := false
	stride := 0
	if p.peek() == ':' {
		pos := p.pos
		p.pos++
		if unicode.IsDigit(p.peek()) {
			num, err := p.parseNumber()
			if err == nil {
				stride = num
				hasStride = true
			} else {
				p.pos = pos
			}
		} else {
			p.pos = pos
		}
	}
	return count, true, stride, hasStride
}

func (p *parser) parseArrayWithStride(name string, count, stride int, token string) (node, error) {
	specs, err := p.fieldSpec(token, 1)
	if err != nil {
		return nil, err
	}
	fieldSize := specs[0].TotalBytes()
	if stride <= 0 || stride < fieldSize {
		return nil, fmt.Errorf("stride must be >= field size for %s", token)
	}
	return &arrayStrideNode{
		spec:     specs[0],
		baseName: name,
		count:    count,
		stride:   stride,
	}, nil
}

func (p *parser) parseStruct(_ string) (node, error) {
	p.pos++ // skip {
	fields := make([]node, 0)

	for {
		if p.peek() == '}' {
			break
		}

		// Parse field: name:type or just type
		fieldNode, err := p.parseStructField()
		if err != nil {
			return nil, err
		}
		fields = append(fields, fieldNode)

		if p.peek() == ',' {
			p.pos++ // skip ,
			continue
		}
		if p.peek() == '}' {
			break
		}
		return nil, fmt.Errorf("expected , or } in struct at %d", p.pos)
	}

	if p.peek() != '}' {
		return nil, fmt.Errorf("missing closing } in struct at %d", p.pos)
	}
	p.pos++ // skip }

	if len(fields) == 1 {
		return fields[0], nil
	}
	return &sequenceNode{children: fields}, nil
}

func (p *parser) parseStructField() (node, error) {
	// Parse optional name: fieldName:type or just type
	var name string
	start := p.pos
	for !p.eof() && p.peek() != ':' && p.peek() != ',' && p.peek() != '}' {
		if !isTypeLetter(p.peek()) && !unicode.IsLetter(p.peek()) && p.peek() != '_' {
			break
		}
		p.pos++
	}

	if p.peek() == ':' {
		name = p.input[start:p.pos]
		p.pos++ // skip :
	} else {
		// No name, reset position
		p.pos = start
	}

	// Parse field type
	if p.peek() == '%' {
		// Nested field (could be another struct or array)
		return p.parseField()
	}

	token := p.parseTypeToken()
	if token == "" {
		return nil, fmt.Errorf("missing field type in struct at %d", p.pos)
	}

	specs, err := p.fieldSpec(token, 1)
	if err != nil {
		return nil, err
	}

	if name != "" {
		specs[0].Name = name
	}

	return &fieldNode{spec: specs[0]}, nil
}

func (p *parser) parseArrayOfStructs(count int, useStar bool, baseName string) (node, error) {
	p.pos++ // skip {

	// Parse struct fields
	structFields := make([]FieldSpec, 0)
	for {
		if p.peek() == '}' {
			break
		}

		// Parse field type
		token := p.parseTypeToken()
		if token == "" {
			return nil, fmt.Errorf("missing field type in array struct at %d", p.pos)
		}

		specs, err := p.fieldSpec(token, 1)
		if err != nil {
			return nil, err
		}
		structFields = append(structFields, specs[0])

		if p.peek() == ',' {
			p.pos++ // skip ,
			continue
		}
		if p.peek() == '}' {
			break
		}
		return nil, fmt.Errorf("expected , or } in array struct at %d", p.pos)
	}

	if p.peek() != '}' {
		return nil, fmt.Errorf("missing closing } in array struct at %d", p.pos)
	}
	p.pos++ // skip }

	return &arrayOfStructsNode{
		count:       count,
		useStar:     useStar,
		baseName:    baseName,
		structSpecs: structFields,
	}, nil
}

func (p *parser) parseArrayWithStar(token, name string) (node, error) {
	// %*type - read until exhausted
	specs, err := p.fieldSpec(token, 1)
	if err != nil {
		return nil, err
	}
	return &arrayWithStarNode{spec: specs[0], baseName: name}, nil
}

func (p *parser) parseImplicitField() (node, error) {
	token := p.parseLengthToken()
	specs, err := p.fieldSpec(token, 1)
	if err != nil {
		return nil, err
	}
	return &fieldNode{spec: specs[0]}, nil
}

func (p *parser) fieldSpec(token string, count int) ([]FieldSpec, error) {
	specs := make([]FieldSpec, 0, count)
	for i := 0; i < count; i++ {
		name := fmt.Sprintf("field_%d", p.nextFieldID())
		switch token {
		case "i":
			specs = append(specs, FieldSpec{Name: name, Type: FieldInt8})
		case "L":
			specs = append(specs, FieldSpec{Name: name, Type: FieldUint8, Kind: FieldKindLength})
		case "u":
			specs = append(specs, FieldSpec{Name: name, Type: FieldUint8})
		case "ii":
			specs = append(specs, FieldSpec{Name: name, Type: FieldInt16LE})
		case "II":
			specs = append(specs, FieldSpec{Name: name, Type: FieldInt16BE})
		case "ll":
			specs = append(specs, FieldSpec{Name: name, Type: FieldUint16LE, Kind: FieldKindLength})
		case "LL":
			specs = append(specs, FieldSpec{Name: name, Type: FieldUint16BE, Kind: FieldKindLength})
		case "uu":
			specs = append(specs, FieldSpec{Name: name, Type: FieldUint16LE})
		case "UU":
			specs = append(specs, FieldSpec{Name: name, Type: FieldUint16BE})
		case "iiii":
			specs = append(specs, FieldSpec{Name: name, Type: FieldInt32LE})
		case "IIII":
			specs = append(specs, FieldSpec{Name: name, Type: FieldInt32BE})
		case "uuuu":
			specs = append(specs, FieldSpec{Name: name, Type: FieldUint32LE})
		case "UUUU":
			specs = append(specs, FieldSpec{Name: name, Type: FieldUint32BE})
		case "iiiiii":
			specs = append(specs, FieldSpec{Name: name, Type: FieldInt64LE})
		case "IIIIII":
			specs = append(specs, FieldSpec{Name: name, Type: FieldInt64BE})
		case "uuuuuu":
			specs = append(specs, FieldSpec{Name: name, Type: FieldUint64LE})
		case "UUUUUU":
			specs = append(specs, FieldSpec{Name: name, Type: FieldUint64BE})
		case "c":
			specs = append(specs, FieldSpec{Name: name, Type: FieldUint8})
		case "cc":
			specs = append(specs, FieldSpec{Name: name, Type: FieldUint16LE})
		case "f":
			specs = append(specs, FieldSpec{Name: name, Type: FieldFloat32})
		case "F":
			specs = append(specs, FieldSpec{Name: name, Type: FieldFloat64})
		case "v":
			specs = append(specs, FieldSpec{Name: name, Type: FieldVarintU})
		case "V":
			specs = append(specs, FieldSpec{Name: name, Type: FieldVarintS})
		case "s":
			specs = append(specs, FieldSpec{Name: name, Type: FieldStringPascal})
		case "S":
			specs = append(specs, FieldSpec{Name: name, Type: FieldStringC})
		default:
			if strings.HasSuffix(token, "s") {
				lengthStr := strings.TrimSuffix(token, "s")
				n, err := strconv.Atoi(lengthStr)
				if err != nil {
					return nil, fmt.Errorf("invalid fixed string size in %s", token)
				}
				specs = append(specs, FieldSpec{Name: name, Type: FieldStringFixed, Size: n})
				continue
			}
			return nil, fmt.Errorf("unknown field token %s", token)
		}
	}
	return specs, nil
}

func (p *parser) buildExpressionNode(name, exprSrc string, isCondition bool) (node, error) {
	baseToken, err := p.findBaseTypeToken(exprSrc)
	if err != nil {
		return nil, err
	}
	var baseSpec *FieldSpec
	if baseToken != "" {
		if isCondition {
			return nil, fmt.Errorf("conditions cannot consume bytes")
		}
		specs, err := p.fieldSpec(baseToken, 1)
		if err != nil {
			return nil, err
		}
		baseSpec = &specs[0]
		if name == "" {
			name = baseSpec.Name
		}
	}
	exprAST, err := parseExpression(exprSrc)
	if err != nil {
		return nil, err
	}
	return &expressionNode{
		name:        name,
		expr:        exprAST,
		exprSrc:     exprSrc,
		baseSpec:    baseSpec,
		baseType:    baseToken,
		isCondition: isCondition,
	}, nil
}

func (p *parser) findBaseTypeToken(exprSrc string) (string, error) {
	// Find the first type token in the expression. If expression starts with parentheses,
	// look inside them. The type token indicates we decode a field first, then use its
	// value in the expression. For example: %length:(uu-2000) means:
	// - Decode a field of type 'uu'
	// - Evaluate expression (uu-2000) where 'uu' refers to the decoded field value
	// - Store result as field 'length'
	trimmed := strings.TrimSpace(exprSrc)
	if len(trimmed) == 0 {
		return "", nil
	}

	reader := strings.NewReader(trimmed)

	// If expression starts with '(', skip to the content inside
	if trimmed[0] == '(' {
		r, _, err := reader.ReadRune()
		if err != nil || r != '(' {
			return "", nil
		}
		// Skip whitespace after '('
		for {
			r, _, err := reader.ReadRune()
			if err != nil {
				return "", nil
			}
			if !unicode.IsSpace(r) {
				reader.UnreadRune()
				break
			}
		}
	}

	var builder strings.Builder
	var nextRune rune
	nextRead := false

	// Read the first identifier
	r, _, err := reader.ReadRune()
	if err != nil {
		return "", nil
	}
	if !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_') {
		// Expression starts with operator - no base type
		return "", nil
	}

	builder.WriteRune(r)
	for {
		curr, _, err := reader.ReadRune()
		if err != nil {
			nextRune = 0
			nextRead = true
			break
		}
		if unicode.IsLetter(curr) || unicode.IsDigit(curr) || curr == '_' {
			builder.WriteRune(curr)
		} else {
			nextRune = curr
			nextRead = true
			break
		}
	}

	token := builder.String()
	for _, tt := range typeTokens {
		if token == tt {
			// Check if the token is followed by operator/paren/EOF - if so, it's a base type
			if !nextRead {
				nextRune, _, _ = reader.ReadRune()
			}
			if nextRune == 0 || nextRune == '(' || nextRune == ')' || nextRune == '+' || nextRune == '-' ||
				nextRune == '*' || nextRune == '/' || nextRune == '<' || nextRune == '>' ||
				nextRune == '=' || nextRune == '!' || nextRune == '&' || nextRune == '|' ||
				unicode.IsSpace(nextRune) {
				return token, nil
			}
		}
	}
	return "", nil
}

func (p *parser) parseTypeToken() string {
	start := p.pos
	for isTypeLetter(p.peek()) {
		p.pos++
	}
	return p.input[start:p.pos]
}

func (p *parser) parseLengthToken() string {
	start := p.pos
	for p.peek() == 'l' || p.peek() == 'L' || p.peek() == 'c' || p.peek() == 'C' {
		p.pos++
	}
	return p.input[start:p.pos]
}

func (p *parser) consumeDigits() string {
	start := p.pos
	for unicode.IsDigit(p.peek()) {
		p.pos++
	}
	return p.input[start:p.pos]
}

func isTypeLetter(ch rune) bool {
	return strings.ContainsRune("uUiIfFsSlLcCvV", ch)
}

func (p *parser) parseNumber() (int, error) {
	start := p.pos
	for unicode.IsDigit(p.peek()) {
		p.pos++
	}
	if start == p.pos {
		return 0, fmt.Errorf("expected number at %d", p.pos)
	}
	value, err := strconv.Atoi(p.input[start:p.pos])
	if err != nil {
		return 0, err
	}
	return value, nil
}

func (p *parser) nextFieldID() int {
	p.fieldIdx++
	return p.fieldIdx
}

func (p *parser) peek() rune {
	if p.pos >= len(p.input) {
		return 0
	}
	return rune(p.input[p.pos])
}

func (p *parser) consumeChar(ch rune) bool {
	if p.peek() == ch {
		p.pos++
		return true
	}
	return false
}

func (p *parser) eof() bool {
	return p.pos >= len(p.input)
}

func isHexDigit(ch rune) bool {
	return (ch >= '0' && ch <= '9') ||
		(ch >= 'a' && ch <= 'f') ||
		(ch >= 'A' && ch <= 'F')
}
