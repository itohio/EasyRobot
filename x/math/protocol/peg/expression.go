package peg

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

type expression interface {
	eval(env *exprEnv) (exprValue, error)
}

type exprValue struct {
	num    float64
	bool   bool
	isBool bool
}

func (v exprValue) asNumber() (float64, bool) {
	if v.isBool {
		return 0, false
	}
	return v.num, true
}

func (v exprValue) asBool() (bool, bool) {
	if v.isBool {
		return v.bool, true
	}
	return v.num != 0, true
}

type exprEnv struct {
	state        State
	currentName  string
	currentValue exprValue
	baseType     string
	cache        map[string]exprValue
}

func (env *exprEnv) lookup(name string) (exprValue, error) {
	if env.cache == nil {
		env.cache = make(map[string]exprValue)
		fields := env.state.Fields()
		for i := len(fields) - 1; i >= 0; i-- {
			val, ok := toExprValue(fields[i].Value)
			if ok {
				env.cache[fields[i].Name] = val
			}
		}
	}
	if name == "" {
		return exprValue{}, fmt.Errorf("empty identifier")
	}
	if strings.EqualFold(name, "value") {
		return env.currentValue, nil
	}
	if env.currentName != "" && name == env.currentName {
		return env.currentValue, nil
	}
	if env.baseType != "" && name == env.baseType {
		return env.currentValue, nil
	}
	if val, ok := env.cache[name]; ok {
		return val, nil
	}
	return exprValue{}, fmt.Errorf("unknown identifier %s", name)
}

func parseExpression(input string) (expression, error) {
	p := &expressionParser{
		input: strings.TrimSpace(input),
	}
	expr, err := p.parseOr()
	if err != nil {
		return nil, err
	}
	p.skipSpaces()
	if p.pos != len(p.input) {
		return nil, fmt.Errorf("unexpected token %s", p.input[p.pos:])
	}
	return expr, nil
}

type expressionParser struct {
	input string
	pos   int
}

func (p *expressionParser) parseOr() (expression, error) {
	left, err := p.parseAnd()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpaces()
		if p.matchString("||") {
			right, err := p.parseAnd()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: "||", left: left, right: right}
			continue
		}
		break
	}
	return left, nil
}

func (p *expressionParser) parseAnd() (expression, error) {
	left, err := p.parseEquality()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpaces()
		if p.matchString("&&") {
			right, err := p.parseEquality()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: "&&", left: left, right: right}
			continue
		}
		break
	}
	return left, nil
}

func (p *expressionParser) parseEquality() (expression, error) {
	left, err := p.parseComparison()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpaces()
		if p.matchString("==") {
			right, err := p.parseComparison()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: "==", left: left, right: right}
			continue
		}
		if p.matchString("!=") {
			right, err := p.parseComparison()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: "!=", left: left, right: right}
			continue
		}
		break
	}
	return left, nil
}

func (p *expressionParser) parseComparison() (expression, error) {
	left, err := p.parseBitwiseOr()
	if err != nil {
		return nil, err
	}
	p.skipSpaces()
	op := p.matchRelOp()
	if op == "" {
		return left, nil
	}
	right, err := p.parseBitwiseOr()
	if err != nil {
		return nil, err
	}
	expr := &binaryExpr{op: op, left: left, right: right}
	current := right
	for {
		p.skipSpaces()
		nextOp := p.matchRelOp()
		if nextOp == "" {
			break
		}
		nextRight, err := p.parseBitwiseOr()
		if err != nil {
			return nil, err
		}
		expr = &binaryExpr{
			op:   "&&",
			left: expr,
			right: &binaryExpr{
				op:    nextOp,
				left:  current,
				right: nextRight,
			},
		}
		current = nextRight
	}
	return expr, nil
}

func (p *expressionParser) parseBitwiseOr() (expression, error) {
	left, err := p.parseBitwiseXor()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpaces()
		// Check for logical OR first (||) - if found, don't consume, break
		if p.pos+1 < len(p.input) && p.input[p.pos] == '|' && p.input[p.pos+1] == '|' {
			break // Let parseOr() handle this
		}
		if p.match('|') { // Single | for bitwise OR
			right, err := p.parseBitwiseXor()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: "|", left: left, right: right}
			continue
		}
		break
	}
	return left, nil
}

func (p *expressionParser) parseBitwiseXor() (expression, error) {
	left, err := p.parseBitwiseAnd()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpaces()
		if p.match('^') {
			right, err := p.parseBitwiseAnd()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: "^", left: left, right: right}
			continue
		}
		break
	}
	return left, nil
}

func (p *expressionParser) parseBitwiseAnd() (expression, error) {
	left, err := p.parseAddSub()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpaces()
		// Check for logical AND first (&&) - if found, don't consume, break
		if p.pos+1 < len(p.input) && p.input[p.pos] == '&' && p.input[p.pos+1] == '&' {
			break // Let parseAnd() handle this
		}
		if p.match('&') { // Single & for bitwise AND
			right, err := p.parseAddSub()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: "&", left: left, right: right}
			continue
		}
		break
	}
	return left, nil
}

func (p *expressionParser) parseShift() (expression, error) {
	left, err := p.parseMulDiv()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpaces()
		if p.matchString("<<") {
			right, err := p.parseMulDiv()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: "<<", left: left, right: right}
			continue
		}
		if p.matchString(">>") {
			right, err := p.parseMulDiv()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: ">>", left: left, right: right}
			continue
		}
		break
	}
	return left, nil
}

func (p *expressionParser) matchRelOp() string {
	if p.matchString("<=") {
		return "<="
	}
	if p.matchString(">=") {
		return ">="
	}
	if p.matchString("<") {
		return "<"
	}
	if p.matchString(">") {
		return ">"
	}
	return ""
}

func (p *expressionParser) parseAddSub() (expression, error) {
	left, err := p.parseShift()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpaces()
		if p.match('+') {
			right, err := p.parseShift()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: "+", left: left, right: right}
			continue
		}
		if p.match('-') {
			right, err := p.parseShift()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: "-", left: left, right: right}
			continue
		}
		break
	}
	return left, nil
}

func (p *expressionParser) parseMulDiv() (expression, error) {
	left, err := p.parseUnary()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpaces()
		if p.match('*') {
			right, err := p.parseUnary()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: "*", left: left, right: right}
			continue
		}
		if p.match('/') {
			right, err := p.parseUnary()
			if err != nil {
				return nil, err
			}
			left = &binaryExpr{op: "/", left: left, right: right}
			continue
		}
		break
	}
	return left, nil
}

func (p *expressionParser) parseUnary() (expression, error) {
	p.skipSpaces()
	if p.match('+') {
		return p.parseUnary()
	}
	if p.match('-') {
		node, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return &unaryExpr{op: '-', expr: node}, nil
	}
	if p.match('!') {
		node, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return &unaryExpr{op: '!', expr: node}, nil
	}
	return p.parsePrimary()
}

func (p *expressionParser) parsePrimary() (expression, error) {
	p.skipSpaces()
	if p.match('(') {
		expr, err := p.parseOr()
		if err != nil {
			return nil, err
		}
		p.skipSpaces()
		if !p.match(')') {
			return nil, fmt.Errorf("missing ) at %d", p.pos)
		}
		return expr, nil
	}
	if p.peek() == 0 {
		return nil, fmt.Errorf("unexpected end of expression")
	}
	// Check for hex constant (0x...)
	if p.peek() == '0' && p.pos+1 < len(p.input) && (p.input[p.pos+1] == 'x' || p.input[p.pos+1] == 'X') {
		return p.parseNumber()
	}
	if unicode.IsDigit(p.peek()) || p.peek() == '.' {
		return p.parseNumber()
	}
	if isIdentStart(p.peek()) {
		return p.parseIdentifier()
	}
	return nil, fmt.Errorf("unexpected token %s at %d", p.input[p.pos:], p.pos)
}

func (p *expressionParser) parseNumber() (expression, error) {
	start := p.pos

	// Check for hex constant (0x...)
	if p.peek() == '0' && p.pos+1 < len(p.input) && (p.input[p.pos+1] == 'x' || p.input[p.pos+1] == 'X') {
		p.pos += 2 // skip "0x" or "0X"
		hasHexDigits := false
		for p.isHexDigit(p.peek()) {
			p.pos++
			hasHexDigits = true
		}
		if !hasHexDigits {
			return nil, fmt.Errorf("invalid hex constant at %d", start)
		}
		hexStr := p.input[start+2 : p.pos] // skip "0x" prefix
		value, err := strconv.ParseUint(hexStr, 16, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid hex constant %s", p.input[start:p.pos])
		}
		return &numberExpr{value: float64(value)}, nil
	}

	// Decimal number parsing
	hasDigits := false
	for unicode.IsDigit(p.peek()) {
		p.pos++
		hasDigits = true
	}
	if p.peek() == '.' {
		p.pos++
		for unicode.IsDigit(p.peek()) {
			p.pos++
			hasDigits = true
		}
	}
	if !hasDigits {
		return nil, fmt.Errorf("invalid number at %d", start)
	}
	value, err := strconv.ParseFloat(p.input[start:p.pos], 64)
	if err != nil {
		return nil, fmt.Errorf("invalid number %s", p.input[start:p.pos])
	}
	return &numberExpr{value: value}, nil
}

func (p *expressionParser) parseIdentifier() (expression, error) {
	start := p.pos
	for isIdentPart(p.peek()) {
		p.pos++
	}
	ident := p.input[start:p.pos]
	switch strings.ToLower(ident) {
	case "true":
		return &boolExpr{value: true}, nil
	case "false":
		return &boolExpr{value: false}, nil
	}
	return &identifierExpr{name: ident}, nil
}

func (p *expressionParser) skipSpaces() {
	for unicode.IsSpace(p.peek()) {
		p.pos++
	}
}

func (p *expressionParser) match(ch byte) bool {
	if p.peek() == rune(ch) {
		p.pos++
		return true
	}
	return false
}

func (p *expressionParser) matchString(s string) bool {
	if strings.HasPrefix(p.input[p.pos:], s) {
		p.pos += len(s)
		return true
	}
	return false
}

func (p *expressionParser) peek() rune {
	if p.pos >= len(p.input) {
		return 0
	}
	return rune(p.input[p.pos])
}

func (p *expressionParser) isHexDigit(ch rune) bool {
	return (ch >= '0' && ch <= '9') ||
		(ch >= 'a' && ch <= 'f') ||
		(ch >= 'A' && ch <= 'F')
}

type numberExpr struct {
	value float64
}

func (n *numberExpr) eval(_ *exprEnv) (exprValue, error) {
	return exprValue{num: n.value}, nil
}

type boolExpr struct {
	value bool
}

func (b *boolExpr) eval(_ *exprEnv) (exprValue, error) {
	return exprValue{bool: b.value, isBool: true}, nil
}

type identifierExpr struct {
	name string
}

func (i *identifierExpr) eval(env *exprEnv) (exprValue, error) {
	return env.lookup(i.name)
}

type unaryExpr struct {
	op   rune
	expr expression
}

func (u *unaryExpr) eval(env *exprEnv) (exprValue, error) {
	val, err := u.expr.eval(env)
	if err != nil {
		return exprValue{}, err
	}
	switch u.op {
	case '-':
		num, ok := val.asNumber()
		if !ok {
			return exprValue{}, fmt.Errorf("cannot negate boolean")
		}
		return exprValue{num: -num}, nil
	case '!':
		boolean, ok := val.asBool()
		if !ok {
			return exprValue{}, fmt.Errorf("cannot use ! on numeric value")
		}
		return exprValue{bool: !boolean, isBool: true}, nil
	default:
		return val, nil
	}
}

type binaryExpr struct {
	op    string
	left  expression
	right expression
}

func (b *binaryExpr) eval(env *exprEnv) (exprValue, error) {
	leftVal, err := b.left.eval(env)
	if err != nil {
		return exprValue{}, err
	}
	rightVal, err := b.right.eval(env)
	if err != nil {
		return exprValue{}, err
	}

	switch b.op {
	case "+", "-", "*", "/":
		l, ok := leftVal.asNumber()
		if !ok {
			return exprValue{}, fmt.Errorf("left operand is not numeric")
		}
		r, ok := rightVal.asNumber()
		if !ok {
			return exprValue{}, fmt.Errorf("right operand is not numeric")
		}
		switch b.op {
		case "+":
			return exprValue{num: l + r}, nil
		case "-":
			return exprValue{num: l - r}, nil
		case "*":
			return exprValue{num: l * r}, nil
		case "/":
			if r == 0 {
				return exprValue{}, fmt.Errorf("division by zero")
			}
			return exprValue{num: l / r}, nil
		}
	case "<<", ">>":
		// Shift operations require integer operands
		l, ok := leftVal.asNumber()
		if !ok {
			return exprValue{}, fmt.Errorf("left operand is not numeric")
		}
		r, ok := rightVal.asNumber()
		if !ok {
			return exprValue{}, fmt.Errorf("right operand is not numeric")
		}
		leftInt := int64(l)
		rightInt := int64(r)
		if rightInt < 0 {
			return exprValue{}, fmt.Errorf("shift count must be non-negative")
		}
		if rightInt >= 64 {
			return exprValue{}, fmt.Errorf("shift count too large")
		}
		var result int64
		if b.op == "<<" {
			result = leftInt << uint(rightInt)
		} else {
			result = leftInt >> uint(rightInt)
		}
		return exprValue{num: float64(result)}, nil
	case "&", "^", "|":
		// Bitwise operations require integer operands
		l, ok := leftVal.asNumber()
		if !ok {
			return exprValue{}, fmt.Errorf("left operand is not numeric")
		}
		r, ok := rightVal.asNumber()
		if !ok {
			return exprValue{}, fmt.Errorf("right operand is not numeric")
		}
		leftInt := int64(l)
		rightInt := int64(r)
		var result int64
		switch b.op {
		case "&":
			result = leftInt & rightInt
		case "^":
			result = leftInt ^ rightInt
		case "|":
			result = leftInt | rightInt
		}
		return exprValue{num: float64(result)}, nil
	case "<", "<=", ">", ">=":
		l, ok := leftVal.asNumber()
		if !ok {
			return exprValue{}, fmt.Errorf("left operand is not numeric")
		}
		r, ok := rightVal.asNumber()
		if !ok {
			return exprValue{}, fmt.Errorf("right operand is not numeric")
		}
		switch b.op {
		case "<":
			return exprValue{bool: l < r, isBool: true}, nil
		case "<=":
			return exprValue{bool: l <= r, isBool: true}, nil
		case ">":
			return exprValue{bool: l > r, isBool: true}, nil
		case ">=":
			return exprValue{bool: l >= r, isBool: true}, nil
		}
	case "==", "!=":
		if leftVal.isBool || rightVal.isBool {
			l, ok1 := leftVal.asBool()
			r, ok2 := rightVal.asBool()
			if !ok1 || !ok2 {
				return exprValue{}, fmt.Errorf("cannot compare boolean with numeric")
			}
			if b.op == "==" {
				return exprValue{bool: l == r, isBool: true}, nil
			}
			return exprValue{bool: l != r, isBool: true}, nil
		}
		l, ok := leftVal.asNumber()
		if !ok {
			return exprValue{}, fmt.Errorf("left operand is not numeric")
		}
		r, ok := rightVal.asNumber()
		if !ok {
			return exprValue{}, fmt.Errorf("right operand is not numeric")
		}
		if b.op == "==" {
			return exprValue{bool: l == r, isBool: true}, nil
		}
		return exprValue{bool: l != r, isBool: true}, nil
	case "&&", "||":
		l, ok := leftVal.asBool()
		if !ok {
			return exprValue{}, fmt.Errorf("left operand is not boolean")
		}
		r, ok := rightVal.asBool()
		if !ok {
			return exprValue{}, fmt.Errorf("right operand is not boolean")
		}
		if b.op == "&&" {
			return exprValue{bool: l && r, isBool: true}, nil
		}
		return exprValue{bool: l || r, isBool: true}, nil
	}
	return exprValue{}, fmt.Errorf("unsupported operator %s", b.op)
}

func toExprValue(value interface{}) (exprValue, bool) {
	switch v := value.(type) {
	case int:
		return exprValue{num: float64(v)}, true
	case int8:
		return exprValue{num: float64(v)}, true
	case int16:
		return exprValue{num: float64(v)}, true
	case int32:
		return exprValue{num: float64(v)}, true
	case int64:
		return exprValue{num: float64(v)}, true
	case uint:
		return exprValue{num: float64(v)}, true
	case uint8:
		return exprValue{num: float64(v)}, true
	case uint16:
		return exprValue{num: float64(v)}, true
	case uint32:
		return exprValue{num: float64(v)}, true
	case uint64:
		return exprValue{num: float64(v)}, true
	case float32:
		return exprValue{num: float64(v)}, true
	case float64:
		return exprValue{num: v}, true
	case bool:
		return exprValue{bool: v, isBool: true}, true
	default:
		return exprValue{}, false
	}
}
