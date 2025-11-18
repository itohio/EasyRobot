package peg

// State captures decoded fields and can be cloned for backtracking.
type State interface {
	Clone() State
	Merge(other State)
	ResetFields()

	AddField(field DecodedField)
	Fields() []DecodedField

	Packet() []byte
	SetPacket(data []byte)
	AppendPacket(data ...byte)

	DeclaredLength() int
	SetDeclaredLength(int)

	MaxLength() int
	SetMaxLength(int)

	CurrentLength() int

	SetDecision(Decision)
	Decision() Decision

	Offset() int
	SetOffset(int)
}

// DecodedField describes a parsed value.
type DecodedField struct {
	Name   string
	Offset int
	Type   FieldType
	Value  interface{}
}

// DefaultState provides a basic implementation.
type DefaultState struct {
	fields         []DecodedField
	packet         []byte
	declaredLength int
	maxLength      int
	decision       Decision
	offset         int
}

// NewDefaultState creates an empty state.
func NewDefaultState() *DefaultState {
	return &DefaultState{
		fields: make([]DecodedField, 0),
		packet: make([]byte, 0),
	}
}

func (s *DefaultState) Clone() State {
	cpFields := make([]DecodedField, len(s.fields))
	copy(cpFields, s.fields)
	return &DefaultState{
		fields:         cpFields,
		packet:         append([]byte{}, s.packet...),
		declaredLength: s.declaredLength,
		maxLength:      s.maxLength,
		offset:         s.offset,
	}
}

func (s *DefaultState) Merge(other State) {
	if o, ok := other.(*DefaultState); ok {
		s.fields = append(s.fields[:0], o.fields...)
		s.packet = append(s.packet[:0], o.packet...)
		s.declaredLength = o.declaredLength
		s.maxLength = o.maxLength
		s.offset = o.offset
	}
}

func (s *DefaultState) AddField(field DecodedField) {
	s.fields = append(s.fields, field)
}

func (s *DefaultState) Fields() []DecodedField {
	return append([]DecodedField{}, s.fields...)
}

func (s *DefaultState) ResetFields() {
	s.fields = s.fields[:0]
}

func (s *DefaultState) Packet() []byte {
	return append([]byte{}, s.packet...)
}

func (s *DefaultState) SetPacket(data []byte) {
	if cap(s.packet) >= len(data) {
		s.packet = s.packet[:len(data)]
		copy(s.packet, data)
		return
	}
	s.packet = append(make([]byte, 0, len(data)), data...)
}

// AppendPacket appends bytes to the packet buffer.
func (s *DefaultState) AppendPacket(data ...byte) {
	s.packet = append(s.packet, data...)
}

func (s *DefaultState) DeclaredLength() int {
	return s.declaredLength
}

func (s *DefaultState) SetDeclaredLength(length int) {
	if length <= 0 {
		return
	}
	if s.maxLength > 0 && length > s.maxLength {
		s.declaredLength = s.maxLength
		return
	}
	s.declaredLength = length
}

func (s *DefaultState) MaxLength() int {
	return s.maxLength
}

func (s *DefaultState) SetMaxLength(length int) {
	if length <= 0 {
		return
	}
	if s.maxLength == 0 || length < s.maxLength {
		s.maxLength = length
		if s.declaredLength > 0 && s.declaredLength > s.maxLength {
			s.declaredLength = s.maxLength
		}
	}
}

func (s *DefaultState) CurrentLength() int {
	return len(s.packet)
}

func (s *DefaultState) SetDecision(dec Decision) {
	s.decision = dec
}

func (s *DefaultState) Decision() Decision {
	return s.decision
}

func (s *DefaultState) Offset() int {
	return s.offset
}

func (s *DefaultState) SetOffset(offset int) {
	s.offset = offset
}
