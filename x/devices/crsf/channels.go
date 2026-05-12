package crsf

const (
	NumChannels     = 16
	channelsPayload = 22 // 16 × 11 bits = 176 bits = 22 bytes

	// Channel value domain
	ChannelMin    uint16 = 0
	ChannelCenter uint16 = 992
	ChannelMax    uint16 = 1984
)

// Channels holds 16 RC channel values in raw CRSF ticks (0..1984).
type Channels struct {
	Ch [NumChannels]uint16
}

// Microseconds converts a channel value to servo pulse width in µs.
func (c *Channels) Microseconds(ch int) uint16 {
	return ticksToMicroseconds(c.Ch[ch])
}

// SetMicroseconds sets a channel value from servo pulse width in µs.
func (c *Channels) SetMicroseconds(ch int, us uint16) {
	c.Ch[ch] = microsecondsToTicks(us)
}

func ticksToMicroseconds(ticks uint16) uint16 {
	return uint16((int(ticks)-int(ChannelCenter))*5/8 + 1500)
}

func microsecondsToTicks(us uint16) uint16 {
	return uint16((int(us)-1500)*8/5 + int(ChannelCenter))
}

// MarshalChannelsTo encodes Channels into dst (must be >= 22 bytes). Returns bytes written.
func MarshalChannelsTo(dst []byte, ch *Channels) int {
	clear(dst[:channelsPayload])
	packChannels(dst[:channelsPayload], ch)
	return channelsPayload
}

// MarshalChannels encodes Channels into a newly allocated 22-byte payload.
func MarshalChannels(ch *Channels) []byte {
	buf := make([]byte, channelsPayload)
	packChannels(buf, ch)
	return buf
}

// UnmarshalChannels decodes a 22-byte payload into Channels.
func UnmarshalChannels(payload []byte) (*Channels, error) {
	if len(payload) < channelsPayload {
		return nil, ErrFrameTooShort
	}
	ch := &Channels{}
	unpackChannels(payload, ch)
	return ch, nil
}

// packChannels packs 16 × 11-bit values into 22 bytes (little-endian bitstream).
func packChannels(dst []byte, ch *Channels) {
	var bitOffset uint
	for i := range NumChannels {
		v := ch.Ch[i] & 0x7FF
		byteIdx := bitOffset / 8
		bitIdx := bitOffset % 8

		dst[byteIdx] |= byte(v << bitIdx)
		dst[byteIdx+1] |= byte(v >> (8 - bitIdx))
		if bitIdx > 5 {
			dst[byteIdx+2] |= byte(v >> (16 - bitIdx))
		}
		bitOffset += 11
	}
}

// unpackChannels unpacks 22 bytes into 16 × 11-bit channel values.
func unpackChannels(src []byte, ch *Channels) {
	var bitOffset uint
	for i := range NumChannels {
		byteIdx := bitOffset / 8
		bitIdx := bitOffset % 8

		raw := uint32(src[byteIdx]) | uint32(src[byteIdx+1])<<8
		if bitIdx > 5 {
			raw |= uint32(src[byteIdx+2]) << 16
		}
		ch.Ch[i] = uint16((raw >> bitIdx) & 0x7FF)
		bitOffset += 11
	}
}

// DecodeChannels extracts Channels from a TypeChannelsPacked frame.
func DecodeChannels(f *Frame) (*Channels, error) {
	if f.Type != TypeChannelsPacked {
		return nil, ErrTypeMismatch
	}
	return UnmarshalChannels(f.Payload)
}

// EncodeChannels builds a complete Frame containing packed RC channels.
func EncodeChannels(ch *Channels) *Frame {
	return &Frame{
		Type:    TypeChannelsPacked,
		Payload: MarshalChannels(ch),
	}
}
