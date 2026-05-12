package crsf

import "encoding/binary"

// MSPHeaderSize is the overhead per MSP-over-CRSF frame (status byte).
const MSPHeaderSize = 1

// MSPStatus encodes the chunking/sequence state in the first byte of MSP payload.
type MSPStatus byte

// MSPStatus bit layout: [7:4]=sequence, [3:2]=reserved, [1]=error, [0]=start
const (
	mspFlagStart MSPStatus = 0x01
	mspFlagError MSPStatus = 0x02
)

// IsStart reports whether this is the first chunk of an MSP message.
func (s MSPStatus) IsStart() bool { return s&mspFlagStart != 0 }

// IsError reports whether this MSP response indicates an error.
func (s MSPStatus) IsError() bool { return s&mspFlagError != 0 }

// Sequence returns the 4-bit sequence counter (0-15).
func (s MSPStatus) Sequence() uint8 { return uint8(s >> 4) }

// NewMSPStatus constructs a status byte from flags.
func NewMSPStatus(seq uint8, start, isError bool) MSPStatus {
	s := MSPStatus(seq&0x0F) << 4
	if start {
		s |= mspFlagStart
	}
	if isError {
		s |= mspFlagError
	}
	return s
}

// MSPMessage is a complete MSP message (before/after CRSF chunking).
type MSPMessage struct {
	Function uint16
	Payload  []byte
}

// MSPFrame represents a single MSP-over-CRSF frame chunk.
type MSPFrame struct {
	Dest    Address
	Origin  Address
	Status  MSPStatus
	Payload []byte // raw MSP data for this chunk
}

// DecodeMSPFrame parses an MSP request (0x7A) or response (0x7B) frame.
func DecodeMSPFrame(f *Frame) (*MSPFrame, error) {
	switch f.Type {
	case TypeMSPRequest, TypeMSPResponse:
	default:
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 4 { // dest + origin + status + at least 1 byte
		return nil, ErrFrameTooShort
	}
	return &MSPFrame{
		Dest:    Address(f.Payload[0]),
		Origin:  Address(f.Payload[1]),
		Status:  MSPStatus(f.Payload[2]),
		Payload: f.Payload[3:],
	}, nil
}

// EncodeMSPFrame creates a CRSF frame wrapping an MSP chunk.
func EncodeMSPFrame(mf *MSPFrame, isResponse bool) *Frame {
	p := make([]byte, 3+len(mf.Payload))
	p[0] = byte(mf.Dest)
	p[1] = byte(mf.Origin)
	p[2] = byte(mf.Status)
	copy(p[3:], mf.Payload)

	t := TypeMSPRequest
	if isResponse {
		t = TypeMSPResponse
	}
	return &Frame{Type: t, Payload: p}
}

// maxMSPChunkPayload is the max MSP data per CRSF frame.
// Extended frame: 64 - sync(1) - len(1) - type(1) - dest(1) - origin(1) - status(1) - crc(1) = 57
const maxMSPChunkPayload = 57

// ChunkMSPRequest splits an MSP message into CRSF request frames.
// The first chunk carries: [function_lo][function_hi][payload_size][data...]
// Subsequent chunks carry only continuation data.
func ChunkMSPRequest(dest, origin Address, msg *MSPMessage, seq uint8) []*Frame {
	header := make([]byte, 3)
	binary.LittleEndian.PutUint16(header[0:2], msg.Function)
	header[2] = byte(len(msg.Payload))
	fullPayload := append(header, msg.Payload...)

	return chunkMSP(dest, origin, fullPayload, seq, false)
}

// ChunkMSPResponse splits an MSP response into CRSF response frames.
func ChunkMSPResponse(dest, origin Address, msg *MSPMessage, seq uint8, isError bool) []*Frame {
	header := make([]byte, 3)
	binary.LittleEndian.PutUint16(header[0:2], msg.Function)
	header[2] = byte(len(msg.Payload))
	fullPayload := append(header, msg.Payload...)

	return chunkMSPResp(dest, origin, fullPayload, seq, isError)
}

func chunkMSP(dest, origin Address, data []byte, seq uint8, isResponse bool) []*Frame {
	var frames []*Frame
	for i := 0; i < len(data); i += maxMSPChunkPayload {
		end := i + maxMSPChunkPayload
		if end > len(data) {
			end = len(data)
		}
		start := i == 0
		mf := &MSPFrame{
			Dest:    dest,
			Origin:  origin,
			Status:  NewMSPStatus(seq, start, false),
			Payload: data[i:end],
		}
		frames = append(frames, EncodeMSPFrame(mf, isResponse))
	}
	return frames
}

func chunkMSPResp(dest, origin Address, data []byte, seq uint8, isError bool) []*Frame {
	var frames []*Frame
	for i := 0; i < len(data); i += maxMSPChunkPayload {
		end := i + maxMSPChunkPayload
		if end > len(data) {
			end = len(data)
		}
		start := i == 0
		mf := &MSPFrame{
			Dest:    dest,
			Origin:  origin,
			Status:  NewMSPStatus(seq, start, isError && start),
			Payload: data[i:end],
		}
		frames = append(frames, EncodeMSPFrame(mf, true))
	}
	return frames
}

// ReassembleMSP reassembles chunked MSP frames into a complete MSPMessage.
// Pass frames in order; the first must have IsStart() true.
func ReassembleMSP(frames []*MSPFrame) (*MSPMessage, error) {
	if len(frames) == 0 || !frames[0].Status.IsStart() {
		return nil, ErrFrameTooShort
	}

	var data []byte
	for _, f := range frames {
		data = append(data, f.Payload...)
	}

	if len(data) < 3 {
		return nil, ErrFrameTooShort
	}
	fn := binary.LittleEndian.Uint16(data[0:2])
	size := int(data[2])
	payload := data[3:]
	if len(payload) < size {
		return nil, ErrFrameTooShort
	}
	return &MSPMessage{
		Function: fn,
		Payload:  payload[:size],
	}, nil
}
