package crsf

import "io"

// Encoder writes CRSF frames to a byte stream.
type Encoder struct {
	w   io.Writer
	buf [MaxFrameSize]byte
}

func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{w: w}
}

// WriteFrame serializes and writes a frame to the underlying writer.
func (e *Encoder) WriteFrame(f *Frame) error {
	n, err := f.Marshal(e.buf[:])
	if err != nil {
		return err
	}
	_, err = e.w.Write(e.buf[:n])
	return err
}

// WriteChannels encodes and writes an RC channels frame with zero extra allocations.
func (e *Encoder) WriteChannels(ch *Channels) error {
	// Build frame directly in e.buf: [sync][len][type][22 bytes payload][crc]
	const frameLen = 2 + channelsPayload // type + payload + crc
	const total = 2 + frameLen           // sync + len + frameLen

	e.buf[0] = SyncByte
	e.buf[1] = byte(frameLen)
	e.buf[2] = byte(TypeChannelsPacked)
	clear(e.buf[3 : 3+channelsPayload])
	packChannels(e.buf[3:3+channelsPayload], ch)
	e.buf[total-1] = CRC8(e.buf[2 : total-1])

	_, err := e.w.Write(e.buf[:total])
	return err
}

func (e *Encoder) WriteBattery(b *Battery) error {
	return e.WriteFrame(EncodeBattery(b))
}

func (e *Encoder) WriteGPS(g *GPS) error {
	return e.WriteFrame(EncodeGPS(g))
}

func (e *Encoder) WriteAttitude(a *Attitude) error {
	return e.WriteFrame(EncodeAttitude(a))
}

func (e *Encoder) WriteBaroAltitude(ba *BaroAltitude) error {
	return e.WriteFrame(EncodeBaroAltitude(ba))
}

func (e *Encoder) WriteLinkStats(ls *LinkStatistics) error {
	return e.WriteFrame(EncodeLinkStats(ls))
}

func (e *Encoder) WriteFlightMode(fm *FlightMode) error {
	return e.WriteFrame(EncodeFlightMode(fm))
}

func (e *Encoder) WriteHeartbeat(hb *Heartbeat) error {
	return e.WriteFrame(EncodeHeartbeat(hb))
}
