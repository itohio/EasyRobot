package crsf

// CommandID identifies a direct command sub-type (first payload byte after extended header).
type CommandID byte

const (
	CmdGeneralFC       CommandID = 0x0A // Flight controller general commands
	CmdGeneralBluetooth CommandID = 0x10
	CmdGeneralWifi     CommandID = 0x11
	CmdGeneralVTX      CommandID = 0x08
	CmdGeneralOSD      CommandID = 0x09
	CmdGeneralRX       CommandID = 0x20 // Receiver-specific commands
)

// SubCommandID provides finer granularity within a CommandID.
type SubCommandID byte

const (
	SubCmdBindRX      SubCommandID = 0x01 // Bind receiver
	SubCmdModelSelect SubCommandID = 0x05 // Select model ID
	SubCmdSetRXBaud   SubCommandID = 0x70 // Propose baud rate change (RX→FC)
	SubCmdConfirmBaud SubCommandID = 0x71 // Confirm baud rate change (FC→RX)
)

// Command represents a CRSF direct command frame (Type 0x32).
type Command struct {
	Dest       Address
	Origin     Address
	ID         CommandID
	SubCommand SubCommandID
	Data       []byte
}

// EncodeCommand creates a Command frame with the proper command CRC8 appended.
// Wire format: [dest][origin][cmdID][subCmd][data...][cmdCRC8]
// The command CRC8 covers: frame type + dest + origin + cmdID + subCmd + data.
func EncodeCommand(cmd *Command) *Frame {
	payloadLen := 2 + 2 + len(cmd.Data) + 1 // dest+origin + id+sub + data + cmdCRC
	p := make([]byte, payloadLen)
	p[0] = byte(cmd.Dest)
	p[1] = byte(cmd.Origin)
	p[2] = byte(cmd.ID)
	p[3] = byte(cmd.SubCommand)
	copy(p[4:], cmd.Data)

	// Command CRC8 covers: type byte + full payload (excluding the crc byte itself)
	crcBuf := make([]byte, 1+payloadLen-1)
	crcBuf[0] = byte(TypeCommand)
	copy(crcBuf[1:], p[:payloadLen-1])
	p[payloadLen-1] = CommandCRC8(crcBuf)

	return &Frame{Type: TypeCommand, Payload: p}
}

// DecodeCommand parses a direct command frame and validates its command CRC8.
func DecodeCommand(f *Frame) (*Command, error) {
	if f.Type != TypeCommand {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 5 { // dest+origin+id+sub+crc
		return nil, ErrFrameTooShort
	}

	p := f.Payload
	// Verify command CRC8
	crcBuf := make([]byte, 1+len(p)-1)
	crcBuf[0] = byte(TypeCommand)
	copy(crcBuf[1:], p[:len(p)-1])
	if CommandCRC8(crcBuf) != p[len(p)-1] {
		return nil, ErrCRCMismatch
	}

	return &Command{
		Dest:       Address(p[0]),
		Origin:     Address(p[1]),
		ID:         CommandID(p[2]),
		SubCommand: SubCommandID(p[3]),
		Data:       p[4 : len(p)-1],
	}, nil
}
