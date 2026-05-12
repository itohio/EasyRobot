package crsf

import "encoding/binary"

// BaudRate represents a CRSF baud rate proposal.
type BaudRate uint32

const (
	Baud400K  BaudRate = 400000
	Baud750K  BaudRate = 750000
	Baud1_5M  BaudRate = 1500000
	Baud2M    BaudRate = 2000000
	Baud2_5M  BaudRate = 2500000
	Baud3M    BaudRate = 3000000
	Baud3_75M BaudRate = 3750000
	Baud4_5M  BaudRate = 4500000
)

// EncodeBaudProposal creates a baud rate proposal command (RX → FC).
// CRSFv3: Command 0x32, sub-address 0x0A (FC general), sub-command 0x70.
func EncodeBaudProposal(dest, origin Address, baud BaudRate) *Frame {
	data := make([]byte, 5)
	data[0] = byte(SubCmdSetRXBaud)
	binary.LittleEndian.PutUint32(data[1:5], uint32(baud))
	return EncodeCommand(&Command{
		Dest:       dest,
		Origin:     origin,
		ID:         CmdGeneralFC,
		SubCommand: SubCmdSetRXBaud,
		Data:       data[1:], // the sub-command byte is in SubCommand field
	})
}

// EncodeBaudConfirm creates a baud confirmation command (FC → RX).
// CRSFv3: Command 0x32, sub-address 0x0A (FC general), sub-command 0x71.
func EncodeBaudConfirm(dest, origin Address, baud BaudRate) *Frame {
	data := make([]byte, 4)
	binary.LittleEndian.PutUint32(data[0:4], uint32(baud))
	return EncodeCommand(&Command{
		Dest:       dest,
		Origin:     origin,
		ID:         CmdGeneralFC,
		SubCommand: SubCmdConfirmBaud,
		Data:       data,
	})
}

// DecodeBaudRate extracts the baud rate from a baud command's Data field.
func DecodeBaudRate(cmd *Command) BaudRate {
	if len(cmd.Data) < 4 {
		return 0
	}
	return BaudRate(binary.LittleEndian.Uint32(cmd.Data[0:4]))
}
