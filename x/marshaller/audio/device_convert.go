package audio

import (
	"encoding/binary"
	"fmt"

	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// pcmToTensor converts PCM bytes to tensor data based on format.
func pcmToTensor(pcmData []byte, format string, channels int, frames int, dtype tensortypes.DataType) (any, error) {
	sampleSize := sampleSizeFromFormat(format)
	if len(pcmData) < frames*channels*sampleSize {
		return nil, fmt.Errorf("insufficient PCM data: got %d bytes, need %d", len(pcmData), frames*channels*sampleSize)
	}

	switch dtype {
	case tensortypes.INT16:
		data := make([]int16, frames*channels)
		if format == "S16" {
			for i := 0; i < frames*channels; i++ {
				data[i] = int16(binary.LittleEndian.Uint16(pcmData[i*2 : (i+1)*2]))
			}
		} else {
			return nil, fmt.Errorf("unsupported format conversion: %s to INT16", format)
		}
		return data, nil

	case tensortypes.INT32:
		data := make([]int32, frames*channels)
		if format == "S32" {
			for i := 0; i < frames*channels; i++ {
				data[i] = int32(binary.LittleEndian.Uint32(pcmData[i*4 : (i+1)*4]))
			}
		} else if format == "S24" {
			// 24-bit samples stored in 32-bit ints
			for i := 0; i < frames*channels; i++ {
				// Read 3 bytes, sign-extend to 32-bit
				b0 := uint32(pcmData[i*3])
				b1 := uint32(pcmData[i*3+1])
				b2 := uint32(pcmData[i*3+2])
				val := int32((b0 | (b1 << 8) | (b2 << 16)))
				if (b2 & 0x80) != 0 {
					val |= ^int32(0xFFFFFF)
				}
				data[i] = val
			}
		} else {
			return nil, fmt.Errorf("unsupported format conversion: %s to INT32", format)
		}
		return data, nil

	case tensortypes.FP32:
		data := make([]float32, frames*channels)
		if format == "F32" {
			for i := 0; i < frames*channels; i++ {
				bits := binary.LittleEndian.Uint32(pcmData[i*4 : (i+1)*4])
				data[i] = float32(bits)
			}
		} else if format == "S16" {
			maxVal := float32(32768.0)
			for i := 0; i < frames*channels; i++ {
				val := int16(binary.LittleEndian.Uint16(pcmData[i*2 : (i+1)*2]))
				data[i] = float32(val) / maxVal
			}
		} else if format == "S32" {
			maxVal := float32(2147483648.0)
			for i := 0; i < frames*channels; i++ {
				val := int32(binary.LittleEndian.Uint32(pcmData[i*4 : (i+1)*4]))
				data[i] = float32(val) / maxVal
			}
		} else {
			return nil, fmt.Errorf("unsupported format conversion: %s to FP32", format)
		}
		return data, nil

	case tensortypes.UINT8:
		data := make([]uint8, frames*channels)
		if format == "U8" {
			copy(data, pcmData[:frames*channels])
		} else {
			return nil, fmt.Errorf("unsupported format conversion: %s to UINT8", format)
		}
		return data, nil

	default:
		return nil, fmt.Errorf("unsupported tensor dtype: %v", dtype)
	}
}

// tensorToPCM converts tensor data to PCM bytes based on format.
func tensorToPCM(tensorData any, format string, channels int, frames int) ([]byte, error) {
	sampleSize := sampleSizeFromFormat(format)
	pcmData := make([]byte, frames*channels*sampleSize)

	switch data := tensorData.(type) {
	case []int16:
		if format == "S16" {
			for i := 0; i < frames*channels; i++ {
				binary.LittleEndian.PutUint16(pcmData[i*2:(i+1)*2], uint16(data[i]))
			}
		} else {
			return nil, fmt.Errorf("unsupported format conversion: INT16 to %s", format)
		}

	case []int32:
		if format == "S32" {
			for i := 0; i < frames*channels; i++ {
				binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], uint32(data[i]))
			}
		} else if format == "S24" {
			// Convert 32-bit to 24-bit
			for i := 0; i < frames*channels; i++ {
				val := data[i]
				pcmData[i*3] = byte(val)
				pcmData[i*3+1] = byte(val >> 8)
				pcmData[i*3+2] = byte(val >> 16)
			}
		} else {
			return nil, fmt.Errorf("unsupported format conversion: INT32 to %s", format)
		}

	case []float32:
		if format == "F32" {
			for i := 0; i < frames*channels; i++ {
				bits := uint32(data[i])
				binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], bits)
			}
		} else if format == "S16" {
			maxVal := float32(32767.0)
			for i := 0; i < frames*channels; i++ {
				val := int16(data[i] * maxVal)
				if val > 32767 {
					val = 32767
				} else if val < -32768 {
					val = -32768
				}
				binary.LittleEndian.PutUint16(pcmData[i*2:(i+1)*2], uint16(val))
			}
		} else if format == "S32" {
			maxVal := float32(2147483647.0)
			for i := 0; i < frames*channels; i++ {
				val := int32(data[i] * maxVal)
				if val > 2147483647 {
					val = 2147483647
				} else if val < -2147483648 {
					val = -2147483648
				}
				binary.LittleEndian.PutUint32(pcmData[i*4:(i+1)*4], uint32(val))
			}
		} else {
			return nil, fmt.Errorf("unsupported format conversion: FP32 to %s", format)
		}

	case []uint8:
		if format == "U8" {
			copy(pcmData, data[:frames*channels])
		} else {
			return nil, fmt.Errorf("unsupported format conversion: UINT8 to %s", format)
		}

	default:
		return nil, fmt.Errorf("unsupported tensor data type: %T", tensorData)
	}

	return pcmData, nil
}

func sampleSizeFromFormat(format string) int {
	switch format {
	case "U8":
		return 1
	case "S16":
		return 2
	case "S24":
		return 3
	case "S32", "F32":
		return 4
	default:
		return 2 // Default to 16-bit
	}
}

