package graph

import (
	"encoding/binary"
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"google.golang.org/protobuf/proto"
)

func readDataEntry(storage types.MappedStorage, offset uint64) (DataType, string, []byte, error) {
	if offset == 0 {
		return 0, "", nil, fmt.Errorf("data offset is zero")
	}
	region, err := storage.Map(0, 0)
	if err != nil {
		return 0, "", nil, fmt.Errorf("failed to map data storage: %w", err)
	}
	defer region.Unmap()

	data := region.Bytes()
	if offset >= uint64(len(data)) {
		return 0, "", nil, fmt.Errorf("data offset %d out of range", offset)
	}

	entryOffset := int64(offset)
	if entryOffset+dataEntryHeaderSize > int64(len(data)) {
		return 0, "", nil, fmt.Errorf("data entry at %d truncated", offset)
	}
	header := data[entryOffset : entryOffset+dataEntryHeaderSize]
	payloadLen := binary.LittleEndian.Uint32(header[0:4])
	entryType := DataType(header[4])
	typeNameLen := binary.LittleEndian.Uint16(header[5:7])

	totalSize := dataEntryHeaderSize + int64(typeNameLen) + int64(payloadLen)
	if entryOffset+totalSize > int64(len(data)) {
		return 0, "", nil, fmt.Errorf("data entry at %d exceeds storage bounds", offset)
	}
	entryBytes := data[entryOffset : entryOffset+totalSize]
	typeName := string(entryBytes[7 : 7+typeNameLen])
	payloadStart := 7 + int(typeNameLen)
	payload := make([]byte, payloadLen)
	copy(payload, entryBytes[payloadStart:payloadStart+int(payloadLen)])

	return entryType, typeName, payload, nil
}

func readDataEntryValue(storage types.MappedStorage, offset uint64, registry map[string]proto.Message) (any, error) {
	if offset == 0 {
		return nil, nil
	}
	entryType, typeName, payload, err := readDataEntry(storage, offset)
	if err != nil {
		return nil, err
	}
	return deserializeData(payload, entryType, typeName, registry)
}
