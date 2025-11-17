package graph

import (
	"encoding/binary"
	"fmt"
)

const (
	// Magic numbers for file format identification
	NodeMagic = "GRAPHND1" // 8 bytes
	EdgeMagic = "GRAPHED1" // 8 bytes
	DataMagic = "GRAPHDT1" // 8 bytes

	// Format version
	FormatVersion = 1

	// Record sizes
	NodeRecordSize = 32
	EdgeRecordSize = 32

	// Header sizes
	NodeHeaderSize = 64
	EdgeHeaderSize = 64
	DataHeaderSize = 64
)

// NodeFileHeader represents the header of a node file
type NodeFileHeader struct {
	Magic          [8]byte
	Version        uint32
	MaxID          int64
	NodeCount      uint64
	DataFileOffset uint64
	Checksum       uint64
	Reserved       [20]byte
}

// EdgeFileHeader represents the header of an edge file
type EdgeFileHeader struct {
	Magic          [8]byte
	Version        uint32
	MaxID          int64
	EdgeCount      uint64
	DataFileOffset uint64
	Checksum       uint64
	Reserved       [20]byte
}

// DataFileHeader represents the header of the data file
type DataFileHeader struct {
	Magic      [8]byte
	Version    uint32
	EntryCount uint64
	Reserved   [44]byte
}

// NodeRecord represents a fixed-size node record (32 bytes)
type NodeRecord struct {
	ID         int64
	DataOffset uint64
	Flags      uint8
	Reserved   [15]byte
}

// EdgeRecord represents a fixed-size edge record (32 bytes)
type EdgeRecord struct {
	FromID     int64
	ToID       int64
	DataOffset uint32
	Flags      uint8
	Reserved   [15]byte
}

// Record flags
const (
	FlagDeleted = 1 << iota
	FlagActive
)

// WriteNodeHeader writes a node file header to a byte slice
func WriteNodeHeader(data []byte, header *NodeFileHeader) error {
	if len(data) < NodeHeaderSize {
		return fmt.Errorf("buffer too small for node header: need %d, got %d", NodeHeaderSize, len(data))
	}

	copy(data[0:8], header.Magic[:])
	binary.LittleEndian.PutUint32(data[8:12], header.Version)
	binary.LittleEndian.PutUint64(data[12:20], uint64(header.MaxID))
	binary.LittleEndian.PutUint64(data[20:28], header.NodeCount)
	binary.LittleEndian.PutUint64(data[28:36], header.DataFileOffset)
	binary.LittleEndian.PutUint64(data[36:44], header.Checksum)
	copy(data[44:64], header.Reserved[:])

	return nil
}

// ReadNodeHeader reads a node file header from a byte slice
func ReadNodeHeader(data []byte) (*NodeFileHeader, error) {
	if len(data) < NodeHeaderSize {
		return nil, fmt.Errorf("buffer too small for node header: need %d, got %d", NodeHeaderSize, len(data))
	}

	header := &NodeFileHeader{}
	copy(header.Magic[:], data[0:8])
	header.Version = binary.LittleEndian.Uint32(data[8:12])
	header.MaxID = int64(binary.LittleEndian.Uint64(data[12:20]))
	header.NodeCount = binary.LittleEndian.Uint64(data[20:28])
	header.DataFileOffset = binary.LittleEndian.Uint64(data[28:36])
	header.Checksum = binary.LittleEndian.Uint64(data[36:44])
	copy(header.Reserved[:], data[44:64])

	// Validate magic
	if string(header.Magic[:]) != NodeMagic {
		return nil, fmt.Errorf("invalid node file magic: expected %s, got %s", NodeMagic, string(header.Magic[:]))
	}

	// Validate version
	if header.Version != FormatVersion {
		return nil, fmt.Errorf("unsupported format version: expected %d, got %d", FormatVersion, header.Version)
	}

	return header, nil
}

// WriteEdgeHeader writes an edge file header to a byte slice
func WriteEdgeHeader(data []byte, header *EdgeFileHeader) error {
	if len(data) < EdgeHeaderSize {
		return fmt.Errorf("buffer too small for edge header: need %d, got %d", EdgeHeaderSize, len(data))
	}

	copy(data[0:8], header.Magic[:])
	binary.LittleEndian.PutUint32(data[8:12], header.Version)
	binary.LittleEndian.PutUint64(data[12:20], uint64(header.MaxID))
	binary.LittleEndian.PutUint64(data[20:28], header.EdgeCount)
	binary.LittleEndian.PutUint64(data[28:36], header.DataFileOffset)
	binary.LittleEndian.PutUint64(data[36:44], header.Checksum)
	copy(data[44:64], header.Reserved[:])

	return nil
}

// ReadEdgeHeader reads an edge file header from a byte slice
func ReadEdgeHeader(data []byte) (*EdgeFileHeader, error) {
	if len(data) < EdgeHeaderSize {
		return nil, fmt.Errorf("buffer too small for edge header: need %d, got %d", EdgeHeaderSize, len(data))
	}

	header := &EdgeFileHeader{}
	copy(header.Magic[:], data[0:8])
	header.Version = binary.LittleEndian.Uint32(data[8:12])
	header.MaxID = int64(binary.LittleEndian.Uint64(data[12:20]))
	header.EdgeCount = binary.LittleEndian.Uint64(data[20:28])
	header.DataFileOffset = binary.LittleEndian.Uint64(data[28:36])
	header.Checksum = binary.LittleEndian.Uint64(data[36:44])
	copy(header.Reserved[:], data[44:64])

	// Validate magic
	if string(header.Magic[:]) != EdgeMagic {
		return nil, fmt.Errorf("invalid edge file magic: expected %s, got %s", EdgeMagic, string(header.Magic[:]))
	}

	// Validate version
	if header.Version != FormatVersion {
		return nil, fmt.Errorf("unsupported format version: expected %d, got %d", FormatVersion, header.Version)
	}

	return header, nil
}

// WriteNodeRecord writes a node record to a byte slice
func WriteNodeRecord(data []byte, record *NodeRecord) error {
	if len(data) < NodeRecordSize {
		return fmt.Errorf("buffer too small for node record: need %d, got %d", NodeRecordSize, len(data))
	}

	binary.LittleEndian.PutUint64(data[0:8], uint64(record.ID))
	binary.LittleEndian.PutUint64(data[8:16], record.DataOffset)
	data[16] = record.Flags
	copy(data[17:32], record.Reserved[:])

	return nil
}

// ReadNodeRecord reads a node record from a byte slice
func ReadNodeRecord(data []byte) (*NodeRecord, error) {
	if len(data) < NodeRecordSize {
		return nil, fmt.Errorf("buffer too small for node record: need %d, got %d", NodeRecordSize, len(data))
	}

	record := &NodeRecord{}
	record.ID = int64(binary.LittleEndian.Uint64(data[0:8]))
	record.DataOffset = binary.LittleEndian.Uint64(data[8:16])
	record.Flags = data[16]
	copy(record.Reserved[:], data[17:32])

	return record, nil
}

// WriteEdgeRecord writes an edge record to a byte slice
func WriteEdgeRecord(data []byte, record *EdgeRecord) error {
	if len(data) < EdgeRecordSize {
		return fmt.Errorf("buffer too small for edge record: need %d, got %d", EdgeRecordSize, len(data))
	}

	binary.LittleEndian.PutUint64(data[0:8], uint64(record.FromID))
	binary.LittleEndian.PutUint64(data[8:16], uint64(record.ToID))
	binary.LittleEndian.PutUint32(data[16:20], record.DataOffset)
	data[20] = record.Flags
	copy(data[21:32], record.Reserved[:])

	return nil
}

// ReadEdgeRecord reads an edge record from a byte slice
func ReadEdgeRecord(data []byte) (*EdgeRecord, error) {
	if len(data) < EdgeRecordSize {
		return nil, fmt.Errorf("buffer too small for edge record: need %d, got %d", EdgeRecordSize, len(data))
	}

	record := &EdgeRecord{}
	record.FromID = int64(binary.LittleEndian.Uint64(data[0:8]))
	record.ToID = int64(binary.LittleEndian.Uint64(data[8:16]))
	record.DataOffset = binary.LittleEndian.Uint32(data[16:20])
	record.Flags = data[20]
	copy(record.Reserved[:], data[21:32])

	return record, nil
}

// CalculateChecksum calculates a simple checksum for header validation
// This is a placeholder - in production, use a proper checksum algorithm
func CalculateChecksum(data []byte) uint64 {
	var sum uint64
	for _, b := range data {
		sum += uint64(b)
	}
	return sum
}

// WriteDataHeader writes a data file header to a byte slice
func WriteDataHeader(data []byte, header *DataFileHeader) error {
	if len(data) < DataHeaderSize {
		return fmt.Errorf("buffer too small for data header: need %d, got %d", DataHeaderSize, len(data))
	}

	copy(data[0:8], header.Magic[:])
	binary.LittleEndian.PutUint32(data[8:12], header.Version)
	binary.LittleEndian.PutUint64(data[12:20], header.EntryCount)
	copy(data[20:64], header.Reserved[:])

	return nil
}

// ReadDataHeader reads a data file header from a byte slice
func ReadDataHeader(data []byte) (*DataFileHeader, error) {
	if len(data) < DataHeaderSize {
		return nil, fmt.Errorf("buffer too small for data header: need %d, got %d", DataHeaderSize, len(data))
	}

	header := &DataFileHeader{}
	copy(header.Magic[:], data[0:8])
	header.Version = binary.LittleEndian.Uint32(data[8:12])
	header.EntryCount = binary.LittleEndian.Uint64(data[12:20])
	copy(header.Reserved[:], data[20:64])

	if string(header.Magic[:]) != DataMagic {
		return nil, fmt.Errorf("invalid data file magic: expected %s, got %s", DataMagic, string(header.Magic[:]))
	}

	if header.Version != FormatVersion {
		return nil, fmt.Errorf("unsupported format version: expected %d, got %d", FormatVersion, header.Version)
	}

	return header, nil
}
