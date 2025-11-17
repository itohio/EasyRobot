package graph

import (
	"testing"
)

func TestNodeHeader(t *testing.T) {
	header := &NodeFileHeader{
		Magic:          [8]byte{'G', 'R', 'A', 'P', 'H', 'N', 'D', '1'},
		Version:        FormatVersion,
		MaxID:          100,
		NodeCount:      50,
		DataFileOffset: 1024,
		Checksum:       12345,
	}

	buf := make([]byte, NodeHeaderSize)
	if err := WriteNodeHeader(buf, header); err != nil {
		t.Fatalf("WriteNodeHeader() failed: %v", err)
	}

	readHeader, err := ReadNodeHeader(buf)
	if err != nil {
		t.Fatalf("ReadNodeHeader() failed: %v", err)
	}

	if readHeader.Version != header.Version {
		t.Errorf("Version mismatch: expected %d, got %d", header.Version, readHeader.Version)
	}
	if readHeader.MaxID != header.MaxID {
		t.Errorf("MaxID mismatch: expected %d, got %d", header.MaxID, readHeader.MaxID)
	}
	if readHeader.NodeCount != header.NodeCount {
		t.Errorf("NodeCount mismatch: expected %d, got %d", header.NodeCount, readHeader.NodeCount)
	}
	if readHeader.DataFileOffset != header.DataFileOffset {
		t.Errorf("DataFileOffset mismatch: expected %d, got %d", header.DataFileOffset, readHeader.DataFileOffset)
	}
	if readHeader.Checksum != header.Checksum {
		t.Errorf("Checksum mismatch: expected %d, got %d", header.Checksum, readHeader.Checksum)
	}
}

func TestNodeHeaderInvalidMagic(t *testing.T) {
	buf := make([]byte, NodeHeaderSize)
	copy(buf[0:8], []byte("INVALID"))

	_, err := ReadNodeHeader(buf)
	if err == nil {
		t.Error("Expected error for invalid magic, got nil")
	}
}

func TestNodeHeaderInvalidVersion(t *testing.T) {
	header := &NodeFileHeader{
		Magic:   [8]byte{'G', 'R', 'A', 'P', 'H', 'N', 'D', '1'},
		Version: 999, // Invalid version
	}

	buf := make([]byte, NodeHeaderSize)
	if err := WriteNodeHeader(buf, header); err != nil {
		t.Fatalf("WriteNodeHeader() failed: %v", err)
	}

	_, err := ReadNodeHeader(buf)
	if err == nil {
		t.Error("Expected error for invalid version, got nil")
	}
}

func TestEdgeHeader(t *testing.T) {
	header := &EdgeFileHeader{
		Magic:          [8]byte{'G', 'R', 'A', 'P', 'H', 'E', 'D', '1'},
		Version:        FormatVersion,
		MaxID:          200,
		EdgeCount:      100,
		DataFileOffset: 2048,
		Checksum:       54321,
	}

	buf := make([]byte, EdgeHeaderSize)
	if err := WriteEdgeHeader(buf, header); err != nil {
		t.Fatalf("WriteEdgeHeader() failed: %v", err)
	}

	readHeader, err := ReadEdgeHeader(buf)
	if err != nil {
		t.Fatalf("ReadEdgeHeader() failed: %v", err)
	}

	if readHeader.Version != header.Version {
		t.Errorf("Version mismatch: expected %d, got %d", header.Version, readHeader.Version)
	}
	if readHeader.MaxID != header.MaxID {
		t.Errorf("MaxID mismatch: expected %d, got %d", header.MaxID, readHeader.MaxID)
	}
	if readHeader.EdgeCount != header.EdgeCount {
		t.Errorf("EdgeCount mismatch: expected %d, got %d", header.EdgeCount, readHeader.EdgeCount)
	}
}

func TestNodeRecord(t *testing.T) {
	record := &NodeRecord{
		ID:         1,
		DataOffset: 100,
		Flags:      FlagActive,
	}

	buf := make([]byte, NodeRecordSize)
	if err := WriteNodeRecord(buf, record); err != nil {
		t.Fatalf("WriteNodeRecord() failed: %v", err)
	}

	readRecord, err := ReadNodeRecord(buf)
	if err != nil {
		t.Fatalf("ReadNodeRecord() failed: %v", err)
	}

	if readRecord.ID != record.ID {
		t.Errorf("ID mismatch: expected %d, got %d", record.ID, readRecord.ID)
	}
	if readRecord.DataOffset != record.DataOffset {
		t.Errorf("DataOffset mismatch: expected %d, got %d", record.DataOffset, readRecord.DataOffset)
	}
	if readRecord.Flags != record.Flags {
		t.Errorf("Flags mismatch: expected %d, got %d", record.Flags, readRecord.Flags)
	}
}

func TestEdgeRecord(t *testing.T) {
	record := &EdgeRecord{
		FromID:     1,
		ToID:       2,
		DataOffset: 200,
		Flags:      FlagActive,
	}

	buf := make([]byte, EdgeRecordSize)
	if err := WriteEdgeRecord(buf, record); err != nil {
		t.Fatalf("WriteEdgeRecord() failed: %v", err)
	}

	readRecord, err := ReadEdgeRecord(buf)
	if err != nil {
		t.Fatalf("ReadEdgeRecord() failed: %v", err)
	}

	if readRecord.FromID != record.FromID {
		t.Errorf("FromID mismatch: expected %d, got %d", record.FromID, readRecord.FromID)
	}
	if readRecord.ToID != record.ToID {
		t.Errorf("ToID mismatch: expected %d, got %d", record.ToID, readRecord.ToID)
	}
	if readRecord.DataOffset != record.DataOffset {
		t.Errorf("DataOffset mismatch: expected %d, got %d", record.DataOffset, readRecord.DataOffset)
	}
	if readRecord.Flags != record.Flags {
		t.Errorf("Flags mismatch: expected %d, got %d", record.Flags, readRecord.Flags)
	}
}

func TestFormatConstants(t *testing.T) {
	if len(NodeMagic) != 8 {
		t.Errorf("NodeMagic should be 8 bytes, got %d", len(NodeMagic))
	}
	if len(EdgeMagic) != 8 {
		t.Errorf("EdgeMagic should be 8 bytes, got %d", len(EdgeMagic))
	}
	if NodeRecordSize != 32 {
		t.Errorf("NodeRecordSize should be 32, got %d", NodeRecordSize)
	}
	if EdgeRecordSize != 32 {
		t.Errorf("EdgeRecordSize should be 32, got %d", EdgeRecordSize)
	}
}
