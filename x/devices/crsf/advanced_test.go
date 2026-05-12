package crsf

import "testing"

func TestCommandCRC8(t *testing.T) {
	if got := CommandCRC8(nil); got != 0 {
		t.Errorf("CommandCRC8(nil) = 0x%02X, want 0x00", got)
	}
	data := []byte{0x32, 0xC8, 0xEC, 0x0A, 0x70}
	crc := CommandCRC8(data)
	if CommandCRC8(data) != crc {
		t.Error("CommandCRC8 not deterministic")
	}
}

func TestLinkStatsRXRoundTrip(t *testing.T) {
	ls := &LinkStatsRX{
		RSSIDB:      -85,
		RSSIPercent: 60,
		LinkQuality: 95,
		SNR:         8,
		RFPower:     3,
	}
	f := EncodeLinkStatsRX(ls)
	got, err := DecodeLinkStatsRX(f)
	if err != nil {
		t.Fatal(err)
	}
	if *got != *ls {
		t.Errorf("got %+v, want %+v", got, ls)
	}
}

func TestLinkStatsTXRoundTrip(t *testing.T) {
	ls := &LinkStatsTX{
		RSSIDB:      -90,
		RSSIPercent: 45,
		LinkQuality: 80,
		SNR:         5,
		RFPower:     4,
		FPS:         500,
	}
	f := EncodeLinkStatsTX(ls)
	got, err := DecodeLinkStatsTX(f)
	if err != nil {
		t.Fatal(err)
	}
	if *got != *ls {
		t.Errorf("got %+v, want %+v", got, ls)
	}
}

func TestPingRoundTrip(t *testing.T) {
	f := EncodePing(AddrFlightController, AddrReceiver)
	dest, origin, err := DecodePing(f)
	if err != nil {
		t.Fatal(err)
	}
	if dest != AddrFlightController {
		t.Errorf("dest = 0x%02X, want 0x%02X", dest, AddrFlightController)
	}
	if origin != AddrReceiver {
		t.Errorf("origin = 0x%02X, want 0x%02X", origin, AddrReceiver)
	}
}

func TestDeviceInfoRoundTrip(t *testing.T) {
	di := &DeviceInfo{
		Dest:            AddrRemoteControl,
		Origin:          AddrReceiver,
		Name:            "ExpressLRS RX",
		SerialNumber:    0x12345678,
		HardwareVersion: 0x00010002,
		SoftwareVersion: 0x00030004,
		ParamCount:      12,
		ProtocolVersion: 1,
	}
	f := EncodeDeviceInfo(di)
	got, err := DecodeDeviceInfo(f)
	if err != nil {
		t.Fatal(err)
	}
	if *got != *di {
		t.Errorf("got %+v,\nwant %+v", got, di)
	}
}

func TestParamReadRoundTrip(t *testing.T) {
	pr := &ParamRead{
		Dest:   AddrReceiver,
		Origin: AddrRemoteControl,
		Index:  3,
		Chunk:  0,
	}
	f := EncodeParamRead(pr)
	got, err := DecodeParamRead(f)
	if err != nil {
		t.Fatal(err)
	}
	if *got != *pr {
		t.Errorf("got %+v, want %+v", got, pr)
	}
}

func TestParamWriteRoundTrip(t *testing.T) {
	pw := &ParamWrite{
		Dest:   AddrReceiver,
		Origin: AddrRemoteControl,
		Index:  5,
		Value:  []byte{0x02},
	}
	f := EncodeParamWrite(pw)
	got, err := DecodeParamWrite(f)
	if err != nil {
		t.Fatal(err)
	}
	if got.Dest != pw.Dest || got.Origin != pw.Origin || got.Index != pw.Index {
		t.Errorf("header mismatch: got %+v, want %+v", got, pw)
	}
	if string(got.Value) != string(pw.Value) {
		t.Errorf("value = %v, want %v", got.Value, pw.Value)
	}
}

func TestParamEntryChunk0RoundTrip(t *testing.T) {
	pe := &ParamEntry{
		Dest:   AddrRemoteControl,
		Origin: AddrReceiver,
		Index:  2,
		Chunks: 1,
		Chunk:  0,
		Parent: 0,
		Type:   ParamTextSelect,
		Name:   "Rate",
		Value:  []byte{0x01, '5', '0', 'H', 'z', ';', '1', '5', '0', 'H', 'z', 0},
	}
	f := EncodeParamEntry(pe)
	got, err := DecodeParamEntry(f)
	if err != nil {
		t.Fatal(err)
	}
	if got.Dest != pe.Dest || got.Origin != pe.Origin {
		t.Errorf("addr mismatch")
	}
	if got.Index != pe.Index || got.Chunks != pe.Chunks || got.Chunk != pe.Chunk {
		t.Errorf("index/chunk mismatch")
	}
	if got.Parent != pe.Parent || got.Type != pe.Type || got.Name != pe.Name {
		t.Errorf("header mismatch: parent=%d type=%d name=%q", got.Parent, got.Type, got.Name)
	}
	if string(got.Value) != string(pe.Value) {
		t.Errorf("value = %v, want %v", got.Value, pe.Value)
	}
}

func TestCommandRoundTrip(t *testing.T) {
	cmd := &Command{
		Dest:       AddrFlightController,
		Origin:     AddrReceiver,
		ID:         CmdGeneralFC,
		SubCommand: SubCmdBindRX,
		Data:       []byte{0xAA, 0xBB},
	}
	f := EncodeCommand(cmd)
	got, err := DecodeCommand(f)
	if err != nil {
		t.Fatal(err)
	}
	if got.Dest != cmd.Dest || got.Origin != cmd.Origin {
		t.Errorf("addr mismatch: got dest=0x%02X origin=0x%02X", got.Dest, got.Origin)
	}
	if got.ID != cmd.ID || got.SubCommand != cmd.SubCommand {
		t.Errorf("cmd mismatch: got ID=0x%02X sub=0x%02X", got.ID, got.SubCommand)
	}
	if string(got.Data) != string(cmd.Data) {
		t.Errorf("data = %v, want %v", got.Data, cmd.Data)
	}
}

func TestCommandCRCValidation(t *testing.T) {
	cmd := &Command{
		Dest:       AddrFlightController,
		Origin:     AddrReceiver,
		ID:         CmdGeneralFC,
		SubCommand: SubCmdBindRX,
		Data:       nil,
	}
	f := EncodeCommand(cmd)
	// Corrupt the command CRC (last byte of payload)
	f.Payload[len(f.Payload)-1] ^= 0xFF
	_, err := DecodeCommand(f)
	if err != ErrCRCMismatch {
		t.Errorf("expected ErrCRCMismatch, got %v", err)
	}
}

func TestBaudProposalRoundTrip(t *testing.T) {
	f := EncodeBaudProposal(AddrFlightController, AddrReceiver, Baud1_5M)
	cmd, err := DecodeCommand(f)
	if err != nil {
		t.Fatal(err)
	}
	baud := DecodeBaudRate(cmd)
	if baud != Baud1_5M {
		t.Errorf("baud = %d, want %d", baud, Baud1_5M)
	}
}

func TestTimingCorrectionRoundTrip(t *testing.T) {
	tc := &TimingCorrection{
		Dest:           AddrFlightController,
		Origin:         AddrReceiver,
		UpdateInterval: 4000,
		Offset:         -200,
	}
	f := EncodeTimingCorrection(tc)
	got, err := DecodeTimingCorrection(f)
	if err != nil {
		t.Fatal(err)
	}
	if *got != *tc {
		t.Errorf("got %+v, want %+v", got, tc)
	}
}

func TestMSPFrameRoundTrip(t *testing.T) {
	mf := &MSPFrame{
		Dest:    AddrFlightController,
		Origin:  AddrRemoteControl,
		Status:  NewMSPStatus(3, true, false),
		Payload: []byte{0x01, 0x00, 0x05, 0xDE, 0xAD, 0xBE, 0xEF, 0x42},
	}
	f := EncodeMSPFrame(mf, false)
	if f.Type != TypeMSPRequest {
		t.Errorf("type = 0x%02X, want 0x%02X", f.Type, TypeMSPRequest)
	}
	got, err := DecodeMSPFrame(f)
	if err != nil {
		t.Fatal(err)
	}
	if got.Dest != mf.Dest || got.Origin != mf.Origin {
		t.Errorf("addr mismatch")
	}
	if got.Status != mf.Status {
		t.Errorf("status = 0x%02X, want 0x%02X", got.Status, mf.Status)
	}
	if string(got.Payload) != string(mf.Payload) {
		t.Errorf("payload mismatch")
	}
}

func TestMSPStatus(t *testing.T) {
	s := NewMSPStatus(7, true, true)
	if !s.IsStart() {
		t.Error("expected start flag")
	}
	if !s.IsError() {
		t.Error("expected error flag")
	}
	if s.Sequence() != 7 {
		t.Errorf("seq = %d, want 7", s.Sequence())
	}

	s2 := NewMSPStatus(0, false, false)
	if s2.IsStart() || s2.IsError() || s2.Sequence() != 0 {
		t.Errorf("unexpected flags: %+v", s2)
	}
}

func TestMSPChunkAndReassemble(t *testing.T) {
	msg := &MSPMessage{
		Function: 100,
		Payload:  make([]byte, 120), // > maxMSPChunkPayload, forces chunking
	}
	for i := range msg.Payload {
		msg.Payload[i] = byte(i)
	}

	frames := ChunkMSPRequest(AddrFlightController, AddrRemoteControl, msg, 5)
	if len(frames) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(frames))
	}

	// Decode frames back into MSPFrames
	var mspFrames []*MSPFrame
	for _, f := range frames {
		mf, err := DecodeMSPFrame(f)
		if err != nil {
			t.Fatal(err)
		}
		mspFrames = append(mspFrames, mf)
	}

	// First should be start
	if !mspFrames[0].Status.IsStart() {
		t.Error("first chunk should have start flag")
	}
	for i := 1; i < len(mspFrames); i++ {
		if mspFrames[i].Status.IsStart() {
			t.Errorf("chunk %d should not have start flag", i)
		}
	}

	got, err := ReassembleMSP(mspFrames)
	if err != nil {
		t.Fatal(err)
	}
	if got.Function != msg.Function {
		t.Errorf("function = %d, want %d", got.Function, msg.Function)
	}
	if len(got.Payload) != len(msg.Payload) {
		t.Fatalf("payload len = %d, want %d", len(got.Payload), len(msg.Payload))
	}
	for i := range got.Payload {
		if got.Payload[i] != msg.Payload[i] {
			t.Errorf("payload[%d] = %d, want %d", i, got.Payload[i], msg.Payload[i])
			break
		}
	}
}

func TestMSPResponseChunking(t *testing.T) {
	msg := &MSPMessage{
		Function: 200,
		Payload:  []byte{0x01, 0x02, 0x03},
	}
	frames := ChunkMSPResponse(AddrRemoteControl, AddrFlightController, msg, 2, false)
	if len(frames) != 1 {
		t.Fatalf("expected 1 frame for small payload, got %d", len(frames))
	}
	if frames[0].Type != TypeMSPResponse {
		t.Errorf("type = 0x%02X, want 0x%02X", frames[0].Type, TypeMSPResponse)
	}
}

func TestSchedulerRoundRobin(t *testing.T) {
	// Use a simple implementation to test without importing rx
	type slot struct {
		fn    func() *Frame
		calls int
	}

	callCount := [3]int{}
	fns := [3]func() *Frame{
		func() *Frame { callCount[0]++; return &Frame{Type: TypeBattery} },
		func() *Frame { callCount[1]++; return nil }, // empty slot
		func() *Frame { callCount[2]++; return &Frame{Type: TypeGPS} },
	}

	// Simulate scheduler logic
	slots := fns[:]
	idx := 0
	next := func() *Frame {
		n := len(slots)
		for range n {
			fn := slots[idx%n]
			idx = (idx + 1) % n
			if f := fn(); f != nil {
				return f
			}
		}
		return nil
	}

	f1 := next()
	if f1 == nil || f1.Type != TypeBattery {
		t.Errorf("first = %v, want Battery", f1)
	}
	f2 := next()
	if f2 == nil || f2.Type != TypeGPS {
		t.Errorf("second = %v, want GPS (skips nil)", f2)
	}
	f3 := next()
	if f3 == nil || f3.Type != TypeBattery {
		t.Errorf("third = %v, want Battery (wraps around)", f3)
	}
}
