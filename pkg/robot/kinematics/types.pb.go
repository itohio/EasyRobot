// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: types.proto

package kinematics

import (
	encoding_binary "encoding/binary"
	fmt "fmt"
	proto "github.com/foxis/EasyRobot/pkg/robot/proto"
	io "io"
	math "math"
	math_bits "math/bits"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.GoGoProtoPackageIsVersion3 // please upgrade the proto package

type DenavitHartenberg struct {
	Index uint32  `protobuf:"varint,1,opt,name=Index,proto3" json:"Index,omitempty"`
	Theta float32 `protobuf:"fixed32,2,opt,name=Theta,proto3" json:"Theta,omitempty"`
	Alpha float32 `protobuf:"fixed32,3,opt,name=Alpha,proto3" json:"Alpha,omitempty"`
	R     float32 `protobuf:"fixed32,4,opt,name=R,proto3" json:"R,omitempty"`
	D     float32 `protobuf:"fixed32,5,opt,name=D,proto3" json:"D,omitempty"`
}

func (m *DenavitHartenberg) Reset()         { *m = DenavitHartenberg{} }
func (m *DenavitHartenberg) String() string { return proto.CompactTextString(m) }
func (*DenavitHartenberg) ProtoMessage()    {}
func (*DenavitHartenberg) Descriptor() ([]byte, []int) {
	return fileDescriptor_d938547f84707355, []int{0}
}
func (m *DenavitHartenberg) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *DenavitHartenberg) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_DenavitHartenberg.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *DenavitHartenberg) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DenavitHartenberg.Merge(m, src)
}
func (m *DenavitHartenberg) XXX_Size() int {
	return m.Size()
}
func (m *DenavitHartenberg) XXX_DiscardUnknown() {
	xxx_messageInfo_DenavitHartenberg.DiscardUnknown(m)
}

var xxx_messageInfo_DenavitHartenberg proto.InternalMessageInfo

func (m *DenavitHartenberg) GetIndex() uint32 {
	if m != nil {
		return m.Index
	}
	return 0
}

func (m *DenavitHartenberg) GetTheta() float32 {
	if m != nil {
		return m.Theta
	}
	return 0
}

func (m *DenavitHartenberg) GetAlpha() float32 {
	if m != nil {
		return m.Alpha
	}
	return 0
}

func (m *DenavitHartenberg) GetR() float32 {
	if m != nil {
		return m.R
	}
	return 0
}

func (m *DenavitHartenberg) GetD() float32 {
	if m != nil {
		return m.D
	}
	return 0
}

type Wheels struct {
	Radius float32 `protobuf:"fixed32,1,opt,name=Radius,proto3" json:"Radius,omitempty"`
	BaseX  float32 `protobuf:"fixed32,2,opt,name=BaseX,proto3" json:"BaseX,omitempty"`
	BaseY  float32 `protobuf:"fixed32,3,opt,name=BaseY,proto3" json:"BaseY,omitempty"`
}

func (m *Wheels) Reset()         { *m = Wheels{} }
func (m *Wheels) String() string { return proto.CompactTextString(m) }
func (*Wheels) ProtoMessage()    {}
func (*Wheels) Descriptor() ([]byte, []int) {
	return fileDescriptor_d938547f84707355, []int{1}
}
func (m *Wheels) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *Wheels) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_Wheels.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *Wheels) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Wheels.Merge(m, src)
}
func (m *Wheels) XXX_Size() int {
	return m.Size()
}
func (m *Wheels) XXX_DiscardUnknown() {
	xxx_messageInfo_Wheels.DiscardUnknown(m)
}

var xxx_messageInfo_Wheels proto.InternalMessageInfo

func (m *Wheels) GetRadius() float32 {
	if m != nil {
		return m.Radius
	}
	return 0
}

func (m *Wheels) GetBaseX() float32 {
	if m != nil {
		return m.BaseX
	}
	return 0
}

func (m *Wheels) GetBaseY() float32 {
	if m != nil {
		return m.BaseY
	}
	return 0
}

type Config struct {
	DH []*DenavitHartenberg `protobuf:"bytes,1,rep,name=DH,proto3" json:"DH,omitempty"`
}

func (m *Config) Reset()         { *m = Config{} }
func (m *Config) String() string { return proto.CompactTextString(m) }
func (*Config) ProtoMessage()    {}
func (*Config) Descriptor() ([]byte, []int) {
	return fileDescriptor_d938547f84707355, []int{2}
}
func (m *Config) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *Config) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_Config.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *Config) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Config.Merge(m, src)
}
func (m *Config) XXX_Size() int {
	return m.Size()
}
func (m *Config) XXX_DiscardUnknown() {
	xxx_messageInfo_Config.DiscardUnknown(m)
}

var xxx_messageInfo_Config proto.InternalMessageInfo

func (m *Config) GetDH() []*DenavitHartenberg {
	if m != nil {
		return m.DH
	}
	return nil
}

func init() {
	proto.RegisterType((*DenavitHartenberg)(nil), "kinematics.DenavitHartenberg")
	proto.RegisterType((*Wheels)(nil), "kinematics.Wheels")
	proto.RegisterType((*Config)(nil), "kinematics.Config")
}

func init() { proto.RegisterFile("types.proto", fileDescriptor_d938547f84707355) }

var fileDescriptor_d938547f84707355 = []byte{
	// 245 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xe2, 0x2e, 0xa9, 0x2c, 0x48,
	0x2d, 0xd6, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0xe2, 0xca, 0xce, 0xcc, 0x4b, 0xcd, 0x4d, 0x2c,
	0xc9, 0x4c, 0x2e, 0x56, 0x2a, 0xe4, 0x12, 0x74, 0x49, 0xcd, 0x4b, 0x2c, 0xcb, 0x2c, 0xf1, 0x48,
	0x2c, 0x2a, 0x49, 0xcd, 0x4b, 0x4a, 0x2d, 0x4a, 0x17, 0x12, 0xe1, 0x62, 0xf5, 0xcc, 0x4b, 0x49,
	0xad, 0x90, 0x60, 0x54, 0x60, 0xd4, 0xe0, 0x0d, 0x82, 0x70, 0x40, 0xa2, 0x21, 0x19, 0xa9, 0x25,
	0x89, 0x12, 0x4c, 0x0a, 0x8c, 0x1a, 0x4c, 0x41, 0x10, 0x0e, 0x48, 0xd4, 0x31, 0xa7, 0x20, 0x23,
	0x51, 0x82, 0x19, 0x22, 0x0a, 0xe6, 0x08, 0xf1, 0x70, 0x31, 0x06, 0x49, 0xb0, 0x80, 0x45, 0x18,
	0x83, 0x40, 0x3c, 0x17, 0x09, 0x56, 0x08, 0xcf, 0x45, 0xc9, 0x87, 0x8b, 0x2d, 0x3c, 0x23, 0x35,
	0x35, 0xa7, 0x58, 0x48, 0x8c, 0x8b, 0x2d, 0x28, 0x31, 0x25, 0xb3, 0xb4, 0x18, 0x6c, 0x11, 0x53,
	0x10, 0x94, 0x07, 0x32, 0xd3, 0x29, 0xb1, 0x38, 0x35, 0x02, 0x66, 0x13, 0x98, 0x03, 0x13, 0x8d,
	0x84, 0xd9, 0x04, 0xe6, 0x28, 0x99, 0x73, 0xb1, 0x39, 0xe7, 0xe7, 0xa5, 0x65, 0xa6, 0x0b, 0xe9,
	0x72, 0x31, 0xb9, 0x78, 0x48, 0x30, 0x2a, 0x30, 0x6b, 0x70, 0x1b, 0xc9, 0xea, 0x21, 0xfc, 0xa8,
	0x87, 0xe1, 0xc1, 0x20, 0x26, 0x17, 0x0f, 0x27, 0x89, 0x13, 0x8f, 0xe4, 0x18, 0x2f, 0x3c, 0x92,
	0x63, 0x7c, 0xf0, 0x48, 0x8e, 0x71, 0xc2, 0x63, 0x39, 0x86, 0x0b, 0x8f, 0xe5, 0x18, 0x6e, 0x3c,
	0x96, 0x63, 0x48, 0x62, 0x03, 0x07, 0x93, 0x31, 0x20, 0x00, 0x00, 0xff, 0xff, 0x95, 0xac, 0xc5,
	0x05, 0x35, 0x01, 0x00, 0x00,
}

func (m *DenavitHartenberg) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *DenavitHartenberg) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *DenavitHartenberg) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.D != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.D))))
		i--
		dAtA[i] = 0x2d
	}
	if m.R != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.R))))
		i--
		dAtA[i] = 0x25
	}
	if m.Alpha != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Alpha))))
		i--
		dAtA[i] = 0x1d
	}
	if m.Theta != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Theta))))
		i--
		dAtA[i] = 0x15
	}
	if m.Index != 0 {
		i = encodeVarintTypes(dAtA, i, uint64(m.Index))
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func (m *Wheels) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *Wheels) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *Wheels) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.BaseY != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.BaseY))))
		i--
		dAtA[i] = 0x1d
	}
	if m.BaseX != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.BaseX))))
		i--
		dAtA[i] = 0x15
	}
	if m.Radius != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Radius))))
		i--
		dAtA[i] = 0xd
	}
	return len(dAtA) - i, nil
}

func (m *Config) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *Config) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *Config) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.DH) > 0 {
		for iNdEx := len(m.DH) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.DH[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintTypes(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func encodeVarintTypes(dAtA []byte, offset int, v uint64) int {
	offset -= sovTypes(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *DenavitHartenberg) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Index != 0 {
		n += 1 + sovTypes(uint64(m.Index))
	}
	if m.Theta != 0 {
		n += 5
	}
	if m.Alpha != 0 {
		n += 5
	}
	if m.R != 0 {
		n += 5
	}
	if m.D != 0 {
		n += 5
	}
	return n
}

func (m *Wheels) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Radius != 0 {
		n += 5
	}
	if m.BaseX != 0 {
		n += 5
	}
	if m.BaseY != 0 {
		n += 5
	}
	return n
}

func (m *Config) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.DH) > 0 {
		for _, e := range m.DH {
			l = e.Size()
			n += 1 + l + sovTypes(uint64(l))
		}
	}
	return n
}

func sovTypes(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozTypes(x uint64) (n int) {
	return sovTypes(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *DenavitHartenberg) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTypes
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: DenavitHartenberg: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: DenavitHartenberg: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Index", wireType)
			}
			m.Index = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTypes
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Index |= uint32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Theta", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.Theta = float32(math.Float32frombits(v))
		case 3:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Alpha", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.Alpha = float32(math.Float32frombits(v))
		case 4:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field R", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.R = float32(math.Float32frombits(v))
		case 5:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field D", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.D = float32(math.Float32frombits(v))
		default:
			iNdEx = preIndex
			skippy, err := skipTypes(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthTypes
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func (m *Wheels) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTypes
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: Wheels: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Wheels: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Radius", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.Radius = float32(math.Float32frombits(v))
		case 2:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field BaseX", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.BaseX = float32(math.Float32frombits(v))
		case 3:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field BaseY", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.BaseY = float32(math.Float32frombits(v))
		default:
			iNdEx = preIndex
			skippy, err := skipTypes(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthTypes
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func (m *Config) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTypes
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: Config: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Config: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field DH", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTypes
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthTypes
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthTypes
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.DH = append(m.DH, &DenavitHartenberg{})
			if err := m.DH[len(m.DH)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipTypes(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthTypes
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func skipTypes(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowTypes
			}
			if iNdEx >= l {
				return 0, io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		wireType := int(wire & 0x7)
		switch wireType {
		case 0:
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowTypes
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				iNdEx++
				if dAtA[iNdEx-1] < 0x80 {
					break
				}
			}
		case 1:
			iNdEx += 8
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowTypes
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				length |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if length < 0 {
				return 0, ErrInvalidLengthTypes
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupTypes
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthTypes
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthTypes        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowTypes          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupTypes = fmt.Errorf("proto: unexpected end of group")
)
