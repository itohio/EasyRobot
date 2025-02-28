// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: pb/types.proto

package pb

import (
	fmt "fmt"
	proto "github.com/itohio/EasyRobot/pkg/robot/proto"
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

type Motor struct {
	PinA      uint32 `protobuf:"varint,1,opt,name=PinA,proto3" json:"PinA,omitempty"`
	PinB      uint32 `protobuf:"varint,2,opt,name=PinB,proto3" json:"PinB,omitempty"`
	PinPWM    uint32 `protobuf:"varint,3,opt,name=PinPWM,proto3" json:"PinPWM,omitempty"`
	PinEnable uint32 `protobuf:"varint,4,opt,name=PinEnable,proto3" json:"PinEnable,omitempty"`
	PinEncA   uint32 `protobuf:"varint,5,opt,name=PinEncA,proto3" json:"PinEncA,omitempty"`
	PinEncB   uint32 `protobuf:"varint,6,opt,name=PinEncB,proto3" json:"PinEncB,omitempty"`
}

func (m *Motor) Reset()         { *m = Motor{} }
func (m *Motor) String() string { return proto.CompactTextString(m) }
func (*Motor) ProtoMessage()    {}
func (*Motor) Descriptor() ([]byte, []int) {
	return fileDescriptor_fcfd97e91e26151a, []int{0}
}
func (m *Motor) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *Motor) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_Motor.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *Motor) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Motor.Merge(m, src)
}
func (m *Motor) XXX_Size() int {
	return m.Size()
}
func (m *Motor) XXX_DiscardUnknown() {
	xxx_messageInfo_Motor.DiscardUnknown(m)
}

var xxx_messageInfo_Motor proto.InternalMessageInfo

func (m *Motor) GetPinA() uint32 {
	if m != nil {
		return m.PinA
	}
	return 0
}

func (m *Motor) GetPinB() uint32 {
	if m != nil {
		return m.PinB
	}
	return 0
}

func (m *Motor) GetPinPWM() uint32 {
	if m != nil {
		return m.PinPWM
	}
	return 0
}

func (m *Motor) GetPinEnable() uint32 {
	if m != nil {
		return m.PinEnable
	}
	return 0
}

func (m *Motor) GetPinEncA() uint32 {
	if m != nil {
		return m.PinEncA
	}
	return 0
}

func (m *Motor) GetPinEncB() uint32 {
	if m != nil {
		return m.PinEncB
	}
	return 0
}

type Config struct {
	Motors    []*Motor `protobuf:"bytes,1,rep,name=Motors,proto3" json:"Motors,omitempty"`
	PWMPeriod uint32   `protobuf:"varint,2,opt,name=PWMPeriod,proto3" json:"PWMPeriod,omitempty"`
}

func (m *Config) Reset()         { *m = Config{} }
func (m *Config) String() string { return proto.CompactTextString(m) }
func (*Config) ProtoMessage()    {}
func (*Config) Descriptor() ([]byte, []int) {
	return fileDescriptor_fcfd97e91e26151a, []int{1}
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

func (m *Config) GetMotors() []*Motor {
	if m != nil {
		return m.Motors
	}
	return nil
}

func (m *Config) GetPWMPeriod() uint32 {
	if m != nil {
		return m.PWMPeriod
	}
	return 0
}

func init() {
	proto.RegisterType((*Motor)(nil), "pb.Motor")
	proto.RegisterType((*Config)(nil), "pb.Config")
}

func init() { proto.RegisterFile("pb/types.proto", fileDescriptor_fcfd97e91e26151a) }

var fileDescriptor_fcfd97e91e26151a = []byte{
	// 215 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xe2, 0x2b, 0x48, 0xd2, 0x2f,
	0xa9, 0x2c, 0x48, 0x2d, 0xd6, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0x62, 0x2a, 0x48, 0x52, 0x9a,
	0xc9, 0xc8, 0xc5, 0xea, 0x9b, 0x5f, 0x92, 0x5f, 0x24, 0x24, 0xc4, 0xc5, 0x12, 0x90, 0x99, 0xe7,
	0x28, 0xc1, 0xa8, 0xc0, 0xa8, 0xc1, 0x1b, 0x04, 0x66, 0x43, 0xc5, 0x9c, 0x24, 0x98, 0xe0, 0x62,
	0x4e, 0x42, 0x62, 0x5c, 0x6c, 0x01, 0x99, 0x79, 0x01, 0xe1, 0xbe, 0x12, 0xcc, 0x60, 0x51, 0x28,
	0x4f, 0x48, 0x86, 0x8b, 0x33, 0x20, 0x33, 0xcf, 0x35, 0x2f, 0x31, 0x29, 0x27, 0x55, 0x82, 0x05,
	0x2c, 0x85, 0x10, 0x10, 0x92, 0xe0, 0x62, 0x07, 0x73, 0x92, 0x1d, 0x25, 0x58, 0xc1, 0x72, 0x30,
	0x2e, 0x42, 0xc6, 0x49, 0x82, 0x0d, 0x59, 0xc6, 0x49, 0xc9, 0x93, 0x8b, 0xcd, 0x39, 0x3f, 0x2f,
	0x2d, 0x33, 0x5d, 0x48, 0x91, 0x8b, 0x0d, 0xec, 0xc8, 0x62, 0x09, 0x46, 0x05, 0x66, 0x0d, 0x6e,
	0x23, 0x4e, 0xbd, 0x82, 0x24, 0x3d, 0xb0, 0x48, 0x10, 0x54, 0x02, 0x6c, 0x7d, 0xb8, 0x6f, 0x40,
	0x6a, 0x51, 0x66, 0x7e, 0x0a, 0xd4, 0xbd, 0x08, 0x01, 0x27, 0x89, 0x13, 0x8f, 0xe4, 0x18, 0x2f,
	0x3c, 0x92, 0x63, 0x7c, 0xf0, 0x48, 0x8e, 0x71, 0xc2, 0x63, 0x39, 0x86, 0x0b, 0x8f, 0xe5, 0x18,
	0x6e, 0x3c, 0x96, 0x63, 0x48, 0x62, 0x03, 0x87, 0x85, 0x31, 0x20, 0x00, 0x00, 0xff, 0xff, 0x68,
	0x29, 0x3e, 0xfa, 0x1d, 0x01, 0x00, 0x00,
}

func (m *Motor) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *Motor) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *Motor) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.PinEncB != 0 {
		i = encodeVarintTypes(dAtA, i, uint64(m.PinEncB))
		i--
		dAtA[i] = 0x30
	}
	if m.PinEncA != 0 {
		i = encodeVarintTypes(dAtA, i, uint64(m.PinEncA))
		i--
		dAtA[i] = 0x28
	}
	if m.PinEnable != 0 {
		i = encodeVarintTypes(dAtA, i, uint64(m.PinEnable))
		i--
		dAtA[i] = 0x20
	}
	if m.PinPWM != 0 {
		i = encodeVarintTypes(dAtA, i, uint64(m.PinPWM))
		i--
		dAtA[i] = 0x18
	}
	if m.PinB != 0 {
		i = encodeVarintTypes(dAtA, i, uint64(m.PinB))
		i--
		dAtA[i] = 0x10
	}
	if m.PinA != 0 {
		i = encodeVarintTypes(dAtA, i, uint64(m.PinA))
		i--
		dAtA[i] = 0x8
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
	if m.PWMPeriod != 0 {
		i = encodeVarintTypes(dAtA, i, uint64(m.PWMPeriod))
		i--
		dAtA[i] = 0x10
	}
	if len(m.Motors) > 0 {
		for iNdEx := len(m.Motors) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.Motors[iNdEx].MarshalToSizedBuffer(dAtA[:i])
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
func (m *Motor) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.PinA != 0 {
		n += 1 + sovTypes(uint64(m.PinA))
	}
	if m.PinB != 0 {
		n += 1 + sovTypes(uint64(m.PinB))
	}
	if m.PinPWM != 0 {
		n += 1 + sovTypes(uint64(m.PinPWM))
	}
	if m.PinEnable != 0 {
		n += 1 + sovTypes(uint64(m.PinEnable))
	}
	if m.PinEncA != 0 {
		n += 1 + sovTypes(uint64(m.PinEncA))
	}
	if m.PinEncB != 0 {
		n += 1 + sovTypes(uint64(m.PinEncB))
	}
	return n
}

func (m *Config) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Motors) > 0 {
		for _, e := range m.Motors {
			l = e.Size()
			n += 1 + l + sovTypes(uint64(l))
		}
	}
	if m.PWMPeriod != 0 {
		n += 1 + sovTypes(uint64(m.PWMPeriod))
	}
	return n
}

func sovTypes(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozTypes(x uint64) (n int) {
	return sovTypes(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *Motor) Unmarshal(dAtA []byte) error {
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
			return fmt.Errorf("proto: Motor: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Motor: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field PinA", wireType)
			}
			m.PinA = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTypes
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.PinA |= uint32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field PinB", wireType)
			}
			m.PinB = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTypes
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.PinB |= uint32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field PinPWM", wireType)
			}
			m.PinPWM = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTypes
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.PinPWM |= uint32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field PinEnable", wireType)
			}
			m.PinEnable = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTypes
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.PinEnable |= uint32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 5:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field PinEncA", wireType)
			}
			m.PinEncA = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTypes
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.PinEncA |= uint32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 6:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field PinEncB", wireType)
			}
			m.PinEncB = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTypes
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.PinEncB |= uint32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
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
				return fmt.Errorf("proto: wrong wireType = %d for field Motors", wireType)
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
			m.Motors = append(m.Motors, &Motor{})
			if err := m.Motors[len(m.Motors)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field PWMPeriod", wireType)
			}
			m.PWMPeriod = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTypes
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.PWMPeriod |= uint32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
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
