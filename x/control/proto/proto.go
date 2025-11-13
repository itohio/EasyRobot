// NOTE: As per https://github.com/tinygo-org/tinygo/issues/447
// Currently as of 2021 gogoproto does not support tinygo

package proto

const Marshal = 0
const GoGoProtoPackageIsVersion3 = 0

type Message interface {
}

type InternalMessageInfo interface {
	DiscardUnknown(interface{})
	Merge(interface{}, interface{})
	Marshal(interface{}, interface{}, interface{}) ([]byte, error)
}

func RegisterFile(string, []byte) {}

func RegisterEnum(string, map[int32]string, map[string]int32) {}

func RegisterType(interface{}, string) {}

func EnumName(map[int32]string, int32) string { return "" }

func CompactTextString(interface{}) string {
	return "nope"
}
