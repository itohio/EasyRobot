package plugin

import (
	"encoding/json"
	"testing"
)

type O0 struct {
	A  int `opts:""`
	B  int
	F1 string `opts:""`
}
type O1 struct {
	O0 `opts:""`
	F0 string  `opts:""`
	F1 int     `opts:""`
	F2 []int   `opts:""`
	F3 int     `opts:""`
	F4 float32 `opts:""`
}
type O2 struct {
	Base  O1                `opts:""`
	C     string            `opts:""`
	D     map[string]string `opts:""`
	E     map[int]string    `opts:""`
	Named int               `opts:"named_value"`
}

func TestMarshalOpts(t *testing.T) {

	data := O2{
		C: "c",
		D: map[string]string{"1": "1", "2": "2", "3": "3"},
		E: map[int]string{1: "1", 2: "2", 3: "3"},
		Base: O1{
			O0: O0{
				A:  1,
				B:  2,
				F1: "f1",
			},
			F0: "f0",
			F1: 123,
			F2: []int{1, 2, 3},
			F3: 321,
			F4: 3.145,
		},
		Named: 321,
	}
	truth := `{"C":"c","D":{"1":"1","2":"2","3":"3"},"E":{"1":"1","2":"2","3":"3"},"F0":"f0","F1":123,"F2":[1,2,3],"F3":321,"F4":3.145,"O0":{"A":1,"F1":"f1"},"named_value":321}`

	result := MarshalOptions(data)
	b, err := json.Marshal(result)
	if err != nil {
		t.Error(err)
	} else {
		if string(b) != truth {
			t.Error("got ", string(b), " expected ", truth)
		}
	}
}

func TestUnmarshal(t *testing.T) {
	data := `{"C":"c","D":{"1":"1","2":"2","3":"3"},"E":null,"F0":"f0","F1":123,"F2":[1,2,3],"F3":321,"F4":3.145,"O0":{"A":1,"B":0,"F1":"f1"},"named_value":321}`
	truth := `{"C":"c","D":{"1":"1","2":"2","3":"3"},"E":null,"F0":"f0","F1":123,"F2":[1,2,3],"F3":321,"F4":3.145,"O0":{"A":1,"F1":"f1"},"named_value":321}`
	var m map[string]interface{}

	err := json.Unmarshal([]byte(data), &m)
	if err != nil {
		t.Error(err)
	}

	structure := O2{}
	fillStruct(&structure, m)

	result := MarshalOptions(structure)
	b, err := json.Marshal(result)
	if err != nil {
		t.Error(err)
	}

	if string(b) != truth {
		t.Error("got ", string(b), " expected ", truth)
	}
	if structure.Base.O0.B != 0 {
		t.Error("got ", structure.Base.O0.B, " expected ", 0)
	}

	if structure.Named != 321 {
		t.Error("got ", structure.Named, " expected ", 321)
	}
}
