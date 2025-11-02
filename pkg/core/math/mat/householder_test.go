package mat

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

func TestMatrix_H1(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix
		col0    int
		lpivot  int
		l1      int
		rangeVal float32
		wantErr bool
		verify  func(m Matrix, up float32, t *testing.T)
	}{
		{
			name: "simple case",
			init: func(t *testing.T) Matrix {
				m := New(3, 3)
				m.Eye()
				return m
			},
			col0:     0,
			lpivot:   0,
			l1:       1,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(m Matrix, up float32, t *testing.T) {
				// After H1, the pivot element should be modified
				if m[0][0] == 0 {
					t.Errorf("H1: pivot element should be modified")
				}
			},
		},
		{
			name: "invalid indices",
			init: func(t *testing.T) Matrix {
				return New(3, 3)
			},
			col0:     0,
			lpivot:   5, // Invalid
			l1:       1,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(m Matrix, up float32, t *testing.T) {
				if up != 0 {
					t.Errorf("H1: should return 0 for invalid indices, got %v", up)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			up, err := m.H1(tt.col0, tt.lpivot, tt.l1, tt.rangeVal)

			if err != nil {
				if !tt.wantErr {
					t.Errorf("Matrix.H1() error = %v", err)
				}
				return
			}

			if tt.verify != nil {
				tt.verify(m, up, t)
			}
		})
	}
}

func TestMatrix_H2(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) (Matrix, vec.Vector)
		col0    int
		lpivot  int
		l1      int
		up      float32
		rangeVal float32
		wantErr bool
	}{
		{
			name: "simple case",
			init: func(t *testing.T) (Matrix, vec.Vector) {
				m := New(3, 3)
				m.Eye()
				zz := make(vec.Vector, 3)
				zz[0] = 1
				zz[1] = 2
				zz[2] = 3
				return m, zz
			},
			col0:     0,
			lpivot:   0,
			l1:       1,
			up:       0.5,
			rangeVal: DefaultRange,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, zz := tt.init(t)
			err := m.H2(tt.col0, tt.lpivot, tt.l1, tt.up, zz, tt.rangeVal)

			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.H2() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestMatrix_H3(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix
		col0    int
		lpivot  int
		l1      int
		up      float32
		col1    int
		rangeVal float32
		wantErr bool
	}{
		{
			name: "simple case",
			init: func(t *testing.T) Matrix {
				m := New(3, 3)
				m.Eye()
				return m
			},
			col0:     0,
			lpivot:   0,
			l1:       1,
			up:       0.5,
			col1:     1,
			rangeVal: DefaultRange,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			err := m.H3(tt.col0, tt.lpivot, tt.l1, tt.up, tt.col1, tt.rangeVal)

			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.H3() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

