// +build sam,xiao

package servos

import (
	"errors"
	"machine"
	"math"

	"tinygo.org/x/drivers/servo"
)

type servos struct {
	motors []servo.Servo
	cfg    []Motor
	pos    []float32
}

var timerMapping = map[machine.Pin]servo.PWM{
	machine.D9:  machine.TCC0,
	machine.D8:  machine.TCC1,
	machine.D10: machine.TCC1,
}

var (
	errInvalidPin      = errors.New("invalid pin")
	errNumberOfParams  = errors.New("number of params")
	errBadCoefficients = errors.New("bad coefficients")
)

func New(cfg []Motor) (servos, error) {
	s := servos{}
	err := s.Configure(cfg)
	return s, err
}

func (s *servos) Configure(cfg []Motor) error {
	s.motors = make([]servo.Servo, len(cfg))
	s.pos = make([]float32, len(cfg))
	for i, c := range cfg {
		timer, valid := timerMapping[machine.Pin(c.Pin)]
		if !valid {
			return errInvalidPin
		}

		us := c.Min*c.Scale + c.Offset
		if us < 0 || us > math.MaxInt16 {
			return errBadCoefficients
		}
		us = c.Max*c.Scale + c.Offset
		if us < 0 || us > math.MaxInt16 {
			return errBadCoefficients
		}

		m, err := servo.New(timer, machine.Pin(c.Pin))
		if err != nil {
			return err
		}
		s.motors[i] = m
		s.pos[i] = c.Default
	}
	s.cfg = cfg

	return nil
}

func (s *servos) Get() ([]float32, error) {
	return s.pos, nil
}

func (s *servos) Set(params []float32) error {
	if len(s.pos) != len(params) {
		return errNumberOfParams
	}

	for i, p := range params {
		s.setServo(i, p)
	}

	return nil
}

func (s *servos) setServo(idx int, param float32) {
	if param < s.cfg[idx].Min {
		param = s.cfg[idx].Min
	}
	if param > s.cfg[idx].Max {
		param = s.cfg[idx].Max
	}
	us := param*s.cfg[idx].Scale + s.cfg[idx].Offset
	s.motors[idx].SetMicroseconds(int16(us))
}
