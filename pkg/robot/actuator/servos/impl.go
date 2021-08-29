// +build sam,xiao

package servos

import (
	"errors"
	"machine"
	"math"

	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
	"tinygo.org/x/drivers/servo"
)

type servos struct {
	motors []servo.Servo
	cfg    []Motor
	pos    []float32
}

var (
	errInvalidPin      = errors.New("invalid pin")
	errNumberOfParams  = errors.New("number of params")
	errBadCoefficients = errors.New("bad coefficients")
)

func New(cfg []Motor) (Actuator, error) {
	s := &servos{}
	err := s.Configure(cfg)
	return s, err
}

func (s *servos) ConfigureKinematics(cfg []kinematics.DenavitHartenberg) error {
	return nil
}

func (s *servos) Configure(cfg []Motor) error {
	oldCfg := s.cfg
	oldMotors := s.motors
	s.motors = make([]servo.Servo, len(cfg))
	s.pos = make([]float32, len(cfg))
	for i, c := range cfg {
		needNewServo := true

		pin := machine.Pin(c.Pin)

		if c.Pin == 0 {
			if oldCfg == nil || len(oldCfg) != len(cfg) {
				return errInvalidPin
			}
			pin = machine.Pin(oldCfg[i].Pin)
			if oldCfg[i].Pin == cfg[i].Pin {
				needNewServo = false
			}
		}

		us := c.Min*c.Scale + c.Offset
		if us < 0 || us > math.MaxInt16 {
			return errBadCoefficients
		}
		us = c.Max*c.Scale + c.Offset
		if us < 0 || us > math.MaxInt16 {
			return errBadCoefficients
		}

		if needNewServo {
			timer, valid := timerMapping(pin)
			if !valid {
				return errInvalidPin
			}

			m, err := servo.New(timer, pin)
			if err != nil {
				return err
			}
			s.motors[i] = m
		} else {
			s.motors[i] = oldMotors[i]
		}
		s.pos[i] = c.Default
	}
	s.cfg = cfg

	return s.Set(s.pos)
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
