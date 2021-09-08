package servos

import (
	"errors"
	"machine"
	"math"

	"github.com/foxis/EasyRobot/pkg/robot/actuator/servos/pb"
	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
	"tinygo.org/x/drivers/servo"
)

type Motor = pb.Motor
type Config = pb.Config
type State = kinematics.State
type TimerMappingFunc func(pin machine.Pin) (*machine.TCC, bool)

var (
	errInvalidPin      = errors.New("invalid pin")
	errNumberOfParams  = errors.New("number of params")
	errBadCoefficients = errors.New("bad coefficients")
)

func (s *Actuator) Configure(cfg []Motor) error {
	oldCfg := s.cfg
	oldMotors := s.motors
	s.motors = make([]servo.Servo, len(cfg))
	pos := [32]float32{}

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
			timer, valid := s.timerMapping(pin)
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
		pos[i] = c.Default
	}
	s.cfg = cfg

	return s.Set(pos[:len(s.motors)])
}

func (s *Actuator) Get() ([]float32, error) {
	return nil, nil
}

func (s *Actuator) setServo(idx int, param float32) {
	if param < s.cfg[idx].Min {
		param = s.cfg[idx].Min
	}
	if param > s.cfg[idx].Max {
		param = s.cfg[idx].Max
	}
	us := param*s.cfg[idx].Scale + s.cfg[idx].Offset
	s.motors[idx].SetMicroseconds(int16(us))
}
