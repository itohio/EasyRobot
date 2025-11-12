// +build !servomotion

package servos

import (
	"tinygo.org/x/drivers/servo"
)

type Actuator struct {
	motors []servo.Servo
	cfg    []Motor

	timerMapping TimerMappingFunc
}

func New(cfg []Motor, timerMapping TimerMappingFunc) (Actuator, error) {
	s := Actuator{
		timerMapping: timerMapping,
	}
	err := s.Configure(cfg)
	return s, err
}

func (s *Actuator) Set(params []float32) error {
	if len(s.motors) != len(params) {
		return errNumberOfParams
	}

	for i, p := range params {
		s.setServo(i, p)
	}

	return nil
}
