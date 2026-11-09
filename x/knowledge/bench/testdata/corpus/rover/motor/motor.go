// Package motor implements the per-wheel PID speed loop.
package motor

// Gains are the PID coefficients of one speed loop.
type Gains struct {
	Kp, Ki, Kd float64
}

// Controller is a PID speed controller with integral clamping.
type Controller struct {
	gains    Gains
	integral float64
	prevErr  float64
	limit    float64
}

// NewController returns a controller whose output is clamped to ±limit.
func NewController(g Gains, limit float64) *Controller {
	return &Controller{gains: g, limit: limit}
}

// Step advances the loop by dt seconds and returns the PWM duty in [-limit, limit].
func (c *Controller) Step(target, measured, dt float64) float64 {
	err := target - measured
	c.integral += err * dt
	deriv := (err - c.prevErr) / dt
	c.prevErr = err
	out := c.gains.Kp*err + c.gains.Ki*c.integral + c.gains.Kd*deriv
	return clamp(out, c.limit)
}

func clamp(v, limit float64) float64 {
	if v > limit {
		return limit
	}
	if v < -limit {
		return -limit
	}
	return v
}
