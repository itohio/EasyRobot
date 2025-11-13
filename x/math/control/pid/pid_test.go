package pid

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/itohio/EasyRobot/x/math/vec"
)

func TestNewPanicsOnLengthMismatch(t *testing.T) {
	t.Parallel()

	p := vec.NewFrom(1, 2)
	i := vec.NewFrom(0.1)
	d := vec.NewFrom(0.01, 0.02)
	min := vec.NewFrom(-1, -1)
	max := vec.NewFrom(1, 1)

	require.Panics(t, func() {
		_ = New(p, i, d, min, max)
	})
}

func TestResetClearsIntegralAndSyncsLastInput(t *testing.T) {
	t.Parallel()

	controller := New(
		vec.NewFrom(1, 2, 3),
		vec.NewFrom(0.1, 0.2, 0.3),
		vec.NewFrom(0.01, 0.02, 0.03),
		vec.NewFrom(-10, -10, -10),
		vec.NewFrom(10, 10, 10),
	)

	controller.Input = vec.NewFrom(4, 5, 6)
	controller.lastInput = vec.NewFrom(1, 1, 1)
	controller.iTerm = vec.NewFrom(3, 3, 3)

	controller.Reset()

	require.Equal(t, vec.NewFrom(0, 0, 0), controller.iTerm)
	require.Equal(t, controller.Input, controller.lastInput)
}

func TestUpdateComputesPIDTerms(t *testing.T) {
	t.Parallel()

	const dt = 0.1

	controller := New(
		vec.NewFrom(2),
		vec.NewFrom(0.5),
		vec.NewFrom(1),
		vec.NewFrom(-10),
		vec.NewFrom(10),
	)

	controller.Target = vec.NewFrom(4)
	controller.Input = vec.NewFrom(1)
	controller.lastInput = vec.NewFrom(0.5)
	controller.iTerm = vec.NewFrom(0)

	controller.Update(dt)

	require.InDelta(t, 0.15, controller.iTerm[0], 1e-6)
	require.InDelta(t, 1.15, controller.Output[0], 1e-6)
	require.InDelta(t, controller.Input[0], controller.lastInput[0], 1e-6)
}

func TestUpdateIsComponentwise(t *testing.T) {
	t.Parallel()

	const dt = 0.05

	controller := New(
		vec.NewFrom(1, 2, 3),
		vec.NewFrom(0.4, 0.5, 0.6),
		vec.NewFrom(0.7, 0.8, 0.9),
		vec.NewFrom(-5, -4, -3),
		vec.NewFrom(5, 4, 3),
	)

	controller.Target = vec.NewFrom(2, 4, 6)
	controller.Input = vec.NewFrom(1, 2, 3)
	controller.lastInput = vec.NewFrom(0.5, 1.5, 2.5)
	controller.iTerm = vec.NewFrom(0, 0, 0)

	controller.Update(dt)

	expectedOutput := vec.NewFrom(
		-5,
		-3.95,
		0.09,
	)

	require.InDeltaSlice(t, expectedOutput, controller.Output, 1e-5)
	require.InDeltaSlice(t, controller.Input, controller.lastInput, 1e-6)
}

func TestIntegralClampingPreventsWindup(t *testing.T) {
	t.Parallel()

	const dt = 0.1

	controller := New(
		vec.NewFrom(0),
		vec.NewFrom(5),
		vec.NewFrom(0),
		vec.NewFrom(-0.5),
		vec.NewFrom(0.5),
	)

	controller.Target = vec.NewFrom(1)
	controller.Input = vec.NewFrom(0)
	controller.lastInput = vec.NewFrom(0)
	controller.iTerm = vec.NewFrom(0)

	for i := 0; i < 10; i++ {
		controller.Update(dt)
	}

	require.InDelta(t, 0.5, controller.iTerm[0], 1e-6)
	require.InDelta(t, 0.5, controller.Output[0], 1e-6)
}

func TestPID1DReset(t *testing.T) {
	t.Parallel()

	controller := New1D(1, 0.5, 0.2, -10, 10)
	controller.Input = 3
	controller.lastInput = 1
	controller.iTerm = 2

	controller.Reset()

	require.Zero(t, controller.iTerm)
	require.Equal(t, controller.Input, controller.lastInput)
}

func TestPID1DUpdate(t *testing.T) {
	t.Parallel()

	const dt = 0.05

	controller := New1D(2, 0.4, 0.6, -5, 5)
	controller.Target = 4
	controller.Input = 1
	controller.lastInput = 0.5
	controller.iTerm = 0.1

	controller.Update(dt)

	require.InDelta(t, 0.16, float64(controller.iTerm), 1e-6)
	require.InDelta(t, 0.16, float64(controller.Output), 1e-6)
	require.InDelta(t, controller.Input, controller.lastInput, 1e-6)
}

func TestPID1DIntegralClamping(t *testing.T) {
	t.Parallel()

	const dt = 0.1

	controller := New1D(0, 5, 0, -0.5, 0.5)
	controller.Target = 1
	controller.Input = 0
	controller.lastInput = 0
	controller.iTerm = 0

	for i := 0; i < 8; i++ {
		controller.Update(dt)
	}

	require.InDelta(t, 0.5, float64(controller.iTerm), 1e-6)
	require.InDelta(t, 0.5, float64(controller.Output), 1e-6)
}
