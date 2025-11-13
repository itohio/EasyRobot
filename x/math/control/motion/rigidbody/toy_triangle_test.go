package rigidbody

import (
	"testing"

	"github.com/chewxy/math32"
	rigidbody "github.com/itohio/EasyRobot/x/math/control/kinematics/rigidbody"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

func TestTriangularChassisFollowsSinusoid(t *testing.T) {
	m := mustPlanner(t)

	const (
		dt          = 0.02
		mass        = 1.0
		radius      = 0.5
		steps       = 200
		pathSamples = 400
		pathStep    = 0.05
		pathFreq    = 0.4
	)

	rbCfg := rigidbody.Config{
		LinearGain:  vec.Vector3D{12, 12, 12},
		AngularGain: vec.Vector3D{16, 16, 16},
		MaxForce:    vec.Vector3D{140, 140, 140},
		MaxTorque:   vec.Vector3D{90, 90, 90},
	}
	inertiaTensor := rigidbody.InertiaSolidCylinder(mass, radius, radius*0.5)
	inertiaTensor = rigidbody.RotateInertia(inertiaTensor, rotationZ(math32.Pi/6))
	rbModel, err := rigidbody.NewModel(mass, inertiaTensor, rbCfg)
	if err != nil {
		t.Fatalf("failed to create rigid body model: %v", err)
	}

	path := make([]vec.Vector3D, pathSamples)
	for i := 0; i < pathSamples; i++ {
		x := float32(i) * pathStep
		y := math32.Sin(pathFreq * x)
		path[i] = vec.Vector3D{x, y, 0}
	}
	if err := m.SetPath(path); err != nil {
		t.Fatalf("SetPath error: %v", err)
	}

	state := mat.New(testStateRows, 1)
	controls := mat.New(testControlRows, 1)
	destination := mat.New(testStateRows, 1)

	rbState := mat.New(6, 1)
	rbDest := mat.New(6, 1)
	rbControls := mat.New(6, 1)

	// Physical state of triangular chassis (body frame velocities).
	pos := vec.Vector3D{0, 0, 0}
	yaw := float32(0)
	bodyVel := vec.Vector3D{0, 0, 0}
	angVel := vec.Vector3D{0, 0, 0}
	time := float32(0)

	maxDistance := float32(0)
	maxYawError := float32(0)

	for stepIdx := 0; stepIdx < steps; stepIdx++ {
		// Write current state into planner matrix.
		state[0][0] = pos[0]
		state[1][0] = pos[1]
		state[2][0] = pos[2]
		state[3][0] = yaw
		state[4][0] = math32.Sqrt(bodyVel[0]*bodyVel[0] + bodyVel[1]*bodyVel[1])
		state[5][0] = time

		if err := m.Forward(state, destination, controls); err != nil {
			t.Fatalf("Forward error at step %d: %v", stepIdx, err)
		}
		if err := m.Backward(state, destination, controls); err != nil {
			t.Fatalf("Backward error at step %d: %v", stepIdx, err)
		}

		desiredSpeed := controls[0][0]
		desiredAngular := controls[1][0]

		desiredLinear := vec.Vector3D{desiredSpeed, 0, 0}
		desiredAngularVec := vec.Vector3D{0, 0, desiredAngular}

		// Populate rigid body state and desired velocities.
		for i := 0; i < 3; i++ {
			rbState[i][0] = bodyVel[i]
			rbState[i+3][0] = angVel[i]
			rbDest[i][0] = desiredLinear[i]
			rbDest[i+3][0] = desiredAngularVec[i]
		}

		if err := rbModel.Backward(rbState, rbDest, rbControls); err != nil {
			t.Fatalf("rigid body Backward error at step %d: %v", stepIdx, err)
		}

		forceBody := vec.Vector3D{rbControls[0][0], rbControls[1][0], rbControls[2][0]}
		torqueBody := vec.Vector3D{rbControls[3][0], rbControls[4][0], rbControls[5][0]}

		// Integrate body-frame linear velocities.
		for i := 0; i < 3; i++ {
			acc := forceBody[i] / mass
			bodyVel[i] += acc * dt
		}

		angAccVec := rbModel.AngularAcceleration(torqueBody)
		for i := 0; i < 3; i++ {
			angVel[i] += angAccVec[i] * dt
		}

		yaw = normalizeAngle(yaw + angVel[2]*dt)

		// Convert body velocity to world frame for position integration.
		vx := bodyVel[0]*math32.Cos(yaw) - bodyVel[1]*math32.Sin(yaw)
		vy := bodyVel[0]*math32.Sin(yaw) + bodyVel[1]*math32.Cos(yaw)
		pos[0] += vx * dt
		pos[1] += vy * dt
		time += dt

		dist, tangentYaw := nearestDistanceAndYaw(pos, path)
		if dist > maxDistance {
			maxDistance = dist
		}
		yawError := math32.Abs(normalizeAngle(yaw - tangentYaw))
		if yawError > maxYawError {
			maxYawError = yawError
		}
	}

	if maxDistance > 0.5 {
		t.Fatalf("path deviation too large: %.3f m", maxDistance)
	}
	if maxYawError > 0.55 {
		t.Fatalf("yaw misalignment too high: %.3f rad", maxYawError)
	}
}

func nearestDistanceAndYaw(pos vec.Vector3D, path []vec.Vector3D) (float32, float32) {
	minDist := float32(math32.MaxFloat32)
	var tangentYaw float32
	for i := 0; i < len(path)-1; i++ {
		a := path[i]
		b := path[i+1]
		proj := projectPoint(pos, a, b)
		dist := distance(pos, proj)
		if dist < minDist {
			minDist = dist
			t := unitTangent(a, b)
			tangentYaw = math32.Atan2(t[1], t[0])
		}
	}
	return minDist, tangentYaw
}

func rotationZ(angle float32) mat.Matrix3x3 {
	return mat.Matrix3x3{}.RotationZ(angle).(mat.Matrix3x3)
}
