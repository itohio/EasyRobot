package vec

import (
	"fmt"

	vecTypes "github.com/itohio/EasyRobot/pkg/core/math/vec/types"
)

func readVector(arg vecTypes.Vector, op string, minLen int) Vector {
	switch v := arg.(type) {
	case Vector:
		if minLen >= 0 && len(v) < minLen {
			panic(fmt.Sprintf("vec.%s: expected minimum length %d, got %d", op, minLen, len(v)))
		}
		return v
	case *Vector2D:
		return v[:]
	case *Vector3D:
		return v[:]
	case *Vector4D:
		return v[:]
	case *Quaternion:
		return v[:]
	default:
		if clone := arg.Clone(); clone != nil {
			if vv, ok := clone.(Vector); ok {
				if minLen >= 0 && len(vv) < minLen {
					panic(fmt.Sprintf("vec.%s: expected minimum length %d, got %d", op, minLen, len(vv)))
				}
				return vv
			}
		}
		panic(fmt.Sprintf("vec.%s: unsupported vector type %T", op, arg))
	}
}

func writeVector(arg vecTypes.Vector, op string, minLen int) Vector {
	switch v := arg.(type) {
	case Vector:
		if minLen >= 0 && len(v) < minLen {
			panic(fmt.Sprintf("vec.%s: expected minimum length %d, got %d", op, minLen, len(v)))
		}
		return v
	case *Vector2D:
		return v[:]
	case *Vector3D:
		return v[:]
	case *Vector4D:
		return v[:]
	case *Quaternion:
		return v[:]
	default:
		panic(fmt.Sprintf("vec.%s: unsupported destination vector type %T", op, arg))
	}
}
