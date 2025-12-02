package gps

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// WGS84 ellipsoid constants
const (
	// WGS84Axis is the semi-major axis of the WGS84 ellipsoid in meters
	WGS84Axis float32 = 6378137.0
	// WGS84Ecc is the first eccentricity squared of the WGS84 ellipsoid
	WGS84Ecc float32 = 0.00669438037928458
	// EarthRadius is the approximate Earth radius in meters (for spherical calculations)
	EarthRadius float32 = 6376500.0
)

// Coord represents a geographic coordinate.
// [0]=Latitude in degrees, [1]=Longitude in degrees, [2]=Altitude in meters.
type Coord vec.Vector3D

// Lat returns the latitude in degrees.
func (c Coord) Lat() float32 {
	return c[0]
}

// Lon returns the longitude in degrees.
func (c Coord) Lon() float32 {
	return c[1]
}

// Alt returns the altitude in meters.
func (c Coord) Alt() float32 {
	return c[2]
}

// ECEF represents Earth-Centered, Earth-Fixed coordinates.
// [0]=X, [1]=Y, [2]=Z, all in meters.
type ECEF vec.Vector3D

// X returns the X coordinate in meters.
func (e ECEF) X() float32 {
	return e[0]
}

// Y returns the Y coordinate in meters.
func (e ECEF) Y() float32 {
	return e[1]
}

// Z returns the Z coordinate in meters.
func (e ECEF) Z() float32 {
	return e[2]
}

// CourseAndDistance contains course (bearing) and distance between two points.
type CourseAndDistance vec.Vector2D

// Distance returns the distance in meters.
func (cd CourseAndDistance) Distance() float32 {
	return cd[0]
}

// Course returns the course/bearing in degrees (0-360).
func (cd CourseAndDistance) Course() float32 {
	return cd[1]
}

// ToECEF converts geographic coordinates to Earth-Centered, Earth-Fixed (ECEF) coordinates.
// Uses WGS84 ellipsoid model.
func ToECEF(coord Coord) ECEF {
	lat := coord.Lat() * math32.Pi / 180.0
	lon := coord.Lon() * math32.Pi / 180.0
	alt := coord.Alt()

	// Calculate prime vertical radius of curvature
	v := WGS84Axis / math32.Sqrt(1.0-WGS84Ecc*math32.Sin(lat)*math32.Sin(lat))

	// Calculate ECEF coordinates
	x := (v + alt) * math32.Cos(lat) * math32.Cos(lon)
	y := (v + alt) * math32.Cos(lat) * math32.Sin(lon)
	z := ((1.0-WGS84Ecc)*v + alt) * math32.Sin(lat)

	return ECEF(vec.Vector3D{x, y, z})
}

// HaversineDistance calculates the great circle distance between two points using the Haversine formula.
// Returns distance in meters.
func HaversineDistance(lat1, lon1, lat2, lon2 float32) float32 {
	lat1Rad := lat1 * math32.Pi / 180.0
	lat2Rad := lat2 * math32.Pi / 180.0
	lon1Rad := lon1 * math32.Pi / 180.0
	lon2Rad := lon2 * math32.Pi / 180.0

	dlat := lat2Rad - lat1Rad
	dlon := lon2Rad - lon1Rad

	a := math32.Sin(dlat/2.0)*math32.Sin(dlat/2.0) +
		math32.Cos(lat1Rad)*math32.Cos(lat2Rad)*
			math32.Sin(dlon/2.0)*math32.Sin(dlon/2.0)

	c := 2.0 * math32.Atan2(math32.Sqrt(a), math32.Sqrt(1.0-a))

	return EarthRadius * c
}

// CourseAndDistanceBetween calculates both the distance and course (bearing) between two points.
// Returns distance in meters and course in degrees (0-360).
func CourseAndDistanceBetween(lat1, lon1, lat2, lon2 float32) CourseAndDistance {
	lat1Rad := lat1 * math32.Pi / 180.0
	lat2Rad := lat2 * math32.Pi / 180.0
	lon1Rad := lon1 * math32.Pi / 180.0
	lon2Rad := lon2 * math32.Pi / 180.0

	dlat := lat2Rad - lat1Rad
	dlon := lon2Rad - lon1Rad

	// Haversine distance
	a := math32.Sin(dlat/2.0)*math32.Sin(dlat/2.0) +
		math32.Cos(lat1Rad)*math32.Cos(lat2Rad)*
			math32.Sin(dlon/2.0)*math32.Sin(dlon/2.0)

	c := 2.0 * math32.Atan2(math32.Sqrt(a), math32.Sqrt(1.0-a))
	distance := EarthRadius * c

	// Course calculation
	course := math32.Atan2(dlon*math32.Cos(lat1Rad), dlat)
	course = math32.Mod(course, 2.0*math32.Pi)
	if course < 0 {
		course += 2.0 * math32.Pi
	}
	courseDeg := course * 180.0 / math32.Pi

	return CourseAndDistance{distance, courseDeg}
}

