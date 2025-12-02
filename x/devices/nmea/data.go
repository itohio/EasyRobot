package nmea

import (
	"strconv"
	"time"
)

// Data contains essential GPS parameters extracted from NMEA sentences.
type Data struct {
	// Time information
	Time      time.Time // UTC time from GPS
	TimeValid bool      // Whether time is valid

	// Position information
	Latitude   float32 // Latitude in degrees (positive = North)
	Longitude  float32 // Longitude in degrees (positive = East)
	Altitude   float32 // Altitude in meters
	PosValid   bool    // Whether position is valid

	// Fix information
	FixQuality int  // Fix quality: 0=invalid, 1=GPS, 2=DGPS, etc.
	NumSatellites int // Number of satellites in view
	HDOP        float32 // Horizontal Dilution of Precision

	// Navigation information
	Speed      float32 // Speed over ground in m/s
	Course     float32 // Course/bearing in degrees (0-360)
	NavValid   bool    // Whether navigation data is valid

	// Additional information
	LastUpdate time.Time // When this data was last updated
}

// ParseGGA parses a GPGGA (Global Positioning System Fix Data) sentence.
func ParseGGA(s *Sentence) (*Data, error) {
	if s.Type != "GPGGA" && s.Type != "GNGGA" {
		return nil, ErrInvalidSentence
	}

	if len(s.Fields) < 14 {
		return nil, ErrInvalidSentence
	}

	data := &Data{
		LastUpdate: s.Timestamp,
	}

	// Parse time (field 0: HHMMSS.SSS)
	if s.Fields[0] != "" {
		timeStr := s.Fields[0]
		if len(timeStr) >= 6 {
			hour, _ := strconv.Atoi(timeStr[0:2])
			min, _ := strconv.Atoi(timeStr[2:4])
			sec, _ := strconv.Atoi(timeStr[4:6])
			now := time.Now().UTC()
			data.Time = time.Date(now.Year(), now.Month(), now.Day(), hour, min, sec, 0, time.UTC)
			data.TimeValid = true
		}
	}

	// Parse latitude (field 1: DDMM.MMMM, field 2: N/S)
	if s.Fields[1] != "" && s.Fields[2] != "" {
		lat, err := parseDMM(s.Fields[1], s.Fields[2])
		if err == nil {
			data.Latitude = lat
			data.PosValid = true
		}
	}

	// Parse longitude (field 3: DDDMM.MMMM, field 4: E/W)
	if s.Fields[3] != "" && s.Fields[4] != "" {
		lon, err := parseDMM(s.Fields[3], s.Fields[4])
		if err == nil {
			data.Longitude = lon
			data.PosValid = true
		}
	}

	// Parse fix quality (field 5)
	if s.Fields[5] != "" {
		quality, _ := strconv.Atoi(s.Fields[5])
		data.FixQuality = quality
		if quality == 0 {
			data.PosValid = false
		}
	}

	// Parse number of satellites (field 6)
	if s.Fields[6] != "" {
		numSat, _ := strconv.Atoi(s.Fields[6])
		data.NumSatellites = numSat
	}

	// Parse HDOP (field 7)
	if s.Fields[7] != "" {
		hdop, _ := strconv.ParseFloat(s.Fields[7], 32)
		data.HDOP = float32(hdop)
	}

	// Parse altitude (field 8: meters, field 9: M)
	if s.Fields[8] != "" {
		alt, _ := strconv.ParseFloat(s.Fields[8], 32)
		data.Altitude = float32(alt)
	}

	return data, nil
}

// ParseRMC parses a GPRMC (Recommended Minimum Course) sentence.
func ParseRMC(s *Sentence) (*Data, error) {
	if s.Type != "GPRMC" && s.Type != "GNRMC" {
		return nil, ErrInvalidSentence
	}

	if len(s.Fields) < 12 {
		return nil, ErrInvalidSentence
	}

	data := &Data{
		LastUpdate: s.Timestamp,
	}

	// Parse time (field 0: HHMMSS.SSS)
	if s.Fields[0] != "" {
		timeStr := s.Fields[0]
		if len(timeStr) >= 6 {
			hour, _ := strconv.Atoi(timeStr[0:2])
			min, _ := strconv.Atoi(timeStr[2:4])
			sec, _ := strconv.Atoi(timeStr[4:6])
			now := time.Now().UTC()
			data.Time = time.Date(now.Year(), now.Month(), now.Day(), hour, min, sec, 0, time.UTC)
			data.TimeValid = true
		}
	}

	// Parse status (field 1: A=valid, V=invalid)
	if s.Fields[1] == "A" {
		data.PosValid = true
		data.NavValid = true
	}

	// Parse latitude (field 2: DDMM.MMMM, field 3: N/S)
	if s.Fields[2] != "" && s.Fields[3] != "" {
		lat, err := parseDMM(s.Fields[2], s.Fields[3])
		if err == nil {
			data.Latitude = lat
		}
	}

	// Parse longitude (field 4: DDDMM.MMMM, field 5: E/W)
	if s.Fields[4] != "" && s.Fields[5] != "" {
		lon, err := parseDMM(s.Fields[4], s.Fields[5])
		if err == nil {
			data.Longitude = lon
		}
	}

	// Parse speed (field 6: knots)
	if s.Fields[6] != "" {
		speedKnots, _ := strconv.ParseFloat(s.Fields[6], 32)
		// Convert knots to m/s: 1 knot = 0.514444 m/s
		data.Speed = float32(speedKnots * 0.514444)
	}

	// Parse course (field 7: degrees)
	if s.Fields[7] != "" {
		course, _ := strconv.ParseFloat(s.Fields[7], 32)
		data.Course = float32(course)
	}

	// Parse date (field 8: DDMMYY)
	if s.Fields[8] != "" && data.TimeValid {
		dateStr := s.Fields[8]
		if len(dateStr) >= 6 {
			day, _ := strconv.Atoi(dateStr[0:2])
			month, _ := strconv.Atoi(dateStr[2:4])
			year, _ := strconv.Atoi(dateStr[4:6])
			year += 2000 // Convert YY to YYYY
			if year < 2000 {
				year += 100 // Handle years 00-99
			}
			data.Time = time.Date(year, time.Month(month), day, data.Time.Hour(), data.Time.Minute(), data.Time.Second(), 0, time.UTC)
		}
	}

	return data, nil
}

// parseDMM parses degrees and decimal minutes format (DDMM.MMMM or DDDMM.MMMM).
// Returns degrees as a float32.
func parseDMM(dmmStr, hemisphere string) (float32, error) {
	if dmmStr == "" {
		return 0, ErrInvalidSentence
	}

	// Find decimal point
	dotIdx := -1
	for i := range dmmStr {
		if dmmStr[i] == '.' {
			dotIdx = i
			break
		}
	}

	if dotIdx < 0 {
		return 0, ErrInvalidSentence
	}

	// Extract degrees (2 digits for lat, 3 for lon)
	degStr := dmmStr[:dotIdx-2]
	deg, err := strconv.Atoi(degStr)
	if err != nil {
		return 0, err
	}

	// Extract minutes (including decimal part)
	minStr := dmmStr[dotIdx-2:]
	min, err := strconv.ParseFloat(minStr, 32)
	if err != nil {
		return 0, err
	}

	// Convert to decimal degrees
	result := float32(deg) + float32(min)/60.0

	// Apply hemisphere
	if hemisphere == "S" || hemisphere == "W" {
		result = -result
	}

	return result, nil
}

// MergeData merges data from multiple sentences, with later data taking precedence.
func MergeData(dst *Data, src *Data) {
	if src.TimeValid {
		dst.Time = src.Time
		dst.TimeValid = true
	}
	if src.PosValid {
		dst.Latitude = src.Latitude
		dst.Longitude = src.Longitude
		dst.Altitude = src.Altitude
		dst.PosValid = true
	}
	if src.FixQuality > 0 {
		dst.FixQuality = src.FixQuality
	}
	if src.NumSatellites > 0 {
		dst.NumSatellites = src.NumSatellites
	}
	if src.HDOP > 0 {
		dst.HDOP = src.HDOP
	}
	if src.NavValid {
		dst.Speed = src.Speed
		dst.Course = src.Course
		dst.NavValid = true
	}
	if src.LastUpdate.After(dst.LastUpdate) {
		dst.LastUpdate = src.LastUpdate
	}
}

