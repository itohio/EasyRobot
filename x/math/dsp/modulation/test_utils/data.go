package test_utils

// MakeData generates a deterministic []byte sequence of the phrase
// "The quick brown fox jumps over the lazy dog" repeated as many times as needed
// to reach the requested length. (Sometimes referred to as "A brown fox..." in the
// prompt, but using the canonical pangram.)
func MakeData(length int) []byte {
	phrase := []byte("The quick brown fox jumps over the lazy dog")
	result := make([]byte, length)
	phraseLen := len(phrase)
	for i := 0; i < length; i++ {
		result[i] = phrase[i%phraseLen]
	}
	return result
}
