// Package token is the identifier-aware tokenizer shared by indexing and
// querying: it keeps ids (ABU-061) and dotted/snake/kebab identifiers whole,
// also emits their parts, splits camelCase, and lightly stems prose words.
// Indexing and querying must use the same functions so terms agree.
package token

import (
	"strings"
	"unicode"
)

// Tokens returns index terms of s in order, with repetitions (for term
// frequency). All terms are lower-case.
func Tokens(s string) []string {
	var out []string
	for _, w := range words(s) {
		out = appendWord(out, w)
	}
	return out
}

// Query returns the unique terms of s in first-seen order.
func Query(s string) []string {
	seen := map[string]bool{}
	var out []string
	for _, t := range Tokens(s) {
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	return out
}

// IDs returns id-like words (ABU-061, O-005, KE-1) in s, upper-cased,
// unique, in first-seen order.
func IDs(s string) []string {
	seen := map[string]bool{}
	var out []string
	for _, w := range words(s) {
		if IsID(w) {
			u := strings.ToUpper(w)
			if !seen[u] {
				seen[u] = true
				out = append(out, u)
			}
		}
	}
	return out
}

// IsID reports whether w looks like a ticket/record id: 1–6 letters, a
// hyphen, 1–5 digits.
func IsID(w string) bool {
	letters, rest, ok := strings.Cut(w, "-")
	if !ok || len(letters) < 1 || len(letters) > 6 || len(rest) < 1 || len(rest) > 5 {
		return false
	}
	for _, r := range letters {
		if r > unicode.MaxASCII || !unicode.IsLetter(r) {
			return false
		}
	}
	for _, r := range rest {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// Count approximates the token count of s (whitespace-separated fields).
func Count(s string) int { return len(strings.Fields(s)) }

func isWordRune(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }

func isConnector(r rune) bool { return r == '_' || r == '-' || r == '.' }

// words splits s into candidate words: runs of letters, digits and inner
// connectors (_ - .), with connectors trimmed from both ends.
func words(s string) []string {
	fields := strings.FieldsFunc(s, func(r rune) bool { return !isWordRune(r) && !isConnector(r) })
	out := fields[:0]
	for _, f := range fields {
		f = strings.TrimFunc(f, isConnector)
		if f != "" {
			out = append(out, f)
		}
	}
	return out
}

func appendWord(out []string, w string) []string {
	if IsID(w) {
		return append(out, strings.ToLower(w))
	}
	parts := strings.FieldsFunc(w, isConnector)
	if len(parts) > 1 {
		out = append(out, strings.ToLower(w))
		for _, p := range parts {
			out = appendSimple(out, p)
		}
		return out
	}
	return appendSimple(out, w)
}

func appendSimple(out []string, p string) []string {
	sub := camelSplit(p)
	if len(sub) > 1 {
		// Keep the whole identifier, plus its stemmed form so a lower-case
		// query ("peerpermissions", stemmed as a prose word) still matches.
		whole := strings.ToLower(p)
		out = append(out, whole)
		if s := stem(whole); s != whole {
			out = append(out, s)
		}
		for _, s := range sub {
			out = appendTerm(out, s)
		}
		return out
	}
	return appendTerm(out, p)
}

func appendTerm(out []string, s string) []string {
	l := strings.ToLower(s)
	if len(l) < 2 && !(len(l) == 1 && l[0] >= '0' && l[0] <= '9') {
		return out
	}
	if stopwords[l] {
		return out
	}
	return append(out, stem(l))
}

// camelSplit splits at lower→Upper and at the last upper of an
// upper-case run followed by lower ("HTTPServer" → HTTP, Server).
func camelSplit(s string) []string {
	rs := []rune(s)
	var out []string
	start := 0
	for i := 1; i < len(rs); i++ {
		prev, cur := rs[i-1], rs[i]
		boundary := unicode.IsLower(prev) && unicode.IsUpper(cur)
		if !boundary && unicode.IsUpper(prev) && unicode.IsUpper(cur) && i+1 < len(rs) && unicode.IsLower(rs[i+1]) {
			boundary = true
		}
		if boundary {
			out = append(out, string(rs[start:i]))
			start = i
		}
	}
	return append(out, string(rs[start:]))
}

// stem applies light English suffix rules to purely alphabetic words until
// none applies, so stem is idempotent and a plural reaches the same term as
// its singular ("settings" → "setting" → "set").
func stem(w string) string {
	for {
		s := stemOnce(w)
		if s == w {
			return w
		}
		w = s
	}
}

// stemOnce strips one suffix; every rule shortens w, so stem terminates.
func stemOnce(w string) string {
	if len(w) <= 3 {
		return w
	}
	for _, r := range w {
		if r < 'a' || r > 'z' {
			return w
		}
	}
	switch {
	case strings.HasSuffix(w, "ies") && len(w) > 4:
		return w[:len(w)-3] + "y"
	case strings.HasSuffix(w, "sses"), strings.HasSuffix(w, "xes"):
		return w[:len(w)-2]
	case strings.HasSuffix(w, "ing") && len(w)-3 >= 3:
		return undouble(w[:len(w)-3])
	case strings.HasSuffix(w, "ed") && len(w)-2 >= 3:
		return undouble(w[:len(w)-2])
	case strings.HasSuffix(w, "s") && !strings.HasSuffix(w, "ss") && !strings.HasSuffix(w, "us") && !strings.HasSuffix(w, "is"):
		return w[:len(w)-1]
	}
	return w
}

func undouble(w string) string {
	n := len(w)
	if n >= 2 && w[n-1] == w[n-2] && strings.IndexByte("bdfgmnprt", w[n-1]) >= 0 {
		return w[:n-1]
	}
	return w
}

var stopwords = map[string]bool{
	"a": true, "an": true, "and": true, "are": true, "as": true, "at": true, "be": true, "by": true,
	"for": true, "from": true, "in": true, "into": true, "is": true, "it": true, "its": true, "of": true,
	"on": true, "or": true, "that": true, "the": true, "this": true, "to": true, "was": true, "were": true,
	"with": true, "what": true, "which": true, "how": true, "does": true, "do": true, "when": true, "where": true,
}
