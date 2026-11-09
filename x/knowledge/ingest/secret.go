package ingest

import (
	"bytes"
	"maps"
	"path"
	"regexp"
	"slices"
	"strings"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// pemBegin and pemEnd delimit an armoured private key (PKCS#1/#8, EC,
// OpenSSH, encrypted, PGP "PRIVATE KEY BLOCK"); every line from BEGIN to
// END is secret.
var (
	pemBegin = regexp.MustCompile(`-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----`)
	pemEnd   = regexp.MustCompile(`-----END [A-Z0-9 ]*-----`)
)

// secretPatterns flag a line that must never be indexed.
var secretPatterns = []*regexp.Regexp{
	pemBegin,
	regexp.MustCompile(`\bS[UAOCN][A-Z2-7]{50,}\b`),                                  // NATS nkey seeds
	regexp.MustCompile(`eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}`), // JWT-shaped
	regexp.MustCompile(`\b(?:AKIA|ASIA)[0-9A-Z]{16}\b`),                              // AWS access key id (long-term, session)
	// AWS secret access key: env/ini (`AWS_SECRET_ACCESS_KEY=…`), JSON
	// (`"SecretAccessKey": "…"`, any case, aws_ prefix optional) and the CLI
	// form (`aws configure set aws_secret_access_key <key>`).
	regexp.MustCompile(`(?i)(?:aws_?)?secret_?access_?key["']?\s*(?::=|[:=]|\s)\s*["']?[A-Za-z0-9/+=]{40}(?:[^A-Za-z0-9/+=]|$)`),
	regexp.MustCompile(`\bgh[pousr]_[A-Za-z0-9]{30,}`),                           // GitHub tokens
	regexp.MustCompile(`\bgithub_pat_[A-Za-z0-9_]{20,}`),                         // GitHub fine-grained PAT
	regexp.MustCompile(`\bxox[abprs]-[A-Za-z0-9-]{10,}`),                         // Slack tokens
	regexp.MustCompile(`https://hooks\.slack\.com/services/[A-Za-z0-9/_-]{10,}`), // Slack webhooks
	regexp.MustCompile(`\bglpat-[A-Za-z0-9_-]{20,}`),                             // GitLab PAT
	regexp.MustCompile(`\b[rs]k_live_[0-9A-Za-z]{10,}`),                          // Stripe live keys
	regexp.MustCompile(`\bAIza[0-9A-Za-z_-]{35}`),                                // Google API keys
}

// secretToken matches OpenAI/Anthropic-style keys ("sk-…", "sk-proj-…",
// "sk-ant-api03-…") not inside a hyphenated identifier; the key also needs
// a part mixing letters and digits.
var secretToken = regexp.MustCompile(`(?:^|[^A-Za-z0-9_-])sk-([A-Za-z0-9_-]{20,})`)

// credURL matches URLs carrying a password: scheme://user:password@host.
var credURL = regexp.MustCompile(`\b[A-Za-z][A-Za-z0-9+.-]*://[^\s/:@]+:([^\s/@]+)@[^\s/@]`)

// secretAssign matches key = value assignments (=, :, :=; quoted keys as
// in JSON). The value counts only when one of its parts (split at - _ / .)
// mixes letters and digits, which keeps identifiers, dates, slugs and paths
// (`token: tokenSourceProviderFactory`, `token: 2026-11-09-release`,
// `password: docs/runbooks/rotation-2026`) out.
var secretAssign = regexp.MustCompile(`(?i)["']?(?:api[_-]?key|secret|token|passw(?:or)?d|private[_-]?key|client[_-]?secret)["']?\s*(?::=|[:=])\s*["']?([A-Za-z0-9_\-/+=]{16,})`)

// mixedPart reports whether a part of v, split at - _ / ., holds both a
// letter and a digit.
func mixedPart(v string) bool {
	for _, p := range strings.FieldsFunc(v, func(r rune) bool { return r == '-' || r == '_' || r == '/' || r == '.' }) {
		if strings.ContainsAny(p, "0123456789") && strings.IndexFunc(p, isLetter) >= 0 {
			return true
		}
	}
	return false
}

func isLetter(r rune) bool { return r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' }

// hasSecret reports whether one line (or a title, crumb or "key: value"
// prop) contains a secret.
func hasSecret(s string) bool {
	for _, re := range secretPatterns {
		if re.MatchString(s) {
			return true
		}
	}
	for _, m := range secretToken.FindAllStringSubmatch(s, -1) {
		if mixedPart(m[1]) {
			return true
		}
	}
	for _, m := range credURL.FindAllStringSubmatch(s, -1) {
		// Placeholders and variable references are not credentials.
		if pw := m[1]; !strings.ContainsAny(pw[:1], "$<{*") {
			return true
		}
	}
	for _, m := range secretAssign.FindAllStringSubmatch(s, -1) {
		if mixedPart(m[1]) {
			return true
		}
	}
	return false
}

// redactSecrets blanks every secret line of src before chunking: a PEM
// private key from its BEGIN line to its END line (to the end of the file
// without one) and every other line hasSecret flags. Blanked lines keep
// their terminator, so line numbers and spans are unchanged. It returns src
// itself when nothing matched, else a redacted copy, and the number of
// secret spans (a key block counts once).
func redactSecrets(src []byte) ([]byte, int) {
	lines := bytes.SplitAfter(src, []byte("\n"))
	var (
		blank []bool
		spans int
	)
	mark := func(from, to int) {
		if blank == nil {
			blank = make([]bool, len(lines))
		}
		for i := from; i <= to; i++ {
			blank[i] = true
		}
		spans++
	}
	for i := 0; i < len(lines); i++ {
		l := lines[i]
		if loc := pemBegin.FindIndex(l); loc != nil {
			end := len(lines) - 1
			if pemEnd.Match(l[loc[1]:]) {
				end = i
			} else {
				for j := i + 1; j < len(lines); j++ {
					if pemEnd.Match(lines[j]) {
						end = j
						break
					}
				}
			}
			mark(i, end)
			i = end
			continue
		}
		if hasSecret(string(l)) {
			mark(i, i)
		}
	}
	if spans == 0 {
		return src, 0
	}
	out := make([]byte, 0, len(src))
	for i, l := range lines {
		if !blank[i] {
			out = append(out, l...)
		} else if bytes.HasSuffix(l, []byte("\n")) {
			out = append(out, '\n')
		}
	}
	return out, spans
}

// dropSecretMeta removes props whose "key: value" matches a secret pattern
// and replaces a matching title, and matching chunk heading crumbs (which
// are indexed too), with the file's base name. It returns the number of
// dropped props, plus one for the title, plus one per scrubbed chunk.
func dropSecretMeta(doc knowledge.DocID, cr *knowledge.ChunkResult) int {
	n := 0
	for _, k := range slices.Sorted(maps.Keys(cr.Props)) {
		if hasSecret(k + ": " + cr.Props[k]) {
			delete(cr.Props, k)
			n++
		}
	}
	base := path.Base(string(doc))
	fallback := strings.TrimSuffix(base, path.Ext(base))
	if hasSecret(cr.Title) {
		cr.Title = fallback
		n++
	}
	for i := range cr.Chunks {
		c := &cr.Chunks[i]
		if !slices.ContainsFunc(c.Heading, hasSecret) {
			continue
		}
		c.Heading = slices.Clone(c.Heading) // crumbs may be shared between chunks
		for j, h := range c.Heading {
			if hasSecret(h) {
				c.Heading[j] = fallback
			}
		}
		n++
	}
	return n
}
