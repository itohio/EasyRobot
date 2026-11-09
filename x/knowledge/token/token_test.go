package token

import (
	"slices"
	"testing"
)

func TestTokensIdentifiers(t *testing.T) {
	cases := []struct {
		in   string
		want []string // must all be present
		not  []string // must be absent
	}{
		{"PeerPermissions", []string{"peerpermissions", "peer", "permission"}, nil},
		{"HTTPServer", []string{"httpserver", "http", "server"}, nil},
		{"abg.svc.find", []string{"abg.svc.find", "abg", "svc", "find"}, nil},
		{"snake_case_name", []string{"snake_case_name", "snake", "case", "name"}, nil},
		{"kebab-case-name", []string{"kebab-case-name", "kebab", "case", "name"}, nil},
		{"ABU-061", []string{"abu-061"}, []string{"abu", "061"}},
		{"see O-005 and KE-1.", []string{"o-005", "ke-1", "see"}, []string{"and"}},
		{"abugaga/internal/keys/keys.go", []string{"abugaga", "internal", "keys.go", "key", "go"}, nil},
		{"permissions running indexed queries", []string{"permission", "run", "index", "query"}, nil},
		{"`Teststore_Name` (foo)", []string{"teststore_name", "teststore", "name", "foo"}, nil},
	}
	for _, c := range cases {
		got := Tokens(c.in)
		for _, w := range c.want {
			if !slices.Contains(got, w) {
				t.Errorf("Tokens(%q) = %v, missing %q", c.in, got, w)
			}
		}
		for _, w := range c.not {
			if slices.Contains(got, w) {
				t.Errorf("Tokens(%q) = %v, must not contain %q", c.in, got, w)
			}
		}
	}
}

func TestQueryUniqueAndStable(t *testing.T) {
	got := Query("peer peer Peer permissions")
	want := []string{"peer", "permission"}
	if !slices.Equal(got, want) {
		t.Fatalf("Query = %v, want %v", got, want)
	}
}

func TestIndexAndQueryAgree(t *testing.T) {
	// Whatever the query produces for a word must be produced by indexing it.
	for _, w := range []string{"Permissions", "keys.go", "ABU-061", "GPTAttention", "data_bytes", "running"} {
		idx := Tokens(w)
		for _, q := range Query(w) {
			if !slices.Contains(idx, q) {
				t.Errorf("query token %q of %q not in index tokens %v", q, w, idx)
			}
		}
	}
}

func TestLowercaseQueryMatchesCamelCaseIndex(t *testing.T) {
	// People type identifiers in lower case; "peerpermissions" must hit
	// a chunk that contains "PeerPermissions".
	idx := Tokens("PeerPermissions GPTAttention")
	for _, q := range []string{"peerpermissions", "PeerPermissions", "gptattention"} {
		for _, term := range Query(q) {
			if !slices.Contains(idx, term) {
				t.Errorf("query %q term %q not in index tokens %v", q, term, idx)
			}
		}
	}
}

func TestIDs(t *testing.T) {
	got := IDs("Fixes ABU-061, see o-005 and KE-1; not 2026-11-09 or v1.2-3 or x-y")
	want := []string{"ABU-061", "O-005", "KE-1"}
	if !slices.Equal(got, want) {
		t.Fatalf("IDs = %v, want %v", got, want)
	}
	if !IsID("ER-017") || IsID("2026-11") || IsID("abc") {
		t.Fatal("IsID misclassifies")
	}
}

func TestCount(t *testing.T) {
	if n := Count("one two three"); n != 3 {
		t.Fatalf("Count = %d, want 3", n)
	}
	if n := Count(""); n != 0 {
		t.Fatalf("Count empty = %d", n)
	}
}

func TestStemPluralMatchesSingular(t *testing.T) {
	pairs := [][2]string{
		{"settings", "setting"}, {"warnings", "warning"}, {"embeddings", "embedding"},
		{"mappings", "mapping"}, {"readings", "reading"}, {"strings", "string"},
		{"permissions", "permission"}, {"queries", "query"}, {"classes", "class"},
		{"boxes", "box"}, {"indexes", "index"}, {"caches", "cache"}, {"chunks", "chunk"},
	}
	for _, p := range pairs {
		if a, b := Tokens(p[0]), Tokens(p[1]); !slices.Equal(a, b) {
			t.Errorf("Tokens(%q) = %v, Tokens(%q) = %v; want equal", p[0], a, p[1], b)
		}
	}
}

func TestStemIdempotent(t *testing.T) {
	words := []string{
		"settings", "setting", "warnings", "embeddings", "embedded", "embed", "mappings",
		"readings", "strings", "string", "matches", "boxes", "indexed", "running", "queries",
		"studies", "classes", "status", "analysis", "speeded", "things", "nested", "stopped",
		"buildings", "fixes", "caches", "permissions", "tested", "hopping",
	}
	for _, w := range words {
		s := stem(w)
		if s2 := stem(s); s2 != s {
			t.Errorf("stem(%q) = %q but stem(%q) = %q", w, s, s, s2)
		}
		if len(s) < 3 {
			t.Errorf("stem(%q) = %q, too short", w, s)
		}
	}
}
