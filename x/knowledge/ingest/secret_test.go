package ingest

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/chunk/golang"
	"github.com/itohio/EasyRobot/x/knowledge/chunk/md"
)

// chunkTexts returns every stored chunk text of every document.
func chunkTexts(t *testing.T, st knowledge.Store) map[knowledge.DocID][]string {
	t.Helper()
	out := make(map[knowledge.DocID][]string)
	for _, id := range docIDs(t, st) {
		cs, err := st.DocChunks(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range cs {
			out[id] = append(out[id], c.Text)
		}
	}
	return out
}

// storedAnywhere reports the documents with a chunk containing s.
func storedAnywhere(texts map[knowledge.DocID][]string, s string) []knowledge.DocID {
	var out []knowledge.DocID
	for id, ts := range texts {
		if slices.ContainsFunc(ts, func(x string) bool { return strings.Contains(x, s) }) {
			out = append(out, id)
		}
	}
	return out
}

// keyBody returns n deterministic 64-character base64 lines.
func keyBody(n int) []string {
	out := make([]string, n)
	for i := range out {
		a := sha256.Sum256([]byte("key line " + strconv.Itoa(i)))
		b := sha256.Sum256(a[:])
		out[i] = base64.StdEncoding.EncodeToString(append(a[:], b[:16]...))[:64]
	}
	return out
}

func TestPrivateKeyAcrossChunksNeverIndexed(t *testing.T) {
	body := keyBody(47)
	key := append(append([]string{"-----BEGIN RSA PRIVATE KEY-----"}, body...), "-----END RSA PRIVATE KEY-----")
	if len(key) != 49 {
		t.Fatalf("fixture key has %d lines", len(key))
	}
	src := "# TLS runbook\n\nInstall the key with the heredoc below, runbookbefore.\n\n```bash\ncat > /etc/tls/server.key <<'EOF'\n" +
		strings.Join(key, "\n") + "\nEOF\nchmod 600 /etc/tls/server.key\n```\n\nThen restart, runbookafter.\n"
	mdc := md.New(md.WithMaxTokens(30))

	// The fixture must really split the key across chunks.
	cr, err := mdc.Chunk(knowledge.Doc{ID: "tls.md"}, []byte(src))
	if err != nil {
		t.Fatal(err)
	}
	holder := func(line string) int {
		for i, c := range cr.Chunks {
			if strings.Contains(c.Text, line) {
				return i
			}
		}
		return -1
	}
	if b, e := holder(key[0]), holder(key[48]); b < 0 || e < 0 || b == e {
		t.Fatalf("fixture: BEGIN in chunk %d, END in chunk %d; want two different chunks", b, e)
	}

	root := t.TempDir()
	writeFile(t, root, "tls.md", src)
	st := newStore(t)
	rep := sync1(t, New(st, Source{Root: root}, WithChunkers(mdc, golang.New()), WithWorkers(2)))
	if rep.SecretsDropped != 1 {
		t.Errorf("SecretsDropped = %d, want 1 (one key span)", rep.SecretsDropped)
	}
	texts := chunkTexts(t, st)
	stored := 0
	for _, l := range key {
		if len(storedAnywhere(texts, l)) > 0 {
			stored++
		}
	}
	if stored != 0 {
		t.Errorf("private-key lines stored: %d/49", stored)
	}
	for _, l := range []string{body[0], body[23], body[46]} {
		if got := search(t, st, l); len(got) != 0 {
			t.Errorf("search key line %q = %v, want none", l, got)
		}
	}
	// Lines around the key stay indexed.
	for q, want := range map[string]string{"runbookbefore": "tls.md", "runbookafter": "tls.md", "chmod": "tls.md"} {
		if got := search(t, st, q); !slices.Equal(got, ids(want)) {
			t.Errorf("search %q = %v, want [%s]", q, got, want)
		}
	}
}

func TestSecretFormatsNeverIndexed(t *testing.T) {
	type planted struct {
		file, secret string
		// block is the full secret text when it spans lines (PEM armour).
		block string
	}
	pgp := "-----BEGIN PGP PRIVATE KEY BLOCK-----\n\nlQOYBGVhc2tleQEIAMx3b2tlbjEyMzQ1Njc4OTBhYmNkZWZnaGlqa2xtbm9wcXJz\ndHV2d3h5ejAxMjM0NTY3ODkwQUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=\n=pgpZ\n-----END PGP PRIVATE KEY BLOCK-----"
	pem := func(kind, b string) string {
		return "-----BEGIN " + kind + " PRIVATE KEY-----\n" + b + "\n-----END " + kind + " PRIVATE KEY-----"
	}
	cases := []planted{
		{file: "pgp.md", secret: "lQOYBGVhc2tleQEIAMx3b2tlbjEyMzQ1Njc4OTBhYmNkZWZnaGlqa2xtbm9wcXJz", block: pgp},
		{file: "openssh.md", secret: "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtz", block: pem("OPENSSH", "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtz")},
		{file: "ec.md", secret: "MHcCAQEEIBkg4LVWM9nuwNSk3yByxZpYRTBnVJk5DVPmIaiFpe1MoAoGCCqGSM49", block: pem("EC", "MHcCAQEEIBkg4LVWM9nuwNSk3yByxZpYRTBnVJk5DVPmIaiFpe1MoAoGCCqGSM49")},
		{file: "enc.md", secret: "MIIFHDBOBgkqhkiG9w0BBQ0wQTApBgkqhkiG9w0BBQwwHAQIq3Vb0Zg5IuYCAggA", block: pem("ENCRYPTED", "MIIFHDBOBgkqhkiG9w0BBQ0wQTApBgkqhkiG9w0BBQwwHAQIq3Vb0Zg5IuYCAggA")},
		{file: "awsenv.md", secret: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", block: "export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"},
		{file: "awsini.md", secret: "je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY", block: "aws_secret_access_key = je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY"},
		{file: "awsjson.md", secret: "Tq9vL2mN8bX4rK7cY1sD6fH3jW0pZ5aE9uG2iQ8o", block: `    "SecretAccessKey": "Tq9vL2mN8bX4rK7cY1sD6fH3jW0pZ5aE9uG2iQ8o",`},
		{file: "awsjsonlc.md", secret: "Rz3kP7wQ1nV9cB5mX2tL8hJ4dF6sA0gY3eU7iO1r", block: `{"secret_access_key":"Rz3kP7wQ1nV9cB5mX2tL8hJ4dF6sA0gY3eU7iO1r"}`},
		{file: "awscli.md", secret: "Hw5nC8xK2pL7vQ3mT9bR1sZ6dY4fJ0gE8uA2iN5o", block: "aws configure set aws_secret_access_key Hw5nC8xK2pL7vQ3mT9bR1sZ6dY4fJ0gE8uA2iN5o"},
		{file: "asia.md", secret: "ASIAQ3EGUKTSAMPLE7XZ", block: "session key ASIAQ3EGUKTSAMPLE7XZ in use"},
		{file: "stripe.md", secret: "sk_" + "live_4eC39HqLyjWDarjtT1zdp7dc", block: "stripe sk_" + "live_4eC39HqLyjWDarjtT1zdp7dc"},
		{file: "stripr.md", secret: "rk_" + "live_51Hc9q2LkVb8Xz3Rt6Yp0QwE", block: "restricted rk_" + "live_51Hc9q2LkVb8Xz3Rt6Yp0QwE"},
		{file: "google.md", secret: "AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY", block: "maps key AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY"},
		{file: "webhook.md", secret: "T024BE7LD/" + "B01ABCD2EFG/xoQ1wA2bC3dE4fG5hI6jK7lM", block: "curl https://hooks.slack.com/services/T024BE7LD/" + "B01ABCD2EFG/xoQ1wA2bC3dE4fG5hI6jK7lM -d @msg.json"},
		{file: "credurl.md", secret: "S3cr3tPassw0rd", block: "DATABASE_URL=postgres://admin:S3cr3tPassw0rd@db.internal:5432/app"},
		{file: "credurl2.md", secret: "hunter2xyzzy", block: "git clone https://deploy:hunter2xyzzy@git.example.org/team/repo.git"},
	}
	root := t.TempDir()
	for _, c := range cases {
		m := strings.TrimSuffix(c.file, ".md")
		writeFile(t, root, c.file, "# "+m+"\n\nbefore"+m+"marker words\n"+c.block+"\nafter"+m+"marker words\n")
	}
	st := newStore(t)
	rep := sync1(t, New(st, Source{Root: root}, testOptions()...))
	if rep.SecretsDropped != len(cases) {
		t.Errorf("SecretsDropped = %d, want %d", rep.SecretsDropped, len(cases))
	}
	texts := chunkTexts(t, st)
	for _, c := range cases {
		m := strings.TrimSuffix(c.file, ".md")
		if got := storedAnywhere(texts, c.secret); len(got) != 0 {
			t.Errorf("%s: secret stored in %v", c.file, got)
		}
		if got := search(t, st, c.secret); len(got) != 0 {
			t.Errorf("%s: search secret = %v, want none", c.file, got)
		}
		// Only the secret lines go: neighbouring lines of the same chunk stay.
		for _, q := range []string{"before" + m + "marker", "after" + m + "marker"} {
			if got := search(t, st, q); !slices.Equal(got, ids(c.file)) {
				t.Errorf("search %q = %v, want [%s]", q, got, c.file)
			}
		}
	}
}

func TestSecretFalsePositivesStayIndexed(t *testing.T) {
	samples := map[string]string{
		"tok.md":     "token: 2026-11-09-knowledge-engine",
		"pw.md":      "password: docs/runbooks/rotation-2026",
		"skid.md":    "the flag --task-sk-abcdefghijklmnopqrstuvwxyz enables it",
		"sklearn.md": "sk-learn-preprocessing-standardscalerpipeline wraps it",
	}
	root := t.TempDir()
	for f, s := range samples {
		writeFile(t, root, f, "# Sample\n\n"+s+"\n")
	}
	st := newStore(t)
	rep := sync1(t, New(st, Source{Root: root}, testOptions()...))
	if rep.SecretsDropped != 0 {
		t.Errorf("SecretsDropped = %d, want 0", rep.SecretsDropped)
	}
	texts := chunkTexts(t, st)
	for f, s := range samples {
		if got := storedAnywhere(texts, s); !slices.Equal(got, ids(f)) {
			t.Errorf("%q stored in %v, want [%s]", s, got, f)
		}
	}
	for q, want := range map[string]string{"knowledge": "tok.md", "runbooks": "pw.md", "enables": "skid.md", "wraps": "sklearn.md"} {
		if got := search(t, st, q); !slices.Equal(got, ids(want)) {
			t.Errorf("search %q = %v, want [%s]", q, got, want)
		}
	}
}
