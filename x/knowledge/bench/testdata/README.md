# Benchmark fixtures

Everything here is synthetic: invented projects, people-free notes and toy Go
code. No real personal data.

- `corpus/` — a fake multi-project notes tree (51 Markdown notes, 10 Go files, a
  `go.mod` for module `example.com/fixture`). Ingest it with an empty scope so
  DocIDs equal the paths below.
- `fixture-questions.json` — 28 questions with known answers in `corpus/`
  (used by the default-build tests).

## Corpus layout

| Area | Content |
|---|---|
| `index.md` | Hub linking every note with wikilinks (graph hub; high in-degree everywhere). |
| `glossary.md` | Short definitions; links to the attitude limiter. |
| `rover/` | Robot firmware project: `README.md` hub (relative Markdown links), 15 design notes in `docs/`, 6 tickets `RV-001…RV-015` in `tickets/`, Go packages `motor`, `imu`, `odometry`, `telemetry`, `safety`. |
| `atlas/` | Multi-tenant HTTP service: `README.md` hub, 10 design notes / runbooks / an incident in `docs/`, 4 tickets `AT-001…AT-008`, Go packages `gateway`, `ratelimit`, `cache`, `auth`. |
| `notes/` | Coordination records: `README.md` (rules), goals `G-001…G-003`, tasks `T-001…T-005`, `decisions.md`, two dated meeting notes. |

Frontmatter carries `id` (tickets, tasks, goals), `title`, `type`,
`status`, `tags: [..]`, `updated: YYYY-MM-DD` and, on newer notes,
`supersedes: <basename>`.

Go imports between fixture packages (for `Imports` edges):
`rover/odometry → rover/imu`; `rover/safety → rover/imu, rover/motor`;
`atlas/gateway → atlas/auth, atlas/cache, atlas/ratelimit`.

Deliberately broken links (for `missing:` nodes and moved-file resolution):

- `rover/docs/imu-fusion.md` links `old/imu-calibration.md` — the file moved to
  `rover/docs/imu-calibration.md` (same basename; a resolver that falls back
  to basename finds it).
- `rover/README.md` links `docs/power-budget.md` and
  `rover/docs/safety-watchdog.md` links `[[power-budget-draft]]` — neither
  exists.
- `atlas/docs/onboarding.md` links `service-map.md` — does not exist.

Vocabulary-mismatch pair: `rover/docs/attitude-limiter.md` (and
`rover/safety/limiter.go`) never say "tipping", "steep", "slopes" or "avoid";
`rover/docs/field-test-2026-09.md` uses exactly those words and links the
limiter. A lexical-only ranker returns the field test; a one-hop link
expansion should surface the limiter.

Recency pairs (same topic, different `updated`): `battery-cutoff-2026-03`
(2026-03-02) / `battery-cutoff-2026-10` (2026-10-14); `pinout-rev-a`
(2025-11-20) / `pinout-rev-b` (2026-08-30); `deploy-runbook-2025`
(2025-12-05) / `deploy-runbook-2026` (2026-09-18); meetings `2026-10-01-sync` /
`2026-11-01-sync`. In each pair the older file sorts first by path, so a
path tie-break never favours the right answer by accident.

## Fixture questions and intended answers

"grep" is the rank the `Grep` baseline measured on 2026-11-15 (k = 8; — = not
in the top 8). Floors in `fixture_test.go` pin these.

| id | type | intended answer | why | grep |
|---|---|---|---|---|
| f01 | identifier | `rover/tickets/RV-007-odometry-slip.md` | Frontmatter `id: RV-007`; other notes only mention the id. | 1 |
| f02 | identifier | `atlas/tickets/AT-003-per-tenant-quotas.md` | Frontmatter `id: AT-003`; README hub and G-002 also mention it. | 2 |
| f03 | identifier | `notes/goals/G-002-atlas-public-beta.md` | Frontmatter `id: G-002`; tasks T-002/T-004 cite it as their goal. | 2 |
| f04 | identifier | `notes/tasks/T-004-token-rotation.md` | Frontmatter `id: T-004`, title "Token rotation". | 1 |
| f05 | identifier | `atlas/ratelimit/limiter.go` | Defines `ratelimit.TokenBucket`. | 1 |
| f06 | identifier | `rover/telemetry/telemetry.go` | Defines `telemetry.EncodeFrame`. | 1 |
| f07 | identifier | `rover/tickets/RV-012-watchdog-reset-loop.md` | Id plus title words. | 1 |
| f08 | paraphrase | `rover/docs/motor-control.md`, `rover/motor/motor.go` | "Controller coefficients chosen" = PID gains tuning (section "Tuning"; `motor.Gains`). | 1 |
| f09 | paraphrase | `rover/tickets/RV-004-imu-drift.md`, `rover/docs/imu-calibration.md` | "Compass direction creep" = heading/yaw drift from gyro bias. | 1 |
| f10 | paraphrase | `atlas/docs/rate-limiting.md`, `atlas/tickets/AT-003-per-tenant-quotas.md` | Per-tenant token buckets so one noisy customer cannot starve the others. |2 |
| f11 | paraphrase | `notes/README.md` | Rule 1: a contributor writes only the named paths. | 1 |
| f12 | paraphrase | `atlas/docs/incidents/2026-10-03-cache-stampede.md` | Key expired, 400 concurrent misses hit PostgreSQL (the "database overload"). | 6 |
| f13 | paraphrase | `rover/tickets/RV-007-odometry-slip.md`, `rover/odometry/odometry.go`, `rover/docs/odometry.md` | "Traction loss" = wheel slip; `odometry.DetectSlip`. | 3 |
| f14 | cross-link | `rover/docs/attitude-limiter.md`, `rover/safety/limiter.go` | Vocabulary mismatch: only the linking field-test note has the question's words. | — |
| f15 | cross-link | `atlas/docs/rate-limiting.md` | AT-003 says "Design note: [[atlas/docs/rate-limiting]]"; the ticket itself is not the answer. | 8 |
| f16 | cross-link | `notes/goals/G-001-rover-field-trial.md` | T-005 (lidar bring-up) has "Parent: [[G-001-…]]". | 4 |
| f17 | cross-link | `rover/docs/imu-calibration.md` | The fusion note links the old moved path `old/imu-calibration.md`. | 3 |
| f18 | cross-link | `atlas/docs/cache-invalidation.md` | AT-005 links its write-up; the note names the root cause (no version prefix). | 4 |
| f19 | recency | `rover/docs/battery-cutoff-2026-10.md` | 3.45 V/cell, `updated: 2026-10-14`, supersedes the March note. | 2 |
| f20 | recency | `rover/docs/pinout-rev-b.md` | Rev B, `updated: 2026-08-30`, supersedes rev A. | 2 |
| f21 | recency | `atlas/docs/deploy-runbook-2026.md` | Container/canary runbook, `updated: 2026-09-18`, supersedes the VM one. | 2 |
| f22 | recency | `notes/meetings/2026-11-01-sync.md` | November decision replaces October's LoRa choice. | 4 |
| f23 | code | `rover/odometry/odometry.go` | `odometry.Integrate` turns tick deltas into a `Pose`. | 1 |
| f24 | code | `atlas/ratelimit/limiter.go` | `(*TokenBucket).refill`. | 1 |
| f25 | code | `atlas/cache/cache.go` | `Cache.Get` compares `now` with `entry.expires`. | 1 |
| f26 | code | `rover/telemetry/telemetry.go` | `crc16` used by `EncodeFrame`/`DecodeFrame`. | 1 |
| f27 | code | `rover/imu/imu.go` | `(*Fusion).Update` is the complementary filter. | 1 |
| f28 | code | `atlas/gateway/gateway.go`, `atlas/auth/token.go` | Gateway middleware calls `auth.Verify` on the bearer token. | 1 |

## Live questions

Questions over a real corpus are not committed (their answers point at private
notes). The same harness runs them from a JSON file of the same shape; for this
repository they cover EasyRobot topics (Kalman P aliasing, DH Jacobian, LD06
decoder, chain adapter crash, Reed–Solomon placeholder, CRSF packing, an IK
failure, `x/math/nn/layers/gpt_attention.go`) plus recency pairs. Report
tables carry ids and scores only.
