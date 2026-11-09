---
id: ER-031
title: "SPI slave interface and feature-frame SPI framing"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W4
milestone: M-b
tags: [easyrobot, ticket, w4, spi]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-031 — SPI slave interface and feature-frame SPI framing

[[tickets/index|Ticket index]] · [[roadmap]] · [[ER-R2]] · [[todo]]

## Goal and evidence

M-b stage 1 host work plus the stage 2 interface ([[W4-k230-feature-stream]], [[2026-11-09-k230-spi-w4]] §3 and "Smallest offline experiment"). Today SPI is master-only and the TinyGo SPI file is empty; no slave, DMA or SD.

- **SPI framing** (`x/devices/spiframe`, pure Go): 24-byte header (magic `0x45524B34`, `frame_seq`, `chunk_idx`/`chunk_cnt`, `payload_len`, flags, `frame_ts`, CRC-32 IEEE), payload ≤ 4072 B padded to 4, a 32-byte frame meta on the first chunk (shape, dtype, layout, scale, zero point, model hash), READY handshake with `latest` delivery.
- `Chunker`, `Assembler`, a host `Recorder` writing complete frames into 64 KB aligned blocks on an `io.WriterAt`, and a replay reader.
- **SPI slave interface** in `x/devices` (receive queue of DMA-sized buffers, READY pin), with a host fake; the ESP32 implementation is stage 2.

## Dependencies and scope

Dependencies: [[ER-027-marshaller-fixes|ER-027]] first (O7); CRC-32 from [[ER-023-coding-fixes|ER-023]] (or `hash/crc32`). Stage 2 needs the TinyGo stage of [[ER-026-build-green|ER-026]]. The feature-frame wire contract (decision 8) and the TinyGo SD recorder stay unticketed.

Not in scope: No ESP32 or K230 run, no protobuf decode on the MCU.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Round trip with random shapes is bit-exact.
- [ ] Property tests: injected bit flips caught 100 % by CRC; a dropped chunk loses exactly that frame; duplicated/reordered chunks rejected; READY drops skip the frame and later frames are intact.
- [ ] Every recorder write is 64 KB aligned (assertion).
- [ ] A host benchmark records `Assembler` + `Recorder` MB/s against the W4 budget (7×7×1280 int8 at 10 fps = 5.02 Mbit/s).

## Effort uncertainty and revision trigger

Product's guess: M-b stage 1 **M** (with ER-027 and the wire contract). Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
