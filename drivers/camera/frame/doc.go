// Package frame implements camera frame format v1: the chunked,
// CRC-protected image stream a camera device writes to its UART and a host
// decodes, shared by every EasyRobot camera firmware (K210, ESP32-S3). This
// comment is the format's canonical specification.
//
// # Stream
//
// A device writes chunks; any bytes between chunks (boot banners, logs) are
// allowed and reach the receiver as text. One image is one frame_seq and
// chunk_cnt chunks. All integers are little-endian.
//
//	Chunk = Header (24 B) || Payload (payload_len B, no padding)
//
//	Header:
//	 off size field
//	  0  u32  magic       0x4D495245, bytes on the wire "ERIM"
//	  4  u32  frame_seq   +1 per image; wraps
//	  8  u16  chunk_idx   0 .. chunk_cnt-1
//	 10  u16  chunk_cnt   ≥ 1
//	 12  u16  payload_len 1 .. 4096
//	 14  u16  flags       bit0 first, bit1 last, bit2 image meta present
//	 16  u32  frame_ts    device clock, µs, low 32 bits (capture start)
//	 20  u32  crc32       CRC-32/IEEE over header bytes 0..19, then the payload
//
//	Image meta, 16 B, at the start of the first chunk's payload (counted in payload_len):
//	  0  u16  width       pixels, ≥ 1
//	  2  u16  height      pixels, ≥ 1
//	  4  u8   format      1 Y8 · 2 RGB565 LE · 3 RGB565 BE · 4 YUYV · 5 JPEG · 6 RGB888 planar
//	  5  u8   version     1
//	  6  u16  stride      bytes per row; 0 = tight; planar: all three planes; ignored for JPEG
//	  8  u32  data_len    image bytes over all chunks, meta excluded (JPEG: exact file length)
//	 12  u8   sensor      0 unknown · 1 OV7740 · 2 OV2640 · 3 OV3660 · 4 OV5640
//	 13  u8×3 reserved    0
//
// # Receiver rules
//
//   - Find the magic and read 24 bytes. Reject payload_len 0 or > 4096,
//     chunk_cnt 0, chunk_idx ≥ chunk_cnt, or inconsistent flags (first and
//     meta exactly on chunk 0 with ≥ 16 payload bytes, last exactly on the
//     last chunk), and rescan from magic+1. Unknown flag bits are ignored.
//   - On a CRC mismatch drop the image under assembly and rescan from
//     magic+1 (the skipped bytes count as text).
//   - A missing, duplicate or out-of-order chunk_idx, or a new frame_seq
//     before last, drops the image (in-order receivers; see resend below).
//   - A meta with another version, a zero dimension, a too-small stride, an
//     odd YUYV width or a data_len mismatch drops the image. Unknown formats
//     are delivered; encoders refuse them.
//   - An image completes when last arrives and 16 + data_len bytes were
//     received. Dropped = images with a valid chunk that never completed;
//     Lost = frame_seq gaps (backward jumps ≥ 2³¹, a reboot, are not counted).
//   - A breaking change gets a new magic; new formats, sensor ids and flag
//     bits are additive in v1.
//
// # Chunk resend (optional, additive)
//
// A receiver may send 16-byte control messages back to a sender that
// supports resend:
//
//	 0  u32  magic       0x4E495245, bytes on the wire "ERIN"
//	 4  u32  frame_seq   image concerned
//	 8  u16  chunk_idx   Nak: chunk to resend; Ack: 0
//	10  u16  kind        1 Nak, 2 Ack
//	12  u32  crc32       CRC-32/IEEE over bytes 0..11
//
// The sender keeps its last image until it receives Ack for that frame_seq
// (or a timeout of its choosing, e.g. 1 s of silence), and answers each Nak
// by writing that chunk again, byte for byte. The receiver collects chunks
// in any order ([Assembler]), Naks the missing ones when the last chunk
// arrives and again after an idle time of a few chunk airtimes, Acks a
// complete image, and gives an image up (Ack) after a bounded number of
// rounds. Images are identified by frame_seq and frame_ts together, so a
// rebooted sender reusing frame_seq is not mistaken for a late resend.
// Senders without resend never read the port.
//
// # Golden vectors
//
// A 2×2 Y8 image, data 01 02 03 04, frame_seq 0, frame_ts 0x11223344,
// sensor OV2640 (CRC 0xCCBD5C3A):
//
//	45 52 49 4d 00 00 00 00 00 00 01 00 14 00 07 00 44 33 22 11 3a 5c bd cc
//	02 00 02 00 01 01 00 00 04 00 00 00 02 00 00 00
//	01 02 03 04
//
// Control messages for frame_seq 7: Nak chunk 3, then Ack:
//
//	45 52 49 4e 07 00 00 00 03 00 01 00 23 01 53 86
//	45 52 49 4e 07 00 00 00 00 00 02 00 0e fd cb bf
//
// The package is TinyGo-friendly: it imports only encoding/binary,
// hash/crc32, io, errors and bufio, and the [Encoder], [Encoder.Resend] and
// [ControlParser] do not allocate.
package frame
