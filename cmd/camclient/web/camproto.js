// camproto.js: the camera frame format v1 (drivers/camera/frame) and the
// webcam example's text protocol, for the client page (Web Serial).
//
// FrameDecoder.push(bytes) consumes a chunk of the serial stream and returns
// events: {type: "image", seq, ts, meta, data} for every complete,
// CRC-checked image, and {type: "text", line} for every text line between
// chunks (the board's "webcam: …" lines). Receiver rules as in Go's
// frame.Decoder: scan for the magic; a header failing the bounds is skipped
// (scan resumes at magic+1); a CRC mismatch drops the image under assembly;
// a missing or out-of-order chunk, a changed count or a new seq before the
// last chunk drops it; an image completes with its last chunk when exactly
// MetaSize + data_len bytes arrived.

export const HeaderSize = 24, MetaSize = 16, MaxPayload = 4096;
export const FlagFirst = 1, FlagLast = 2, FlagMeta = 4;
export const FormatJPEG = 5;
const MAGIC = [0x45, 0x52, 0x49, 0x4d]; // "ERIM"

const crcTable = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

// crc32 continues an IEEE CRC-32 over b (start with crc = 0).
export function crc32(b, crc = 0) {
  let c = ~crc >>> 0;
  for (let i = 0; i < b.length; i++) c = crcTable[(c ^ b[i]) & 0xff] ^ (c >>> 8);
  return ~c >>> 0;
}

export class FrameDecoder {
  constructor() {
    this.buf = new Uint8Array(0);
    this.text = []; // bytes of the text line in progress
    this.img = null; // {seq, cnt, next, ts, meta, data, off}
    this.stats = { images: 0, crcErrors: 0, dropped: 0, textBytes: 0 };
  }

  push(bytes) {
    const merged = new Uint8Array(this.buf.length + bytes.length);
    merged.set(this.buf, 0);
    merged.set(bytes, this.buf.length);
    this.buf = merged;
    const out = [];
    let i = 0;
    const b = this.buf;
    while (i < b.length) {
      const m = this.findMagic(b, i);
      if (m < 0) {
        // Hold back only trailing bytes that may start a magic ("E", "ER",
        // "ERI"), so a final text line is not delayed.
        let keep = 0;
        for (let k = Math.min(3, b.length - i); k > 0 && !keep; k--) {
          let match = true;
          for (let j = 0; j < k; j++) if (b[b.length - k + j] !== MAGIC[j]) match = false;
          if (match) keep = k;
        }
        this.toText(b, i, b.length - keep, out);
        i = b.length - keep;
        break;
      }
      if (m > i) this.toText(b, i, m, out);
      i = m;
      if (b.length - i < HeaderSize) break;
      const dv = new DataView(b.buffer, b.byteOffset + i, HeaderSize);
      const h = {
        seq: dv.getUint32(4, true), idx: dv.getUint16(8, true), cnt: dv.getUint16(10, true),
        plen: dv.getUint16(12, true), flags: dv.getUint16(14, true), ts: dv.getUint32(16, true), crc: dv.getUint32(20, true),
      };
      if (!validHeader(h)) {
        this.toText(b, i, i + 1, out);
        i += 1;
        continue;
      }
      if (b.length - i < HeaderSize + h.plen) break;
      const payload = b.subarray(i + HeaderSize, i + HeaderSize + h.plen);
      if (crc32(payload, crc32(b.subarray(i, i + 20))) !== h.crc) {
        this.stats.crcErrors++;
        this.drop();
        this.toText(b, i, i + 1, out);
        i += 1;
        continue;
      }
      this.text = []; // a valid chunk ends any partial text (e.g. bytes of a torn chunk)
      const done = this.chunk(h, payload);
      if (done) out.push(done);
      i += HeaderSize + h.plen;
    }
    this.buf = this.buf.slice(i);
    return out;
  }

  findMagic(b, from) {
    for (let i = from; i <= b.length - 4; i++) {
      if (b[i] === MAGIC[0] && b[i + 1] === MAGIC[1] && b[i + 2] === MAGIC[2] && b[i + 3] === MAGIC[3]) return i;
    }
    return -1;
  }

  toText(b, from, to, out) {
    this.stats.textBytes += to - from;
    for (let i = from; i < to; i++) {
      const c = b[i];
      if (c === 10 || c === 13) {
        if (this.text.length) {
          out.push({ type: "text", line: new TextDecoder().decode(new Uint8Array(this.text)) });
          this.text = [];
        }
      } else if (this.text.length < 512) {
        this.text.push(c);
      }
    }
  }

  drop() {
    if (this.img) this.stats.dropped++;
    this.img = null;
  }

  chunk(h, payload) {
    if (h.idx === 0) {
      this.drop();
      const dv = new DataView(payload.buffer, payload.byteOffset, MetaSize);
      const meta = {
        width: dv.getUint16(0, true), height: dv.getUint16(2, true), format: dv.getUint8(4),
        version: dv.getUint8(5), stride: dv.getUint16(6, true), dataLen: dv.getUint32(8, true), sensor: dv.getUint8(12),
      };
      // Same bounds as Go's Meta.Validate; allocate at most 1 MiB up front
      // and grow as data arrives, as Go's decoder does (a crafted header
      // must not reserve hundreds of MB).
      if (meta.version !== 1 || meta.width === 0 || meta.height === 0 || meta.dataLen === 0 ||
          meta.dataLen > h.cnt * MaxPayload - MetaSize) {
        this.stats.dropped++;
        return null;
      }
      this.img = { seq: h.seq, cnt: h.cnt, next: 0, ts: h.ts, meta, data: new Uint8Array(Math.min(meta.dataLen, 1 << 20)), off: 0 };
      payload = payload.subarray(MetaSize);
    } else if (!this.img || h.seq !== this.img.seq || h.cnt !== this.img.cnt || h.idx !== this.img.next) {
      this.drop();
      return null;
    }
    const img = this.img;
    if (img.off + payload.length > img.meta.dataLen) {
      this.drop();
      return null;
    }
    if (img.off + payload.length > img.data.length) {
      const grown = new Uint8Array(Math.min(img.meta.dataLen, Math.max(img.data.length * 2, img.off + payload.length)));
      grown.set(img.data.subarray(0, img.off));
      img.data = grown;
    }
    img.data.set(payload, img.off);
    img.off += payload.length;
    img.next = h.idx + 1;
    if (h.idx !== h.cnt - 1) return null;
    this.img = null;
    if (img.off !== img.meta.dataLen) {
      this.stats.dropped++;
      return null;
    }
    this.stats.images++;
    return { type: "image", seq: img.seq, ts: img.ts, meta: img.meta, data: img.data.subarray(0, img.off) };
  }
}

function validHeader(h) {
  if (h.plen === 0 || h.plen > MaxPayload || h.cnt === 0 || h.idx >= h.cnt) return false;
  const first = h.idx === 0, last = h.idx === h.cnt - 1;
  if (!!(h.flags & FlagFirst) !== first || !!(h.flags & FlagLast) !== last || !!(h.flags & FlagMeta) !== first) return false;
  return !first || h.plen >= MetaSize;
}

// parseStatus turns "webcam: status k=v k=v …" into an object, or null.
export function parseStatus(line) {
  const p = "webcam: status ";
  if (!line.startsWith(p)) return null;
  const st = {};
  for (const f of line.slice(p.length).trim().split(/\s+/)) {
    const k = f.indexOf("=");
    if (k > 0) st[f.slice(0, k)] = f.slice(k + 1);
  }
  return st;
}

// Settings shared with grab (tmp/badgecam/settings.json): size 0-4,
// quality 1-63, exposure "auto" or µs, gain "auto" or ×, wb "auto" or
// [r, g, b] gains (0-3.99), mirror, flip.
export const defaultSettings = { size: 0, quality: 30, exposure: "auto", gain: 1, wb: "auto", mirror: true, flip: false };

// White-balance presets: esp32-camera's OV3660 set_wb_mode register values
// (0x3400/02/04) as gains (register / 1024).
export const wbPresets = {
  sunny: [1.47, 1.02, 1.31],
  cloudy: [1.58, 1.02, 1.23],
  office: [1.28, 1.02, 1.59],
  home: [1.03, 0.98, 1.77],
};

// settingsCommands returns the webcam command lines that apply s.
export function settingsCommands(s) {
  return [
    "Z " + s.size,
    "Q " + s.quality,
    "E " + (s.exposure === "auto" ? "a" : Math.round(s.exposure)),
    "I " + (s.gain === "auto" ? "a" : s.gain),
    "W " + (s.wb === "auto" || s.wb === undefined ? "a" : s.wb.map((g) => g.toFixed(2)).join(" ")),
    "M " + (s.mirror ? 1 : 0) + " " + (s.flip ? 1 : 0),
  ];
}

// White balance from colour temperature and tint. The OV3660 gains that
// neutralise light of temperature T follow, to a good approximation, a
// log-linear line in mired (1e6 / T) through esp32-camera's presets
// (cloudy ≈ 6500 K, sunny ≈ 5500 K, office ≈ 4000 K, home ≈ 2800 K):
//   ln R = 0.781 − 0.00210 · mired,  ln B = −0.069 + 0.00179 · mired.
// Tint −1 (green) … +1 (magenta) scales the green gain by 1 − 0.25 · tint.
// The preset temperatures are nominal, so the K scale is approximate.
const WB_R0 = 0.781, WB_R1 = -0.0021, WB_B0 = -0.069, WB_B1 = 0.00179, WB_TINT = 0.25;
const clampGain = (g) => Math.min(3.99, Math.max(0, g));

export function wbFromTemp(kelvin, tint) {
  const mired = 1e6 / kelvin;
  return [
    clampGain(Math.exp(WB_R0 + WB_R1 * mired)),
    clampGain(1 - WB_TINT * tint),
    clampGain(Math.exp(WB_B0 + WB_B1 * mired)),
  ].map((g) => Math.round(g * 100) / 100);
}

// tempFromGains is the inverse: the temperature and tint closest to gains
// [r, g, b] (r and b normalised by g first), for starting the sliders from
// what auto white balance is using.
export function tempFromGains([r, g, b]) {
  // Gains of 0 are valid on the sensor; keep the logs finite (verify: G = 0 gave "NaN K").
  [r, g, b] = [r, g, b].map((x) => Math.max(0.01, +x || 0));
  const rn = r / g, bn = b / g;
  const m1 = (Math.log(rn) - WB_R0) / WB_R1, m2 = (Math.log(bn) - WB_B0) / WB_B1;
  const mired = Math.min(500, Math.max(100, (m1 + m2) / 2));
  const tint = Math.min(1, Math.max(-1, (1 - g) / WB_TINT));
  return { kelvin: Math.round(1e6 / mired / 50) * 50, tint: Math.round(tint * 100) / 100 };
}
