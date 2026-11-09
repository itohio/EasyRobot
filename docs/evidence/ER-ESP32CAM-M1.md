---
task: ER-ESP32CAM-MOTION
step: M1 SD card alive (read-only)
owner: esp32cam
date: 2026-11-09
board: Seeed XIAO ESP32-S3 Sense, /dev/cu.usbmodem101
---

# ER-ESP32CAM M1 — SD card alive: identity, size, root listing, free space (read-only)

**Check (owner's):** list the root directory and print the card size and free space over serial. **Pass on the replacement card.** The first card is recorded as **faulty**.

| | Replacement card (owner swapped it in) |
|---|---|
| Identity (CID) | SanDisk (MID 0x03, OEM "SD"), product **SN64G**, rev 8.6, made 2022-12 (MDT 0x16C, decoded by hand; the driver's date field reads "12 6") |
| Card capacity (CSD v2) | **63,864,045,568 B** (63.86 GB, SDXC) |
| Partition table | MBR, one partition: type 0x07, start LBA 32,768, 124,702,720 sectors (63,847 MB) |
| File system | **exFAT**: 512 B sectors, 256 sectors per cluster (128 KiB), 486,992 clusters, volume serial **3161-6464** |
| Root directory | 1 entry: `System Volume Information` (directory) |
| Free space | **486,986 clusters = 63,830,228,992 B (63.83 GB)**, i.e. 6 clusters (768 KiB) used |
| Writes | none: the write/erase counter of the read-only block device stays at 0 |

To compare on the Mac: `diskutil info` should show about 63.9 GB, exFAT, volume UUID/serial `3161-6464`, and `System Volume Information` at the root. Not compared yet.

## 1. The first card: faulty

The first card ("SMISD", 6.13 GB) answered its ID commands, but returned the same 512 bytes for every block: a copy of its CID and CSD registers followed by zeros. There was no partition table or file system anywhere (first run of this report, logs `m1/sdprobe-run{1..5}.log`).

The replacement card works with the same wiring, driver, SPI speed and probe. So the fault was in the first card, not on the board or in the code. The owner skipped the Mac check of the old card.

## 2. exFAT needed a FatFs build option (local copy of tinyfs)

The card is exFAT, as SDXC cards are formatted out of the box. tinyfs v0.5.0 ships FatFs with `FF_FS_EXFAT 0`, so the first run on this card read the MBR fine but could not mount the volume.

Fix (scratch only): a copy of the cached `tinygo.org/x/tinyfs@v0.5.0` at `tmp/esp32cam/sd/tinyfs-exfat/`, wired in with `replace tinygo.org/x/tinyfs => ./tinyfs-exfat` in `tmp/esp32cam/sd/go.mod`. Only the build settings changed. `fatfs/ffconf.h` lines 241–242:

```
#define FF_FS_EXFAT     1
#define FF_INTDEF       2 /* QWORD comes from integer.h (unsigned long long); needed for exFAT */
```

Without the second line, `ff.h` stops with "exFAT feature wants C99 or later": the tinyfs copy of `ff.h` has commented out FatFs's C99 detection. `integer.h` already defines a 64-bit `QWORD`. The `examples/` and `littlefs/` directories of the copy were dropped, since they're unused. Nothing new was downloaded for this.

Probe size with exFAT: 177,189 B flash, 13,104 B RAM. SHA-256 `9f21868e…cdad8e26`.

## 3. Read-only guarantee (unchanged)

`tmp/esp32cam/sd/sdprobe/main.go`:
- FatFs gets the card only through a wrapper whose `WriteAt` and `EraseBlocks` return an error and count the attempt. The count stayed at **0**: mount, root listing and free-space count did not try to write.
- The probe's own card access: init (CMD0/8/41/58/16), CMD9/CMD10 (read CSD/CID), CMD17 (read one block). It never sends a write or erase command. `Format`, `Mkdir`, `Remove` and writable `OpenFile` are never called.

## 4. Output (log `tmp/esp32cam/m1/sdprobe-newcard-exfat.log`)

```
sd: raw CID 03 53 44 53 4e 36 34 47 86 5b 58 dd a1 01 6c 43
sd: raw CSD 40 0e 00 32 db 79 00 01 db d3 7f 80 0a 40 40 f9
sd: card type SDHC/SDXC (CSD v2) capacity bytes 63864045568 = 63864 MB
sd: CID manufacturer 3 OEM 851 product SN64G rev 8.6 date 12 6
sd: sector 0 signature 0xaa55
sd: partition 1 type 0x07 start LBA 32768 sectors 124702720 = 63847 MB
sd: exFAT volume length sectors 124702720 = 63847 MB, clusters 486992 serial 828466276
sd: volume at LBA 32768 OEM EXFAT fs exFAT bytes/sector 512 sectors/cluster 256
sd: root directory, 1 entries:
sd:   dir  0 System Volume Information
sd: free clusters 486986 free bytes 63830228992 = 63830 MB
sd: write/erase attempts refused: 0
```

The probe repeats every 15 s with identical output (two cycles logged). A raw read of sector 0 via single-byte SPI transfers matches the driver's read.

Pins, as in the first run: SCK GPIO7, MISO GPIO8, MOSI GPIO9, CS GPIO21 (shared with the user LED, which is never touched). Driver `tinygo.org/x/drivers` v0.36.0 `sdcard` on `machine.SPI1`, at 4 MHz after init.

## 5. Decisions for the owner / Leader

1. **exFAT in repo code (from M2 on, for writing under `/MOTION/`).** FatFs supports exFAT read and write; only the build option is off in tinyfs v0.5.0. Options:
   - (a) keep a two-line patched copy of tinyfs in the repo (`third_party/` + `replace`), noting the upstream version;
   - (b) propose upstream that tinyfs make exFAT an option, and use (a) until then;
   - (c) the owner reformats the card to FAT32 on the Mac, which I may not do.

   Recommendation: **(a) now, (b) later.** A 64 GB card is normal for SDXC, and exFAT avoids FAT32's 4 GB file limit.
2. **Card swap note:** the probe initialises the card only at boot. After any swap, the board needs a reset; `mon.py --reset` did that here.

## M1b FAT32

**Decision (owner, via Leader, 2026-11-09):** "I'll format fat32". The owner reformats the card himself; I drop the exFAT patch and the vendored-tinyfs plan. Done: the `replace` line was removed from `tmp/esp32cam/sd/go.mod` and the local copy `tinyfs-exfat/` deleted, so the probe builds against **stock tinyfs v0.5.0**. §2 and decision 1 above are superseded.

The probe also prints the FAT32 boot-sector identity: volume serial (BS_VolID), label (BS_VolLab), total sectors, reserved sectors, FAT count and sectors per FAT. Build: 173,313 B flash; SHA-256 `addfba0e…ce1258b1`.

**Run 1, right after the task (board reset):** the card is **still the exFAT volume**, serial 828466276 (`0x31616464`), with the same partition as in §0. Stock tinyfs then reports `mount: fatfs: (13) There is no valid FAT volume`, as expected without exFAT support. So the reformatted card isn't in the board yet. Log: `tmp/esp32cam/m1/sdprobe-fat32.log`.

```
sd: card type SDHC/SDXC (CSD v2) capacity bytes 63864045568 = 63864 MB
sd: partition 1 type 0x07 start LBA 32768 sectors 124702720 = 63847 MB
sd: exFAT volume length sectors 124702720 = 63847 MB, clusters 486992 serial 828466276
sd: volume at LBA 32768 OEM EXFAT fs exFAT bytes/sector 512 sectors/cluster 256
sd: mount: fatfs: (13) There is no valid FAT volume
```

**Run 2, on the card the owner put in (board reset, read-only): pass.** Log `tmp/esp32cam/m1/sdprobe-fat32-run2.log`, finished 18:09:50.

**Run 3, at Leader's request (board reset 18:14:55, finished 18:15:30; log `sdprobe-fat32-run3.log`): identical to run 2.** Same CID and CSD, FAT32 partition, label, root entries, and free clusters (242,515). Write/erase attempts 0.

| | Third card |
|---|---|
| Identity (CID) | MID 0x6F, OID 0x0013, product "SMI" (a generic Silicon Motion card), made 2019-07 (MDT 0x137, decoded by hand) |
| Card capacity (CSD v2) | **8,052,539,392 B** (8.05 GB, SDHC) |
| Partition table | MBR, one partition: type **0x0B (FAT32)**, start LBA 8,192, 15,720,448 sectors (8,048 MB) |
| File system | **FAT32**: 512 B sectors, 64 sectors per cluster (32 KiB), 4,354 reserved sectors, 2 FATs × 1,919 sectors, 245,504 clusters |
| Volume label / serial | label **`H4N_SD`**; serial **0000-0000** (BS_VolID is zero; the formatting tool didn't set one); boot-sector OEM name blank |
| Root directory | 4 directories: `STEREO`, `4CH`, `MTR`, `System Volume Information` |
| Free space | **242,515 clusters = 7,946,731,520 B (7.95 GB)**; 2,989 clusters (≈ 98 MB) in use |
| Writes | none: write/erase attempts 0 |

```
sd: raw CID 6f 00 13 53 4d 49 20 20 00 00 00 00 00 01 37 25
sd: card type SDHC/SDXC (CSD v2) capacity bytes 8052539392 = 8052 MB
sd: partition 1 type 0x0b start LBA 8192 sectors 15720448 = 8048 MB
sd: FAT32 volume serial 0x0000-0000 label H4N_SD total sectors 15720448 reserved 4354 FATs 2 sectors/FAT 1919
sd: volume at LBA 8192 OEM  fs FAT32 bytes/sector 512 sectors/cluster 64
sd: root directory, 4 entries:
sd:   dir  0 STEREO
sd:   dir  0 4CH
sd:   dir  0 MTR
sd:   dir  0 System Volume Information
sd: free clusters 242515 free bytes 7946731520 = 7946 MB
sd: write/erase attempts refused: 0
```

- **Stock tinyfs v0.5.0 mounts it with no patch**, so the FAT32 path needed by M2 works as shipped.
- **This card holds existing content:**
  - the label and folders (`STEREO`, `4CH`, `MTR`) are a Zoom H4n recorder's layout `[INFERENCE]`, with about 98 MB in use;
  - the folders were not opened; directory sizes show as 0 because FAT doesn't store them;
  - M2 will only add `/MOTION/` at the root and never touch these folders. Whether adding a folder to a recorder's card is fine is the owner's call (decision below).

**M1b decision for the owner:** M2 creates `/MOTION/` on this H4n card, with nothing outside it touched. Is that OK, or should M2 wait for a blank card? Recommendation: OK. FAT allows any extra root folder, and the H4n should ignore unknown folders `[INFERENCE]`.

## 6. Boundaries and slips

- **No writes** to either card (§3). No format. Existing files untouched.
- **Shell slip** (reported to Leader at the commit step): I ran `set -e` in the persistent tool shell while making the commits (040f47b…b4fcba5), which README rule 4 forbids. I reset it with `set +e` right after; later commands weren't affected.
- **Board** left running the read-only exFAT `sdprobe`.

## 7. Network contact (disclosed)

- First run:
  - `proxy.golang.org` tinyfs version list and `@latest`;
  - `go get` of `tinygo.org/x/tinyfs` v0.5.0 (`h1:1nLSvtvvoNcK3Ii+Dib9MVe4y/Fnp2iMgzbKgADGfbs=`) and its indirect dependency `github.com/google/shlex` v0.0.0-20191202100458-e7afc7fbc510 (`h1:El6M4kTTCOh6aBiKaUGG7oYTSPP8MxqL4YI3kZKwcP4=`);
  - two read-only doc fetches for the pins;
  - one web search.
- This rerun: none.
