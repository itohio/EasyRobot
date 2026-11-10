//go:build esp32s3

package psram

// ROM function addresses (ESP32-S3 ROM, ESP-IDF v5.3.1
// components/esp_rom/esp32s3/ld/esp32s3.rom.ld). TinyGo's esp32s3.ld does
// not define them, so this cgo preamble defines them as absolute symbols;
// the body-less //export functions in psram_esp32s3.go resolve to them.
// That way the package builds with the stock xiao-esp32s3 target.

/*
__asm__(
	".global esp_rom_opiflash_exec_cmd\n.set esp_rom_opiflash_exec_cmd, 0x400008b8\n"
	".global esp_rom_opiflash_pin_config\n.set esp_rom_opiflash_pin_config, 0x40000894\n"
	".global esp_rom_spi_set_dtr_swap_mode\n.set esp_rom_spi_set_dtr_swap_mode, 0x4000093c\n"
	".global rom_Cache_Suspend_DCache\n.set rom_Cache_Suspend_DCache, 0x400018b4\n"
	".global Cache_Resume_DCache\n.set Cache_Resume_DCache, 0x400018c0\n"
	".global Cache_Invalidate_Addr\n.set Cache_Invalidate_Addr, 0x400016b0\n"
	".global rom_Cache_WriteBack_Addr\n.set rom_Cache_WriteBack_Addr, 0x400016c8\n"
	".global Cache_Suspend_DCache_Autoload\n.set Cache_Suspend_DCache_Autoload, 0x40001734\n"
	".global Cache_Resume_DCache_Autoload\n.set Cache_Resume_DCache_Autoload, 0x40001740\n"
);
*/
import "C"
