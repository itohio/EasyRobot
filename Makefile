# Define the paths to the source code and build artifacts
BUILD_PATH=./build

# Define the build flags for go build
BUILD_FLAGS=-ldflags="-s -w"

# Define the command for running tests
TEST_CMD=go test -race ./...

# Define CGO setting
CGO_ENABLED?=0

.PHONY: all
all: test

.PHONY: proto
proto:
	cd proto && buf generate

.PHONY: gen
gen:
	# Run go generate to generate any required files
	go generate ./...

.PHONY: mocks
mocks:
	mockery

.PHONY: vulncheck
vulncheck:
	@mkdir -p $(BUILD_PATH)
	GOBIN=$(BUILD_PATH) go install golang.org/x/vuln/cmd/govulncheck@latest
	$(BUILD_PATH)/govulncheck ./...

.PHONY: test
test:
	$(TEST_CMD)

.PHONY: cover
cover:
	go test -coverprofile=.coverage.tmp ./...
	cat .coverage.tmp | grep -Ev '/mock_|/.*options.go' > .coverage
	go tool cover -func=.coverage

.PHONY: watch
watch:
	air --build.cmd "go build -o $(BUILD_PATH)/app" --build.bin "$(BUILD_PATH)/app" --build.exclude_dir "proto,scripts,build"

.PHONY: docs
docs:
	go run ./docs/tools/vault -docs docs -repo . gen

.PHONY: docs-check
docs-check:
	go run ./docs/tools/vault -docs docs -repo . check

.PHONY: docs-stale
docs-stale:
	go run ./docs/tools/vault -docs docs -repo . stale

.PHONY: clean
clean:
	# Remove the build artifacts
	rm -rf $(BUILD_PATH)
	rm -f .coverage .coverage.tmp

.PHONY: install-tools
install-tools:
	go install github.com/vektra/mockery/v2@latest
	go install github.com/air-verse/air@latest
	go install golang.org/x/vuln/cmd/govulncheck@latest
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install github.com/bufbuild/buf/cmd/buf@latest

.PHONY: flash-manipulator
flash-manipulator:
	@echo "Will run: tinygo flash -target=xiao ./drivers/manipulator"
	@read -p "Flash connected XIAO? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		tinygo flash -target=xiao ./drivers/manipulator; \
	else \
		echo "Aborted."; \
	fi


# UnitV (M5Stack, Kendryte K210). Build from the repo root: the target's
# linker-script path is relative to the working directory (drivers/unitv/doc.go).
UNITV_PROG ?= hello
UNITV_PORT ?=
KTOOL ?= ktool.py

.PHONY: unitv
unitv:
	@mkdir -p build
	tinygo build -target=drivers/unitv/target/unitv.json -size=short -o build/unitv-$(UNITV_PROG).bin ./drivers/unitv/examples/$(UNITV_PROG)

# qi8 ESP32-S3 PIE backend checks: the layer suite on the host emulation of
# the PIE kernels (test-only tag qi8pieemu), then a TinyGo S3 compile gate
# (nnbench, PIE build; -print-allocs=qi8 must print nothing).
.PHONY: qi8-pie
qi8-pie:
	go test -race -tags qi8pieemu ./x/math/primitive/qi8/ ./x/math/nn/qnet/...
	@mkdir -p build
	tinygo build -target=cmd/nnbench/target/xiao-esp32s3-noradio.json -print-allocs=qi8 -o build/nnbench-s3-pie.bin ./cmd/nnbench

# The chip kernels' C requantisation (requant_c.h) on the host: built with
# cgo under UndefinedBehaviorSanitizer (test-only tag qi8ctest) and compared
# with the Go reference. Needs CGO_ENABLED=1 and a C compiler with UBSan
# (clang or gcc); it fails, rather than skipping, when the test cannot run.
.PHONY: qi8-ctest
qi8-ctest:
	@test "$$(go env CGO_ENABLED)" = 1 || { echo "qi8-ctest: cgo is off; the C requantisation test needs CGO_ENABLED=1 and a C compiler with UBSan"; exit 1; }
	@mkdir -p build
	go vet -tags qi8ctest ./x/math/primitive/qi8/
	go test -count=1 -tags qi8ctest -run TestCRequantizeMatchesGo -v ./x/math/primitive/qi8/ > build/qi8-ctest.out 2>&1 || { cat build/qi8-ctest.out; exit 1; }
	@grep -E '^(--- |ok|FAIL)|cases equal' build/qi8-ctest.out
	@grep -q "cases equal" build/qi8-ctest.out || { echo "qi8-ctest: TestCRequantizeMatchesGo did not run"; exit 1; }

.PHONY: flash-unitv
flash-unitv: unitv
	@test -n "$(UNITV_PORT)" || { echo "Set UNITV_PORT to the UnitV's serial port (macOS: /dev/cu.usbserial-…)."; exit 1; }
	@echo "Will run: $(KTOOL) -p $(UNITV_PORT) -B kd233 -b 115200 -a 0 build/unitv-$(UNITV_PROG).bin"
	@read -p "Flash connected UnitV? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		$(KTOOL) -p $(UNITV_PORT) -B kd233 -b 115200 -a 0 build/unitv-$(UNITV_PROG).bin; \
	else \
		echo "Aborted."; \
	fi

.PHONY: help
help:
	@echo "Makefile for EasyRobot project"
	@echo ""
	@echo "Usage:"
	@echo "  make install-tools    Install required development tools"
	@echo "  make proto            Generate code from proto files (requires buf)"
	@echo "  make gen              Run go generate to generate necessary files"
	@echo "  make mocks            Update interface mocks using mockery"
	@echo "  make test             Run all tests"
	@echo "  make cover            Run test coverage analysis"
	@echo "  make vulncheck        Check for known vulnerabilities"
	@echo "  make watch            Watch for changes and rebuild (requires air)"
	@echo "  make docs             Regenerate vault tables, counts and docs/todo.md"
	@echo "  make docs-check       Check the vault (links, frontmatter, TODO tags, generated blocks)"
	@echo "  make docs-stale       List notes whose cited source files changed since their revision"
	@echo "  make clean            Remove build artifacts"
	@echo "  make flash-manipulator Flash drivers/manipulator to XIAO via tinygo (asks y/N)"
	@echo "  make unitv [UNITV_PROG=hello]  Build a UnitV (K210) example with tinygo into build/"
	@echo "  make flash-unitv       Flash it with ktool (needs UNITV_PORT, KTOOL; asks y/N)"
	@echo "  make qi8-pie          Test the qi8 PIE backend on its host emulation and compile it for the S3 (tinygo)"
	@echo "  make qi8-ctest        Test the S3 kernels' C requantisation on the host against Go (cgo, UBSan)"
	@echo "  make help             Display this help message"

.DEFAULT_GOAL := help

