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
	@echo "  make clean            Remove build artifacts"
	@echo "  make help             Display this help message"

.DEFAULT_GOAL := help

