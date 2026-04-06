.PHONY: help check test fmt lint doc ci cov cov-report install-dev

help:
	@echo "Available targets:"
	@echo "  check        Check workspace compiles"
	@echo "  test         Run all tests"
	@echo "  fmt          Format code"
	@echo "  lint         Run clippy with warnings as errors"
	@echo "  doc          Build and open docs in browser"
	@echo "  cov          Generate HTML coverage report and open it"
	@echo "  cov-report   Print coverage summary as text (paste into Claude Code)"
	@echo "  ci           Run all CI checks locally (fmt, clippy, test, doc)"
	@echo "  install-dev  Install local dev tools (cargo-llvm-cov)"

# Check workspace compiles
check:
	cargo check --workspace

# Run all tests
test:
	cargo test --workspace

# Format code
fmt:
	cargo fmt --all

# Lint
lint:
	cargo clippy --workspace -- -D warnings

# Build and open docs in browser
doc:
	cargo doc --no-deps -p wavekat-core --all-features --open

# Install local dev tools (run once after cloning)
install-dev:
	rustup component add llvm-tools-preview
	cargo install cargo-llvm-cov

# Generate HTML coverage report and open it in the browser
cov:
	cargo llvm-cov --all-features --open

# Print uncovered lines only (paste into Claude Code to write tests)
cov-report:
	@cargo llvm-cov --all-features --text 2>/dev/null | grep -E '\|[[:space:]]+0\|' || echo "No uncovered lines."

# Run all CI checks locally (mirrors .github/workflows/ci.yml)
ci:
	cargo fmt --all -- --check
	cargo clippy --workspace -- -D warnings
	cargo test --workspace
	cargo doc --no-deps -p wavekat-core --all-features
