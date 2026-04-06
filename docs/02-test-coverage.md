# 02 — Test Coverage

## Overview

Code coverage is measured on every CI run using [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov) and reported to [Codecov](https://codecov.io/gh/wavekat/wavekat-core).

## How It Works

The `coverage` job in `.github/workflows/ci.yml`:

1. Installs `cargo-llvm-cov` via [`taiki-e/install-action`](https://github.com/taiki-e/install-action).
2. Runs the full test suite with LLVM instrumentation across all features:
   ```sh
   cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
   ```
3. Uploads the LCOV report to Codecov.

## Viewing Coverage

- **Badge** — shown in `README.md`, reflects the latest `main` branch coverage.
- **Codecov dashboard** — `https://codecov.io/gh/wavekat/wavekat-core` for line-by-line detail and PR diff coverage.

## Setup (one-time)

1. Sign in to [codecov.io](https://codecov.io) with your GitHub account.
2. Enable the `wavekat/wavekat-core` repository.
3. Add the `CODECOV_TOKEN` secret to the GitHub repository settings.

## Running Locally

```sh
# Install dev tools once after cloning
make install-dev

# Generate an HTML report and open it
make cov
```
