# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-04-21

### Added

- Python 3.14 classifier and support

### Changed

- Relax `torch` upper bound (`<2.7` → `<3`) so 3.14-compatible PyTorch resolves
- Relax `torchvision` upper bound (`<0.22` → `<1`) for the `[video]` extra
- Relax `torchao` upper bound (`<0.10` → `<1`) for the `[optimized]` extra
- Rename PyPI distribution references in README to `cognitive-scoring`

### Fixed

- Replace deprecated `asyncio.get_event_loop()` with `get_running_loop()` in HTTP server (4 call sites) — required for Python 3.14

## [1.0.0] - 2026-04-20

### Added

- `BrainAPI` — two-line Python interface for text → brain activity scoring
- `BrainResult` — structured output with region scores, composites, and classification
- FastAPI HTTP server with lazy model loading and auto-unload after idle timeout
- `BrainAtlas` — Destrieux atlas mapping with 10 functional region groups
- `BrainStateClassifier` — Pearson correlation-based emotional state classification
- Apple Silicon (MPS) compatibility patches for all feature extractors
- Persistent configuration manager (`~/Library/Application Support/TribeV2/`)
- macOS menu bar app with native settings GUI
- LaunchAgent manager for "Start at Login" support
- Feature caching by content hash for instant re-analysis
- Comprehensive wiki documentation (11 pages)
- PyPI release workflow via GitHub Actions with OIDC Trusted Publishers

### Notes

- Based on Meta's TRIBE v2 multimodal brain encoding model
- Requires Python 3.11+
- First run downloads ~10 GB of model weights (cached in `~/.cache/huggingface/`)
- Licensed under CC BY-NC 4.0 (non-commercial use only)

[Unreleased]: https://github.com/suncloudsmoon/cognitive-scoring/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/suncloudsmoon/cognitive-scoring/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/suncloudsmoon/cognitive-scoring/releases/tag/v1.0.0
