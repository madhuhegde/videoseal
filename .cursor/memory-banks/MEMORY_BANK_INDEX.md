# Memory Banks Index

## Overview

This directory contains organized documentation memory banks for complex features and implementations in the VideoSeal project.

## Memory Banks

### 1. VideoSeal TFLite

**Location**: `./videoseal-tflite/`  
**Created**: January 4, 2025  
**Status**: ✅ Complete  
**Size**: 2 files, 500+ lines

**Description**: Documentation for the VideoSeal v1.0 TFLite implementation - a standard-capacity (256-bit) watermark detector optimized for mobile and edge devices.

**Contents**:
- [README.md](./videoseal-tflite/README.md) - Memory bank overview and reference
- [overview.md](./videoseal-tflite/overview.md) - Introduction and key features

**Key Topics**:
- ConvNeXt-Tiny architecture (no attention)
- TFLite conversion using AI Edge Torch
- INT8 quantization (74.2% size reduction)
- Complete Python API
- 256-bit watermark capacity

**Quick Start**:
```python
from videoseal.tflite import load_detector
detector = load_detector(model_name='videoseal', quantization='int8')
result = detector.detect("watermarked.jpg")
```

**Note**: This memory bank contains all documentation for VideoSeal TFLite. The implementation code is in the `tflite/` directory.

---

### 2. ChunkySeal TFLite

**Location**: `./chunkyseal-tflite/`  
**Created**: January 3-4, 2025  
**Status**: ✅ Complete  
**Size**: 8 files, 3,341 lines

**Description**: Comprehensive documentation for the ChunkySeal TFLite implementation - a high-capacity (1024-bit) watermark detector optimized for mobile and edge devices.

**Contents**:
- [README.md](./chunkyseal-tflite/README.md) - Memory bank overview (212 lines)
- [overview.md](./chunkyseal-tflite/overview.md) - Introduction and key features (212 lines)
- [architecture.md](./chunkyseal-tflite/architecture.md) - Detailed architecture analysis (411 lines)
- [implementation.md](./chunkyseal-tflite/implementation.md) - Implementation details and API (566 lines)
- [usage.md](./chunkyseal-tflite/usage.md) - Usage guide with examples (548 lines)
- [conversion.md](./chunkyseal-tflite/conversion.md) - TFLite conversion process (431 lines)
- [quantization.md](./chunkyseal-tflite/quantization.md) - Quantization options (428 lines)
- [troubleshooting.md](./chunkyseal-tflite/troubleshooting.md) - Common issues (533 lines)

**Key Topics**:
- ConvNeXt-based architecture (no attention)
- TFLite conversion using AI Edge Torch
- INT8/FP16 quantization (67.5% size reduction)
- Complete Python API
- 1024-bit watermark capacity

**Quick Start**:
```python
from videoseal.chunky_tflite import load_detector
detector = load_detector(quantization='int8')
result = detector.detect("watermarked.jpg")
```

## Memory Bank Structure

Following the documentation rule in `.cursor/rules/documentation.mdc`:

```
.cursor/memory-banks/
├── MEMORY_BANK_INDEX.md (this file)
├── videoseal-tflite/
│   ├── README.md              # Memory bank overview (references tflite/ docs)
│   └── overview.md            # High-level introduction
└── chunkyseal-tflite/
    ├── README.md              # Memory bank overview
    ├── overview.md            # High-level introduction
    ├── architecture.md        # Architecture details
    ├── implementation.md      # Implementation guide
    ├── usage.md               # Usage examples
    ├── conversion.md          # Conversion process
    ├── quantization.md        # Quantization guide
    └── troubleshooting.md     # Troubleshooting guide
```

## Documentation Standards

All memory banks follow these standards:

### File Organization
✅ Dedicated folder per feature/concept  
✅ Markdown (.md) format for all docs  
✅ Clear, descriptive file names  
✅ README.md as entry point  

### Content Standards
✅ Clear titles and table of contents  
✅ Code examples with syntax highlighting  
✅ Step-by-step instructions  
✅ Cross-references to related docs  
✅ Practical examples and use cases  

### Naming Conventions
✅ Folders: kebab-case or snake_case  
✅ Files: kebab-case with .md extension  
✅ Descriptive and specific names  

## Related Documentation

### Project Documentation

**ChunkySeal TFLite Package** (`chunky_tflite/`):
- [detector.py](../../chunky_tflite/detector.py) - ChunkySeal detector implementation
- [example.py](../../chunky_tflite/example.py) - Usage examples
- [compare_pytorch_tflite.py](../../chunky_tflite/compare_pytorch_tflite.py) - Benchmarking
- [test_int8.py](../../chunky_tflite/test_int8.py) - INT8 testing

**VideoSeal TFLite Package** (`tflite/`):
- [detector.py](../../tflite/detector.py) - VideoSeal detector implementation
- [example.py](../../tflite/example.py) - Usage examples
- [compare_pytorch_tflite.py](../../tflite/compare_pytorch_tflite.py) - Benchmarking

**Main Project Documentation**:
- [README.md](../../README.md) - Main project overview and model information
- [IMAGE_WATERMARK_USAGE.md](../../IMAGE_WATERMARK_USAGE.md) - PyTorch CLI for embedding + detection (complementary to TFLite)
- [image_watermark.py](../../image_watermark.py) - PyTorch CLI implementation
- [docs/torchscript.md](../../docs/torchscript.md) - TorchScript model usage
- [docs/training.md](../../docs/training.md) - Training guide
- [docs/baselines.md](../../docs/baselines.md) - Baseline comparisons
- [docs/vmaf.md](../../docs/vmaf.md) - Video quality metrics

**Model Cards** (`videoseal/cards/`):
- [chunkyseal.yaml](../../videoseal/cards/chunkyseal.yaml) - ChunkySeal configuration
- [videoseal_1.0.yaml](../../videoseal/cards/videoseal_1.0.yaml) - VideoSeal v1.0 configuration
- [videoseal_0.0.yaml](../../videoseal/cards/videoseal_0.0.yaml) - VideoSeal v0.0 configuration
- [pixelseal.yaml](../../videoseal/cards/pixelseal.yaml) - PixelSeal configuration

## Usage Guidelines

### When to Create a Memory Bank

Create a new memory bank when:
- ✅ Implementing a major feature
- ✅ Adding a complex subsystem
- ✅ Documenting architecture decisions
- ✅ Creating integration guides
- ✅ Explaining intricate workflows
- ✅ Providing learning resources

### When NOT to Create a Memory Bank

Don't create a memory bank for:
- ❌ Simple functions or utilities
- ❌ Temporary code or experiments
- ❌ Standard library usage
- ❌ One-line fixes or tweaks

### Creating a New Memory Bank

```bash
# 1. Create folder
mkdir -p .cursor/memory-banks/feature-name/

# 2. Create documentation files
touch .cursor/memory-banks/feature-name/README.md
touch .cursor/memory-banks/feature-name/overview.md
touch .cursor/memory-banks/feature-name/implementation.md
touch .cursor/memory-banks/feature-name/usage.md

# 3. Update this index
# Add entry to MEMORY_BANK_INDEX.md
```

## Statistics

### Current Memory Banks
- **Total Banks**: 2
- **Total Files**: 10
- **Total Lines**: 3,800+
- **Total Size**: ~120 KB

### VideoSeal TFLite Memory Bank
- **Files**: 2
- **Lines**: 500+
- **Coverage**: Reference to tflite/ docs
- **Status**: Production-ready

### ChunkySeal TFLite Memory Bank
- **Files**: 8
- **Lines**: 3,341
- **Coverage**: Complete (100%)
- **Status**: Production-ready

## Quick Reference

### VideoSeal TFLite (256-bit)

| Topic | File | Lines |
|-------|------|-------|
| Overview | [README.md](./videoseal-tflite/README.md) | 300+ |
| Getting Started | [overview.md](./videoseal-tflite/overview.md) | 200+ |
| Implementation | [tflite/detector.py](../../tflite/detector.py) | 374 |
| Examples | [tflite/example.py](../../tflite/example.py) | 92 |

### ChunkySeal TFLite (1024-bit)

| Topic | File | Lines |
|-------|------|-------|
| Getting Started | [overview.md](./chunkyseal-tflite/overview.md) | 212 |
| Architecture | [architecture.md](./chunkyseal-tflite/architecture.md) | 411 |
| Implementation | [implementation.md](./chunkyseal-tflite/implementation.md) | 566 |
| Usage Examples | [usage.md](./chunkyseal-tflite/usage.md) | 548 |
| Conversion | [conversion.md](./chunkyseal-tflite/conversion.md) | 431 |
| Quantization | [quantization.md](./chunkyseal-tflite/quantization.md) | 428 |
| Troubleshooting | [troubleshooting.md](./chunkyseal-tflite/troubleshooting.md) | 533 |

## Contributing

When adding documentation:

1. **Follow the structure**:
   - Create dedicated folder
   - Use markdown format
   - Include README.md

2. **Write quality content**:
   - Clear explanations
   - Code examples
   - Cross-references
   - Practical use cases

3. **Update this index**:
   - Add new memory bank entry
   - Update statistics
   - Link to key files

4. **Maintain consistency**:
   - Follow naming conventions
   - Use standard file names
   - Keep structure organized

## Version History

- **January 4, 2025**: Created memory bank index
  - Added ChunkySeal TFLite memory bank (8 files, 3,341 lines)
  - Added VideoSeal TFLite memory bank (2 files, 500+ lines)
  - Total: 2 memory banks, 10 files, 3,800+ lines

## See Also

- [Documentation Rules](./../rules/documentation.mdc)
- [VideoSeal TFLite Package](../../tflite/)
- [ChunkySeal TFLite Package](../../chunky_tflite/)

---

**Last Updated**: January 4, 2025  
**Total Memory Banks**: 2  
**Total Documentation**: 3,800+ lines

