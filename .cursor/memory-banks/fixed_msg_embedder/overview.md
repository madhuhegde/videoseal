# Fixed-Size Message Processor for VideoSeal Embedder

## Overview

This memory bank documents the implementation of a **TFLite-compatible fixed-size message processor** for the VideoSeal 1.0 UNet embedder, enabling full on-device watermark embedding without server dependency.

## Problem Statement

The original VideoSeal embedder failed TFLite conversion due to **dynamic tensor operations** in the message processor:

```python
# Original (FAILED)
indices = torch.arange(msg.shape[-1])  # ❌ Runtime-dependent size
msg_aux = msg_aux.repeat(1, 1, H, W)   # ❌ Runtime-dependent dimensions
```

**Error**: `RuntimeError: Sizes of tensors must match except in dimension 1`

## Solution

Implemented a **TFLite-friendly message processor** using:
1. **Pre-computed indices** as buffers (no `torch.arange` at runtime)
2. **Explicit concatenation** instead of tile/expand/repeat for spatial broadcasting
3. **Hardcoded spatial dimensions** (32×32 for 256px images)

**Breakthrough** (Jan 4, 2026): Explicit concatenation solves BROADCAST_TO runtime error!

## Key Achievements

✅ **Successful TFLite conversion** of VideoSeal UNet embedder  
✅ **Model loads and runs** - no BROADCAST_TO errors  
✅ **PSNR issue resolved** - fixed attenuation (43.29 dB quality)  
✅ **Excellent detection** - 97.7% accuracy  
✅ **No retraining required** - uses original model weights  
✅ **Production ready** for mobile deployment

## Architecture Confirmation

**VideoSeal 1.0 uses PURE CNN architectures**:
- **Embedder**: UNetMsg (no attention mechanisms)
- **Detector**: ConvNeXt-Tiny (no attention mechanisms)

This is why TFLite conversion succeeded with excellent results.

## Model Specifications

| Component | Value |
|-----------|-------|
| **Embedder Type** | UNetMsg (Pure CNN) |
| **Message Capacity** | 256 bits |
| **TFLite Model Size** | 90.42 MB (FLOAT32) |
| **Input** | Image [1,3,256,256] + Message [1,256] |
| **Output** | Watermarked Image [1,3,256,256] |
| **Conversion Time** | ~2 minutes |
| **Operations** | 56.592 G ops (28.296 G MACs) |
| **Quality (PSNR)** | 43.29 dB (vs PyTorch) |
| **Detection Accuracy** | 97.7% |
| **Attenuation** | Fixed (0.11) |

## Attenuation Solution

✅ **Fixed attenuation applied** - constant factor (0.11) replaces dynamic JND module

**Original Issue**: JND module uses boolean indexing (not TFLite-compatible)  
**Solution**: Apply fixed attenuation factor based on average heatmap value  
**Result**: PSNR 43.29 dB, 97.7% detection accuracy  
**Status**: Production-ready quality

See [psnr-fix.md](./psnr-fix.md) for complete details.

## Documentation Structure

```
.cursor/memory-banks/fixed_msg_embedder/
├── overview.md                # This file - high-level summary
├── problem-analysis.md        # Detailed error analysis
├── tensor-dimensions.md       # Dimension analysis across models
├── solution-design.md         # Implementation design
├── implementation.md          # Step-by-step implementation
├── usage.md                   # Usage examples
├── troubleshooting.md         # Common issues and solutions
├── broadcast-to-solution.md   # ✨ BROADCAST_TO fix (explicit concatenation)
├── psnr-fix.md                # ✨ PSNR fix (fixed attenuation)
├── int8-limitation.md         # INT8 quantization limitation
├── workarounds.md             # Practical workarounds for INT8
└── embedder-conversion.md     # VideoSeal vs ChunkySeal comparison
```

## Quick Links

- **✨ BROADCAST_TO Solution**: [broadcast-to-solution.md](./broadcast-to-solution.md) - Conversion fix
- **✨ PSNR Fix**: [psnr-fix.md](./psnr-fix.md) - Quality fix **START HERE!**
- **Problem Analysis**: [problem-analysis.md](./problem-analysis.md)
- **Solution Design**: [solution-design.md](./solution-design.md)
- **Implementation Guide**: [implementation.md](./implementation.md)
- **Usage Examples**: [usage.md](./usage.md)
- **VideoSeal vs ChunkySeal**: [embedder-conversion.md](./embedder-conversion.md)
- **Architecture Verification**: `../../../videoseal_clone/TFLITE_ARCHITECTURE_VERIFICATION.md`

## Status

**Status**: ✅ **PRODUCTION READY**  
**Date**: January 4, 2026  
**Version**: VideoSeal 1.0

---

*See individual documentation files for detailed information.*

