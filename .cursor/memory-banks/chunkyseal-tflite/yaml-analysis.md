# YAML Configuration Analysis - VideoSeal Model Variants

## Overview

This document analyzes the YAML configuration files for all VideoSeal model variants to confirm which modules (UNetMsg, ImageEncoderViT, ConvNeXt) are used in each model's embedder and detector/extractor.

## Model YAML Files

All model configurations are located in `videoseal/cards/`:
- `videoseal_0.0.yaml` - Legacy baseline (96-bit)
- `videoseal_1.0.yaml` - Stable release (256-bit)
- `pixelseal.yaml` - Adversarial-only training (256-bit)
- `chunkyseal.yaml` - High-capacity (1024-bit)

---

## VideoSeal v0.0 (Legacy)

**File**: `videoseal/cards/videoseal_0.0.yaml`

### Embedder Configuration
```yaml
embedder:
  model: vae
  params:
    # VAE-based embedder (not UNetMsg)
```

**Module Used**: `VAEEncoder` (from `videoseal/modules/vae.py`)  
**Architecture**: VAE-based  
**Notes**: Different from v1.0, uses VAE instead of UNet

### Extractor Configuration
```yaml
extractor:
  model: vit
  params:
    encoder:
      img_size: 256
      patch_size: 16
      embed_dim: 384
      depth: 12
      num_heads: 6
```

**Module Used**: `ImageEncoderViT` (from `videoseal/modules/vit.py`)  
**Architecture**: Vision Transformer with **attention mechanisms**  
**Key Features**:
- ✅ Has attention (12 transformer blocks)
- ✅ Uses `Attention` class with multi-head self-attention
- ✅ Patch embedding + positional encoding
- ❌ **NOT TFLite-friendly** (complex attention operations)

---

## VideoSeal v1.0 (Stable)

**File**: `videoseal/cards/videoseal_1.0.yaml`

### Embedder Configuration
```yaml
embedder:
  model: unet_msg
  params:
    in_channels: 1  # Y channel only
    out_channels: 1
    z_channels: 64
    num_blocks: 4
    z_channels_mults: [1, 2, 4, 4]
```

**Module Used**: `UNetMsg` (from `videoseal/modules/unet.py`)  
**Architecture**: UNet-based message embedder  
**Key Features**:
- Pure CNN (ResNet blocks + skip connections)
- Processes Y channel (YUV color space)
- Message processor for embedding watermark bits

### Extractor Configuration
```yaml
extractor:
  model: convnext_tiny
  params:
    encoder:
      depths: [3, 3, 9, 3]  # 18 blocks total
      dims: [96, 192, 384, 768]
    pixel_decoder:
      upscale_stages: [1]
```

**Module Used**: `ConvnextExtractor` (from `videoseal/models/extractor.py`)  
**Architecture**: ConvNeXt-Tiny (Pure CNN)  
**Key Features**:
- ❌ No attention mechanisms
- ✅ Pure CNN (depthwise conv + pointwise conv)
- ✅ **TFLite-friendly**
- 18 ConvNeXt blocks across 4 stages

---

## PixelSeal (Adversarial-Only)

**File**: `videoseal/cards/pixelseal.yaml`

### Embedder Configuration
```yaml
embedder:
  model: unet_msg
  params:
    in_channels: 1  # Y channel only
    out_channels: 1
    z_channels: 64
    num_blocks: 4
    z_channels_mults: [1, 2, 4, 4]
```

**Module Used**: `UNetMsg` (from `videoseal/modules/unet.py`)  
**Architecture**: Same as VideoSeal v1.0  
**Notes**: Identical embedder architecture to v1.0

### Extractor Configuration
```yaml
extractor:
  model: convnext_tiny
  params:
    encoder:
      depths: [3, 3, 9, 3]  # 18 blocks total
      dims: [96, 192, 384, 768]
    pixel_decoder:
      upscale_stages: [1]
```

**Module Used**: `ConvnextExtractor` (from `videoseal/models/extractor.py`)  
**Architecture**: ConvNeXt-Tiny (Pure CNN)  
**Notes**: Identical extractor architecture to v1.0, but trained with adversarial-only approach

---

## ChunkySeal (High-Capacity)

**File**: `videoseal/cards/chunkyseal.yaml`

### Embedder Configuration
```yaml
embedder:
  model: unet_msg
  params:
    in_channels: 3  # RGB (all channels)
    out_channels: 3
    z_channels: 64
    num_blocks: 4
    z_channels_mults: [1, 2, 4, 4]
```

**Module Used**: `UNetMsg` (from `videoseal/modules/unet.py`)  
**Architecture**: UNet-based message embedder  
**Key Difference**: Processes RGB (3 channels) instead of Y channel only

### Extractor Configuration
```yaml
extractor:
  model: convnext_chunky
  params:
    proportional_dim: true
    encoder:
      stem_stride: 2
      depths: [3, 3, 27, 3]  # 36 blocks total
      dims: [128, 256, 512, 1024]  # Base dims (scaled by sqrt(nbits/128))
    pixel_decoder:
      pixelwise: false
      upscale_stages: [1]
      sigmoid_output: false
```

**Module Used**: `ConvnextExtractor` (from `videoseal/models/extractor.py`)  
**Architecture**: ConvNeXt-Chunky (Pure CNN, scaled)  
**Key Features**:
- ❌ No attention mechanisms
- ✅ Pure CNN (same structure as ConvNeXt-Tiny)
- ✅ **TFLite-friendly**
- 36 ConvNeXt blocks (2× VideoSeal v1.0)
- Proportionally scaled dimensions for 1024-bit capacity

**Scaled Dimensions** (for 1024 bits):
```
Base:   [128,  256,  512,  1024]
Scaled: [362,  724,  1448, 2896]  # Multiplied by sqrt(1024/128) ≈ 2.83
```

---

## Summary Table

| Model | Embedder | Extractor | Attention? | TFLite-Friendly? | Capacity |
|-------|----------|-----------|------------|------------------|----------|
| **VideoSeal v0.0** | VAE | ImageEncoderViT | ✅ Yes | ❌ No | 96 bits |
| **VideoSeal v1.0** | UNetMsg | ConvNeXt-Tiny | ❌ No | ✅ Yes | 256 bits |
| **PixelSeal** | UNetMsg | ConvNeXt-Tiny | ❌ No | ✅ Yes | 256 bits |
| **ChunkySeal** | UNetMsg | ConvNeXt-Chunky | ❌ No | ✅ Yes | 1024 bits |

---

## Module Usage Confirmation

### UNetMsg (videoseal/modules/unet.py)

**Used by**:
- ✅ VideoSeal v1.0 (embedder)
- ✅ PixelSeal (embedder)
- ✅ ChunkySeal (embedder)

**NOT used by**:
- ❌ VideoSeal v0.0 (uses VAE instead)

**Key Characteristics**:
- Pure CNN architecture
- ResNet blocks with skip connections
- Message processor for embedding watermark bits
- Downsampling + bottleneck + upsampling

### ImageEncoderViT (videoseal/modules/vit.py)

**Used by**:
- ✅ VideoSeal v0.0 (extractor)

**NOT used by**:
- ❌ VideoSeal v1.0 (uses ConvNeXt-Tiny)
- ❌ PixelSeal (uses ConvNeXt-Tiny)
- ❌ ChunkySeal (uses ConvNeXt-Chunky)

**Key Characteristics**:
- Vision Transformer architecture
- **Has attention mechanisms** (`Attention` class)
- Multi-head self-attention
- Patch embedding + positional encoding
- 12 transformer blocks (depth=12)
- **NOT TFLite-friendly** due to attention operations

### ConvnextExtractor (videoseal/models/extractor.py)

**Used by**:
- ✅ VideoSeal v1.0 (ConvNeXt-Tiny, 18 blocks)
- ✅ PixelSeal (ConvNeXt-Tiny, 18 blocks)
- ✅ ChunkySeal (ConvNeXt-Chunky, 36 blocks)

**NOT used by**:
- ❌ VideoSeal v0.0 (uses ImageEncoderViT)

**Key Characteristics**:
- Pure CNN architecture (no attention)
- Depthwise convolution + pointwise convolution
- Layer normalization + GELU activation
- **TFLite-friendly** (easy quantization)
- Proportional scaling for different capacities

---

## Architecture Evolution

### VideoSeal v0.0 → v1.0 Transition

**Embedder Change**:
- **v0.0**: VAE-based
- **v1.0**: UNetMsg-based
- **Reason**: Better control over message embedding

**Extractor Change**:
- **v0.0**: ImageEncoderViT (with attention)
- **v1.0**: ConvNeXt-Tiny (pure CNN)
- **Reason**: Better TFLite compatibility, easier quantization

**Impact**:
- ✅ v1.0 can be converted to TFLite with INT8 quantization
- ❌ v0.0 is difficult to convert due to attention mechanisms

### VideoSeal v1.0 → ChunkySeal Scaling

**Embedder**:
- Same UNetMsg architecture
- Changed from Y channel (1 channel) to RGB (3 channels)

**Extractor**:
- Same ConvNeXt architecture
- Scaled from 18 blocks to 36 blocks
- Proportionally scaled dimensions (2.83× for 1024 bits)

**Impact**:
- ✅ Both are TFLite-friendly
- ✅ ChunkySeal maintains quantization compatibility
- ⚠️ ChunkySeal is 29× larger (960 MB vs 32.90 MB INT8)

---

## TFLite Conversion Implications

### Why ConvNeXt Works for TFLite

1. **No Attention Mechanisms**:
   - Pure CNN operations (conv, norm, activation)
   - No dynamic attention maps
   - No complex matrix operations

2. **Standard Operations**:
   - Depthwise convolution (supported)
   - Pointwise convolution (supported)
   - Layer normalization (supported)
   - GELU activation (supported)

3. **Quantization-Friendly**:
   - All operations have well-defined quantization schemes
   - Channelwise quantization works well
   - Minimal accuracy loss with INT8

### Why ImageEncoderViT is Difficult for TFLite

1. **Attention Mechanisms**:
   - Multi-head self-attention (complex)
   - Dynamic attention maps
   - Matrix multiplications with softmax

2. **Positional Encoding**:
   - Absolute positional embeddings
   - Relative positional embeddings (optional)
   - Complex indexing operations

3. **Quantization Challenges**:
   - Attention weights are sensitive to quantization
   - Softmax operations lose precision with INT8
   - Significant accuracy degradation

---

## Verification from Source Code

### Extractor Factory (videoseal/models/extractor.py)

```python
def build_extractor(model: str, **params):
    if model == "vit":
        return SegmentationExtractor(
            image_encoder=ImageEncoderViT(**params["encoder"]),
            # Uses ImageEncoderViT from vit.py
        )
    elif model in ["convnext_tiny", "convnext_chunky"]:
        return ConvnextExtractor(
            encoder_name=model,
            # Uses ConvNeXt (no attention)
        )
```

### Embedder Factory (videoseal/models/embedder.py)

```python
def build_embedder(model: str, **params):
    if model == "vae":
        return VAEEncoder(**params)
    elif model == "unet_msg":
        return UNetMsg(**params)
        # Uses UNetMsg from unet.py
```

---

## Conclusion

### Models with Attention (NOT TFLite-friendly)
- ❌ **VideoSeal v0.0**: Uses `ImageEncoderViT` with attention mechanisms

### Models without Attention (TFLite-friendly)
- ✅ **VideoSeal v1.0**: Uses `ConvNeXt-Tiny` (pure CNN)
- ✅ **PixelSeal**: Uses `ConvNeXt-Tiny` (pure CNN)
- ✅ **ChunkySeal**: Uses `ConvNeXt-Chunky` (pure CNN, scaled)

### Embedder Usage
- **UNetMsg** (pure CNN): VideoSeal v1.0, PixelSeal, ChunkySeal
- **VAE**: VideoSeal v0.0 only

### Key Takeaway

The transition from VideoSeal v0.0 to v1.0 was crucial for TFLite deployment:
- Replaced Vision Transformer (with attention) with ConvNeXt (pure CNN)
- Enabled successful TFLite conversion with INT8 quantization
- Maintained high accuracy while improving inference speed
- ChunkySeal inherited this TFLite-friendly architecture

---

**Last Updated**: January 4, 2025  
**Source Files**: `videoseal/cards/*.yaml`, `videoseal/modules/*.py`, `videoseal/models/*.py`

