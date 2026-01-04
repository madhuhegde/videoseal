# ChunkySeal TFLite Architecture

## Table of Contents
- [Overview](#overview)
- [Architecture Components](#architecture-components)
- [ConvNeXt-Chunky Encoder](#convnext-chunky-encoder)
- [Pixel Decoder](#pixel-decoder)
- [No Attention Mechanisms](#no-attention-mechanisms)
- [Comparison with VideoSeal](#comparison-with-videoseal)
- [Proportional Scaling](#proportional-scaling)
- [Model Flow](#model-flow)

## Overview

ChunkySeal uses a **pure CNN architecture** based on ConvNeXt V2, with no attention mechanisms or transformers. This design choice enables efficient TFLite conversion and excellent quantization support.

## Architecture Components

### High-Level Architecture

```
Input (RGB Image)
    ↓
ConvNeXt-Chunky Encoder
    ├── Stem (Conv2d, stride=2)
    ├── Stage 1: 3 blocks
    ├── Stage 2: 3 blocks
    ├── Stage 3: 27 blocks  ← Most computation here
    └── Stage 4: 3 blocks
    ↓
Pixel Decoder
    ├── Upsampling
    └── Convolutions
    ↓
Output (1025 channels)
    ├── Channel 0: Detection mask
    └── Channels 1-1024: Message bits
```

### Model Configuration

From `chunkyseal.yaml`:
```yaml
extractor:
  model: convnext_chunky
  params:
    proportional_dim: true
    encoder:
      stem_stride: 2
      depths: [3, 3, 27, 3]
      dims: [128, 256, 512, 1024]
    pixel_decoder:
      pixelwise: false
      upscale_stages: [1]
      sigmoid_output: false
```

## ConvNeXt-Chunky Encoder

### Stage Configuration

| Stage | Blocks | Base Dims | Scaled Dims (1024-bit) | Resolution |
|-------|--------|-----------|------------------------|------------|
| Stem | 1 | - | 362 | 128×128 |
| Stage 1 | 3 | 128 | 362 | 128×128 |
| Stage 2 | 3 | 256 | 724 | 64×64 |
| Stage 3 | 27 | 512 | 1448 | 32×32 |
| Stage 4 | 3 | 1024 | 2896 | 16×16 |

**Total**: 36 ConvNeXt blocks

### ConvNeXt Block Structure

Each ConvNeXt block consists of:

```python
class Block(nn.Module):
    def __init__(self, dim):
        # 1. Depthwise Convolution (7×7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # 2. Layer Normalization
        self.norm = LayerNorm(dim)
        
        # 3. Pointwise Expansion (1×1, 4× expansion)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        
        # 4. GELU Activation
        self.act = nn.GELU()
        
        # 5. Global Response Normalization
        self.grn = GRN(4 * dim)
        
        # 6. Pointwise Projection (1×1)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # 7. Residual Connection
        self.drop_path = DropPath()
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)  # Residual
        return x
```

**Key Point**: No attention mechanisms - only convolutions!

### Downsampling

Between stages, resolution is reduced by 2×:
```python
downsample = nn.Sequential(
    LayerNorm(dims[i]),
    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
)
```

## Pixel Decoder

### Purpose
Upsamples the encoder features to predict:
1. Detection mask (1 channel)
2. Message bits (1024 channels)

### Configuration
```yaml
pixel_decoder:
  pixelwise: false        # Global prediction, not per-pixel
  upscale_stages: [1]     # Upsample once
  sigmoid_output: false   # Raw logits output
  embed_dim: 2896         # Matches encoder output
  nbits: 1024             # Message length
```

### Architecture
```python
class PixelDecoder(nn.Module):
    def __init__(self, embed_dim=2896, nbits=1024):
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 3, padding=1),
        )
        
        # Output head
        self.head = nn.Conv2d(embed_dim // 4, 1 + nbits, 1)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.head(x)
        # x shape: (B, 1025, H, W)
        return x
```

## No Attention Mechanisms

### Why No Attention?

ChunkySeal deliberately avoids attention mechanisms:

1. **TFLite Compatibility**: Attention operations are complex to convert
2. **Quantization**: CNNs quantize better than transformers
3. **Efficiency**: ConvNeXt matches ViT performance without attention
4. **Simplicity**: No positional embeddings or attention masks needed

### What ChunkySeal Does NOT Have

❌ Multi-head self-attention  
❌ Query-Key-Value projections  
❌ Attention weights  
❌ Positional embeddings  
❌ Transformer blocks  

### What ChunkySeal DOES Have

✅ Depthwise convolutions  
✅ Pointwise convolutions  
✅ Layer normalization  
✅ GELU activation  
✅ Global Response Normalization (GRN)  
✅ Residual connections  

## Comparison with VideoSeal

### VideoSeal v0.0 (Vision Transformer)

```
Input → Patch Embedding → Positional Encoding
    ↓
12 × Transformer Block:
    - Multi-Head Self-Attention  ← ATTENTION
    - MLP (Feed-forward)
    ↓
Pixel Decoder → Output (257 channels)
```

**Has Attention**: ✅ Yes (12 transformer blocks)

### VideoSeal v1.0 (ConvNeXt-Tiny)

```
Input → ConvNeXt Encoder (18 blocks)
    ↓
Pixel Decoder → Output (257 channels)
```

**Has Attention**: ❌ No (pure CNN)

### ChunkySeal (ConvNeXt-Chunky)

```
Input → ConvNeXt-Chunky Encoder (36 blocks, scaled)
    ↓
Pixel Decoder → Output (1025 channels)
```

**Has Attention**: ❌ No (pure CNN, scaled)

### Architecture Comparison Table

| Feature | VideoSeal v0.0 | VideoSeal v1.0 | ChunkySeal |
|---------|---------------|----------------|------------|
| **Encoder** | ViT | ConvNeXt-Tiny | ConvNeXt-Chunky |
| **Attention** | ✅ Yes | ❌ No | ❌ No |
| **Blocks** | 12 (Transformer) | 18 (ConvNeXt) | 36 (ConvNeXt) |
| **Parameters** | ~28M | ~28M | ~200M |
| **Capacity** | 96 bits | 256 bits | 1024 bits |
| **TFLite** | ⚠️ Harder | ✅ Easy | ✅ Easy |

## Proportional Scaling

### Scaling Formula

ChunkySeal scales dimensions based on message length:

```python
multiplier = sqrt(nbits / 128)

# For 1024 bits:
multiplier = sqrt(1024 / 128) = sqrt(8) ≈ 2.83

# Apply to base dimensions:
base_dims = [128, 256, 512, 1024]
scaled_dims = [int(dim * 2.83) for dim in base_dims]
            = [362, 724, 1448, 2896]
```

### Why Square Root?

- **Linear scaling** (2×) would be too aggressive (4× params)
- **Square root** (√2×) provides balanced growth (~2× params)
- Maintains reasonable model size while increasing capacity

### Scaling Comparison

| Model | Capacity | Multiplier | Stage 4 Dims | Parameters |
|-------|----------|------------|--------------|------------|
| VideoSeal v1.0 | 256 bits | 1.41× | 1082 | ~28M |
| ChunkySeal | 1024 bits | 2.83× | 2896 | ~200M |

**Result**: 4× capacity requires ~7× parameters (due to proportional scaling)

## Model Flow

### Input Processing

```python
# Input: RGB image (B, 3, 256, 256)
# Range: [0, 1]

# Preprocessing
imgs = imgs * 2 - 1  # Convert to [-1, 1]
```

### Encoder Forward Pass

```python
def forward(self, x):
    # Stem: 256×256 → 128×128
    x = self.stem(x)  # (B, 362, 128, 128)
    
    # Stage 1: 128×128 (3 blocks)
    x = self.stage1(x)  # (B, 362, 128, 128)
    x = self.downsample1(x)  # (B, 724, 64, 64)
    
    # Stage 2: 64×64 (3 blocks)
    x = self.stage2(x)  # (B, 724, 64, 64)
    x = self.downsample2(x)  # (B, 1448, 32, 32)
    
    # Stage 3: 32×32 (27 blocks) ← Most computation
    x = self.stage3(x)  # (B, 1448, 32, 32)
    x = self.downsample3(x)  # (B, 2896, 16, 16)
    
    # Stage 4: 16×16 (3 blocks)
    x = self.stage4(x)  # (B, 2896, 16, 16)
    
    return x
```

### Decoder Forward Pass

```python
def forward(self, x):
    # Input: (B, 2896, 16, 16)
    
    # Upsample and reduce channels
    x = self.upsample(x)  # (B, 724, 32, 32)
    
    # Predict mask + message
    x = self.head(x)  # (B, 1025, 32, 32)
    
    # Global average pooling
    x = x.mean(dim=[2, 3])  # (B, 1025)
    
    return x
```

### Output Format

```python
# Output shape: (B, 1025)
# Channel 0: Detection confidence
# Channels 1-1024: Message bits (logits)

confidence = output[:, 0]  # (B,)
message_logits = output[:, 1:]  # (B, 1024)
message = (message_logits > 0).int()  # Binary message
```

## Computational Complexity

### FLOPs Estimation

From conversion output:
```
Estimated count of arithmetic ops: 1229.241 G ops
Equivalently: 614.621 G MACs
```

**Interpretation**: ~1.2 trillion operations per image

### Stage-wise Breakdown

| Stage | Blocks | Resolution | Channels | FLOPs (est.) |
|-------|--------|------------|----------|--------------|
| Stem | 1 | 128×128 | 362 | ~10 G |
| Stage 1 | 3 | 128×128 | 362 | ~50 G |
| Stage 2 | 3 | 64×64 | 724 | ~100 G |
| Stage 3 | 27 | 32×32 | 1448 | ~900 G | ← 73% of compute
| Stage 4 | 3 | 16×16 | 2896 | ~150 G |
| Decoder | - | Various | - | ~20 G |

**Key Insight**: Stage 3 (27 blocks) dominates computation

## Memory Requirements

### Model Weights

| Component | Parameters | Size (FLOAT32) | Size (INT8) |
|-----------|------------|----------------|-------------|
| Encoder | ~195M | 780 MB | 195 MB |
| Decoder | ~5M | 20 MB | 5 MB |
| **Total** | **~200M** | **~800 MB** | **~200 MB** |

### Inference Memory

| Phase | Memory Usage |
|-------|--------------|
| Model Loading | 2.95 GB (FLOAT32) / 960 MB (INT8) |
| Activations | ~2-4 GB |
| **Peak** | **~4-6 GB (FLOAT32)** / **~2-3 GB (INT8)** |

## TFLite Conversion Implications

### Why ConvNeXt Converts Well

1. **Pure Convolutions**: All operations are standard conv2d
2. **No Dynamic Shapes**: Fixed input/output sizes
3. **No Attention**: No complex attention operations
4. **Standard Ops**: All ops have TFLite equivalents

### Conversion Success Factors

✅ **Depthwise Conv**: Native TFLite support  
✅ **Pointwise Conv**: Standard conv2d  
✅ **Layer Norm**: Supported in TFLite  
✅ **GELU**: Supported activation  
✅ **Residual**: Simple add operation  

### Why ViT Would Be Harder

⚠️ **Attention**: Complex matmul + softmax  
⚠️ **QKV Projections**: Multiple linear layers  
⚠️ **Positional Encoding**: Custom operations  
⚠️ **Dynamic Attention**: Runtime-dependent shapes  

## See Also

- [Implementation Details](./implementation.md)
- [Quantization Guide](./quantization.md)
- [Usage Examples](./usage.md)
- [Module Usage Analysis](../../chunky_tflite/MODULE_USAGE_ANALYSIS.md)
- [Architecture Analysis](../../chunky_tflite/ARCHITECTURE_ANALYSIS.md)

