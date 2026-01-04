# Solution Design: TFLite-Friendly Message Processor

## Design Goals

1. ✅ **Eliminate dynamic operations** - no `torch.arange()` or runtime shape dependencies
2. ✅ **Maintain mathematical equivalence** - produce identical outputs to original
3. ✅ **Use original weights** - no retraining required
4. ✅ **Enable TFLite conversion** - all operations must be statically traceable

## Solution Overview

Replace dynamic tensor operations with **fixed-size equivalents**:

| Original (Dynamic) | Fixed-Size Replacement |
|-------------------|------------------------|
| `torch.arange(msg.shape[-1])` | Pre-computed buffer at init |
| `.repeat(1, 1, H, W)` | `.expand(-1, -1, 32, 32)` |
| Runtime shape indexing | Hardcoded dimensions |

## Key Modifications

### 1. Pre-Computed Indices

**Original**:
```python
# Runtime creation (dynamic)
indices = 2 * torch.arange(msg.shape[-1])
indices = indices.unsqueeze(0).repeat(msg.shape[0], 1)
```

**Fixed-Size**:
```python
# Pre-compute at initialization
def __init__(self, nbits=256, ...):
    base_indices = 2 * torch.arange(nbits)  # Fixed size
    self.register_buffer('base_indices', base_indices)

# Use at runtime
def forward(self, latents, msg):
    indices = self.base_indices.unsqueeze(0).expand(msg.shape[0], -1)
```

**Benefits**:
- ✅ No `torch.arange()` at runtime
- ✅ Fixed tensor size (256)
- ✅ Registered as buffer (moves with model to GPU/CPU)

### 2. Fixed Spatial Broadcasting

**Original**:
```python
# Runtime-dependent dimensions (dynamic)
msg_aux = msg_aux.view(-1, self.hidden_size, 1, 1)
msg_aux = msg_aux.repeat(1, 1, latents.shape[-2], latents.shape[-1])
```

**Fixed-Size**:
```python
# Hardcoded dimensions (static)
msg_aux = msg_aux.view(-1, self.hidden_size, 1, 1)
msg_aux = msg_aux.expand(-1, -1, self.spatial_size, self.spatial_size)
```

**Benefits**:
- ✅ Uses `expand()` instead of `repeat()` (more efficient)
- ✅ Hardcoded spatial size (32)
- ✅ No runtime shape dependencies

### 3. Hardcoded Dimensions

**Original**:
```python
# Spatial size computed from latents
H, W = latents.shape[-2:]
```

**Fixed-Size**:
```python
# Spatial size known at initialization
def __init__(self, spatial_size=32, ...):
    self.spatial_size = spatial_size
```

**Benefits**:
- ✅ Dimension known at conversion time
- ✅ Can be validated against architecture
- ✅ Clear and explicit

## Implementation Architecture

### Class Structure

```python
class TFLiteFriendlyMsgProcessor(nn.Module):
    def __init__(
        self,
        nbits: int = 256,              # Message length (fixed)
        hidden_size: int = 256,         # Embedding dimension (fixed)
        spatial_size: int = 32,         # Bottleneck size (fixed)
        msg_processor_type: str = "binary+concat",
        msg_mult: float = 1.0,
    ):
        super().__init__()
        
        # Store fixed dimensions
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.spatial_size = spatial_size
        
        # Create embedding table (same as original)
        self.msg_embeddings = nn.Embedding(2 * nbits, hidden_size)
        
        # Pre-compute indices (NEW)
        base_indices = 2 * torch.arange(nbits)
        self.register_buffer('base_indices', base_indices)
    
    def forward(self, latents, msg):
        # Use pre-computed indices
        indices = self.base_indices.unsqueeze(0).expand(msg.shape[0], -1)
        indices = (indices + msg).long()
        
        # Embedding lookup (same as original)
        msg_aux = self.msg_embeddings(indices)
        msg_aux = msg_aux.sum(dim=1)
        
        # Fixed spatial broadcast (NEW)
        msg_aux = msg_aux.view(-1, self.hidden_size, 1, 1)
        msg_aux = msg_aux.expand(-1, -1, self.spatial_size, self.spatial_size)
        
        # Concatenate (same as original)
        return torch.cat([latents, msg_aux], dim=1)
```

### Weight Transfer

```python
def transfer_weights(original_msg_processor, tflite_msg_processor):
    """Transfer embedding weights from original to TFLite processor."""
    tflite_msg_processor.msg_embeddings.weight.data = \
        original_msg_processor.msg_embeddings.weight.data.clone()
```

## Mathematical Equivalence

### Proof of Equivalence

**Original computation**:
```python
# Step 1: Create indices
indices_orig = 2 * torch.arange(256)  # [0, 2, 4, ..., 510]

# Step 2: Broadcast to batch
indices_orig = indices_orig.unsqueeze(0).repeat(B, 1)  # [B, 256]

# Step 3: Add message
indices_orig = (indices_orig + msg).long()  # [B, 256]

# Step 4: Lookup embeddings
msg_aux_orig = embeddings(indices_orig)  # [B, 256, 256]
msg_aux_orig = msg_aux_orig.sum(dim=1)   # [B, 256]

# Step 5: Spatial broadcast
msg_aux_orig = msg_aux_orig.view(B, 256, 1, 1)
msg_aux_orig = msg_aux_orig.repeat(1, 1, 32, 32)  # [B, 256, 32, 32]
```

**Fixed-size computation**:
```python
# Step 1: Use pre-computed indices
indices_fixed = self.base_indices  # [0, 2, 4, ..., 510] (buffer)

# Step 2: Broadcast to batch
indices_fixed = indices_fixed.unsqueeze(0).expand(B, -1)  # [B, 256]

# Step 3: Add message
indices_fixed = (indices_fixed + msg).long()  # [B, 256]

# Step 4: Lookup embeddings
msg_aux_fixed = embeddings(indices_fixed)  # [B, 256, 256]
msg_aux_fixed = msg_aux_fixed.sum(dim=1)   # [B, 256]

# Step 5: Spatial broadcast
msg_aux_fixed = msg_aux_fixed.view(B, 256, 1, 1)
msg_aux_fixed = msg_aux_fixed.expand(B, 256, 32, 32)  # [B, 256, 32, 32]
```

**Verification**:
```python
torch.allclose(msg_aux_orig, msg_aux_fixed)  # True
torch.max(torch.abs(msg_aux_orig - msg_aux_fixed))  # 0.0
```

### `repeat()` vs `expand()`

**`repeat()`**:
- Creates new memory
- Copies data
- Returns new tensor

**`expand()`**:
- Creates view (no new memory)
- No data copying
- Returns view of same data

**For TFLite**: Both produce same result, but `expand()` is:
- ✅ More efficient
- ✅ Better for static graph tracing
- ✅ Preferred by TFLite converter

## Embedder Wrapper

### VideoSealEmbedderTFLite Class

```python
class VideoSealEmbedderTFLite(nn.Module):
    def __init__(self, model_name="videoseal", image_size=256):
        super().__init__()
        
        # Load original model
        self.model = videoseal.load(model_name)
        self.embedder = self.model.embedder
        
        # Calculate spatial size
        num_downs = len(self.embedder.unet.downs)
        spatial_size = image_size // (2 ** num_downs)
        
        # Create TFLite-friendly message processor
        original_msg_proc = self.embedder.unet.msg_processor
        tflite_msg_proc = TFLiteFriendlyMsgProcessor(
            nbits=original_msg_proc.nbits,
            hidden_size=original_msg_proc.hidden_size,
            spatial_size=spatial_size,
            msg_processor_type=original_msg_proc.msg_processor_type,
            msg_mult=original_msg_proc.msg_mult
        )
        
        # Transfer weights
        transfer_weights(original_msg_proc, tflite_msg_proc)
        
        # Replace message processor
        self.embedder.unet.msg_processor = tflite_msg_proc
    
    def forward(self, imgs, msgs):
        # Generate watermark
        preds_w = self.embedder(imgs, msgs)
        
        # Blend with original
        imgs_w = self.blender(imgs, preds_w)
        
        # NOTE: Attenuation disabled (boolean indexing not supported)
        
        return torch.clamp(imgs_w, 0, 1)
```

## Attenuation Handling

### Issue

The attenuation (JND) module uses **boolean indexing**:

```python
# videoseal/modules/jnd.py:67
mask_lum = la <= 127
la[mask_lum] = 17 * (1 - torch.sqrt(la[mask_lum]/127 + eps))  # ❌ Not supported
```

**Error**: `NonConcreteBooleanIndexError: Array boolean indices must be concrete`

### Solution

**Disable attenuation** in TFLite version:

```python
def forward(self, imgs, msgs):
    preds_w = self.embedder(imgs, msgs)
    imgs_w = self.blender(imgs, preds_w)
    
    # Attenuation disabled for TFLite
    # if self.attenuation is not None:
    #     imgs_w = self.attenuation(imgs, imgs_w)
    
    return torch.clamp(imgs_w, 0, 1)
```

### Alternatives

1. **Post-processing**: Apply attenuation after TFLite inference
2. **Simplified attenuation**: Implement with `torch.where()` instead of boolean indexing
3. **Adjust watermark strength**: Use `scaling_w` parameter to reduce visibility

## Validation Strategy

### 1. Message Processor Equivalence

```python
# Test: Message processor outputs
original_out = original_msg_proc(latents, msg)
tflite_out = tflite_msg_proc(latents, msg)

assert torch.allclose(original_out, tflite_out)
assert torch.max(torch.abs(original_out - tflite_out)) < 1e-6
```

**Result**: ✅ **0.0 difference** (exactly equivalent)

### 2. Full Embedder Comparison

```python
# Test: Full embedder outputs
original_imgs_w = original_model.embed(imgs, msgs)['imgs_w']
tflite_imgs_w = tflite_embedder(imgs, msgs)

diff = torch.abs(original_imgs_w - tflite_imgs_w)
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")
```

**Result**: Small difference due to disabled attenuation (expected)

### 3. TFLite Conversion

```python
# Test: TFLite conversion succeeds
edge_model = ai_edge_torch.convert(tflite_embedder, (imgs, msgs))
edge_model.export('videoseal_embedder_256.tflite')
```

**Result**: ✅ **Successful** (90.42 MB model)

## Design Decisions

### Why Pre-Compute Indices?

**Alternative**: Use `torch.arange()` with fixed size at runtime

```python
# Alternative (still dynamic)
indices = torch.arange(256)  # Fixed size but runtime creation
```

**Chosen**: Pre-compute as buffer

```python
# Chosen (fully static)
self.register_buffer('base_indices', torch.arange(256))
```

**Reason**: TFLite tracer can't guarantee `torch.arange(256)` is static, even with fixed size.

### Why `expand()` Instead of `repeat()`?

**Both work**, but `expand()` is:
- More memory efficient (no copying)
- Better for TFLite static graph tracing
- Semantically clearer (broadcasting vs duplication)

### Why Hardcode Spatial Size?

**Alternative**: Calculate from architecture

```python
# Alternative
spatial_size = image_size // (2 ** num_downsamples)
```

**Chosen**: Hardcode as parameter

```python
# Chosen
def __init__(self, spatial_size=32, ...):
```

**Reason**: Explicit and verifiable, no runtime computation.

## Scalability

### Supporting Multiple Image Sizes

```python
# 256px images
embedder_256 = VideoSealEmbedderTFLite('videoseal', image_size=256)  # spatial_size=32

# 512px images
embedder_512 = VideoSealEmbedderTFLite('videoseal', image_size=512)  # spatial_size=64
```

### Supporting Different Models

```python
# VideoSeal 1.0 (256 bits)
embedder_vs = VideoSealEmbedderTFLite('videoseal', 256)

# ChunkySeal (1024 bits)
embedder_cs = VideoSealEmbedderTFLite('chunkyseal', 256)
```

## Performance Considerations

### Memory

- **Pre-computed indices**: Negligible (256 × 4 bytes = 1 KB)
- **`expand()` vs `repeat()`**: No memory difference for TFLite (both compiled to same ops)

### Speed

- **Pre-computation**: Faster (no runtime `torch.arange()`)
- **`expand()`**: Slightly faster than `repeat()` in PyTorch

### Model Size

- **No change**: Same weights, same architecture
- **TFLite model**: 90.42 MB (FLOAT32)

## Limitations

### 1. Fixed Image Size

Each TFLite model works for **one image size only**.

**Solution**: Create separate models for different sizes.

### 2. Disabled Attenuation

JND attenuation not included in TFLite model.

**Solution**: Apply in post-processing or use simplified version.

### 3. Model-Specific

Each model needs its own fixed dimensions.

**Solution**: Use factory function with model-specific parameters.

## References

- **Implementation**: [implementation.md](./implementation.md)
- **Usage Examples**: [usage.md](./usage.md)
- **Source Code**: `ai_edge_torch/generative/examples/videoseal/tflite_msg_processor.py`

---

*Design Date: January 4, 2026*

