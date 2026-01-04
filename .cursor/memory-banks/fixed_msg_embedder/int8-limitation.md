# INT8 Quantization Limitation

## Overview

The VideoSeal embedder TFLite model **cannot be quantized to INT8** due to a `BROADCAST_TO` operation incompatibility in TensorFlow Lite's INT8 quantizer.

**Status**: ⚠️ **Not Supported**  
**Date**: January 4, 2026

---

## The Issue

### Error Message

```
RuntimeError: tensorflow/lite/kernels/broadcast_to.cc
Output shape must be broadcastable from input shape.
Node number 36 (BROADCAST_TO) failed to prepare.
```

### Root Cause

The fixed-size message processor uses `expand()` for spatial broadcasting:

```python
# In TFLiteFriendlyMsgProcessor.forward()
msg_aux = msg_aux.view(-1, self.hidden_size, 1, 1)
msg_aux = msg_aux.expand(-1, -1, self.spatial_size, self.spatial_size)  # ← Creates BROADCAST_TO
```

This operation:
- ✅ Works perfectly in **FLOAT32**
- ❌ **Fails** during INT8 quantization

### Why It Fails

TFLite's INT8 quantizer has stricter requirements:
1. All operations must have well-defined INT8 implementations
2. Some broadcasting operations are not fully supported in INT8
3. The BROADCAST_TO operation fails during `allocate_tensors()`

---

## Current Status

| Model | Format | Size | Status |
|-------|--------|------|--------|
| **Embedder** | FLOAT32 | 90.42 MB | ✅ **Production Ready** |
| **Embedder** | INT8 | N/A | ❌ **Not Supported** |
| **Detector** | FLOAT32 | 127.57 MB | ✅ Available |
| **Detector** | INT8 | 32.90 MB | ✅ Available (74.2% reduction) |

**Key Point**: The detector INT8 quantization works perfectly. Only the embedder has this limitation.

---

## Why Detector Works But Embedder Doesn't

### Detector (Works with INT8)

- **Architecture**: ConvNeXt-Tiny (pure CNN)
- **Operations**: Standard convolutions, normalization, activations
- **All operations**: Well-defined INT8 implementations
- **Result**: ✅ Successful INT8 quantization

### Embedder (Fails with INT8)

- **Architecture**: UNet (encoder-decoder)
- **Operations**: Includes BROADCAST_TO from message processor
- **Issue**: BROADCAST_TO lacks proper INT8 support in TFLite
- **Result**: ❌ INT8 quantization fails

---

## Attempted Solutions

### Attempt 1: Direct INT8 Conversion

```python
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

quantized_model = converter.convert()  # ❌ Fails
```

**Result**: `BROADCAST_TO failed to prepare`

### Attempt 2: Hybrid Quantization

```python
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS  # Fallback for unsupported ops
]
```

**Result**: Same error during model loading

### Attempt 3: ai-edge-torch Quantization

```python
quant_config = ai_edge_torch.config.QuantConfig(
    representative_dataset=representative_dataset
)
edge_model = ai_edge_torch.convert(model, inputs, quant_config=quant_config)
```

**Result**: API not available in current ai-edge-torch version

---

## Impact Analysis

### What This Means

1. **FLOAT32 embedder works perfectly** - Full functionality, no issues
2. **INT8 detector works perfectly** - Great for on-device detection
3. **Only embedder INT8 is blocked** - Not a critical limitation

### Deployment Impact

| Use Case | Impact | Severity |
|----------|--------|----------|
| **Mobile apps** | Low - Use hybrid architecture | ⚠️ Minor |
| **Edge devices** | Medium - Use FLOAT32 embedder | ⚠️ Acceptable |
| **Server-side** | None - Use PyTorch | ✅ No impact |
| **Detection only** | None - INT8 detector works | ✅ No impact |

---

## Technical Details

### BROADCAST_TO Operation Location

**File**: `tflite_msg_processor.py`  
**Line**: ~95 (in `forward()` method)

```python
def forward(self, latents: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
    # ... message embedding creation ...
    
    # Spatial broadcast (creates BROADCAST_TO)
    msg_aux = msg_aux.view(-1, self.hidden_size, 1, 1)
    msg_aux = msg_aux.expand(-1, -1, self.spatial_size, self.spatial_size)  # ← Issue
    
    # Concatenate with latents
    return torch.cat([latents, msg_aux], dim=1)
```

### TFLite Graph Analysis

When converted to TFLite, the `expand()` operation becomes:
- **Operation**: BROADCAST_TO
- **Node number**: 36
- **Input shape**: `[1, 256, 1, 1]`
- **Output shape**: `[1, 256, 32, 32]`
- **INT8 status**: ❌ Not supported

### Why expand() Was Chosen

The `expand()` operation was chosen over `repeat()` because:
1. More memory efficient (creates view, not copy)
2. Better for TFLite static graph tracing
3. Semantically clearer (broadcasting vs duplication)

However, this created the INT8 incompatibility.

---

## Comparison: FLOAT32 vs INT8

### FLOAT32 Embedder (Available)

**Specifications**:
- Size: 90.42 MB
- Operations: 56.592 G ops
- Inference time: ~100-200ms (CPU)
- Memory: ~200 MB runtime

**Advantages**:
- ✅ Works perfectly
- ✅ Full functionality
- ✅ No accuracy loss
- ✅ Production ready

**Disadvantages**:
- ❌ Larger file size
- ❌ Slower inference than INT8 would be
- ❌ Higher memory usage

### INT8 Embedder (Not Available)

**Expected specifications** (if it worked):
- Size: ~23 MB (75% reduction)
- Inference time: ~25-50ms (CPU)
- Memory: ~50 MB runtime

**Why it would be better**:
- Smaller app size
- Faster inference
- Lower memory usage
- Better battery life

**Why it doesn't work**:
- BROADCAST_TO operation incompatibility

---

## Future Solutions

### Potential Fix 1: Replace expand() with repeat()

**Idea**: Use `repeat()` instead of `expand()` to create TILE operation instead of BROADCAST_TO

```python
# Current (uses BROADCAST_TO)
msg_aux = msg_aux.expand(-1, -1, 32, 32)

# Alternative (uses TILE)
msg_aux = msg_aux.repeat(1, 1, 32, 32)
```

**Pros**:
- TILE operation may have better INT8 support
- Mathematically equivalent
- No retraining needed

**Cons**:
- Not tested
- May still fail
- Requires re-conversion

**Status**: ⚠️ Not tested, theoretical solution

### Potential Fix 2: Manual Broadcasting

**Idea**: Implement broadcasting manually with explicit operations

```python
# Manual broadcasting with multiplication
msg_aux = msg_aux.view(-1, hidden_size, 1, 1)
msg_aux = msg_aux * torch.ones(1, 1, 32, 32)  # Explicit broadcast
```

**Status**: Not tested

### Potential Fix 3: Wait for TFLite Update

**Idea**: Future TensorFlow Lite versions may improve BROADCAST_TO INT8 support

**Status**: Monitor TensorFlow Lite releases

### Potential Fix 4: Use ONNX Runtime

**Idea**: Convert to ONNX format instead of TFLite

**Pros**: Better INT8 support for various operations  
**Cons**: Different deployment pipeline, not TFLite

**Status**: Alternative approach, not pursued

---

## Recommendations

### For Mobile Apps

**Use hybrid architecture**:
- Server: PyTorch embedder (best quality)
- Mobile: INT8 detector (32.90 MB)

**Benefits**:
- Smallest mobile footprint
- Best quality (with attenuation)
- Fast on-device detection

See: [workarounds.md](./workarounds.md) for implementation details

### For Edge Devices

**Use FLOAT32 embedder**:
- On-device: FLOAT32 embedder (90.42 MB) + INT8 detector (32.90 MB)
- Total: 123 MB

**Benefits**:
- Fully offline
- Reliable
- Acceptable size for edge devices

### For Research

**Try model modification**:
- Replace `expand()` with `repeat()`
- Test INT8 quantization
- Measure accuracy vs original

**May unlock**: INT8 support for future versions

---

## Related Documentation

- **Workarounds Guide**: [workarounds.md](./workarounds.md) - Practical solutions
- **Solution Design**: [solution-design.md](./solution-design.md) - Why expand() was used
- **Implementation**: [implementation.md](./implementation.md) - Code details
- **Troubleshooting**: [troubleshooting.md](./troubleshooting.md) - Common issues

---

## Conclusion

The INT8 quantization limitation is **not a critical blocker** for VideoSeal deployment:

1. **FLOAT32 embedder works perfectly** (90.42 MB)
2. **INT8 detector works perfectly** (32.90 MB)
3. **Multiple workarounds available** (hybrid, FLOAT32, FP16)
4. **Production-ready solutions exist** for all use cases

The limitation is specific to the embedder's BROADCAST_TO operation and does not affect the overall functionality or deployment viability of VideoSeal TFLite models.

---

*Last Updated: January 4, 2026*  
*Status: FLOAT32 only, INT8 not supported*

