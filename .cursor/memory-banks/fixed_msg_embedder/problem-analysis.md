# Problem Analysis: UNet Embedder TFLite Conversion Failure

## Original Error

```
RuntimeError: Sizes of tensors must match except in dimension 1. 
Expected 32 in dimension 1 but got 1 for tensor number 1 in the list
```

**Location**: `videoseal/modules/msg_processor.py`, line 115  
**Operation**: `torch.cat([latents, msg_aux], dim=1)`

## Root Cause

The message processor uses **dynamic tensor operations** that are incompatible with TFLite's static graph tracing:

### 1. Dynamic Index Creation

```python
# Original code (lines 90-98)
indices = 2 * torch.arange(msg.shape[-1])  # ❌ Runtime-dependent
indices = indices.unsqueeze(0).repeat(msg.shape[0], 1)
indices = (indices + msg).long()
```

**Problem**: `torch.arange(msg.shape[-1])` creates a tensor whose size depends on runtime values, which TFLite cannot trace statically.

### 2. Dynamic Spatial Broadcasting

```python
# Original code (lines 112-115)
msg_aux = msg_aux.view(-1, self.hidden_size, 1, 1)
msg_aux = msg_aux.repeat(1, 1, latents.shape[-2], latents.shape[-1])  # ❌ Runtime-dependent
```

**Problem**: `.repeat()` with `latents.shape[-2]` and `latents.shape[-1]` depends on runtime tensor shapes.

### 3. Incorrect Tracing

The TFLite tracer incorrectly inferred `msg_aux` shape as `[1, 1, H, W]` instead of `[1, 256, H, W]`, causing the concatenation error.

## Why This Happened

### TFLite Static Graph Requirements

TFLite requires **all tensor shapes to be known at conversion time**:

```python
# TFLite expects:
msg_aux = torch.zeros(1, 256, 32, 32)  # ✅ Fixed dimensions

# But got:
msg_aux = torch.zeros(1, msg.shape[-1], H, W)  # ❌ Dynamic dimensions
```

### Dynamic Operations Are Problematic

1. **`torch.arange(n)`**: Size depends on runtime value `n`
2. **`.repeat(1, 1, H, W)`**: Dimensions depend on runtime values `H`, `W`
3. **Shape indexing**: `tensor.shape[-1]` is treated as dynamic

## Error Trace Analysis

```python
# Expected behavior:
latents: [1, 128, 32, 32]
msg_aux:  [1, 256, 32, 32]  # Should be 256 channels
output:   [1, 384, 32, 32]  # 128 + 256 = 384

# Actual traced behavior:
latents: [1, 128, 32, 32]
msg_aux:  [1,   1, 32, 32]  # ❌ Traced as 1 channel
output:   ERROR - dimension mismatch
```

## Why Original Code Worked in PyTorch

PyTorch uses **dynamic graph execution**:
- Shapes are computed at runtime
- `torch.arange()` can create tensors of any size
- `.repeat()` can broadcast to any dimensions

But TFLite uses **static graph compilation**:
- All shapes must be known at conversion time
- Dynamic operations must be eliminated
- Tensor dimensions must be hardcoded

## Comparison with Detector

### Detector (Succeeded)

The detector conversion succeeded because it uses **only static operations**:

```python
# ConvNeXt detector
x = self.conv(x)      # ✅ Fixed kernel sizes
x = self.norm(x)      # ✅ Fixed dimensions
x = self.gelu(x)      # ✅ Element-wise operation
```

**No dynamic operations** → TFLite conversion succeeded

### Embedder (Failed)

The embedder failed because of **dynamic message processing**:

```python
# Message processor
indices = torch.arange(msg.shape[-1])  # ❌ Dynamic
msg_aux = msg_aux.repeat(1, 1, H, W)   # ❌ Dynamic
```

**Dynamic operations** → TFLite conversion failed

## Architecture Verification

### VideoSeal 1.0 Architecture

**Embedder**: UNetMsg (Pure CNN)
- ✅ NO attention mechanisms
- ✅ Pure CNN operations (Conv2d, BatchNorm, ReLU)
- ❌ Dynamic message processor (the problem)

**Detector**: ConvNeXt-Tiny (Pure CNN)
- ✅ NO attention mechanisms
- ✅ Pure CNN operations (Depthwise conv, GELU)
- ✅ All static operations

**Key Insight**: The embedder architecture itself is TFLite-friendly (pure CNN), but the **message processor implementation** used dynamic operations.

## Attempted Workarounds

### 1. Tracing with Fixed Inputs ❌

```python
# Tried: Provide fixed-size sample inputs
sample_msg = torch.zeros(1, 256)  # Fixed size
edge_model = ai_edge_torch.convert(model, (img, sample_msg))
```

**Result**: Failed - tracer still detected dynamic operations inside the model.

### 2. JIT Scripting ❌

```python
# Tried: Use TorchScript
scripted = torch.jit.script(model)
```

**Result**: Failed - TorchScript doesn't eliminate dynamic operations for TFLite.

### 3. ONNX Intermediate Format ❌

```python
# Tried: PyTorch → ONNX → TFLite
torch.onnx.export(model, ...)
```

**Result**: Not attempted - ONNX also requires static shapes.

## Solution Preview

The solution is to **modify the PyTorch model** to use fixed-size operations:

```python
# Solution: Pre-compute indices as buffers
base_indices = 2 * torch.arange(256)  # ✅ Fixed at init
self.register_buffer('base_indices', base_indices)

# Solution: Use expand() instead of repeat()
msg_aux = msg_aux.expand(-1, -1, 32, 32)  # ✅ Fixed dimensions
```

See [solution-design.md](./solution-design.md) for complete details.

## Key Takeaways

1. **TFLite requires static graphs** - all shapes must be known at conversion time
2. **Dynamic operations are incompatible** - `torch.arange()`, `.repeat()` with runtime shapes
3. **The architecture is fine** - UNetMsg is pure CNN, TFLite-friendly
4. **Only the message processor needs fixing** - replace dynamic operations with static equivalents
5. **No retraining needed** - can transfer weights from original model

## References

- **Error Location**: `videoseal/modules/msg_processor.py:90-115`
- **Detailed Analysis**: `../../../videoseal_clone/UNET_EMBEDDER_CONVERSION_ANALYSIS.md`
- **Solution**: [solution-design.md](./solution-design.md)

---

*Analysis Date: January 4, 2026*

