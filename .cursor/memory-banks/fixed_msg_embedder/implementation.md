# Implementation Guide: TFLite-Friendly Message Processor

## Step-by-Step Implementation

This guide walks through the complete implementation of the TFLite-compatible VideoSeal embedder.

## Prerequisites

```bash
# Required packages
pip install torch ai-edge-torch videoseal

# Environment
micromamba activate local_tf_env  # or your TensorFlow environment
```

## Step 1: Create TFLite-Friendly Message Processor

### File: `tflite_msg_processor.py`

```python
import torch
import torch.nn as nn


class TFLiteFriendlyMsgProcessor(nn.Module):
    """
    TFLite-compatible message processor with fixed-size operations.
    
    Replaces dynamic tensor operations with static equivalents while
    maintaining mathematical equivalence to the original implementation.
    """
    
    def __init__(
        self,
        nbits: int = 256,
        hidden_size: int = 256,
        spatial_size: int = 32,
        msg_processor_type: str = "binary+concat",
        msg_mult: float = 1.0,
    ):
        super().__init__()
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.spatial_size = spatial_size
        self.msg_mult = msg_mult
        
        # Parse message processor type
        self.msg_processor_type = msg_processor_type if nbits > 0 else "none+_"
        self.msg_type = self.msg_processor_type.split("+")[0]
        self.msg_agg = self.msg_processor_type.split("+")[1]
        
        # Create embedding table (same as original)
        if self.msg_type.startswith("no"):
            self.msg_embeddings = torch.tensor([])
        elif self.msg_type.startswith("bin"):
            self.msg_embeddings = nn.Embedding(2 * nbits, hidden_size)
        elif self.msg_type.startswith("gau"):
            self.msg_embeddings = nn.Embedding(nbits, hidden_size)
        else:
            raise ValueError(f"Invalid msg_processor_type: {self.msg_processor_type}")
        
        # Pre-compute base indices (KEY MODIFICATION)
        if self.msg_type.startswith("bin"):
            base_indices = 2 * torch.arange(nbits)
            self.register_buffer('base_indices', base_indices)
        elif self.msg_type.startswith("gau"):
            base_indices = torch.arange(nbits)
            self.register_buffer('base_indices', base_indices)
    
    def forward(self, latents: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """Apply message embeddings to latents."""
        if self.nbits == 0:
            return latents
        
        # Create message embeddings
        if self.msg_type.startswith("bin"):
            # Use pre-computed indices (no torch.arange)
            indices = self.base_indices.unsqueeze(0).expand(msg.shape[0], -1)
            indices = (indices + msg).long()
            
            # Embedding lookup
            msg_aux = self.msg_embeddings(indices)  # [B, nbits, hidden_size]
            msg_aux = msg_aux.sum(dim=1)            # [B, hidden_size]
            
        elif self.msg_type.startswith("gau"):
            # Gaussian message processing
            msg_aux = self.msg_embeddings(self.base_indices)
            msg_aux = torch.einsum("kd, bk -> bd", msg_aux, msg)
        else:
            raise ValueError(f"Invalid msg_type: {self.msg_type}")
        
        # Spatial broadcast (KEY MODIFICATION: hardcoded dimensions, use expand)
        msg_aux = msg_aux.view(-1, self.hidden_size, 1, 1)
        msg_aux = msg_aux.expand(-1, -1, self.spatial_size, self.spatial_size)
        
        # Apply to latents
        if self.msg_agg == "concat":
            latents = torch.cat([latents, self.msg_mult * msg_aux], dim=1)
        elif self.msg_agg == "add":
            latents = latents + self.msg_mult * msg_aux
        else:
            raise ValueError(f"Invalid msg_agg: {self.msg_agg}")
        
        return latents
```

### Key Changes from Original

1. **Pre-computed indices**:
   ```python
   # Original: torch.arange(msg.shape[-1])
   # New: self.base_indices (pre-computed buffer)
   ```

2. **Fixed spatial broadcast**:
   ```python
   # Original: .repeat(1, 1, latents.shape[-2], latents.shape[-1])
   # New: .expand(-1, -1, self.spatial_size, self.spatial_size)
   ```

3. **Hardcoded dimensions**:
   ```python
   # Original: Computed from runtime shapes
   # New: Passed as init parameters
   ```

## Step 2: Weight Transfer Function

```python
def transfer_weights(original_msg_processor, tflite_msg_processor):
    """Transfer weights from original to TFLite-friendly processor."""
    if hasattr(original_msg_processor, 'msg_embeddings') and \
       hasattr(tflite_msg_processor, 'msg_embeddings'):
        tflite_msg_processor.msg_embeddings.weight.data = \
            original_msg_processor.msg_embeddings.weight.data.clone()
        print("✓ Embedding weights transferred")
    else:
        print("⚠ No embeddings to transfer")
```

## Step 3: Create Embedder Wrapper

### File: `videoseal_models.py` (add to existing file)

```python
from tflite_msg_processor import TFLiteFriendlyMsgProcessor, transfer_weights

class VideoSealEmbedderTFLite(nn.Module):
    """TFLite-compatible VideoSeal embedder."""
    
    def __init__(self, model_name="videoseal", image_size=256, eval_mode=True):
        super().__init__()
        
        print(f"Loading VideoSeal model: {model_name}")
        self.model = videoseal.load(model_name)
        
        # Extract components
        self.embedder = self.model.embedder
        self.blender = self.model.blender
        self.attenuation = self.model.attenuation
        self.rgb2yuv = RGB2YUV()
        self.yuv_mode = self.embedder.yuv
        self.image_size = image_size
        
        # Calculate spatial size at bottleneck
        num_downs = len(self.embedder.unet.downs)
        spatial_size = image_size // (2 ** num_downs)
        
        print(f"UNet architecture:")
        print(f"  Downsample layers: {num_downs}")
        print(f"  Spatial size at bottleneck: {spatial_size}x{spatial_size}")
        
        # Create TFLite-friendly message processor
        original_msg_proc = self.embedder.unet.msg_processor
        self.tflite_msg_processor = TFLiteFriendlyMsgProcessor(
            nbits=original_msg_proc.nbits,
            hidden_size=original_msg_proc.hidden_size,
            spatial_size=spatial_size,
            msg_processor_type=original_msg_proc.msg_processor_type,
            msg_mult=original_msg_proc.msg_mult
        )
        
        # Transfer weights
        transfer_weights(original_msg_proc, self.tflite_msg_processor)
        
        # Replace message processor in UNet
        self.embedder.unet.msg_processor = self.tflite_msg_processor
        
        print(f"✓ TFLite-friendly message processor installed")
        
        if eval_mode:
            self.eval()
    
    def forward(self, imgs, msgs):
        """Embed watermark into images."""
        # Generate watermark
        if self.yuv_mode:
            imgs_yuv = self.rgb2yuv(imgs)
            preds_w = self.embedder(imgs_yuv[:, 0:1], msgs)
        else:
            preds_w = self.embedder(imgs, msgs)
        
        # Blend watermark with original image
        imgs_w = self.blender(imgs, preds_w)
        
        # NOTE: Attenuation disabled for TFLite (boolean indexing not supported)
        # if self.attenuation is not None:
        #     imgs_w = self.attenuation(imgs, imgs_w)
        
        # Clamp to valid range
        imgs_w = torch.clamp(imgs_w, 0, 1)
        
        return imgs_w
```

## Step 4: Factory Function

```python
def create_embedder_tflite(model_name="videoseal", image_size=256):
    """Create TFLite-compatible VideoSeal embedder."""
    model = VideoSealEmbedderTFLite(
        model_name=model_name,
        image_size=image_size,
        eval_mode=True
    )
    model.eval()
    return model
```

## Step 5: Test Message Processor

```python
# Test standalone message processor
msg_proc = TFLiteFriendlyMsgProcessor(
    nbits=256,
    hidden_size=256,
    spatial_size=32
)

# Test inputs
latents = torch.rand(1, 128, 32, 32)
msg = torch.randint(0, 2, (1, 256)).float()

# Forward pass
with torch.no_grad():
    output = msg_proc(latents, msg)

print(f"Input: {latents.shape}")
print(f"Message: {msg.shape}")
print(f"Output: {output.shape}")  # Should be [1, 384, 32, 32]
```

**Expected output**:
```
Input: torch.Size([1, 128, 32, 32])
Message: torch.Size([1, 256])
Output: torch.Size([1, 384, 32, 32])
```

## Step 6: Test Full Embedder

```python
import videoseal
from videoseal_models import create_embedder_tflite

# Load original model
original_model = videoseal.load('videoseal')

# Create TFLite embedder
tflite_embedder = create_embedder_tflite('videoseal', 256)

# Test inputs
imgs = torch.rand(1, 3, 256, 256)
msgs = torch.randint(0, 2, (1, 256)).float()

# Compare outputs
with torch.no_grad():
    out_original = original_model.embed(imgs, msgs=msgs, is_video=False)['imgs_w']
    out_tflite = tflite_embedder(imgs, msgs)

# Check difference
diff = torch.abs(out_original - out_tflite)
print(f"Max difference: {diff.max():.6f}")
print(f"Mean difference: {diff.mean():.6f}")
```

**Expected output**:
```
Max difference: 0.046645  # Small difference due to disabled attenuation
Mean difference: 0.010602
```

## Step 7: Test Message Processor Equivalence

```python
# Load model
model = videoseal.load('videoseal')
original_msg_proc = model.embedder.unet.msg_processor

# Create TFLite processor
tflite_msg_proc = TFLiteFriendlyMsgProcessor(
    nbits=original_msg_proc.nbits,
    hidden_size=original_msg_proc.hidden_size,
    spatial_size=32,
    msg_processor_type=original_msg_proc.msg_processor_type,
    msg_mult=original_msg_proc.msg_mult
)
transfer_weights(original_msg_proc, tflite_msg_proc)

# Test
latents = torch.rand(1, 128, 32, 32)
msg = torch.randint(0, 2, (1, 256)).float()

with torch.no_grad():
    out_original = original_msg_proc(latents, msg)
    out_tflite = tflite_msg_proc(latents, msg)

# Verify exact equivalence
diff = torch.abs(out_original - out_tflite)
print(f"Max difference: {diff.max():.10f}")  # Should be 0.0
```

**Expected output**:
```
Max difference: 0.0000000000  # Exactly equivalent!
```

## Step 8: Convert to TFLite

```python
import ai_edge_torch

# Create embedder
embedder = create_embedder_tflite('videoseal', 256)

# Sample inputs
sample_img = torch.rand(1, 3, 256, 256)
sample_msg = torch.randint(0, 2, (1, 256)).float()

# Convert to TFLite
print("Converting to TFLite (this may take 2-5 minutes)...")
edge_model = ai_edge_torch.convert(embedder, (sample_img, sample_msg))

# Export
output_file = "videoseal_embedder_256.tflite"
edge_model.export(output_file)

# Check file size
import os
file_size = os.path.getsize(output_file) / (1024 * 1024)
print(f"✓ Model saved: {output_file}")
print(f"  File size: {file_size:.2f} MB")
```

**Expected output**:
```
Converting to TFLite (this may take 2-5 minutes)...
✓ Model saved: videoseal_embedder_256.tflite
  File size: 90.42 MB
```

## Step 9: Verify TFLite Model

```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="videoseal_embedder_256.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:")
for i, detail in enumerate(input_details):
    print(f"  {i}: {detail['name']} - {detail['shape']} - {detail['dtype']}")

print("\nOutput details:")
for i, detail in enumerate(output_details):
    print(f"  {i}: {detail['name']} - {detail['shape']} - {detail['dtype']}")

# Test inference
img_np = np.random.rand(1, 3, 256, 256).astype(np.float32)
msg_np = np.random.randint(0, 2, (1, 256)).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], img_np)
interpreter.set_tensor(input_details[1]['index'], msg_np)

interpreter.invoke()

output_np = interpreter.get_tensor(output_details[0]['index'])
print(f"\n✓ Inference successful")
print(f"  Output shape: {output_np.shape}")
print(f"  Output range: [{output_np.min():.3f}, {output_np.max():.3f}]")
```

## Troubleshooting

### Error: "Sizes of tensors must match"

**Cause**: Dynamic operations still present  
**Solution**: Verify all `torch.arange()` calls are replaced with buffers

### Error: "NonConcreteBooleanIndexError"

**Cause**: Attenuation module not disabled  
**Solution**: Comment out attenuation in forward pass

### Error: "Module has no attribute 'downs'"

**Cause**: Trying to access non-existent attributes  
**Solution**: Use correct attribute names for your model version

### Conversion takes too long (>10 minutes)

**Cause**: Large model or slow system  
**Solution**: Normal for first conversion, use faster machine if available

## File Structure

```
ai_edge_torch/generative/examples/videoseal/
├── tflite_msg_processor.py          # NEW - Fixed-size message processor
├── videoseal_models.py               # UPDATED - Added VideoSealEmbedderTFLite
└── videoseal_embedder_tflite_256.tflite  # OUTPUT - TFLite model
```

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Conversion time** | ~2 minutes |
| **Model size** | 90.42 MB (FLOAT32) |
| **Operations** | 56.592 G ops |
| **Message processor equivalence** | 0.0 difference |
| **Full embedder difference** | <0.05 (due to disabled attenuation) |

## Next Steps

1. **Quantization**: Implement FP16/INT8 quantization
2. **Mobile deployment**: Test on Android/iOS devices
3. **Attenuation**: Implement TFLite-compatible version
4. **Multi-size support**: Create models for 512×512, 1024×1024

## References

- **Solution Design**: [solution-design.md](./solution-design.md)
- **Usage Examples**: [usage.md](./usage.md)
- **Troubleshooting**: [troubleshooting.md](./troubleshooting.md)

---

*Implementation Date: January 4, 2026*

