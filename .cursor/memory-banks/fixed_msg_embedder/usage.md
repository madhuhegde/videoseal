# Usage Guide: TFLite VideoSeal Embedder

## Quick Start

### Python (PyTorch)

```python
from videoseal_models import create_embedder_tflite
import torch

# Create embedder
embedder = create_embedder_tflite('videoseal', 256)

# Prepare inputs
imgs = torch.rand(1, 3, 256, 256)  # RGB image in [0, 1]
msgs = torch.randint(0, 2, (1, 256)).float()  # 256-bit binary message

# Embed watermark
with torch.no_grad():
    imgs_watermarked = embedder(imgs, msgs)

print(f"Watermarked image shape: {imgs_watermarked.shape}")
```

### Python (TFLite)

```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='videoseal_embedder_256.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare inputs (numpy arrays)
img = np.random.rand(1, 3, 256, 256).astype(np.float32)
msg = np.random.randint(0, 2, (1, 256)).astype(np.float32)

# Set inputs
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.set_tensor(input_details[1]['index'], msg)

# Run inference
interpreter.invoke()

# Get output
watermarked_img = interpreter.get_tensor(output_details[0]['index'])
print(f"Watermarked image shape: {watermarked_img.shape}")
```

---

## Complete Examples

### Example 1: Embed Watermark in Image

```python
import torch
import videoseal
from videoseal_models import create_embedder_tflite
from PIL import Image
import torchvision.transforms as transforms

# Load image
img_pil = Image.open('input.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
img_tensor = transform(img_pil).unsqueeze(0)  # [1, 3, 256, 256]

# Create message (256 bits)
msg = torch.randint(0, 2, (1, 256)).float()
print(f"Message: {msg[0][:16]}...")  # Show first 16 bits

# Load embedder
embedder = create_embedder_tflite('videoseal', 256)

# Embed watermark
with torch.no_grad():
    img_watermarked = embedder(img_tensor, msg)

# Save watermarked image
img_watermarked_pil = transforms.ToPILImage()(img_watermarked[0])
img_watermarked_pil.save('output_watermarked.jpg')

print("✓ Watermark embedded successfully")
```

### Example 2: Batch Processing

```python
import torch
from videoseal_models import create_embedder_tflite

# Create embedder
embedder = create_embedder_tflite('videoseal', 256)

# Batch of images
batch_size = 4
imgs = torch.rand(batch_size, 3, 256, 256)

# Same message for all images
msg = torch.randint(0, 2, (1, 256)).float()
msgs = msg.repeat(batch_size, 1)  # [4, 256]

# Embed watermarks
with torch.no_grad():
    imgs_watermarked = embedder(imgs, msgs)

print(f"Processed {batch_size} images")
print(f"Output shape: {imgs_watermarked.shape}")
```

### Example 3: Different Messages per Image

```python
import torch
from videoseal_models import create_embedder_tflite

# Create embedder
embedder = create_embedder_tflite('videoseal', 256)

# Batch of images
imgs = torch.rand(4, 3, 256, 256)

# Different message for each image
msgs = torch.randint(0, 2, (4, 256)).float()

# Embed watermarks
with torch.no_grad():
    imgs_watermarked = embedder(imgs, msgs)

print("✓ Embedded different watermarks in each image")
```

### Example 4: Convert to TFLite

```python
import torch
import ai_edge_torch
from videoseal_models import create_embedder_tflite

# Create embedder
embedder = create_embedder_tflite('videoseal', 256)

# Sample inputs for tracing
sample_img = torch.rand(1, 3, 256, 256)
sample_msg = torch.randint(0, 2, (1, 256)).float()

# Convert to TFLite
print("Converting to TFLite...")
edge_model = ai_edge_torch.convert(embedder, (sample_img, sample_msg))

# Export
edge_model.export('videoseal_embedder_256.tflite')

import os
file_size = os.path.getsize('videoseal_embedder_256.tflite') / (1024 * 1024)
print(f"✓ TFLite model saved: {file_size:.2f} MB")
```

### Example 5: TFLite Inference (Full Pipeline)

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='videoseal_embedder_256.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
img_pil = Image.open('input.jpg').convert('RGB').resize((256, 256))
img_np = np.array(img_pil).astype(np.float32) / 255.0  # Normalize to [0, 1]
img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
img_np = np.expand_dims(img_np, 0)  # Add batch dimension

# Create message
msg_np = np.random.randint(0, 2, (1, 256)).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img_np)
interpreter.set_tensor(input_details[1]['index'], msg_np)
interpreter.invoke()

# Get output
output_np = interpreter.get_tensor(output_details[0]['index'])

# Postprocess
output_np = np.clip(output_np[0], 0, 1)  # Clamp to [0, 1]
output_np = np.transpose(output_np, (1, 2, 0))  # CHW -> HWC
output_np = (output_np * 255).astype(np.uint8)

# Save
output_pil = Image.fromarray(output_np)
output_pil.save('output_watermarked_tflite.jpg')

print("✓ TFLite inference complete")
```

### Example 6: Compare PyTorch vs TFLite

```python
import torch
import tensorflow as tf
import numpy as np
from videoseal_models import create_embedder_tflite

# Create PyTorch embedder
embedder_pt = create_embedder_tflite('videoseal', 256)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='videoseal_embedder_256.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test inputs
img_pt = torch.rand(1, 3, 256, 256)
msg_pt = torch.randint(0, 2, (1, 256)).float()

# PyTorch inference
with torch.no_grad():
    output_pt = embedder_pt(img_pt, msg_pt)

# TFLite inference
img_np = img_pt.numpy()
msg_np = msg_pt.numpy()

interpreter.set_tensor(input_details[0]['index'], img_np)
interpreter.set_tensor(input_details[1]['index'], msg_np)
interpreter.invoke()
output_tflite = interpreter.get_tensor(output_details[0]['index'])

# Compare
diff = np.abs(output_pt.numpy() - output_tflite)
print(f"Max difference: {diff.max():.6f}")
print(f"Mean difference: {diff.mean():.6f}")
print(f"Median difference: {np.median(diff):.6f}")
```

---

## API Reference

### `create_embedder_tflite()`

```python
def create_embedder_tflite(model_name="videoseal", image_size=256):
    """
    Create TFLite-compatible VideoSeal embedder.
    
    Args:
        model_name: VideoSeal model variant ('videoseal', 'pixelseal', 'chunkyseal')
        image_size: Input image size (default: 256)
    
    Returns:
        VideoSealEmbedderTFLite instance ready for conversion
    """
```

**Supported Models**:
- `'videoseal'`: VideoSeal 1.0 (256 bits)
- `'pixelseal'`: PixelSeal (256 bits)
- `'chunkyseal'`: ChunkySeal (1024 bits)

**Example**:
```python
embedder = create_embedder_tflite('videoseal', 256)
```

### `VideoSealEmbedderTFLite`

```python
class VideoSealEmbedderTFLite(nn.Module):
    def __init__(self, model_name="videoseal", image_size=256, eval_mode=True):
        """
        TFLite-compatible VideoSeal embedder.
        
        Args:
            model_name: Model variant
            image_size: Input image size
            eval_mode: Whether to set model to eval mode
        """
    
    def forward(self, imgs, msgs):
        """
        Embed watermark into images.
        
        Args:
            imgs: Tensor [B, 3, H, W] in range [0, 1]
            msgs: Tensor [B, nbits] with binary values {0, 1}
        
        Returns:
            imgs_w: Watermarked images [B, 3, H, W] in range [0, 1]
        """
```

**Example**:
```python
embedder = VideoSealEmbedderTFLite('videoseal', 256)
imgs_w = embedder(imgs, msgs)
```

### `TFLiteFriendlyMsgProcessor`

```python
class TFLiteFriendlyMsgProcessor(nn.Module):
    def __init__(
        self,
        nbits: int = 256,
        hidden_size: int = 256,
        spatial_size: int = 32,
        msg_processor_type: str = "binary+concat",
        msg_mult: float = 1.0,
    ):
        """
        TFLite-compatible message processor.
        
        Args:
            nbits: Number of message bits
            hidden_size: Embedding dimension
            spatial_size: Spatial dimension at bottleneck
            msg_processor_type: Type of message processing
            msg_mult: Multiplier for message embeddings
        """
    
    def forward(self, latents, msg):
        """
        Apply message embeddings to latents.
        
        Args:
            latents: Feature maps [B, C, H, W]
            msg: Binary message [B, nbits]
        
        Returns:
            Latents with message embeddings [B, C+hidden_size, H, W]
        """
```

---

## Input/Output Specifications

### PyTorch Embedder

**Inputs**:
- `imgs`: `torch.Tensor` of shape `[B, 3, H, W]`, dtype `float32`, range `[0, 1]`
- `msgs`: `torch.Tensor` of shape `[B, nbits]`, dtype `float32`, values `{0, 1}`

**Output**:
- `imgs_w`: `torch.Tensor` of shape `[B, 3, H, W]`, dtype `float32`, range `[0, 1]`

### TFLite Model

**Inputs**:
- Input 0 (image): `float32[1, 3, 256, 256]`, range `[0, 1]`
- Input 1 (message): `float32[1, 256]`, values `{0, 1}`

**Output**:
- Output 0 (watermarked image): `float32[1, 3, 256, 256]`, range `[0, 1]`

---

## Common Use Cases

### 1. Content Protection

```python
# Embed unique ID in each image
user_id = 12345
msg = torch.tensor([int(b) for b in format(user_id, '0256b')]).float().unsqueeze(0)
imgs_w = embedder(imgs, msg)
```

### 2. Batch Watermarking

```python
# Watermark entire dataset
for batch_imgs in dataloader:
    msgs = torch.randint(0, 2, (batch_imgs.size(0), 256)).float()
    batch_imgs_w = embedder(batch_imgs, msgs)
    # Save batch_imgs_w...
```

### 3. Mobile Deployment

```python
# Convert once, deploy on mobile
edge_model = ai_edge_torch.convert(embedder, (sample_img, sample_msg))
edge_model.export('videoseal_embedder_mobile.tflite')
# Deploy to Android/iOS app
```

---

## Performance Tips

### 1. Batch Processing

Process multiple images at once for better throughput:

```python
# Good: Batch of 16
imgs = torch.rand(16, 3, 256, 256)
msgs = torch.randint(0, 2, (16, 256)).float()
imgs_w = embedder(imgs, msgs)

# Less efficient: One at a time
for img in imgs:
    img_w = embedder(img.unsqueeze(0), msg)
```

### 2. GPU Acceleration

```python
# Move to GPU
embedder = embedder.cuda()
imgs = imgs.cuda()
msgs = msgs.cuda()

# Inference
imgs_w = embedder(imgs, msgs)
```

### 3. Mixed Precision

```python
# Use FP16 for faster inference
with torch.cuda.amp.autocast():
    imgs_w = embedder(imgs, msgs)
```

---

## Limitations

### 1. Fixed Image Size

Each TFLite model works for **one image size only**:
- 256×256 model: Only accepts 256×256 images
- 512×512 model: Only accepts 512×512 images

**Solution**: Resize images before inference or create multiple models.

### 2. Disabled Attenuation

JND attenuation is not included in TFLite model.

**Impact**: Watermark may be more visible.

**Mitigation**:
- Adjust watermark strength
- Apply attenuation in post-processing
- Use simplified TFLite-compatible attenuation

### 3. Model-Specific

Each model variant needs its own TFLite conversion:
- VideoSeal 1.0: 256 bits
- ChunkySeal: 1024 bits

---

## Troubleshooting

### Issue: Output values outside [0, 1]

**Solution**: Outputs are clamped to [0, 1] automatically. If you see values outside this range, check your input preprocessing.

### Issue: TFLite model file not found

**Solution**: Ensure you've run the conversion step and the file path is correct.

### Issue: Shape mismatch

**Solution**: Verify input shapes match model expectations (e.g., [1, 3, 256, 256] for images).

---

## References

- **Implementation Guide**: [implementation.md](./implementation.md)
- **Solution Design**: [solution-design.md](./solution-design.md)
- **Troubleshooting**: [troubleshooting.md](./troubleshooting.md)

---

*Last Updated: January 4, 2026*

