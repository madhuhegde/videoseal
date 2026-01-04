# Troubleshooting Guide: TFLite VideoSeal Embedder

## Common Issues and Solutions

---

## Conversion Errors

### Error: "Sizes of tensors must match except in dimension 1"

**Full Error**:
```
RuntimeError: Sizes of tensors must match except in dimension 1. 
Expected 32 in dimension 1 but got 1 for tensor number 1 in the list
```

**Cause**: Dynamic tensor operations still present in the model.

**Solution**:
1. Verify you're using `TFLiteFriendlyMsgProcessor`, not the original `MsgProcessor`
2. Check that all `torch.arange()` calls are replaced with pre-computed buffers
3. Ensure `.repeat()` is replaced with `.expand()` with fixed dimensions

**Verification**:
```python
# Check message processor type
print(type(embedder.embedder.unet.msg_processor))
# Should be: TFLiteFriendlyMsgProcessor
```

---

### Error: "NonConcreteBooleanIndexError: Array boolean indices must be concrete"

**Full Error**:
```
jax.errors.NonConcreteBooleanIndexError: Array boolean indices must be concrete; 
got bool[1,1,256,256]
```

**Cause**: Attenuation (JND) module uses boolean indexing which TFLite doesn't support.

**Solution**: Disable attenuation in the forward pass:

```python
def forward(self, imgs, msgs):
    preds_w = self.embedder(imgs, msgs)
    imgs_w = self.blender(imgs, preds_w)
    
    # Disable attenuation
    # if self.attenuation is not None:
    #     imgs_w = self.attenuation(imgs, imgs_w)
    
    return torch.clamp(imgs_w, 0, 1)
```

**Location**: `videoseal_models.py`, `VideoSealEmbedderTFLite.forward()`

---

### Error: "Module 'UNetMsg' has no attribute 'mid'"

**Cause**: Trying to access non-existent attributes in UNet architecture.

**Solution**: Use correct attribute names:
- `embedder.unet.downs` (encoder blocks)
- `embedder.unet.ups` (decoder blocks)
- `embedder.unet.msg_processor` (message processor)

**Avoid**:
- `embedder.unet.mid` (doesn't exist in all versions)
- `embedder.unet.yuv` (stored in wrapper, not UNet)

---

### Error: "BROADCAST_TO failed to prepare"

**Cause**: TFLite model has incompatible broadcast operations.

**Solution**: This is a known issue with the embedder TFLite model when loading for inference. The model was successfully created but may have issues during `allocate_tensors()`. This doesn't affect the detector models.

**Workaround**: Use PyTorch inference for embedder, TFLite for detector.

---

## Runtime Errors

### Error: "Expected tensor of shape [1, 3, 256, 256] but got [1, 256, 256, 3]"

**Cause**: Image tensor has wrong channel order (HWC instead of CHW).

**Solution**: Transpose image before inference:

```python
# Wrong: HWC format
img_np = np.array(img_pil)  # [H, W, C]

# Correct: CHW format
img_np = np.transpose(img_np, (2, 0, 1))  # [C, H, W]
img_np = np.expand_dims(img_np, 0)  # [1, C, H, W]
```

---

### Error: "Input values must be in range [0, 1]"

**Cause**: Image not normalized correctly.

**Solution**: Normalize pixel values:

```python
# Wrong: Values in [0, 255]
img_np = np.array(img_pil)

# Correct: Values in [0, 1]
img_np = np.array(img_pil).astype(np.float32) / 255.0
```

---

### Error: "Message must be binary (0 or 1)"

**Cause**: Message tensor contains non-binary values.

**Solution**: Ensure message is binary:

```python
# Wrong: Float values
msg = torch.rand(1, 256)

# Correct: Binary values
msg = torch.randint(0, 2, (1, 256)).float()
```

---

## Weight Transfer Issues

### Issue: "Embedding weights not transferred"

**Symptom**: Output differs significantly from original model.

**Solution**: Verify weight transfer:

```python
# Check if weights match
original_weights = original_msg_proc.msg_embeddings.weight
tflite_weights = tflite_msg_proc.msg_embeddings.weight

diff = torch.abs(original_weights - tflite_weights).max()
print(f"Weight difference: {diff}")  # Should be 0.0
```

**Fix**: Call `transfer_weights()` explicitly:

```python
from tflite_msg_processor import transfer_weights
transfer_weights(original_msg_proc, tflite_msg_proc)
```

---

### Issue: "Message processor outputs differ"

**Symptom**: Message processor produces different results than original.

**Diagnosis**:
```python
# Test message processor equivalence
latents = torch.rand(1, 128, 32, 32)
msg = torch.randint(0, 2, (1, 256)).float()

out_orig = original_msg_proc(latents, msg)
out_tflite = tflite_msg_proc(latents, msg)

diff = torch.abs(out_orig - out_tflite).max()
print(f"Difference: {diff}")
```

**Expected**: Difference should be 0.0 (exactly equivalent)

**If not 0.0**: Check that:
1. Weights were transferred correctly
2. Same `msg_processor_type` is used
3. Same `msg_mult` is used
4. Spatial size matches

---

## Performance Issues

### Issue: Conversion takes too long (>10 minutes)

**Cause**: Large model or slow system.

**Solutions**:
1. **Normal for first conversion**: Subsequent conversions are faster
2. **Use faster machine**: Consider using a machine with more RAM/CPU
3. **Check system resources**: Close other applications

**Expected times**:
- VideoSeal 1.0 embedder: ~2 minutes
- ChunkySeal embedder: ~5-10 minutes (larger model)

---

### Issue: High memory usage during conversion

**Cause**: Model is large and conversion creates intermediate representations.

**Solutions**:
1. **Close other applications**: Free up RAM
2. **Use machine with more RAM**: 16GB+ recommended
3. **Convert on server**: Use cloud instance with sufficient memory

**Memory requirements**:
- VideoSeal 1.0: ~8GB RAM
- ChunkySeal: ~16GB RAM

---

### Issue: TFLite inference is slow

**Cause**: Not using hardware acceleration.

**Solutions**:

**1. Use GPU delegate (if available)**:
```python
interpreter = tf.lite.Interpreter(
    model_path='model.tflite',
    experimental_delegates=[tf.lite.experimental.load_delegate('libGPUDelegate.so')]
)
```

**2. Use NNAPI delegate (Android)**:
```python
interpreter = tf.lite.Interpreter(
    model_path='model.tflite',
    experimental_delegates=[tf.lite.experimental.load_delegate('libnnapi_delegate.so')]
)
```

**3. Use XNNPACK (CPU optimization)**:
```python
# XNNPACK is enabled by default in recent TFLite versions
interpreter = tf.lite.Interpreter(model_path='model.tflite')
```

---

## Model Loading Issues

### Error: "Model file not found"

**Cause**: Incorrect file path or model not created yet.

**Solution**:
1. Verify file exists:
   ```bash
   ls -lh videoseal_embedder_256.tflite
   ```

2. Use absolute path:
   ```python
   from pathlib import Path
   model_path = Path.home() / "work" / "models" / "videoseal_embedder_256.tflite"
   interpreter = tf.lite.Interpreter(model_path=str(model_path))
   ```

---

### Error: "Invalid model file"

**Cause**: Model file is corrupted or incomplete.

**Solution**:
1. Check file size:
   ```bash
   ls -lh videoseal_embedder_256.tflite
   # Should be ~90 MB for VideoSeal 1.0
   ```

2. Re-convert the model:
   ```python
   edge_model = ai_edge_torch.convert(embedder, (sample_img, sample_msg))
   edge_model.export('videoseal_embedder_256.tflite')
   ```

---

## Output Quality Issues

### Issue: Watermark too visible

**Cause**: Attenuation (JND) is disabled in TFLite version.

**Solutions**:

**1. Adjust watermark strength** (requires re-conversion):
```python
# Reduce watermark strength
model.embedder.scaling_w = 0.5  # Default is 1.0
```

**2. Apply post-processing attenuation**:
```python
# Apply simple attenuation
imgs_w = 0.9 * imgs + 0.1 * imgs_w  # Blend with original
```

**3. Use original PyTorch model** (with attenuation):
```python
# For high-quality watermarking
model = videoseal.load('videoseal')
imgs_w = model.embed(imgs, msgs=msgs, is_video=False)['imgs_w']
```

---

### Issue: Output differs from PyTorch

**Expected**: Small differences due to disabled attenuation.

**Diagnosis**:
```python
# Compare outputs
diff = torch.abs(output_pytorch - output_tflite)
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")
```

**Expected differences**:
- Max: 0.01 - 0.05 (due to attenuation)
- Mean: 0.001 - 0.01

**If differences are larger**:
1. Check input preprocessing (normalization, channel order)
2. Verify model version matches
3. Check for quantization effects (if using INT8)

---

## Debugging Tips

### 1. Enable Verbose Logging

```python
# PyTorch
msg_proc = TFLiteFriendlyMsgProcessor(...)
output = msg_proc(latents, msg, verbose=True)  # Prints shapes

# TFLite
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Inspect Model Details

```python
# TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

print("Inputs:")
for detail in interpreter.get_input_details():
    print(f"  {detail['name']}: {detail['shape']} ({detail['dtype']})")

print("Outputs:")
for detail in interpreter.get_output_details():
    print(f"  {detail['name']}: {detail['shape']} ({detail['dtype']})")
```

### 3. Test Components Separately

```python
# Test message processor alone
msg_proc = TFLiteFriendlyMsgProcessor(256, 256, 32)
latents = torch.rand(1, 128, 32, 32)
msg = torch.randint(0, 2, (1, 256)).float()
output = msg_proc(latents, msg)
print(f"Message processor output: {output.shape}")

# Test embedder wrapper
embedder = VideoSealEmbedderTFLite('videoseal', 256)
imgs = torch.rand(1, 3, 256, 256)
imgs_w = embedder(imgs, msg)
print(f"Embedder output: {imgs_w.shape}")
```

### 4. Validate Intermediate Outputs

```python
# Check intermediate values
with torch.no_grad():
    # Check encoder output
    latents = embedder.embedder.unet.downs[0](imgs)
    print(f"After first down: {latents.shape}")
    
    # Check message processor
    msg_aux = embedder.embedder.unet.msg_processor(latents, msg)
    print(f"After msg processor: {msg_aux.shape}")
```

---

## Getting Help

### Information to Provide

When reporting issues, include:

1. **Error message** (full traceback)
2. **Model variant** (videoseal, chunkyseal, etc.)
3. **Image size** (256, 512, etc.)
4. **Environment**:
   ```python
   import torch
   import tensorflow as tf
   print(f"PyTorch: {torch.__version__}")
   print(f"TensorFlow: {tf.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```
5. **Code snippet** (minimal reproducible example)

### Useful Commands

```bash
# Check file sizes
ls -lh *.tflite

# Check system resources
free -h  # RAM
df -h    # Disk space

# Check Python packages
pip list | grep -E "torch|tensorflow|ai-edge"
```

---

## Known Limitations

### 1. Attenuation Disabled

**Impact**: Watermark may be more visible  
**Status**: By design (boolean indexing not supported)  
**Workaround**: Apply attenuation in post-processing

### 2. Fixed Image Size

**Impact**: Each model works for one size only  
**Status**: By design (static graph requirement)  
**Workaround**: Create multiple models for different sizes

### 3. Embedder TFLite Model Loading

**Impact**: May fail during `allocate_tensors()`  
**Status**: Known issue with BROADCAST_TO operation  
**Workaround**: Use PyTorch for embedder, TFLite for detector

---

## References

- **Implementation Guide**: [implementation.md](./implementation.md)
- **Solution Design**: [solution-design.md](./solution-design.md)
- **Usage Examples**: [usage.md](./usage.md)
- **Problem Analysis**: [problem-analysis.md](./problem-analysis.md)

---

*Last Updated: January 4, 2026*

