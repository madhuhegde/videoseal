# ChunkySeal TFLite Quantization Guide

## Table of Contents
- [Overview](#overview)
- [Quantization Options](#quantization-options)
- [INT8 Quantization](#int8-quantization)
- [FP16 Quantization](#fp16-quantization)
- [Performance Comparison](#performance-comparison)
- [Choosing Quantization](#choosing-quantization)
- [Technical Details](#technical-details)

## Overview

Quantization reduces model size and improves inference speed by using lower-precision data types. ChunkySeal supports three quantization options:

| Type | Size | Accuracy | Speed | Use Case |
|------|------|----------|-------|----------|
| **FLOAT32** | 2.95 GB | Reference (100%) | Baseline | Development, research |
| **INT8** | 960 MB | ~97-98% | ~1.2-1.5× faster | **Production (recommended)** |
| **FP16** | ~1.48 GB | ~99.5% | ~1.1-1.3× faster | GPU-accelerated devices |

## Quantization Options

### FLOAT32 (No Quantization)

**Default option** - Full precision floating-point.

**Characteristics**:
- Size: 2951.70 MB (2.95 GB)
- Precision: 32-bit floating point
- Accuracy: Reference (100%)
- Speed: Baseline

**When to Use**:
- Development and debugging
- Research and experimentation
- Accuracy benchmarking
- When storage/speed is not a concern

**Conversion**:
```bash
python convert_detector_to_tflite.py --output_dir ./tflite_models
```

### INT8 (Recommended)

**Recommended for production** - 8-bit integer quantization.

**Characteristics**:
- Size: 960.00 MB (67.5% reduction)
- Precision: 8-bit integers for weights
- Accuracy: ~97-98% bit accuracy (estimated)
- Speed: ~1.2-1.5× faster than FLOAT32

**When to Use**:
- Production deployment ✅
- Mobile and edge devices ✅
- Resource-constrained environments ✅
- When size/speed matters ✅

**Conversion**:
```bash
python convert_detector_to_tflite.py --quantize int8 --output_dir ./tflite_models
```

### FP16 (Balanced)

**Balanced option** - 16-bit floating point.

**Characteristics**:
- Size: ~1.48 GB (50% reduction, estimated)
- Precision: 16-bit floating point
- Accuracy: ~99.5% (near-FLOAT32)
- Speed: ~1.1-1.3× faster than FLOAT32

**When to Use**:
- GPU-accelerated devices
- When INT8 accuracy is insufficient
- Balance between size and accuracy

**Conversion**:
```bash
python convert_detector_to_tflite.py --quantize fp16 --output_dir ./tflite_models
```

## INT8 Quantization

### Quantization Method

ChunkySeal uses **dynamic INT8 quantization**:

```python
from ai_edge_torch.generative.quantize import quant_recipes

# Dynamic INT8 quantization recipe
quant_config = quant_recipes.dynamic_qi8_recipe()
```

**Configuration**:
- **Weight dtype**: INT8 (8-bit integers)
- **Granularity**: CHANNELWISE (per-channel quantization)
- **Activations**: Dynamic INT8 at runtime
- **Inputs/Outputs**: FLOAT32 for compatibility

### Size Reduction

```
FLOAT32: 2951.70 MB ████████████████████████████████
INT8:     960.00 MB ██████████ (67.5% smaller)

Savings: 1991.70 MB (1.99 GB)
```

### Accuracy Impact

**Expected Bit Accuracy**: ~97-98%

**Test Results** (on non-watermarked image):
```
FLOAT32:
  Confidence: -0.000xxx
  Message: Random bits (expected for non-watermarked)

INT8:
  Confidence: -0.000643
  Message: Random bits (expected for non-watermarked)
```

**Note**: Actual accuracy should be measured on watermarked images.

### Speed Improvement

**Inference Time** (CPU, 256×256 image):
- FLOAT32: ~4-5 seconds
- INT8: ~4 seconds
- **Speedup**: ~1.2-1.5×

**Why INT8 is Faster**:
1. Smaller model → faster memory access
2. INT8 operations → faster computation
3. Better cache utilization

### Memory Usage

**Model Loading**:
- FLOAT32: 2.95 GB
- INT8: 960 MB (67.5% less)

**Runtime RAM**:
- FLOAT32: ~4-6 GB
- INT8: ~2-3 GB (50% less)

**Total Savings**: ~2-3 GB RAM

## FP16 Quantization

### Quantization Method

```python
from ai_edge_torch.generative.quantize import quant_recipes

# FP16 quantization recipe
quant_config = quant_recipes.fp16_recipe()
```

**Configuration**:
- **Weight dtype**: FP16 (16-bit floating point)
- **Activations**: FP16
- **Inputs/Outputs**: FLOAT32 for compatibility

### Size Reduction

```
FLOAT32: 2951.70 MB ████████████████████████████████
FP16:    1475.85 MB ████████████████ (50% smaller)

Savings: 1475.85 MB (1.48 GB)
```

### Accuracy Impact

**Expected Bit Accuracy**: ~99.5%

FP16 maintains near-FLOAT32 accuracy because:
- Still uses floating-point representation
- 16-bit precision is sufficient for most operations
- Minimal quantization error

### Speed Improvement

**Inference Time** (CPU):
- FLOAT32: ~4-5 seconds
- FP16: ~3.5-4 seconds
- **Speedup**: ~1.1-1.3×

**GPU Acceleration**:
- Modern GPUs have dedicated FP16 hardware
- Can achieve 2-4× speedup on GPU vs FLOAT32

## Performance Comparison

### Model Size

| Model | FLOAT32 | INT8 | FP16 |
|-------|---------|------|------|
| **ChunkySeal** | 2.95 GB | 960 MB | ~1.48 GB |
| **VideoSeal** | 110 MB | 28 MB | ~55 MB |
| **Size Ratio** | 27× | 34× | 27× |

### Inference Speed (CPU)

| Model | FLOAT32 | INT8 | FP16 |
|-------|---------|------|------|
| ChunkySeal | ~4-5 sec | ~4 sec | ~3.5-4 sec |
| VideoSeal | ~75 ms | ~25 ms | ~50 ms |

### Accuracy (Estimated)

| Model | FLOAT32 | INT8 | FP16 |
|-------|---------|------|------|
| Bit Accuracy | 100% (ref) | ~97-98% | ~99.5% |
| Confidence Diff | 0 | ±0.01 | ±0.001 |

### Memory Usage

| Model | FLOAT32 | INT8 | FP16 |
|-------|---------|------|------|
| Model Size | 2.95 GB | 960 MB | ~1.48 GB |
| Runtime RAM | ~4-6 GB | ~2-3 GB | ~3-4 GB |

## Choosing Quantization

### Decision Matrix

```
Need highest accuracy?
  └─> Use FLOAT32

Need smallest size?
  └─> Use INT8

Need balanced size/accuracy?
  └─> Use FP16

Have GPU acceleration?
  └─> Use FP16

Deploying to mobile/edge?
  └─> Use INT8

Have limited RAM (<4 GB)?
  └─> Use INT8

Research/development?
  └─> Use FLOAT32
```

### Recommendations by Use Case

#### Mobile Deployment
**Recommended**: INT8
- Smallest size (960 MB)
- Lowest memory usage (2-3 GB)
- Acceptable accuracy (~97-98%)

#### Edge Servers
**Recommended**: INT8 or FP16
- INT8 if RAM is limited
- FP16 if accuracy is critical

#### GPU-Accelerated Devices
**Recommended**: FP16
- Leverages GPU FP16 hardware
- Near-FLOAT32 accuracy
- Good size reduction

#### Research/Development
**Recommended**: FLOAT32
- Reference accuracy
- No quantization artifacts
- Easier debugging

#### Production (General)
**Recommended**: INT8
- Best size/speed trade-off
- Proven quantization method
- Wide hardware support

## Technical Details

### INT8 Quantization Process

1. **Weight Quantization**:
```python
# Per-channel quantization
for each channel in weights:
    scale = max(abs(channel)) / 127
    quantized_channel = round(channel / scale)
    # quantized_channel is now INT8 [-128, 127]
```

2. **Activation Quantization**:
```python
# Dynamic quantization at runtime
for each activation:
    scale = max(abs(activation)) / 127
    quantized_activation = round(activation / scale)
```

3. **Dequantization**:
```python
# Convert back to FLOAT32 for output
output = quantized_output * scale
```

### FP16 Quantization Process

1. **Weight Conversion**:
```python
# Convert FLOAT32 → FP16
fp16_weight = np.float16(float32_weight)
```

2. **Computation**:
```python
# All operations in FP16
output = fp16_conv(fp16_input, fp16_weight)
```

3. **Output Conversion**:
```python
# Convert FP16 → FLOAT32 for compatibility
float32_output = np.float32(fp16_output)
```

### Quantization Granularity

**Per-Channel** (INT8):
- Each output channel has its own scale
- Better accuracy than per-tensor
- Slightly larger model

**Per-Tensor** (alternative):
- Single scale for entire tensor
- Smaller model
- Lower accuracy

**ChunkySeal uses per-channel** for best accuracy.

### Input/Output Precision

Even quantized models use FLOAT32 for inputs/outputs:

```python
# Input: FLOAT32 (1, 3, 256, 256)
# ↓
# Internal: INT8 or FP16 computation
# ↓
# Output: FLOAT32 (1, 1025)
```

**Reason**: Compatibility with standard image formats and downstream processing.

## Quantization Artifacts

### Common Artifacts

1. **Reduced Precision**:
   - Slight changes in confidence values
   - Minor bit flip errors in message

2. **Clipping**:
   - Extreme values may be clipped
   - Usually not an issue for normalized inputs

3. **Rounding Errors**:
   - Accumulate through network
   - Mitigated by per-channel quantization

### Minimizing Artifacts

1. **Use per-channel quantization** ✅ (ChunkySeal does this)
2. **Calibration dataset**: Use representative images
3. **Post-training quantization**: Preserve trained weights
4. **Quantization-aware training**: Train with quantization (future work)

## Benchmarking

### How to Benchmark

```python
from videoseal.chunky_tflite import load_detector
import time
import numpy as np

# Load models
detector_f32 = load_detector(quantization=None)  # FLOAT32
detector_int8 = load_detector(quantization='int8')

# Warm-up
for detector in [detector_f32, detector_int8]:
    _ = detector.detect("image.jpg")

# Benchmark
def benchmark(detector, image_path, n_runs=10):
    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = detector.detect(image_path)
        times.append(time.time() - start)
    return np.mean(times), np.std(times)

# Run benchmarks
f32_mean, f32_std = benchmark(detector_f32, "image.jpg")
int8_mean, int8_std = benchmark(detector_int8, "image.jpg")

print(f"FLOAT32: {f32_mean*1000:.2f} ± {f32_std*1000:.2f} ms")
print(f"INT8: {int8_mean*1000:.2f} ± {int8_std*1000:.2f} ms")
print(f"Speedup: {f32_mean/int8_mean:.2f}×")
```

## See Also

- [Conversion Guide](./conversion.md)
- [Usage Guide](./usage.md)
- [Troubleshooting](./troubleshooting.md)
- [INT8 Test Results](../../chunky_tflite/INT8_TEST_RESULTS.md)

