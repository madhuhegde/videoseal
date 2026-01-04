# ChunkySeal TFLite Conversion Guide

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Conversion Process](#conversion-process)
- [Conversion Scripts](#conversion-scripts)
- [Model Output](#model-output)
- [Verification](#verification)
- [Troubleshooting Conversion](#troubleshooting-conversion)

## Overview

This guide covers the process of converting ChunkySeal PyTorch models to TensorFlow Lite format using Google's AI Edge Torch framework.

**Location**: `~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/chunkyseal/`

## Prerequisites

### System Requirements

| Resource | FLOAT32 | INT8 |
|----------|---------|------|
| RAM | 15-20 GB | 15-20 GB |
| Disk Space | 3+ GB | 3+ GB |
| Time | 5-10 min | 5-10 min |

**Note**: Both conversions require similar RAM during the process. The size reduction only applies to the final model file.

### Software Requirements

```bash
# Python environment
Python 3.11+

# Core packages
ai-edge-torch
torch >= 2.3.1
tensorflow
videoseal (from source)

# Environment
micromamba or conda
```

### Setup Environment

```bash
# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate local_tf_env

# Verify packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import ai_edge_torch; print('AI Edge Torch: OK')"
```

## Conversion Process

### Step 1: Prepare Checkpoint

The ChunkySeal checkpoint is large (13 GB full, 2.95 GB detector-only).

**Option A: Use Detector-Only Checkpoint (Recommended)**

```bash
# Location: /mnt/shared/shared/ChunkySeal/chunkyseal_detector_only.pth
# Size: 2.95 GB
# Contains: Only detector weights (no embedder, no training artifacts)
```

**Option B: Extract from Full Checkpoint**

```bash
cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/chunkyseal

python extract_detector_weights.py
# Input: /mnt/shared/shared/ChunkySeal/chunkyseal_checkpoint.pth (13 GB)
# Output: /mnt/shared/shared/ChunkySeal/chunkyseal_detector_only.pth (2.95 GB)
```

### Step 2: Convert to FLOAT32

```bash
cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/chunkyseal

# Set Python path
export PYTHONPATH=/home/madhuhegde/work/videoseal/videoseal:$PYTHONPATH

# Run conversion
python convert_detector_to_tflite.py --output_dir ./chunkyseal_tflite
```

**Output**:
- File: `chunkyseal_detector_chunkyseal_256.tflite`
- Size: 2.95 GB
- Time: ~5-10 minutes

### Step 3: Convert to INT8 (Recommended)

```bash
cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/chunkyseal

# Set Python path
export PYTHONPATH=/home/madhuhegde/work/videoseal/videoseal:$PYTHONPATH

# Run INT8 conversion
python convert_detector_to_tflite.py --quantize int8 --output_dir ./chunkyseal_tflite
```

**Output**:
- File: `chunkyseal_detector_chunkyseal_256_int8.tflite`
- Size: 960 MB (67.5% reduction)
- Time: ~5-10 minutes

### Step 4: Verify Conversion

```bash
# Test FLOAT32 model
python verify_detector_tflite.py \
    --tflite_model ./chunkyseal_tflite/chunkyseal_detector_chunkyseal_256.tflite

# Test INT8 model
python verify_detector_tflite.py \
    --tflite_model ./chunkyseal_tflite/chunkyseal_detector_chunkyseal_256_int8.tflite
```

## Conversion Scripts

### Main Conversion Script

**File**: `convert_detector_to_tflite.py`

**Usage**:
```bash
python convert_detector_to_tflite.py [OPTIONS]

Options:
  --output_dir OUTPUT_DIR     Output directory for TFLite models
  --model_name MODEL_NAME     Model variant (default: chunkyseal)
  --image_size IMAGE_SIZE     Input image size (default: 256)
  --quantize {int8,fp16}      Quantization type (default: None/FLOAT32)
  --no_simple                 Use dynamic version instead of simple
```

**Examples**:
```bash
# FLOAT32 (default)
python convert_detector_to_tflite.py --output_dir ./tflite_models

# INT8 quantization
python convert_detector_to_tflite.py --quantize int8 --output_dir ./tflite_models

# FP16 quantization
python convert_detector_to_tflite.py --quantize fp16 --output_dir ./tflite_models

# Custom image size
python convert_detector_to_tflite.py --image_size 512 --output_dir ./tflite_models
```

### Conversion Flow

```python
# 1. Load PyTorch model
model = create_detector(model_name='chunkyseal')

# 2. Create sample input
sample_input = torch.randn(1, 3, 256, 256)

# 3. Convert to TFLite
if quantize == 'int8':
    # INT8 quantization
    edge_model = ai_edge_torch.convert(
        model,
        sample_input,
        quant_config=quant_recipes.dynamic_qi8_recipe()
    )
elif quantize == 'fp16':
    # FP16 quantization
    edge_model = ai_edge_torch.convert(
        model,
        sample_input,
        quant_config=quant_recipes.fp16_recipe()
    )
else:
    # FLOAT32 (no quantization)
    edge_model = ai_edge_torch.convert(model, sample_input)

# 4. Export to TFLite
edge_model.export(output_path)
```

### Model Wrapper

**File**: `chunkyseal_models.py`

**Purpose**: Wraps PyTorch model for TFLite conversion

```python
class ChunkySealDetectorWrapper(nn.Module):
    """
    Wrapper for ChunkySeal Detector optimized for TFLite conversion.
    """
    
    def __init__(self, model_name="chunkyseal", eval_mode=True):
        super().__init__()
        
        # Load from detector-only checkpoint if available
        checkpoint_paths = [
            Path("/mnt/shared/shared/ChunkySeal/chunkyseal_detector_only.pth"),
            Path.home() / ".cache" / "videoseal" / "chunkyseal_detector_only.pth",
        ]
        
        checkpoint_found = None
        for checkpoint_path in checkpoint_paths:
            if checkpoint_path.exists():
                checkpoint_found = checkpoint_path
                break
        
        if checkpoint_found:
            self.model = setup_model_from_checkpoint(str(checkpoint_found))
        else:
            self.model = videoseal.load(model_name)
        
        if eval_mode:
            self.eval()
    
    def forward(self, imgs):
        """Forward pass for watermark detection."""
        # Preprocess
        imgs = imgs * 2 - 1  # [0, 1] → [-1, 1]
        
        # Extract features
        outputs = self.model.detect(imgs, is_video=False)
        preds = outputs["preds"]
        
        # Global average pooling
        preds = preds.mean(dim=[2, 3])  # (B, 1025, H, W) → (B, 1025)
        
        return preds
```

## Model Output

### FLOAT32 Model

```
File: chunkyseal_detector_chunkyseal_256.tflite
Size: 2951.70 MB (2.95 GB)
Quantization: FLOAT32
Input: (1, 3, 256, 256) float32
Output: (1, 1025) float32
```

### INT8 Model

```
File: chunkyseal_detector_chunkyseal_256_int8.tflite
Size: 960.00 MB
Quantization: INT8 (dynamic)
Input: (1, 3, 256, 256) float32
Output: (1, 1025) float32
```

**Note**: Even INT8 models use FLOAT32 for inputs/outputs for compatibility.

### Model Metadata

```python
# Conversion output
INFO: Estimated count of arithmetic ops: 1229.241 G ops
INFO: Equivalently: 614.621 G MACs
```

## Verification

### Verification Script

**File**: `verify_detector_tflite.py`

**Usage**:
```bash
python verify_detector_tflite.py \
    --tflite_model ./chunkyseal_tflite/chunkyseal_detector_chunkyseal_256_int8.tflite \
    --image_path ../../../videoseal/assets/imgs/1.jpg
```

### Verification Checks

1. **Model Loading**: Verify TFLite model loads correctly
2. **Shape Validation**: Check input/output shapes
3. **Inference Test**: Run sample inference
4. **Output Range**: Verify output is reasonable
5. **Consistency**: Multiple runs produce same results

### Example Verification Output

```
Loading TFLite model...
✓ Model loaded successfully

Model Information:
  Input shape: (1, 3, 256, 256)
  Output shape: (1, 1025)
  Model size: 960.00 MB

Running inference...
✓ Inference complete

Results:
  Confidence: -0.000643
  Message (first 32 bits): [1 1 0 1 0 1 0 0 ...]
  Total 1 bits: 506/1024

✓ Verification complete
```

## Troubleshooting Conversion

### Out of Memory (OOM)

**Symptom**: Process killed with exit code 137

**Solutions**:
1. Increase VM RAM to 20+ GB
2. Close other applications
3. Use detector-only checkpoint (2.95 GB vs 13 GB)
4. Add swap space

```bash
# Check available RAM
free -h

# Increase VM RAM in virtualization settings
# Then restart VM
```

### Checkpoint Not Found

**Symptom**: `FileNotFoundError: Model card 'chunkyseal' not found`

**Solutions**:
1. Use detector-only checkpoint explicitly
2. Set correct working directory
3. Update checkpoint path in script

```python
# In chunkyseal_models.py
checkpoint_paths = [
    Path("/mnt/shared/shared/ChunkySeal/chunkyseal_detector_only.pth"),
    # Add your checkpoint path here
]
```

### Module Not Found

**Symptom**: `ModuleNotFoundError: No module named 'videoseal'`

**Solution**: Add videoseal to Python path

```bash
export PYTHONPATH=/home/madhuhegde/work/videoseal/videoseal:$PYTHONPATH
```

### Conversion Hangs

**Symptom**: Conversion process appears stuck

**Possible Causes**:
1. Insufficient RAM (swapping to disk)
2. Large model size
3. Complex operations

**Solutions**:
1. Monitor system resources: `htop`
2. Be patient - conversion takes 5-10 minutes
3. Ensure adequate RAM (15-20 GB)

### Verification Fails

**Symptom**: Verification script reports errors

**Checks**:
1. Model file exists and is complete
2. File size matches expected size
3. TensorFlow/TFLite is installed correctly
4. Input image exists and is valid

## Conversion Best Practices

### 1. Use Detector-Only Checkpoint

✅ **Recommended**: 2.95 GB detector-only checkpoint  
❌ **Avoid**: 13 GB full checkpoint (includes training artifacts)

### 2. Adequate RAM

Ensure 20+ GB RAM for smooth conversion:
- FLOAT32: 15-20 GB peak
- INT8: 15-20 GB peak (similar to FLOAT32)

### 3. Verify After Conversion

Always run verification script after conversion:
```bash
python verify_detector_tflite.py --tflite_model <model_path>
```

### 4. Test Both Quantizations

Convert and test both FLOAT32 and INT8:
- FLOAT32: Reference accuracy
- INT8: Production deployment

### 5. Monitor Conversion

Watch conversion progress:
```bash
# In another terminal
watch -n 1 'ps aux | grep python'
htop
```

## See Also

- [Quantization Guide](./quantization.md)
- [Implementation Details](./implementation.md)
- [Troubleshooting](./troubleshooting.md)
- [Conversion Summary](../../chunky_tflite/CONVERSION_SUMMARY.md)

