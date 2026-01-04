# Image Watermarking with VideoSeal - Usage Guide

## Overview
The `image_watermark.py` script provides a simple command-line interface to embed and detect watermarks in images using the VideoSeal v1.0 model.

## Setup

### Environment
Use the `uvq_env` environment with all dependencies installed:
```bash
source /home/madhuhegde/work/UVQ/uvq_env/bin/activate
```

### Model Weights
The VideoSeal v1.0 model weights are automatically downloaded on first use:
- **URL**: https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth
- **Cached at**: `ckpts/videoseal_y_256b_img.pth` (218 MB)
- **Auto-download**: Happens automatically when you first run the script

## Usage

### 1. Embed Watermark
Add an invisible 256-bit watermark to an image:

```bash
cd /home/madhuhegde/work/videoseal/videoseal
source /home/madhuhegde/work/UVQ/uvq_env/bin/activate

python image_watermark.py --embed --input image.jpg --output watermarked.jpg
```

**Output:**
- Watermarked image saved to specified output path
- Displays embedded message (256-bit binary vector)
- Original image quality preserved (imperceptible watermark)

### 2. Detect Watermark
Extract and verify the watermark from a watermarked image:

```bash
python image_watermark.py --detect --input watermarked.jpg
```

**Output:**
- Detected message (256-bit binary vector)
- Detection confidence score
- Detection mask score

### 3. Use GPU (Optional)
For faster processing with CUDA-enabled GPU:

```bash
python image_watermark.py --embed --input image.jpg --output watermarked.jpg --device cuda
```

## Example Session

```bash
# Navigate to videoseal directory
cd /home/madhuhegde/work/videoseal/videoseal

# Activate environment
source /home/madhuhegde/work/UVQ/uvq_env/bin/activate

# Embed watermark
python image_watermark.py --embed --input assets/imgs/1.jpg --output outputs/1_watermarked.jpg

# Detect watermark
python image_watermark.py --detect --input outputs/1_watermarked.jpg
```

## Test Results

Successfully tested with `assets/imgs/1.jpg` (1080x1904 image):

**Embedding:**
```
✓ Watermarked image saved to: outputs/1_watermarked.jpg
✓ Embedded message (256 bits): [0 0 0 0 1 1 0 0 1 0 1 1 1 1 1 1 ...]
  Message shape: torch.Size([1, 256])
```

**Detection:**
```
✓ Watermark detected!
✓ Detected message (256 bits): [0. 0. 0. 0. 1. 1. 0. 0. 1. 0. 1. 1. 1. 1. 1. 1. ...]
  Message shape: torch.Size([256])
  Detection confidence: 9.4372
  Detection mask score: -0.0507
```

✅ **Result**: Embedded and detected messages match perfectly!

## Features

- ✅ **Automatic model download**: Downloads VideoSeal v1.0 weights on first use
- ✅ **Invisible watermarks**: 256-bit capacity with imperceptible changes
- ✅ **High accuracy**: Successfully detects watermarks from watermarked images
- ✅ **Simple CLI**: Easy-to-use command-line interface
- ✅ **GPU support**: Optional CUDA acceleration
- ✅ **Robust**: Works with various image sizes and formats

## Model Information

- **Model**: VideoSeal v1.0 (256-bit, stable)
- **Capacity**: 256 bits per image
- **Method**: Additive blending with JND-based attenuation
- **Architecture**: UNet embedder + ConvNeXt extractor
- **Paper**: [VideoSeal: Open and Efficient Video Watermarking](https://arxiv.org/abs/2412.09492)

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- Any format supported by PIL/Pillow

## Notes

- The watermark is a random 256-bit binary vector generated during embedding
- The same watermark can be detected from the watermarked image
- Detection confidence > 5.0 typically indicates successful watermark detection
- The watermark is robust to common image operations (JPEG compression, resizing, etc.)

## Troubleshooting

### Missing Dependencies
If you get import errors, ensure all dependencies are installed:
```bash
pip install omegaconf einops timm==0.9.16 tqdm
```

### Model Download Issues
If automatic download fails, manually download the model:
```bash
wget https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth -P ckpts/
mv ckpts/y_256b_img.pth ckpts/videoseal_y_256b_img.pth
```

### CUDA Not Available
If GPU is requested but not available, the script automatically falls back to CPU.

## Help

For full command-line options:
```bash
python image_watermark.py --help
```

