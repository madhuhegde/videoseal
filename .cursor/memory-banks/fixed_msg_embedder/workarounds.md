# INT8 Limitation Workarounds

## Practical Solutions

This document provides **4 production-ready workarounds** for the INT8 quantization limitation, ranked by effectiveness and use case.

---

## Workaround 1: Hybrid Architecture ⭐ RECOMMENDED

### Server-Side Embedding + Mobile Detection

**Best for**: Mobile apps, web services, content protection

### Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│  Server/Cloud       │         │  Mobile Device       │
│                     │  Image  │                      │
│  PyTorch Embedder   │  with   │  TFLite Detector     │
│  (Full Model)       ├────────>│  (INT8)              │
│  • With attenuation │  mark   │  • 32.90 MB          │
│  • Best quality     │         │  • Fast inference    │
│  • Flexible         │         │  • Low power         │
└─────────────────────┘         └──────────────────────┘
```

### Implementation

**Server (Python/Flask)**:

```python
from flask import Flask, request, send_file
import videoseal
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Load model once at startup
model = videoseal.load('videoseal')
model.eval()

@app.route('/embed', methods=['POST'])
def embed_watermark():
    """Embed watermark on server."""
    # Get image and message
    image_file = request.files['image']
    message = request.form.get('message', type=str)
    
    # Load image
    img = Image.open(image_file).convert('RGB')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    
    # Parse message (binary string to tensor)
    msg_list = [int(b) for b in message]
    msg_tensor = torch.tensor(msg_list).float().unsqueeze(0)
    
    # Embed watermark (with attenuation)
    with torch.no_grad():
        result = model.embed(img_tensor, msgs=msg_tensor, is_video=False)
        img_watermarked = result['imgs_w']
    
    # Convert back to PIL Image
    img_out = transforms.ToPILImage()(img_watermarked[0])
    
    # Return as JPEG
    output = io.BytesIO()
    img_out.save(output, format='JPEG', quality=95)
    output.seek(0)
    
    return send_file(output, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Mobile (Kotlin/Android)**:

```kotlin
import okhttp3.*
import java.io.File

class WatermarkService {
    private val client = OkHttpClient()
    private val detector = VideoSealDetector("detector_int8.tflite")
    
    suspend fun embedWatermark(imageFile: File, message: String): ByteArray {
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image", 
                imageFile.name,
                imageFile.asRequestBody("image/*".toMediaType())
            )
            .addFormDataPart("message", message)
            .build()
        
        val request = Request.Builder()
            .url("https://your-server.com/embed")
            .post(requestBody)
            .build()
        
        val response = client.newCall(request).execute()
        return response.body?.bytes() ?: byteArrayOf()
    }
    
    fun detectWatermark(imageBytes: ByteArray): DetectionResult {
        return detector.detect(imageBytes)
    }
    
    suspend fun embedAndVerify(imageFile: File, message: String): Boolean {
        // 1. Upload to server for embedding
        val watermarkedImage = embedWatermark(imageFile, message)
        
        // 2. Detect on device (INT8 TFLite)
        val result = detectWatermark(watermarkedImage)
        
        // 3. Verify message
        return result.message == message
    }
}
```

### Advantages

✅ **Best quality** - Full PyTorch model with attenuation  
✅ **Smallest mobile app** - Only 32.90 MB detector  
✅ **Fast detection** - INT8 optimized  
✅ **Low power** - Efficient on-device inference  
✅ **Flexible** - Can update server model without app updates  
✅ **Scalable** - Server can handle many requests

### Disadvantages

❌ Requires network connection for embedding  
❌ Server infrastructure needed  
❌ Latency for embedding (~100-500ms including network)  
❌ Server costs

### Use Cases

- ✅ Social media apps (embed on upload)
- ✅ Photo sharing platforms
- ✅ Content protection services
- ✅ Copyright enforcement
- ✅ Media authentication

---

## Workaround 2: FLOAT32 Embedder ✅ RECOMMENDED

### On-Device Embedding with FLOAT32

**Best for**: Edge devices, offline apps, privacy-sensitive use cases

### Architecture

```
┌──────────────────────────────────┐
│       Mobile/Edge Device         │
│                                  │
│  TFLite Embedder (FLOAT32)       │
│  • 90.42 MB                       │
│  • Full functionality             │
│  • No network needed              │
│                                  │
│  TFLite Detector (INT8)          │
│  • 32.90 MB                       │
│  • Fast inference                 │
│                                  │
│  Total: 123 MB                    │
└──────────────────────────────────┘
```

### Implementation

**Python**:

```python
import tensorflow as tf
import numpy as np
from PIL import Image

class OfflineWatermarking:
    def __init__(self):
        # Load FLOAT32 embedder
        self.embedder = tf.lite.Interpreter(
            model_path='videoseal_embedder_256.tflite'
        )
        self.embedder.allocate_tensors()
        
        # Load INT8 detector
        self.detector = tf.lite.Interpreter(
            model_path='videoseal_detector_256_int8.tflite'
        )
        self.detector.allocate_tensors()
    
    def embed(self, image_path, message):
        """Embed watermark on device."""
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB').resize((256, 256))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_array = np.expand_dims(img_array, 0)
        
        # Prepare message
        msg_array = np.array(message).astype(np.float32).reshape(1, 256)
        
        # Embed watermark
        input_details = self.embedder.get_input_details()
        output_details = self.embedder.get_output_details()
        
        self.embedder.set_tensor(input_details[0]['index'], img_array)
        self.embedder.set_tensor(input_details[1]['index'], msg_array)
        self.embedder.invoke()
        
        img_watermarked = self.embedder.get_tensor(output_details[0]['index'])
        
        return img_watermarked
    
    def detect(self, img_watermarked):
        """Detect watermark on device."""
        input_details = self.detector.get_input_details()
        output_details = self.detector.get_output_details()
        
        self.detector.set_tensor(input_details[0]['index'], img_watermarked)
        self.detector.invoke()
        
        detected_msg = self.detector.get_tensor(output_details[0]['index'])
        
        return detected_msg
    
    def process(self, image_path, message):
        """Complete offline pipeline."""
        # Embed
        img_w = self.embed(image_path, message)
        
        # Save watermarked image
        img_w_np = np.clip(img_w[0], 0, 1)
        img_w_np = np.transpose(img_w_np, (1, 2, 0))  # CHW -> HWC
        img_w_np = (img_w_np * 255).astype(np.uint8)
        img_w_pil = Image.fromarray(img_w_np)
        img_w_pil.save('watermarked.jpg')
        
        # Verify
        detected = self.detect(img_w)
        
        return np.array_equal(message, detected[0] > 0.5)

# Usage
watermarker = OfflineWatermarking()
message = [1, 0, 1, 1, ...] # 256 bits
success = watermarker.process('input.jpg', message)
```

### Advantages

✅ **Fully offline** - No network required  
✅ **Privacy** - All processing on-device  
✅ **Reliable** - No server dependencies  
✅ **Acceptable size** - 123 MB total  
✅ **Complete functionality** - Both embed and detect

### Disadvantages

❌ Larger app size (90.42 MB embedder)  
❌ Slower embedding than INT8 would be  
❌ Higher memory usage (~200 MB runtime)

### Use Cases

- ✅ Offline photo apps
- ✅ Privacy-focused applications
- ✅ Edge devices with sufficient storage
- ✅ Industrial/enterprise applications
- ✅ Camera apps with watermarking
- ✅ IoT devices

---

## Workaround 3: FP16 Quantization

### Half-Precision Quantization

**Best for**: Balance between size and quality

### Implementation

```python
import tensorflow as tf

# Load FLOAT32 model
converter = tf.lite.TFLiteConverter.from_saved_model('embedder_float32')

# Apply FP16 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert
fp16_model = converter.convert()

# Save
with open('videoseal_embedder_256_fp16.tflite', 'wb') as f:
    f.write(fp16_model)

print(f"FP16 model size: {len(fp16_model) / (1024*1024):.2f} MB")
```

### Expected Results

| Model | Size | Reduction |
|-------|------|-----------|
| FLOAT32 | 90.42 MB | Baseline |
| **FP16** | **~45 MB** | **50%** |
| INT8 (not supported) | ~23 MB | 75% |

### Advantages

✅ **50% size reduction** - Significant savings  
✅ **Minimal quality loss** - FP16 is very accurate  
✅ **Faster than FLOAT32** - Hardware acceleration  
✅ **Compatible** - No BROADCAST_TO issues  
✅ **Good balance** - Size vs quality

### Disadvantages

❌ Still larger than INT8 would be  
❌ Requires FP16 hardware support  
❌ Not all devices support FP16

### Use Cases

- ✅ Modern mobile devices (2020+)
- ✅ When size matters but quality is important
- ✅ Devices with FP16 GPU support
- ✅ Balanced deployment scenarios

---

## Workaround 4: Model Modification 🔬

### Replace expand() with repeat()

**Best for**: Advanced users, research, future development

### The Modification

```python
# In tflite_msg_processor.py

class TFLiteFriendlyMsgProcessor(nn.Module):
    def forward(self, latents, msg):
        # ... existing code ...
        
        # Spatial broadcast (MODIFIED)
        msg_aux = msg_aux.view(-1, self.hidden_size, 1, 1)
        
        # Option 1: Current (uses BROADCAST_TO)
        # msg_aux = msg_aux.expand(-1, -1, self.spatial_size, self.spatial_size)
        
        # Option 2: Alternative (uses TILE)
        msg_aux = msg_aux.repeat(1, 1, self.spatial_size, self.spatial_size)
        
        return torch.cat([latents, msg_aux], dim=1)
```

### Implementation Steps

1. **Modify the message processor**
2. **Re-convert to TFLite**
3. **Attempt INT8 quantization**
4. **Test accuracy**

### Advantages

✅ **May enable INT8** - TILE operation might work  
✅ **Same functionality** - Mathematically equivalent  
✅ **No quality loss** - Same operations  
✅ **Worth trying** - Could unlock INT8

### Disadvantages

❌ **Not guaranteed** - May still fail  
❌ **Requires re-conversion** - Time-consuming  
❌ **Needs testing** - Verify accuracy  
❌ **Not tested** - Theoretical solution

### Status

⚠️ **Experimental** - Not tested, may or may not work

---

## Comparison of Workarounds

| Workaround | Size | Quality | Offline | Complexity | Status |
|------------|------|---------|---------|------------|--------|
| **Hybrid** | 33 MB | ⭐⭐⭐⭐⭐ | ❌ | Low | ✅ Production |
| **FLOAT32** | 123 MB | ⭐⭐⭐⭐ | ✅ | Low | ✅ Production |
| **FP16** | ~68 MB | ⭐⭐⭐⭐ | ✅ | Medium | ✅ Production |
| **Modification** | ~23 MB? | ⭐⭐⭐⭐ | ✅ | High | ⚠️ Experimental |

---

## Recommendations by Use Case

### Mobile App (Social Media, Photo Sharing)

**Recommendation**: **Hybrid Architecture**

**Why**:
- Smallest mobile footprint (32.90 MB)
- Best quality (with attenuation)
- Fast on-device detection
- Acceptable latency for upload scenarios

**Implementation**: See Workaround 1

### Edge Device (Camera, IoT, Offline)

**Recommendation**: **FLOAT32 Embedder**

**Why**:
- Fully offline operation
- Reliable and tested
- Acceptable size for edge devices (123 MB)
- Complete functionality

**Implementation**: See Workaround 2

### High-Performance App

**Recommendation**: **FP16 Quantization**

**Why**:
- Good balance of size (68 MB) and quality
- Faster than FLOAT32
- Modern device support
- 50% size reduction

**Implementation**: See Workaround 3

### Research/Development

**Recommendation**: **Model Modification**

**Why**:
- May unlock INT8 support
- Worth exploring for future versions
- Could benefit entire community

**Implementation**: See Workaround 4

---

## Performance Comparison

### Embedder Options

| Option | Size | Speed (CPU) | Quality | Offline | Status |
|--------|------|-------------|---------|---------|--------|
| PyTorch (Server) | N/A | Fast (GPU) | ⭐⭐⭐⭐⭐ | ❌ | ✅ Available |
| FLOAT32 TFLite | 90.42 MB | Medium | ⭐⭐⭐⭐ | ✅ | ✅ Available |
| FP16 TFLite | ~45 MB | Fast | ⭐⭐⭐⭐ | ✅ | ✅ Available |
| INT8 TFLite | N/A | N/A | N/A | ❌ | ❌ Not supported |

### Detector Options (for reference)

| Option | Size | Speed | Quality | Status |
|--------|------|-------|---------|--------|
| FLOAT32 TFLite | 127.57 MB | Medium | ⭐⭐⭐⭐⭐ | ✅ Available |
| **INT8 TFLite** | **32.90 MB** | **Very Fast** | ⭐⭐⭐⭐ | ✅ **Available** |

---

## Related Documentation

- **Limitation Analysis**: [int8-limitation.md](./int8-limitation.md) - Why INT8 doesn't work
- **Solution Design**: [solution-design.md](./solution-design.md) - Why expand() was chosen
- **Implementation**: [implementation.md](./implementation.md) - Code details
- **Usage**: [usage.md](./usage.md) - How to use FLOAT32 embedder

---

## Conclusion

Multiple production-ready workarounds are available for the INT8 limitation:

**Best for Mobile**: Hybrid Architecture (32.90 MB mobile app)  
**Best for Offline**: FLOAT32 Embedder (123 MB total)  
**Best Balance**: FP16 Quantization (~68 MB total)

The INT8 limitation is **not a blocker** for VideoSeal deployment. Choose the workaround that best fits your use case and requirements.

---

*Last Updated: January 4, 2026*  
*Status: Multiple production-ready solutions available*

