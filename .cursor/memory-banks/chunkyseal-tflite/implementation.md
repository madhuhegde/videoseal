# ChunkySeal TFLite Implementation Details

## Table of Contents
- [Overview](#overview)
- [Package Structure](#package-structure)
- [Core Classes](#core-classes)
- [API Design](#api-design)
- [Implementation Files](#implementation-files)
- [Code Examples](#code-examples)

## Overview

The ChunkySeal TFLite implementation provides a complete Python API for running ChunkySeal watermark detection on edge devices using TensorFlow Lite.

**Location**: `/home/madhuhegde/work/videoseal/videoseal/chunky_tflite/`

## Package Structure

```
chunky_tflite/
├── __init__.py                      # Package exports
├── detector.py                      # Main detector class
├── example.py                       # Usage examples
├── compare_pytorch_tflite.py        # Benchmarking script
├── test_int8.py                     # INT8 testing
├── README.md                        # User documentation
├── ARCHITECTURE_ANALYSIS.md         # Architecture details
├── IMPLEMENTATION_SUMMARY.md        # Implementation summary
├── INT8_TEST_RESULTS.md            # Test results
├── MODULE_USAGE_ANALYSIS.md        # Module usage
└── YAML_CONFIRMATION.md            # YAML analysis
```

## Core Classes

### ChunkySealDetectorTFLite

Main class for TFLite-based watermark detection.

**File**: `detector.py` (404 lines)

**Purpose**: Provides a high-level API for loading and running ChunkySeal TFLite models.

**Key Features**:
- Automatic quantization detection
- Multiple input formats (PIL, numpy, path)
- Batch processing support
- Message extraction in multiple formats
- Watermark verification

**Class Definition**:
```python
class ChunkySealDetectorTFLite:
    """
    TFLite-based ChunkySeal Detector for high-capacity watermark detection.
    
    Attributes:
        model_path: Path to TFLite model
        image_size: Input image size (default: 256)
        quantization: Detected quantization type (FLOAT32/INT8/FP16)
        interpreter: TFLite interpreter instance
        MESSAGE_LENGTH: 1024 (class constant)
    """
    
    MESSAGE_LENGTH = 1024  # ChunkySeal capacity
    
    def __init__(self, model_path: str, image_size: int = 256):
        # Load TFLite model
        # Detect quantization type
        # Allocate tensors
        # Validate shapes
```

## API Design

### Design Principles

1. **Consistency**: Match VideoSeal TFLite API
2. **Simplicity**: Easy-to-use methods
3. **Flexibility**: Support multiple input formats
4. **Type Safety**: Type hints throughout
5. **Documentation**: Comprehensive docstrings

### Method Signatures

#### Core Methods

```python
def detect(
    image: Union[Image.Image, np.ndarray, str, Path],
    threshold: float = 0.0
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Detect watermark in an image.
    
    Returns:
        {
            'confidence': float,
            'message': np.ndarray (1024,),
            'message_logits': np.ndarray (1024,),
            'predictions': np.ndarray (1025,)
        }
    """
```

```python
def detect_batch(
    images: list,
    threshold: float = 0.0
) -> list:
    """Detect watermarks in multiple images."""
```

```python
def extract_message(
    image: Union[Image.Image, np.ndarray, str, Path],
    threshold: float = 0.0,
    format: str = 'binary'
) -> Union[np.ndarray, str, int]:
    """
    Extract message in specified format.
    
    Formats:
        - 'binary': np.ndarray of 0s and 1s
        - 'hex': hexadecimal string
        - 'int': integer value
        - 'bits': bit string
    """
```

```python
def verify_watermark(
    image: Union[Image.Image, np.ndarray, str, Path],
    expected_message: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.5
) -> Tuple[bool, float, Optional[float]]:
    """
    Verify watermark presence.
    
    Returns:
        (is_watermarked, confidence, bit_accuracy)
    """
```

#### Utility Methods

```python
def preprocess_image(
    image: Union[Image.Image, np.ndarray, str, Path]
) -> np.ndarray:
    """Preprocess image for TFLite inference."""
```

```python
def get_model_info() -> Dict[str, Union[str, int, float]]:
    """Get model metadata."""
```

### Helper Function

```python
def load_detector(
    model_path: Optional[Union[str, Path]] = None,
    model_name: str = 'chunkyseal',
    image_size: int = 256,
    quantization: Optional[str] = None,
    models_dir: Optional[Union[str, Path]] = None
) -> ChunkySealDetectorTFLite:
    """
    Load a ChunkySeal TFLite detector.
    
    Auto-detects model location if not specified.
    """
```

## Implementation Files

### 1. `__init__.py` (35 lines)

**Purpose**: Package initialization and exports

```python
from .detector import ChunkySealDetectorTFLite, load_detector

__all__ = [
    'ChunkySealDetectorTFLite',
    'load_detector',
]
```

### 2. `detector.py` (404 lines)

**Purpose**: Main detector implementation

**Key Components**:
- `ChunkySealDetectorTFLite` class
- `load_detector()` helper function
- Image preprocessing
- Quantization detection
- Model info extraction

**Code Structure**:
```python
# Imports (lines 1-26)
import tensorflow as tf
from pathlib import Path
import numpy as np
from PIL import Image

# Main Class (lines 28-360)
class ChunkySealDetectorTFLite:
    MESSAGE_LENGTH = 1024
    
    def __init__(self, model_path, image_size=256):
        # Load model, detect quantization, validate
    
    def preprocess_image(self, image):
        # Handle PIL/numpy/path inputs
        # Resize to 256×256
        # Normalize to [0, 1]
        # Convert HWC → CHW
    
    def detect(self, image, threshold=0.0):
        # Preprocess
        # Run TFLite inference
        # Extract confidence + message
    
    def detect_batch(self, images, threshold=0.0):
        # Process multiple images
    
    def extract_message(self, image, threshold=0.0, format='binary'):
        # Extract in specified format
    
    def verify_watermark(self, image, expected_message=None, ...):
        # Verify watermark presence
    
    def get_model_info(self):
        # Return model metadata
    
    def _detect_quantization(self):
        # Auto-detect from filename

# Helper Function (lines 362-404)
def load_detector(...):
    # Auto-detect model path
    # Handle multiple locations
    # Return detector instance
```

### 3. `example.py` (98 lines)

**Purpose**: Simple usage demonstration

**Features**:
- Load FLOAT32 and INT8 models
- Detect watermarks
- Verify watermarks
- Extract messages in different formats
- Display model information

**Structure**:
```python
def main():
    # 1. Load FLOAT32 detector
    # 2. Try loading INT8 detector
    # 3. Display model info
    # 4. Load test image
    # 5. Detect watermark
    # 6. Verify watermark
    # 7. Extract message in multiple formats
    # 8. Show capacity comparison
```

### 4. `compare_pytorch_tflite.py` (349 lines)

**Purpose**: Benchmark PyTorch vs TFLite performance

**Features**:
- Create watermarked images using PyTorch
- Detect with both PyTorch and TFLite
- Compare accuracy (bit accuracy, confidence)
- Compare inference speed
- Support for different quantization types

**Main Functions**:
```python
def create_watermarked_image(image_path, output_path, model_name):
    # Use PyTorch to embed watermark
    # Return embedded message

def detect_pytorch(image_path, model_name):
    # Detect using PyTorch
    # Return results + timing

def detect_tflite(image_path, model_path):
    # Detect using TFLite
    # Return results + timing

def compare_results(pytorch_result, tflite_result, original_message):
    # Compare confidence
    # Compare bit accuracy
    # Compare inference time
    # Display summary
```

### 5. `test_int8.py` (Created during testing)

**Purpose**: Quick INT8 model testing

**Features**:
- Load INT8 model
- Run multiple inference iterations
- Measure timing
- Verify consistency

## Code Examples

### Example 1: Basic Detection

```python
from videoseal.chunky_tflite import load_detector

# Load detector
detector = load_detector(quantization='int8')

# Detect watermark
result = detector.detect("watermarked.jpg")

print(f"Confidence: {result['confidence']:.3f}")
print(f"Message: {result['message'][:32]}")  # First 32 of 1024 bits
print(f"Total 1s: {result['message'].sum()}/1024")
```

### Example 2: Batch Processing

```python
from videoseal.chunky_tflite import load_detector
from pathlib import Path

# Load detector
detector = load_detector(quantization='int8')

# Get image paths
image_paths = list(Path("images/").glob("*.jpg"))

# Process batch
results = detector.detect_batch(image_paths)

# Display results
for path, result in zip(image_paths, results):
    print(f"{path.name}: confidence={result['confidence']:.3f}")
```

### Example 3: Message Verification

```python
import numpy as np
from videoseal.chunky_tflite import load_detector

# Load detector
detector = load_detector()

# Expected message (1024 bits)
expected_message = np.random.randint(0, 2, size=1024)

# Verify watermark
is_watermarked, confidence, bit_accuracy = detector.verify_watermark(
    "watermarked.jpg",
    expected_message=expected_message,
    confidence_threshold=0.5
)

print(f"Watermarked: {is_watermarked}")
print(f"Confidence: {confidence:.3f}")
print(f"Bit accuracy: {bit_accuracy*100:.2f}%")
```

### Example 4: Multiple Message Formats

```python
from videoseal.chunky_tflite import load_detector

detector = load_detector()

# Extract in different formats
binary = detector.extract_message("image.jpg", format='binary')
hex_str = detector.extract_message("image.jpg", format='hex')
integer = detector.extract_message("image.jpg", format='int')
bits = detector.extract_message("image.jpg", format='bits')

print(f"Binary: {binary[:32]}")  # First 32 bits
print(f"Hex: {hex_str[:34]}...")  # First 32 hex chars
print(f"Int: {str(integer)[:50]}...")  # First 50 digits
print(f"Bits: {bits[:64]}...")  # First 64 bits
```

### Example 5: Model Information

```python
from videoseal.chunky_tflite import load_detector

detector = load_detector(quantization='int8')

# Get model info
info = detector.get_model_info()

print(f"Model: {info['model_name']}")
print(f"Quantization: {info['quantization']}")
print(f"Size: {info['model_size_mb']:.2f} MB")
print(f"Capacity: {info['message_length']} bits")
print(f"Input shape: {info['input_shape']}")
print(f"Output shape: {info['output_shape']}")
```

## Implementation Decisions

### 1. Quantization Auto-Detection

**Decision**: Automatically detect quantization from filename

**Implementation**:
```python
def _detect_quantization(self) -> str:
    filename = self.model_path.name.lower()
    if '_int8' in filename:
        return 'INT8'
    elif '_fp16' in filename:
        return 'FP16'
    else:
        return 'FLOAT32'
```

**Rationale**: Simplifies API - users don't need to specify quantization

### 2. Flexible Input Handling

**Decision**: Support PIL, numpy, and path inputs

**Implementation**:
```python
def preprocess_image(self, image):
    # Handle path
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    
    # Handle numpy
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Now process PIL Image
    # ...
```

**Rationale**: Maximum flexibility for users

### 3. Multiple Message Formats

**Decision**: Support binary, hex, int, and bits formats

**Implementation**:
```python
def extract_message(self, image, threshold=0.0, format='binary'):
    result = self.detect(image, threshold)
    message = result['message']
    
    if format == 'binary':
        return message
    elif format == 'hex':
        message_int = int(''.join(map(str, message)), 2)
        return hex(message_int)
    elif format == 'int':
        return int(''.join(map(str, message)), 2)
    elif format == 'bits':
        return ''.join(map(str, message))
```

**Rationale**: Different use cases need different formats

### 4. Consistent API with VideoSeal

**Decision**: Match VideoSeal TFLite API exactly

**Rationale**:
- Easy migration between models
- Familiar interface for users
- Code reusability

## Testing

### Unit Tests

Currently manual testing via:
- `example.py` - Basic functionality
- `test_int8.py` - INT8 model testing
- `compare_pytorch_tflite.py` - Accuracy verification

### Test Coverage

✅ Model loading (FLOAT32, INT8)  
✅ Image preprocessing  
✅ Inference execution  
✅ Message extraction  
✅ Batch processing  
✅ Multiple input formats  
✅ Quantization detection  

## Performance Considerations

### Memory Management

```python
# TFLite interpreter is created once
self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
self.interpreter.allocate_tensors()

# Tensors are reused across inferences
# No need to reallocate for each image
```

### Batch Processing

```python
# Currently processes images sequentially
# TFLite doesn't support dynamic batching well
def detect_batch(self, images, threshold=0.0):
    results = []
    for image in images:
        result = self.detect(image, threshold)
        results.append(result)
    return results
```

**Future Improvement**: Could optimize with true batching if TFLite supports it

## Error Handling

### Model Loading

```python
if not self.model_path.exists():
    raise FileNotFoundError(f"Model not found: {self.model_path}")
```

### Shape Validation

```python
expected_shape = (1, 3, image_size, image_size)
actual_shape = tuple(self.input_details[0]['shape'])
if actual_shape != expected_shape:
    raise ValueError(f"Model expects {expected_shape}, got {actual_shape}")
```

### Format Validation

```python
if format not in ['binary', 'hex', 'int', 'bits']:
    raise ValueError(f"Unknown format: {format}")
```

## See Also

- [Architecture Details](./architecture.md)
- [Usage Guide](./usage.md)
- [Quantization Guide](./quantization.md)
- [Troubleshooting](./troubleshooting.md)

