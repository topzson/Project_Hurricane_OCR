
# Hurricane OCR - Thai License Plate Recognition 🚗

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/Rattatammanoon/hurricane-ocr-v1)
[![Thai](https://img.shields.io/badge/Language-Thai-red.svg)](https://en.wikipedia.org/wiki/Thai_language)
[![Accuracy](https://img.shields.io/badge/Accuracy-86.7%25-brightgreen.svg)](https://huggingface.co/Rattatammanoon/hurricane-ocr-v1)

**State-of-the-art Thai License Plate OCR powered by Typhoon-OCR 1.5**

</div>

---

## 📋 Model Description

**Hurricane OCR** is a high-performance OCR model specifically fine-tuned for reading **Thai license plates**. Built on top of [SCB-10X's Typhoon-OCR 1.5 (2B)](https://huggingface.co/scb10x/typhoon-ocr1.5-2b) using **LoRA (Low-Rank Adaptation)**, this model efficiently extracts structured information from license plate images with **86.7% accuracy**.

### 🎯 Extracted Fields

| Field | Description | Example |
|-------|-------------|---------|
| 🔤 **Plate Number** | Full license plate number | `กก 1234` |
| 📝 **Characters** | Thai characters only | `กก` |
| 🔢 **Digits** | Numeric digits only | `1234` |
| 📍 **Province** | Province name in Thai | `กรุงเทพมหานคร` |

---

## 🚀 Quick Start

### Installation

```bash
pip install transformers peft torch pillow
```

### Basic Usage

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image

# Load processor and base model
base_model_name = "scb10x/typhoon-ocr1.5-2b"
processor = AutoProcessor.from_pretrained(base_model_name)
base_model = AutoModelForVision2Seq.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "Rattatammanoon/hurricane-ocr-v1")
model.eval()

# Process license plate image
image = Image.open("license_plate.jpg").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(model.device)

# Generate OCR output
with torch.no_grad():
    generated_ids = model.generate(
        pixel_values,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )
    
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

### Batch Processing

```python
# Process multiple plates at once
images = [Image.open(f"plate{i}.jpg").convert("RGB") for i in range(5)]
pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(model.device)

with torch.no_grad():
    generated_ids = model.generate(pixel_values, max_length=512, num_beams=4)
    
texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
for i, text in enumerate(texts):
    print(f"Plate {i+1}: {text}")
```

---

## 📊 Training Details

| Parameter | Value |
|-----------|-------|
| **Base Model** | [scb10x/typhoon-ocr1.5-2b](https://huggingface.co/scb10x/typhoon-ocr1.5-2b) |
| **Fine-tuning Method** | LoRA (Low-Rank Adaptation) via PEFT |
| **Training Data** | 304 Thai license plate images |
| **Accuracy** | **86.7%** |
| **Languages** | Thai (primary), English (digits) |
| **Framework** | PyTorch + Transformers + PEFT |
| **Training Date** | December 2025 |

### Performance Metrics

- ✅ **Character Accuracy**: 86.7%
- ✅ **Optimized for**: Thai license plates (new & old formats)
- ✅ **Best Performance**: Clean, cropped plate images (200-400px width)

---

## 📝 Output Format

The model outputs structured markdown text:

```markdown
**Plate Number:** กก 1234
**Characters:** กก
**Digits:** 1234
**Province:** กรุงเทพมหานคร
```

You can easily parse this output:

```python
import re

# Parse the OCR output
lines = text.strip().split('\n')
result = {}
for line in lines:
    if '**' in line:
        key, value = line.split(':', 1)
        key = key.strip('*').strip()
        result[key] = value.strip()

print(result)
# {'Plate Number': 'กก 1234', 'Characters': 'กก', ...}
```

---

## 🔧 Advanced Usage

### Custom Generation Parameters

Fine-tune the OCR output quality:

```python
generated_ids = model.generate(
    pixel_values,
    max_length=512,
    num_beams=5,              # Increase for better quality (slower)
    temperature=0.7,          # Lower = more deterministic
    top_p=0.9,
    repetition_penalty=1.2,
    early_stopping=True
)
```

### Integration with Detection Pipeline

Combine with a license plate detector (e.g., YOLOv8):

```python
from ultralytics import YOLO

# 1. Detect license plate region
detector = YOLO("path/to/plate_detector.pt")
results = detector("car_image.jpg")

# 2. Extract and process each detected plate
for result in results:
    for box in result.boxes:
        # Crop plate region
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = image.crop((x1, y1, x2, y2))
        
        # 3. Run Hurricane OCR
        pixel_values = processor(images=plate_crop, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Detected plate: {text}")
```

---

## 💡 Performance Tips

| Tip | Description |
|-----|-------------|
| ✅ **Image Size** | Resize plates to 200-400px width for best results |
| ✅ **Image Quality** | Use clear, well-lit images |
| ✅ **Preprocessing** | Crop tightly around the plate |
| ✅ **Batch Processing** | Process multiple images at once for efficiency |
| ⚠️ **Limitations** | Optimized for Thai plates only |

### Recommended Preprocessing

```python
from PIL import Image, ImageEnhance

def preprocess_plate(image_path):
    img = Image.open(image_path).convert("RGB")
    
    # Resize if needed (maintain aspect ratio)
    if img.width > 400:
        ratio = 400 / img.width
        new_size = (400, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    # Enhance contrast (optional)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    return img
```

---

## 📈 Use Cases

- 🚗 **Parking Management Systems**
- 🚦 **Traffic Monitoring & Analysis**
- 🏢 **Access Control Systems**
- 📊 **Vehicle Fleet Management**
- 🚓 **Law Enforcement Applications**

---

## ⚠️ Limitations

- Designed specifically for **Thai license plates** (both old and new formats)
- Performance may degrade with:
  - Very low resolution images
  - Heavily obscured or damaged plates
  - Extreme lighting conditions
  - Non-standard plate formats

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions or improvements:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This model is licensed under **Apache 2.0**. See [LICENSE](LICENSE) for details.

- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ❗ Must include license and copyright notice

---

## 📚 Citation

If you use Hurricane OCR in your research or project, please cite:

```bibtex
@misc{hurricane-ocr-v1-2025,
  author = {Rattatammanoon},
  title = {Hurricane OCR - Thai License Plate Recognition},
  year = {2025},
  publisher = {Hugging Face},
  journal = {Hugging Face Model Hub},
  howpublished = {\url{https://huggingface.co/Rattatammanoon/hurricane-ocr-v1}}
}
```

---

## 🙏 Acknowledgments

This project builds upon excellent work from:

- **Base Model**: [SCB-10X Typhoon-OCR 1.5 (2B)](https://huggingface.co/scb10x/typhoon-ocr1.5-2b)
- **Framework**: [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- **PEFT Library**: [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft)
- **Training Dataset**: Custom Thai license plate dataset

Special thanks to the SCB-10X team for developing Typhoon-OCR! 🙏

---

## 📧 Contact & Support

- 👨‍💻 **GitHub**: [@topzson](https://github.com/topzson)
- 🐛 **Issues**: [Report here](https://github.com/topzson/hurricane-ocr/issues)
- 💬 **Discussions**: [HuggingFace Discussions](https://huggingface.co/Rattatammanoon/hurricane-ocr-v1/discussions)

---

<div align="center">

### Made with ❤️ for Thai License Plate Recognition

**Star ⭐ this model if you find it useful!**

</div>
