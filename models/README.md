# 📁 Models Directory

Place your YOLO model files in this directory for automatic detection by the Streamlit app.

## 🔧 Supported Model Formats

- **`.pt`** - PyTorch models (most common)
- **`.onnx`** - ONNX format models  
- **`.engine`** - TensorRT optimized models

## 📥 How to Add Models

1. Download or train your YOLO models
2. Copy the model files to this `models/` directory
3. The Streamlit app will automatically detect them

## ✅ Model Validation

Run the model validator to check if your models are working:

```bash
python validate_models.py
```

This will:
- Check all model files in this directory
- Validate file integrity
- Test model loading
- Display model information (classes, task type, etc.)
- Provide troubleshooting tips for problematic models

## 🐛 Common Issues & Solutions

### Error: "file in archive is not in a subdirectory"
- **Cause**: Corrupted model file or incorrect saving format
- **Solution**: Re-download or re-train the model

### Error: "PytorchStreamReader failed"  
- **Cause**: Corrupted or invalid model file
- **Solution**: Re-download the model from original source

### Model loads but no classes shown
- **Cause**: Model not fully trained or missing class information
- **Solution**: Use a pre-trained model or complete training

## 📚 Where to Get Models

### Pre-trained Models
- [Ultralytics YOLO Models](https://github.com/ultralytics/ultralytics)
- [YOLO Official Hub](https://hub.ultralytics.com/)

### Popular Models
- `yolov8n.pt` - YOLOv8 Nano (fastest)
- `yolov8s.pt` - YOLOv8 Small  
- `yolov8m.pt` - YOLOv8 Medium
- `yolov8l.pt` - YOLOv8 Large
- `yolov8x.pt` - YOLOv8 Extra Large (most accurate)

### Segmentation Models  
- `yolov8n-seg.pt` - Nano segmentation
- `yolov8s-seg.pt` - Small segmentation
- `yolov8m-seg.pt` - Medium segmentation

### Custom Models
If you have custom trained models, ensure they were saved properly using:
```python
model.save('your_model_name.pt')
```

## 🎯 Quick Start

1. Add a model file (e.g., `yolov8n.pt`) to this directory
2. Run: `python validate_models.py` to verify
3. Start the app: `./run_app.sh`
4. Select your model from the dropdown in the app