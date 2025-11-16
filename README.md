# 🎯 YOLO Multi-Model Tester

A modern Streamlit web application for testing multiple YOLO models with an intuitive interface. Supports detection, segmentation, and classification tasks.

## ✨ Features

- 🤖 **Multi-Model Support** - Automatically detects and lists all YOLO models in the models/ directory
- 🔄 **Dynamic Model Selection** - Switch between different models with different class sets
- 🖼️ **Image Upload & Processing** - Upload images and get instant detection/segmentation results
- 🎨 **Visual Results** - See segmentation masks overlaid on original images with bounding boxes
- 📊 **Detection Statistics** - View detailed stats including class counts and confidence scores
- ⚙️ **Adjustable Parameters** - Fine-tune confidence and IoU thresholds in real-time
- 🗂️ **Batch Processing** - Process multiple images at once
- 🛡️ **Model Validation** - Built-in model validation with detailed error messages
- 📱 **Responsive Design** - Works on desktop and mobile devices

## 🚀 Quick Start

### Option 1: Using the Run Script (Recommended)
```bash
./run_app.sh
```

### Option 2: Manual Start
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the app
streamlit run app.py
```

## 📋 Requirements

- Python 3.8+
- YOLO11n segmentation model file (`yolo11n-seg.pt`)
- All dependencies are automatically installed

## 🎮 How to Use

1. **Start the app** using one of the methods above
2. **Open your browser** to `http://localhost:8501`
3. **Upload an image** using the file uploader
4. **Adjust settings** in the sidebar (confidence, IoU threshold)
5. **Click "Run Segmentation"** to process your image
6. **View results** with segmentation masks and statistics

## 🔧 Configuration

### Inference Parameters
- **Confidence Threshold**: Minimum confidence score for detections (0.1 - 1.0)
- **IoU Threshold**: IoU threshold for Non-Maximum Suppression (0.1 - 1.0)

### Supported Image Formats
- JPG/JPEG
- PNG
- BMP
- TIFF

## 📊 Output Information

The app provides:
- **Visual Results**: Original image with segmentation masks and bounding boxes
- **Detection Count**: Total number of objects detected
- **Class Statistics**: Breakdown by object class with confidence scores
- **Interactive Charts**: Bar charts showing detection distribution

## 🛠️ Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure `yolo11n-seg.pt` is in the same directory as `app.py`

2. **Dependencies missing**
   - Run: `pip install -r requirements.txt`

3. **App won't start**
   - Check that port 8501 is not in use
   - Try: `streamlit run app.py --server.port 8502`

### Performance Tips

- Use smaller images for faster processing
- Adjust confidence threshold based on your use case
- Lower IoU threshold for more detections

## 📁 File Structure

```
testing model/
├── app.py              # Main Streamlit application
├── yolo11n-seg.pt      # YOLO11n segmentation model
├── requirements.txt    # Python dependencies
├── run_app.sh         # Quick start script
└── README.md          # This file
```

## 🎯 Model Information

This app is designed to work with YOLO11n segmentation models that can detect and segment multiple object classes. The model supports:

- Object detection with bounding boxes
- Instance segmentation with pixel-level masks
- Multi-class recognition
- Confidence scoring

## 💡 Tips for Best Results

1. **Image Quality**: Use clear, well-lit images for better detection
2. **Object Size**: Ensure objects are not too small in the frame
3. **Confidence Tuning**: Lower confidence for more detections, higher for precision
4. **Batch Processing**: Use the batch feature for testing multiple images

## 🔗 Additional Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

🎉 **Happy Testing!** Upload your images and explore the segmentation capabilities of your YOLO11n model.