#!/bin/bash

# YOLO Model Downloader
# Downloads popular pre-trained YOLO models for testing

echo "🎯 YOLO Model Downloader"
echo "========================"
echo ""

# Create models directory if it doesn't exist
mkdir -p models

echo "📥 Available models to download:"
echo ""
echo "1. YOLOv8n Detection (6.2 MB) - Fast, good for testing"
echo "2. YOLOv8s Detection (21.5 MB) - Balanced speed/accuracy"  
echo "3. YOLOv8n Segmentation (6.7 MB) - Fast segmentation"
echo "4. YOLOv8s Segmentation (23.8 MB) - Better segmentation"
echo "5. Download all of the above"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "📦 Downloading YOLOv8n Detection model..."
        cd models
        curl -L -o yolov8n.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        echo "✅ Downloaded: yolov8n.pt"
        ;;
    2)
        echo "📦 Downloading YOLOv8s Detection model..."
        cd models
        curl -L -o yolov8s.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
        echo "✅ Downloaded: yolov8s.pt"
        ;;
    3)
        echo "📦 Downloading YOLOv8n Segmentation model..."
        cd models
        curl -L -o yolov8n-seg.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"
        echo "✅ Downloaded: yolov8n-seg.pt"
        ;;
    4)
        echo "📦 Downloading YOLOv8s Segmentation model..."
        cd models
        curl -L -o yolov8s-seg.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt"
        echo "✅ Downloaded: yolov8s-seg.pt"
        ;;
    5)
        echo "📦 Downloading all models..."
        cd models
        echo "Downloading YOLOv8n Detection..."
        curl -L -o yolov8n.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        echo "Downloading YOLOv8s Detection..."
        curl -L -o yolov8s.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
        echo "Downloading YOLOv8n Segmentation..."
        curl -L -o yolov8n-seg.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"
        echo "Downloading YOLOv8s Segmentation..."
        curl -L -o yolov8s-seg.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt"
        echo "✅ Downloaded all models!"
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🔍 Validating downloaded models..."
cd ..
python validate_models.py

echo ""
echo "🎉 Done! You can now run the Streamlit app:"
echo "   ./run_app.sh"