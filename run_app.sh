#!/bin/bash

# YOLO11n Segmentation Streamlit App Runner
echo "🎯 Starting YOLO11n Segmentation Tester..."
echo "================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the setup first."
    exit 1
fi

# Check if model file exists
if [ ! -f "yolo11n-seg.pt" ]; then
    echo "❌ Model file 'yolo11n-seg.pt' not found!"
    echo "Please ensure the model file is in the current directory."
    exit 1
fi

echo "✅ Model file found: yolo11n-seg.pt"
echo "🚀 Starting Streamlit app..."
echo ""
echo "📱 The app will open in your default browser"
echo "🌐 Or visit: http://localhost:8501"
echo ""
echo "💡 Press Ctrl+C to stop the server"
echo "================================================="

# Activate virtual environment and run the app
source .venv/bin/activate
streamlit run app.py