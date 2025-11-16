#!/usr/bin/env python3
"""
YOLO Model Validator
Validates YOLO model files in the models directory and provides detailed information.
"""

import os
import sys
from ultralytics import YOLO
import torch

def validate_model(model_path):
    """Validate a single YOLO model file"""
    print(f"\n🔍 Validating: {model_path}")
    print("-" * 50)
    
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            print(f"❌ File not found: {model_path}")
            return False
        
        # Check file size
        file_size = os.path.getsize(model_path)
        print(f"📁 File size: {file_size / (1024*1024):.1f} MB")
        
        if file_size < 1000:
            print(f"⚠️  File seems very small ({file_size} bytes) - may be corrupted")
            return False
        
        # Try to load the model
        print("🔄 Loading model...")
        model = YOLO(model_path)
        
        # Get model information
        task = getattr(model, 'task', 'unknown')
        names = getattr(model, 'names', {})
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Task: {task}")
        print(f"🏷️  Classes: {len(names)}")
        
        if names:
            print(f"📝 Class names: {list(names.values())[:10]}{'...' if len(names) > 10 else ''}")
        else:
            print("⚠️  No class names found")
        
        # Try a dummy inference to verify model works
        print("🧪 Testing inference...")
        import numpy as np
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)
        print("✅ Inference test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
        # Provide specific error guidance
        error_str = str(e)
        if "file in archive is not in a subdirectory" in error_str:
            print("\n💡 This error usually means:")
            print("   • The model file has corrupted internal structure")
            print("   • The model was saved incorrectly")
            print("   • Try re-downloading or re-training the model")
        elif "PytorchStreamReader" in error_str:
            print("\n💡 This error usually means:")
            print("   • The file is corrupted or not a valid PyTorch model")
            print("   • Try re-downloading the model")
        
        return False

def main():
    """Main validation function"""
    models_dir = "models"
    
    print("🎯 YOLO Model Validator")
    print("=" * 50)
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory '{models_dir}' not found!")
        print("Please create the models directory and add your model files.")
        return
    
    # Find all model files
    model_files = []
    for ext in ['*.pt', '*.onnx', '*.engine']:
        import glob
        model_files.extend(glob.glob(os.path.join(models_dir, ext)))
    
    if not model_files:
        print(f"❌ No model files found in '{models_dir}' directory!")
        print("Supported formats: .pt, .onnx, .engine")
        print("\n💡 Add your YOLO model files to the models/ directory")
        return
    
    print(f"Found {len(model_files)} model file(s):")
    for model_file in model_files:
        print(f"  • {os.path.basename(model_file)}")
    
    # Validate each model
    valid_models = 0
    for model_file in model_files:
        if validate_model(model_file):
            valid_models += 1
    
    print(f"\n📊 Summary:")
    print(f"Total models: {len(model_files)}")
    print(f"Valid models: {valid_models}")
    print(f"Invalid models: {len(model_files) - valid_models}")
    
    if valid_models == 0:
        print("\n❌ No valid models found!")
        print("The Streamlit app will not work without valid model files.")
    elif valid_models == len(model_files):
        print("\n✅ All models are valid!")
        print("The Streamlit app should work properly.")
    else:
        print(f"\n⚠️  Some models are invalid.")
        print("Remove or replace invalid models for better app performance.")

if __name__ == "__main__":
    main()