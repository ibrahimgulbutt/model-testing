#!/usr/bin/env python3
"""
Script to inspect YOLO model classes and understand where they come from
"""

from ultralytics import YOLO
import json

def inspect_yolo_classes():
    """Inspect and display YOLO model class information"""
    
    print("🎯 YOLO Model Class Inspector")
    print("=" * 50)
    
    # Load the model
    try:
        print("📂 Loading model: yolo11n-seg.pt")
        model = YOLO("yolo11n-seg.pt")
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 1. Show where classes come from
    print("🔍 CLASS INFORMATION SOURCE:")
    print("-" * 30)
    
    # Check if model has names attribute
    if hasattr(model, 'names'):
        print("📋 Classes are stored in: model.names")
        print(f"📊 Data type: {type(model.names)}")
        print(f"🔢 Total classes: {len(model.names)}")
        
        # Show the structure
        print(f"\n📝 Structure example:")
        print(f"   model.names[0] = '{model.names[0]}'")
        print(f"   model.names[1] = '{model.names[1]}'")
        print(f"   model.names[2] = '{model.names[2]}'")
        print("   ...")
        
    else:
        print("❌ Model doesn't have 'names' attribute")
        return
    
    print("\n" + "=" * 50)
    
    # 2. Show complete class list
    print("📜 COMPLETE CLASS LIST:")
    print("-" * 25)
    
    # Display classes in a nice format
    classes = list(model.names.values())
    
    # Show in columns for better readability
    cols = 4
    for i in range(0, len(classes), cols):
        row = classes[i:i+cols]
        formatted_row = []
        for j, cls in enumerate(row):
            class_id = i + j
            formatted_row.append(f"{class_id:2d}: {cls:<15}")
        print("  ".join(formatted_row))
    
    print("\n" + "=" * 50)
    
    # 3. Show dataset information
    print("📚 DATASET INFORMATION:")
    print("-" * 23)
    print("🎓 These classes come from the COCO (Common Objects in Context) dataset")
    print("🌐 COCO is a large-scale object detection, segmentation dataset")
    print("📊 It contains 80 object categories (classes)")
    print("🔗 More info: https://cocodataset.org/")
    
    # 4. Show how to access classes programmatically
    print("\n💻 PROGRAMMATIC ACCESS:")
    print("-" * 25)
    print("# Load model")
    print("model = YOLO('yolo11n-seg.pt')")
    print("")
    print("# Get all class names")
    print("class_names = list(model.names.values())")
    print("")
    print("# Get specific class by ID")
    print("class_0 = model.names[0]  # 'person'")
    print("class_1 = model.names[1]  # 'bicycle'")
    print("")
    print("# Get class ID by name")
    print("person_id = list(model.names.keys())[list(model.names.values()).index('person')]")
    
    # 5. Save classes to JSON file
    print("\n💾 SAVING CLASSES TO FILE:")
    print("-" * 27)
    
    # Create a more detailed class info
    class_info = {
        "model_file": "yolo11n-seg.pt",
        "total_classes": len(model.names),
        "dataset_source": "COCO (Common Objects in Context)",
        "classes": {}
    }
    
    for class_id, class_name in model.names.items():
        class_info["classes"][str(class_id)] = {
            "id": class_id,
            "name": class_name
        }
    
    # Save to JSON
    with open("model_classes.json", "w") as f:
        json.dump(class_info, f, indent=2)
    
    print("📁 Saved detailed class info to: model_classes.json")
    
    # 6. Show class categories
    print("\n🏷️ CLASS CATEGORIES:")
    print("-" * 18)
    
    # Group classes by category (simplified categorization)
    categories = {
        "People & Body Parts": ["person"],
        "Vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
        "Animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
        "Food & Kitchen": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
        "Furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
        "Electronics": ["tv", "laptop", "mouse", "keyboard", "remote", "cell phone", "microwave", "oven", "toaster", "refrigerator", "hair drier"],
        "Sports & Recreation": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"],
        "Household Items": ["book", "clock", "vase", "scissors", "teddy bear", "toothbrush", "sink"],
        "Accessories & Clothing": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
        "Urban Objects": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"]
    }
    
    for category, items in categories.items():
        found_items = [item for item in items if item in classes]
        if found_items:
            print(f"\n  🔸 {category}:")
            for item in found_items:
                class_id = list(model.names.keys())[list(model.names.values()).index(item)]
                print(f"     {class_id:2d}: {item}")
    
    print("\n" + "=" * 50)
    print("🎉 Class inspection complete!")
    print("💡 The app uses model.names to get class information dynamically")

if __name__ == "__main__":
    inspect_yolo_classes()