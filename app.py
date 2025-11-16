import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import tempfile
import os
import glob

# Configure Streamlit page
st.set_page_config(
    page_title="YOLO Multi-Model Tester",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_available_models():
    """Get list of available model files in the models directory"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    # Look for common model file extensions
    model_patterns = [
        os.path.join(models_dir, "*.pt"),
        os.path.join(models_dir, "*.onnx"),
        os.path.join(models_dir, "*.engine")
    ]
    
    models = []
    for pattern in model_patterns:
        models.extend(glob.glob(pattern))
    
    # Return just the filenames, not full paths
    return [os.path.basename(model) for model in models if os.path.isfile(model)]

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model(model_path):
    """Load the selected YOLO model with comprehensive error handling"""
    try:
        full_path = os.path.join("models", model_path)
        if not os.path.exists(full_path):
            return None, {"error": f"Model file '{full_path}' not found!"}
        
        # Check file size
        file_size = os.path.getsize(full_path)
        if file_size < 1000:  # Less than 1KB, probably not a valid model
            return None, {"error": f"Model file '{model_path}' seems too small ({file_size} bytes). It may be corrupted."}
        
        # Try to load the model with detailed error handling
        model = YOLO(full_path)
        
        # Validate the model was loaded properly
        if not hasattr(model, 'model') or model.model is None:
            return None, {"error": f"Model '{model_path}' loaded but appears to be invalid or corrupted."}
        
        # Get model info
        model_info = {
            'path': full_path,
            'name': model_path,
            'task': getattr(model, 'task', 'detect'),  # default to detect
            'names': getattr(model, 'names', {}),
            'nc': len(getattr(model, 'names', {})),
            'file_size': f"{file_size / (1024*1024):.1f} MB",
            'success': True
        }
        
        # Additional validation
        if not model_info['names']:
            model_info['warning'] = "No class names found - model may not be fully trained"
        
        return model, model_info
        
    except RuntimeError as e:
        error_msg = str(e)
        if "file in archive is not in a subdirectory" in error_msg:
            return None, {
                "error": f"Model '{model_path}' has an invalid internal structure. This usually means:\n" +
                        "• The model file is corrupted\n" +
                        "• The model was saved incorrectly\n" +
                        "• The file is not a valid YOLO model\n\n" +
                        "Try re-downloading or re-training the model."
            }
        elif "PytorchStreamReader" in error_msg:
            return None, {
                "error": f"Model '{model_path}' appears to be corrupted or not a valid PyTorch model file."
            }
        else:
            return None, {
                "error": f"Runtime error loading '{model_path}':\n{error_msg}"
            }
    except Exception as e:
        error_msg = str(e)
        return None, {
            "error": f"Unexpected error loading '{model_path}':\n{error_msg}\n\n" +
                    "Please check that this is a valid YOLO model file."
        }

def process_image(image, model, confidence=0.25, iou_threshold=0.45):
    """Process image with YOLO11n segmentation"""
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Run inference
        results = model(
            img_array,
            conf=confidence,
            iou=iou_threshold,
            verbose=False
        )
        
        return results[0] if results else None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def draw_segmentation_results(image, result):
    """Draw segmentation masks and bounding boxes on image"""
    if result is None or result.masks is None:
        return image
    
    # Convert PIL to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Get masks and boxes
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
    
    # Generate colors for each class
    colors = []
    for i in range(len(masks)):
        colors.append((
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255))
        ))
    
    # Draw masks
    for i, mask in enumerate(masks):
        # Resize mask to match image dimensions
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Create colored mask
        colored_mask = np.zeros_like(img)
        colored_mask[mask_binary == 1] = colors[i]
        
        # Blend with original image
        img = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, colors[i], 2)
    
    # Draw bounding boxes and labels
    for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        
        # Get class name
        class_name = result.names[int(cls)] if hasattr(result, 'names') else f"Class {int(cls)}"
        label = f"{class_name}: {conf:.2f}"
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[i], 2)
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), colors[i], -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def create_detection_stats(result):
    """Create detection statistics"""
    if result is None or result.boxes is None:
        return None
    
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    
    # Count detections per class
    class_counts = {}
    class_confidences = {}
    
    for cls, conf in zip(classes, confidences):
        class_name = result.names[int(cls)] if hasattr(result, 'names') else f"Class {int(cls)}"
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_confidences[class_name] = []
        
        class_counts[class_name] += 1
        class_confidences[class_name].append(conf)
    
    return class_counts, class_confidences

def main():
    st.title("🎯 YOLO Multi-Model Tester")
    st.markdown("Select a model and upload an image to test YOLO object detection/segmentation")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Model Selection & Settings")
        
        # Model selection
        st.subheader("🤖 Available Models")
        available_models = get_available_models()
        
        if not available_models:
            st.error("❌ No model files found in 'models/' directory!")
            st.info("Please add .pt, .onnx, or .engine model files to the 'models/' folder")
            return
        
        # Model selector
        selected_model = st.selectbox(
            "Choose a model:",
            available_models,
            help="Select a YOLO model from the models directory"
        )
        
        if selected_model:
            # Load the selected model
            model, model_info = load_model(selected_model)
            
            if model is not None and model_info is not None and model_info.get('success'):
                st.success("✅ Model loaded successfully!")
                
                # Model information
                st.subheader("📋 Model Information")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.metric("Model", model_info['name'])
                    st.metric("Task", model_info['task'].title())
                    st.metric("File Size", model_info['file_size'])
                with col2:
                    st.metric("Classes", model_info['nc'])
                
                # Show warnings if any
                if 'warning' in model_info:
                    st.warning(f"⚠️ {model_info['warning']}")
                
                # Display class names if available
                if model_info['names']:
                    st.subheader("🏷️ Detectable Classes")
                    class_names = list(model_info['names'].values())
                    st.write(f"Total classes: {len(class_names)}")
                    
                    # Show classes in expandable section for better space usage
                    with st.expander("View all classes", expanded=False):
                        # Show classes in a more compact format
                        cols = st.columns(2)
                        for i, class_name in enumerate(class_names):
                            col = cols[i % 2]
                            col.write(f"• {class_name}")
                else:
                    st.info("ℹ️ No class information available for this model")
                    
            elif model_info and 'error' in model_info:
                st.error("❌ Failed to load selected model")
                st.error(model_info['error'])
                
                # Provide helpful suggestions
                st.subheader("💡 Troubleshooting Tips")
                st.markdown("""
                **Common solutions:**
                1. **Re-download the model** from the original source
                2. **Check file integrity** - ensure the download completed successfully
                3. **Try a different model format** (.pt, .onnx, .engine)
                4. **Verify model compatibility** with your YOLO version
                5. **Re-train the model** if it's a custom model
                
                **Valid model sources:**
                - [Ultralytics YOLO Models](https://github.com/ultralytics/ultralytics)
                - [YOLO Official Repository](https://github.com/ultralytics/yolov5)
                - Your own trained models (ensure proper saving format)
                """)
                return
            else:
                st.error("❌ Unknown error loading model")
                return
        else:
            st.warning("Please select a model to continue")
            return
        
        st.divider()
        
        # Inference parameters
        st.subheader("🔧 Inference Parameters")
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="IoU threshold for Non-Maximum Suppression"
        )
        
        st.divider()
        
        # Example images
        st.subheader("📸 Try Example Images")
        if st.button("🏠 Use sample image (if available)"):
            st.info("Upload your own image below!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image for segmentation analysis"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", width='stretch')
            
            # Image info
            st.info(f"Image size: {image.size[0]}x{image.size[1]} pixels")
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Processing Results")
            
            # Process button
            if st.button("🚀 Run Detection/Segmentation", type="primary"):
                with st.spinner("Processing image..."):
                    result = process_image(image, model, confidence, iou_threshold)
                    
                    if result is not None:
                        # Draw results
                        result_image = draw_segmentation_results(image, result)
                        task_type = "Segmentation" if hasattr(result, 'masks') and result.masks is not None else "Detection"
                        st.image(result_image, caption=f"{task_type} Results", width='stretch')
                        
                        # Detection statistics
                        stats = create_detection_stats(result)
                        if stats:
                            class_counts, class_confidences = stats
                            
                            st.subheader("📊 Detection Statistics")
                            
                            # Summary metrics
                            total_detections = sum(class_counts.values())
                            st.metric("Total Detections", total_detections)
                            
                            if class_counts:
                                # Class distribution
                                col_a, col_b = st.columns([1, 1])
                                
                                with col_a:
                                    st.write("**Detections per Class:**")
                                    for class_name, count in class_counts.items():
                                        avg_conf = np.mean(class_confidences[class_name])
                                        st.write(f"• {class_name}: {count} (avg conf: {avg_conf:.2f})")
                                
                                with col_b:
                                    # Create bar chart
                                    fig = px.bar(
                                        x=list(class_counts.keys()),
                                        y=list(class_counts.values()),
                                        title="Detections by Class",
                                        labels={'x': 'Class', 'y': 'Count'}
                                    )
                                    fig.update_layout(height=300)
                                    st.plotly_chart(fig, width='stretch')
                        else:
                            st.warning("No objects detected. Try adjusting the confidence threshold.")
                    else:
                        st.error("Failed to process image. Please try again.")
    
    # Additional features
    st.divider()
    
    # Batch processing section
    with st.expander("🗂️ Batch Processing (Multiple Images)", expanded=False):
        st.markdown("Upload multiple images for batch processing")
        
        uploaded_files = st.file_uploader(
            "Choose multiple images...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if uploaded_files and len(uploaded_files) > 1:
            if st.button("🔄 Process All Images", type="secondary"):
                progress_bar = st.progress(0)
                results_container = st.container()
                
                with results_container:
                    st.subheader("Batch Results")
                    
                    for i, file in enumerate(uploaded_files):
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        with st.expander(f"📸 {file.name}", expanded=False):
                            img = Image.open(file)
                            col_orig, col_result = st.columns([1, 1])
                            
                            with col_orig:
                                st.image(img, caption="Original", width='stretch')
                            
                            with col_result:
                                result = process_image(img, model, confidence, iou_threshold)
                                if result is not None:
                                    result_img = draw_segmentation_results(img, result)
                                    st.image(result_img, caption="Segmented", width='stretch')
                                    
                                    # Quick stats
                                    if result.boxes is not None:
                                        detections = len(result.boxes)
                                        st.metric("Detections", detections)
                                else:
                                    st.error("Processing failed")
                
                st.success("✅ Batch processing completed!")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎯 YOLO11n Segmentation Tester | Built with Streamlit</p>
        <p>Upload images to test object detection and instance segmentation capabilities</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()