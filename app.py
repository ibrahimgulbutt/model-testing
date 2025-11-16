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

# Configure Streamlit page
st.set_page_config(
    page_title="YOLO11n Segmentation Tester",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    """Load the YOLO11n segmentation model"""
    try:
        model_path = "yolo11n-seg.pt"
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found!")
            return None
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

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
    st.title("🎯 YOLO11n Segmentation Tester")
    st.markdown("Upload an image to test the YOLO11n segmentation model")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model info
        st.subheader("📋 Model Information")
        model = load_model()
        
        if model is not None:
            st.success("✅ Model loaded successfully!")
            st.info(f"Model type: YOLO11n Segmentation")
            
            # Display class names if available
            if hasattr(model, 'names') and model.names:
                st.subheader("🏷️ Detectable Classes")
                class_names = list(model.names.values())
                st.write(f"Total classes: {len(class_names)}")
                
                # Show classes in a more compact format
                cols = st.columns(2)
                for i, class_name in enumerate(class_names):
                    col = cols[i % 2]
                    col.write(f"• {class_name}")
        else:
            st.error("❌ Failed to load model")
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
            if st.button("🚀 Run Segmentation", type="primary"):
                with st.spinner("Processing image..."):
                    result = process_image(image, model, confidence, iou_threshold)
                    
                    if result is not None:
                        # Draw results
                        result_image = draw_segmentation_results(image, result)
                        st.image(result_image, caption="Segmentation Results", width='stretch')
                        
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