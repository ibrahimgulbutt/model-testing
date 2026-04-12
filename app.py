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
    """Get list of available model files in the models directory (including subdirectories)"""
    models_dir = Path("models")
    if not models_dir.exists():
        return []

    # Recursively find all model files
    models = []
    for ext in ("*.pt", "*.onnx", "*.engine"):
        for p in models_dir.rglob(ext):
            if p.is_file():
                # Store as relative path from models_dir for display
                models.append(str(p.relative_to(models_dir)))

    return sorted(models)


@st.cache_resource
def load_model(model_rel_path):
    """Load the selected YOLO model with comprehensive error handling.
    model_rel_path is relative to the 'models/' directory.
    """
    try:
        full_path = os.path.join("models", model_rel_path)
        if not os.path.exists(full_path):
            return None, {"error": f"Model file '{full_path}' not found!"}

        file_size = os.path.getsize(full_path)
        if file_size < 1000:
            return None, {"error": f"Model file '{model_rel_path}' seems too small ({file_size} bytes). It may be corrupted."}

        model = YOLO(full_path)

        if not hasattr(model, 'model') or model.model is None:
            return None, {"error": f"Model '{model_rel_path}' loaded but appears to be invalid or corrupted."}

        model_info = {
            'path': full_path,
            'name': model_rel_path,
            'task': getattr(model, 'task', 'detect'),
            'names': getattr(model, 'names', {}),
            'nc': len(getattr(model, 'names', {})),
            'file_size': f"{file_size / (1024*1024):.1f} MB",
            'success': True
        }

        if not model_info['names']:
            model_info['warning'] = "No class names found - model may not be fully trained"

        return model, model_info

    except RuntimeError as e:
        error_msg = str(e)
        if "file in archive is not in a subdirectory" in error_msg:
            return None, {
                "error": (
                    f"Model '{model_rel_path}' has an invalid internal structure. This usually means:\n"
                    "• The model file is corrupted\n"
                    "• The model was saved incorrectly\n"
                    "• The file is not a valid YOLO model\n\n"
                    "Try re-downloading or re-training the model."
                )
            }
        elif "PytorchStreamReader" in error_msg:
            return None, {"error": f"Model '{model_rel_path}' appears to be corrupted or not a valid PyTorch model file."}
        else:
            return None, {"error": f"Runtime error loading '{model_rel_path}':\n{error_msg}"}
    except Exception as e:
        return None, {"error": f"Unexpected error loading '{model_rel_path}':\n{str(e)}\n\nPlease check that this is a valid YOLO model file."}


def process_image(image, model, confidence=0.25, iou_threshold=0.45):
    """Run inference on a PIL image."""
    try:
        img_array = np.array(image)
        results = model(img_array, conf=confidence, iou=iou_threshold, verbose=False)
        return results[0] if results else None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


def draw_results(image, result):
    """Draw segmentation masks and/or bounding boxes on image."""
    if result is None:
        return image

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

    np.random.seed(42)
    colors = [
        (int(np.random.randint(50, 255)), int(np.random.randint(50, 255)), int(np.random.randint(50, 255)))
        for _ in range(max(len(boxes), 1))
    ]

    # Draw segmentation masks if present
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(img)
            colored_mask[mask_binary == 1] = colors[i % len(colors)]
            img = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, colors[i % len(colors)], 2)

    # Draw bounding boxes and labels
    for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        color = colors[i % len(colors)]
        class_name = result.names[int(cls)] if hasattr(result, 'names') else f"Class {int(cls)}"
        label = f"{class_name}: {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def create_detection_stats(result):
    """Return (class_counts, class_confidences) dicts."""
    if result is None or result.boxes is None:
        return None

    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_counts = {}
    class_confidences = {}

    for cls, conf in zip(classes, confidences):
        class_name = result.names[int(cls)] if hasattr(result, 'names') else f"Class {int(cls)}"
        class_counts.setdefault(class_name, 0)
        class_confidences.setdefault(class_name, [])
        class_counts[class_name] += 1
        class_confidences[class_name].append(conf)

    return class_counts, class_confidences


def render_model_info(model_info, label=""):
    """Render model metadata in sidebar."""
    if label:
        st.caption(label)
    c1, c2 = st.columns(2)
    c1.metric("Task", model_info['task'].title())
    c1.metric("Size", model_info['file_size'])
    c2.metric("Classes", model_info['nc'])
    if 'warning' in model_info:
        st.warning(f"⚠️ {model_info['warning']}")
    if model_info.get('names'):
        with st.expander("View classes", expanded=False):
            cols = st.columns(2)
            for i, name in enumerate(model_info['names'].values()):
                cols[i % 2].write(f"• {name}")


def render_result_panel(label, result_image, result, tag=""):
    """Render detection result image + stats."""
    task_type = "Segmentation" if (result is not None and result.masks is not None) else "Detection"
    st.image(result_image, caption=f"{label} — {task_type}", use_container_width=True)

    stats = create_detection_stats(result)
    if stats:
        class_counts, class_confidences = stats
        total = sum(class_counts.values())
        st.metric("Total Detections", total)
        if class_counts:
            ca, cb = st.columns([1, 1])
            with ca:
                st.write("**Per class:**")
                for cls, cnt in class_counts.items():
                    avg_conf = np.mean(class_confidences[cls])
                    st.write(f"• {cls}: {cnt} (avg {avg_conf:.2f})")
            with cb:
                fig = px.bar(
                    x=list(class_counts.keys()),
                    y=list(class_counts.values()),
                    title="Detections by Class",
                    labels={'x': 'Class', 'y': 'Count'}
                )
                fig.update_layout(height=280)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{tag}")
    else:
        st.warning("No objects detected. Try lowering the confidence threshold.")


# ─── SIDEBAR ────────────────────────────────────────────────────────────────

def main():
    st.title("🎯 YOLO Multi-Model Tester")

    available_models = get_available_models()

    with st.sidebar:
        st.header("⚙️ Settings")

        if not available_models:
            st.error("❌ No model files found in 'models/' directory!")
            st.info("Add .pt, .onnx, or .engine files to models/ or its subdirectories.")
            return

        # ── Mode toggle ──────────────────────────────────────────────────────
        mode = st.radio(
            "Test mode",
            ["Single model", "Compare two models"],
            index=0,
            help="Single: run one model. Compare: run both models side-by-side."
        )

        st.divider()

        # ── Model A ──────────────────────────────────────────────────────────
        st.subheader("🤖 Model A")
        model_a_name = st.selectbox("Model A", available_models, key="model_a")
        model_a, info_a = load_model(model_a_name)

        if model_a is None:
            st.error(f"❌ {info_a.get('error', 'Unknown error')}")
            return
        st.success("✅ Loaded")
        render_model_info(info_a, label=model_a_name)

        # ── Model B (comparison mode only) ───────────────────────────────────
        model_b, info_b = None, None
        if mode == "Compare two models":
            st.divider()
            st.subheader("🤖 Model B")
            default_b = available_models[1] if len(available_models) > 1 else available_models[0]
            model_b_name = st.selectbox("Model B", available_models, index=available_models.index(default_b), key="model_b")
            model_b, info_b = load_model(model_b_name)
            if model_b is None:
                st.error(f"❌ {info_b.get('error', 'Unknown error')}")
                return
            st.success("✅ Loaded")
            render_model_info(info_b, label=model_b_name)

        st.divider()

        # ── Inference params ─────────────────────────────────────────────────
        st.subheader("🔧 Inference Parameters")
        confidence = st.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)

    # ─── MAIN AREA ───────────────────────────────────────────────────────────

    if mode == "Single model":
        _single_model_ui(model_a, model_a_name, confidence, iou_threshold)
    else:
        _compare_models_ui(
            model_a, model_a_name,
            model_b, model_b_name,
            confidence, iou_threshold
        )

    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#666;padding:10px;'>"
        "🎯 YOLO Multi-Model Tester | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


# ─── SINGLE MODEL UI ─────────────────────────────────────────────────────────

def _single_model_ui(model, model_name, confidence, iou_threshold):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload Image")
        uploaded = st.file_uploader(
            "Choose an image…",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            key="single_upload"
        )
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Original", use_container_width=True)
            st.info(f"Size: {image.size[0]}×{image.size[1]} px")

    with col2:
        if uploaded:
            st.subheader(f"🔍 Results — {model_name}")
            if st.button("🚀 Run Detection/Segmentation", type="primary", key="single_run"):
                with st.spinner("Running inference…"):
                    result = process_image(image, model, confidence, iou_threshold)
                    if result is not None:
                        result_image = draw_results(image, result)
                        render_result_panel(model_name, result_image, result, tag="single")
                    else:
                        st.error("Inference failed. Please try again.")

    _batch_section(model, confidence, iou_threshold)


# ─── COMPARISON UI ────────────────────────────────────────────────────────────

def _compare_models_ui(model_a, name_a, model_b, name_b, confidence, iou_threshold):
    st.subheader("📤 Upload Image for Comparison")
    uploaded = st.file_uploader(
        "Choose an image…",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        key="compare_upload"
    )

    if not uploaded:
        return

    image = Image.open(uploaded)
    st.image(image, caption=f"Original — {image.size[0]}×{image.size[1]} px", use_container_width=True)

    if st.button("🚀 Run Both Models", type="primary", key="compare_run"):
        with st.spinner("Running inference on both models…"):
            result_a = process_image(image, model_a, confidence, iou_threshold)
            result_b = process_image(image, model_b, confidence, iou_threshold)

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader(f"Model A — {name_a}")
            if result_a is not None:
                img_a = draw_results(image, result_a)
                render_result_panel(name_a, img_a, result_a, tag="cmp_a")
            else:
                st.error("Model A inference failed.")

        with col_b:
            st.subheader(f"Model B — {name_b}")
            if result_b is not None:
                img_b = draw_results(image, result_b)
                render_result_panel(name_b, img_b, result_b, tag="cmp_b")
            else:
                st.error("Model B inference failed.")

        # ── Side-by-side summary table ────────────────────────────────────
        if result_a is not None and result_b is not None:
            st.divider()
            st.subheader("📊 Comparison Summary")

            def _summary(result, name):
                if result.boxes is None:
                    return {"Model": name, "Total Detections": 0, "Unique Classes": 0, "Avg Confidence": "-"}
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                return {
                    "Model": name,
                    "Total Detections": len(confs),
                    "Unique Classes": len(set(classes.astype(int))),
                    "Avg Confidence": f"{confs.mean():.3f}" if len(confs) else "-",
                    "Has Masks": "Yes" if result.masks is not None else "No",
                }

            import pandas as pd
            df = pd.DataFrame([_summary(result_a, name_a), _summary(result_b, name_b)])
            st.table(df.set_index("Model"))


# ─── BATCH SECTION ────────────────────────────────────────────────────────────

def _batch_section(model, confidence, iou_threshold):
    st.divider()
    with st.expander("🗂️ Batch Processing (Multiple Images)", expanded=False):
        uploaded_files = st.file_uploader(
            "Choose multiple images…",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            key="batch_upload"
        )

        if uploaded_files and len(uploaded_files) > 1:
            if st.button("🔄 Process All Images", type="secondary"):
                progress_bar = st.progress(0)
                for i, file in enumerate(uploaded_files):
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    with st.expander(f"📸 {file.name}", expanded=False):
                        img = Image.open(file)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(img, caption="Original", use_container_width=True)
                        with c2:
                            result = process_image(img, model, confidence, iou_threshold)
                            if result is not None:
                                st.image(draw_results(img, result), caption="Result", use_container_width=True)
                                if result.boxes is not None:
                                    st.metric("Detections", len(result.boxes))
                            else:
                                st.error("Processing failed")
                st.success("✅ Batch processing completed!")


if __name__ == "__main__":
    main()
