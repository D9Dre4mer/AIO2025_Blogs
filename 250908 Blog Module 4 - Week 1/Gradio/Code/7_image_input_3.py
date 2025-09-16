import gradio as gr
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def apply_filter(image, filter_type):
    """Apply various filters to an image."""
    if image is None:
        return None
    
    if filter_type == "Original":
        return image
    elif filter_type == "Grayscale":
        return image.convert('L').convert('RGB')
    elif filter_type == "Blur":
        return image.filter(ImageFilter.GaussianBlur(radius=3))
    elif filter_type == "Sharpen":
        return image.filter(ImageFilter.SHARPEN)
    elif filter_type == "Brightness":
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.5)
    elif filter_type == "Contrast":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)
    elif filter_type == "Sepia":
        # Convert to numpy array for sepia effect
        img_array = np.array(image)
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia_img = img_array.dot(sepia_filter.T)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return Image.fromarray(sepia_img)
    elif filter_type == "Vintage":
        # Apply vintage effect (sepia + slight blur + vignette)
        img_array = np.array(image)
        # Sepia
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia_img = img_array.dot(sepia_filter.T)
        sepia_img = np.clip(sepia_img * 0.9, 0, 255).astype(np.uint8)
        vintage_image = Image.fromarray(sepia_img)
        # Add slight blur
        vintage_image = vintage_image.filter(ImageFilter.GaussianBlur(radius=0.5))
        return vintage_image
    
    return image

def create_comparison(image, filter1, filter2):
    """Create a comparison between original and filtered image."""
    if image is None:
        return None
    
    # Apply filters
    image1 = apply_filter(image, filter1)
    image2 = apply_filter(image, filter2)
    
    return (image1, image2)


choices = ["Original", "Grayscale", "Blur", "Sharpen", 
           "Brightness", "Contrast", "Sepia", "Vintage"]

with gr.Blocks(title="Image Slider Demo") as demo:
    gr.Markdown("# 🔄 Image Slider Filter Comparison")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", type="pil")
            with gr.Row():
                filter1_dropdown = gr.Dropdown(
                    choices=choices,
                    value="Original",
                    label="Left Side Filter"
                )
                filter2_dropdown = gr.Dropdown(
                    choices=choices,
                    value="Grayscale",
                    label="Right Side Filter"
                )
            
            compare_btn = gr.Button("Create Comparison", variant="primary")
        
        with gr.Column():
            comparison_slider = gr.ImageSlider(
                label="Drag to Compare Filters",
                elem_id="comparison",
            )
    
    # Event handlers for filter comparison
    compare_btn.click(
        fn=create_comparison,
        inputs=[input_image, filter1_dropdown, filter2_dropdown],
        outputs=[comparison_slider]
    )
    
    # Auto-update when filters change
    filter1_dropdown.change(
        fn=create_comparison,
        inputs=[input_image, filter1_dropdown, filter2_dropdown],
        outputs=[comparison_slider]
    )
    
    filter2_dropdown.change(
        fn=create_comparison,
        inputs=[input_image, filter1_dropdown, filter2_dropdown],
        outputs=[comparison_slider]
    )


demo.launch()
