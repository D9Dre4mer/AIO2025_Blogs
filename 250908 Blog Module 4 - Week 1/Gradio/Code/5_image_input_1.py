import gradio as gr
import numpy as np
from PIL import Image

def get_image_info(image):
    """Get basic information about an image"""
    if image is None:
        return "No image provided"
    
    # Handle different image types
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        return f"Image size: {width}x{height}, Channels: {channels}, Type: NumPy array"
    elif isinstance(image, Image.Image):
        width, height = image.size
        return f"Image size: {width}x{height}, Mode: {image.mode}, Type: PIL Image"
    else:
        return f"Image type: {type(image)}"

demo = gr.Interface(
    fn=get_image_info,
    inputs=gr.Image(label="Upload an Image", type="pil"),
    outputs=gr.Textbox(label="Image Information"),
    title="Image Information Tool",
    description="Upload an image to see its basic information"
)

demo.launch()