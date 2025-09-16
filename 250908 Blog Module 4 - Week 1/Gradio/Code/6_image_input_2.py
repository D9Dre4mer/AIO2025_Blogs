import gradio as gr

def process_image(image_data):
    if image_data is None:
        return None
    return image_data.get('composite', None)

with gr.Blocks(title="Simple Image Editor") as demo:
    gr.Markdown("# 🖼️ Simple Image Editor")
    
    with gr.Row():
        image_editor = gr.ImageEditor(
            label="Edit Your Image",
            type="pil",
            sources=["upload", "webcam"],  # Allow upload and webcam
            transforms=["crop"],           # Enable cropping
            brush=gr.Brush(default_size=10),  # Drawing brush
            eraser=gr.Eraser(default_size=10) # Eraser tool
        )
        output_image = gr.Image(label="Edited Result", type="pil")
    
    image_editor.change(
        fn=process_image,
        inputs=[image_editor],
        outputs=[output_image]
    )

demo.launch()