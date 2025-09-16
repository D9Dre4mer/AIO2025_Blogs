import gradio as gr
import os

def process_file(file):
    """Process uploaded file and return basic information."""
    if file is None:
        return "No file uploaded"
    
    # Get basic file info
    file_name = os.path.basename(file.name)
    file_size = os.path.getsize(file.name)
    
    # Format file size
    if file_size < 1024:
        size_str = f"{file_size} bytes"
    elif file_size < 1024 * 1024:
        size_str = f"{file_size / 1024:.2f} KB"
    else:
        size_str = f"{file_size / (1024 * 1024):.2f} MB"
    
    # Return file information
    info = f"✅ **File uploaded successfully!**\n\n"
    info += f"📁 **File Name:** {file_name}\n"
    info += f"📏 **File Size:** {size_str}\n"
    info += f"📂 **File Path:** {file.name}"
    
    return info

def process_multiple_files(files):
    """Process multiple uploaded files."""
    if not files:
        return "No files uploaded"
    
    info = f"✅ **{len(files)} files uploaded successfully!**\n\n"
    
    total_size = 0
    for i, file in enumerate(files, 1):
        if file is not None:
            file_name = os.path.basename(file.name)
            file_size = os.path.getsize(file.name)
            total_size += file_size
            
            if file_size < 1024:
                size_str = f"{file_size} bytes"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.2f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.2f} MB"
            
            info += f"{i}. **{file_name}** - {size_str}\n"
    
    # Total size
    if total_size < 1024:
        total_size_str = f"{total_size} bytes"
    elif total_size < 1024 * 1024:
        total_size_str = f"{total_size / 1024:.2f} KB"
    else:
        total_size_str = f"{total_size / (1024 * 1024):.2f} MB"
    
    info += f"\n📊 **Total Size:** {total_size_str}"
    
    return info

def process_pair(single_file, multiple_files):
    return process_file(single_file), process_multiple_files(multiple_files)


demo = gr.Interface(
    fn=process_pair,
    inputs=[
        gr.File(label="Single file", file_count="single"),
        gr.File(label="Multiple files", file_count="multiple"),
    ],
    outputs=[
        gr.Markdown(label="Single file info"),
        gr.Markdown(label="Multiple files info"),
    ],
    title="Simple File Upload Demo",
    description="Upload a file or multiple files to see basic information.",
    live=True,
)

demo.launch()
