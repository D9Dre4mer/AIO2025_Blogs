import gradio as gr

def execute_python(code):
    """Execute Python code safely"""
    try:
        import io
        import sys
        
        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        # Execute the code
        exec(code)
        
        # Get the output
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        return output if output else "Code executed successfully (no output)"
        
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=execute_python,
    inputs=gr.Code(
        label="Python Code",
        language="python",
        value="print('Hello, World!')\nprint(2 + 2)",
        lines=5
    ),
    outputs=gr.Textbox(label="Output", lines=5),
    title="Simple Python Code Runner",
    description="Enter Python code and see the output"
)

demo.launch()