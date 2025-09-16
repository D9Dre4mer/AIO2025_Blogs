import gradio as gr

def greet(name):
    return "Hello, " + name

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Enter your name", 
                      placeholder="Type your name here..."),
    outputs=gr.Textbox(label="Greeting"),
)

demo.launch()