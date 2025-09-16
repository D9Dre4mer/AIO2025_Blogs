import gradio as gr

def greet(name: str, times: float) -> str:
	n = int(times or 1)
	name = name or "there"
	return "\n".join([f"Hello, {name}!" for _ in range(max(n, 1))])


with gr.Blocks(title="Blocks + Column: simplest") as demo:
	gr.Markdown("## Simple Blocks + Column demo")

	with gr.Column():
		name = gr.Textbox(label="Name", placeholder="Type your name")
		times = gr.Slider(1, 5, value=1, step=1, label="Times")
		btn = gr.Button("Greet")
		out = gr.Textbox(label="Output", interactive=False)

	btn.click(greet, inputs=[name, times], outputs=out)

demo.launch()
