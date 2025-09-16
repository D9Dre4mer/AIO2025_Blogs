import gradio as gr

def greet(name: str, times: float) -> str:
	n = int(times or 1)
	name = name or "there"
	return "\n".join([f"Hello, {name}!" for _ in range(max(n, 1))])


with gr.Blocks(title="Blocks + Row: simplest") as demo:
	gr.Markdown("## Simple Blocks + Row demo")

	with gr.Row():
		name = gr.Textbox(label="Name", placeholder="Type your name")
		times = gr.Number(value=1, label="Times")

	with gr.Row():
		btn = gr.Button("Greet")
		out = gr.Textbox(label="Output", interactive=False)

	# btn.click(greet, inputs=[name, times], outputs=out)

demo.launch()