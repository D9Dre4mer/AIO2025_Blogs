import gradio as gr

def greet(name: str, times: float) -> str:
	n = int(times or 1)
	name = name or "there"
	return "\n".join([f"Hello, {name}!" for _ in range(max(n, 1))])

def live_preview(name: str) -> str:
	return f"Typing: {name or ''}"

def on_times_change(times: float) -> str:
	return f"Times set to {int(times or 1)}"

with gr.Blocks(
	title="Event Handling: simplest",
	theme=gr.themes.Soft()
) as demo:
	gr.Markdown("## Event Handling demo")
	with gr.Row():
		with gr.Column():
			name = gr.Textbox(label="Name", placeholder="Type your name and press Enter")
			times = gr.Slider(1, 5, value=1, step=1, label="Times")
			greet_btn = gr.Button("Greet")
		with gr.Column():
			live_md = gr.Markdown("(Start typing to see live preview)")
			times_md = gr.Markdown("Times set to 1")
			out = gr.Textbox(label="Output", interactive=False)

	# Events
	name.input(live_preview, inputs=name, outputs=live_md)          
	name.submit(greet, inputs=[name, times], outputs=out)        
	greet_btn.click(greet, inputs=[name, times], outputs=out)       
	times.change(on_times_change, inputs=times, outputs=times_md) 

demo.launch()