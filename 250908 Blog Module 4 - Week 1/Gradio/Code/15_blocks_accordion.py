import gradio as gr

def greet(name: str, mood: str) -> str:
	name = (name or "there").strip()
	mood = (mood or "okay").strip().lower()
	return f"Hi {name}! Nice to hear you're feeling {mood}."

def square(x: float | int | None) -> float:
	if x is None:
		return 0.0
	try:
		return float(x) * float(x)
	except Exception:
		return 0.0


with gr.Blocks(title="Accordion Demo") as demo:
	gr.Markdown("# Gradio Blocks – Accordion Demo")
	gr.Markdown("This simple app shows how to group UI "
                "in expandable sections using `gr.Accordion`.")

	with gr.Accordion("Greet Me", open=True):
		name = gr.Textbox(label="Your name",
					       value="Alice", 
						   placeholder="Type your name…")
		mood = gr.Radio(["Great", "Good", "OK", "Tired"], 
				       label="How are you?", 
					   value="Great")
		greet_btn = gr.Button("Greet")
		greet_out = gr.Textbox(label="Message", interactive=False)
		greet_btn.click(greet, [name, mood], greet_out)

	with gr.Accordion("Quick Utilities", open=False):
		with gr.Row():
			num = gr.Number(label="Number to square", value=10)
			square_btn = gr.Button("Square it")
		result = gr.Number(label="Result", interactive=False)
		square_btn.click(square, num, result)

demo.launch()
