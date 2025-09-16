import gradio as gr

def echo(text: str) -> str:
    text = text or ""
    return f"You said: {text}"


def add(a: float, b: float) -> float:
    return (a or 0) + (b or 0)


with gr.Blocks(title="Blocks + Tabs: simplest") as demo:
    gr.Markdown("## Simple Blocks + Tabs demo")

    with gr.Tabs():
        with gr.TabItem("Echo"):
            with gr.Row():
                inp = gr.Textbox(label="Text", placeholder="Type something")
                btn1 = gr.Button("Echo")
            out1 = gr.Textbox(label="Output", interactive=False)
            btn1.click(echo, inputs=inp, outputs=out1)

        with gr.TabItem("Add"):
            with gr.Row():
                a = gr.Number(value=0, label="A")
                b = gr.Number(value=0, label="B")
                btn2 = gr.Button("Add")
            out2 = gr.Number(label="Sum", interactive=False)
            btn2.click(add, inputs=[a, b], outputs=out2)

demo.launch()
