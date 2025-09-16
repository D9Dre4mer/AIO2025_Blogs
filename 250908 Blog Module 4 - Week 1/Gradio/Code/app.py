import gradio as gr
from typing import Optional, Tuple, Any
import numpy as np
from PIL import Image, ImageOps

def _to_pil(img: Any) -> Optional[Any]:
	if img is None:
		return None
	if Image is None:
		raise RuntimeError("Pillow not available. Please install 'pillow'.")
	if isinstance(img, Image.Image):
		return img
	arr = np.asarray(img)
	if not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (3, 4))):
		raise ValueError("Unsupported image array shape")
	return Image.fromarray(arr.astype(np.uint8))


def preprocess_image(img: Any, max_side: int = 512, progress: Optional[gr.Progress] = None) -> Optional[Any]:
	if img is None:
		gr.Warning("Please upload an image first.")
		return None
	if progress:
		progress(0, desc="Loading image…")
	pil = _to_pil(img)
	if pil is None:
		return None
	if progress:
		progress(0.3, desc="Resizing…")
	# Keep aspect ratio, cap the longest side
	w, h = pil.size
	scale = min(1.0, max_side / max(w, h))
	if scale < 1.0:
		pil = pil.resize((int(w * scale), int(h * scale)))
	if progress:
		progress(0.7, desc="Auto-contrast…")
	pil = ImageOps.autocontrast(pil)
	if progress:
		progress(1.0, desc="Done")
	return pil


def detect_edges(img: Any, strength: float = 1.0, progress: Optional[gr.Progress] = None) -> Optional[Any]:
	if img is None:
		gr.Warning("Please run Preprocess first or upload an image.")
		return None
	pil = _to_pil(img).convert("L")  # grayscale
	if progress:
		progress(0.2, desc="Computing gradients…")
	arr = np.asarray(pil, dtype=np.float32)
	# Use numpy gradient as a simple edge detector (fast and dependency-free)
	gy, gx = np.gradient(arr)
	mag = np.hypot(gx, gy)
	mag *= (255.0 / (mag.max() + 1e-6))
	if progress:
		progress(0.7, desc="Applying strength…")
	mag = np.clip(mag * float(max(0.1, strength)), 0, 255).astype(np.uint8)
	out = Image.fromarray(mag)
	if progress:
		progress(1.0, desc="Done")
	return out


def enhance_image(img: Any, progress: Optional[gr.Progress] = None) -> Optional[Any]:
	if img is None:
		gr.Warning("Please run Detect Edges first.")
		return None
	pil = _to_pil(img)
	if progress:
		progress(0.5, desc="Enhancing…")
	# Simple enhancement via auto-contrast again; could be extended
	pil = ImageOps.autocontrast(pil)
	if progress:
		progress(1.0, desc="Done")
	return pil


def run_all_image(image: Any, strength: float = 1.0, progress: Optional[gr.Progress] = None):
	if image is None:
		gr.Warning("Please upload an image.")
		return None, None, None
	# Use the same progress object for simplicity
	p = preprocess_image(image, progress=progress)
	e = detect_edges(p, strength=strength, progress=progress)
	h = enhance_image(e, progress=progress)
	return p, e, h


# -----------------------
# Text pipeline helpers
# -----------------------
def clean_text(text: str) -> str:
	if not text:
		gr.Warning("Please enter text.")
		return ""
	# Normalize whitespace and quotes
	cleaned = " ".join(text.strip().split())
	return cleaned


def summarize_text(text: str, max_sentences: int = 2) -> str:
	if not text:
		gr.Warning("Please clean the text first.")
		return ""
	# Naive sentence-based summarization: pick first N sentences
	import re

	sents = re.split(r"(?<=[.!?])\s+", text)
	summary = " ".join(sents[: max(1, int(max_sentences))])
	return summary


def sentiment(text: str) -> Tuple[str, float]:
	if not text:
		gr.Warning("Please provide text.")
		return ("neutral", 0.0)
	# Tiny lexicon-based scorer
	pos = {"good", "great", "excellent", "amazing", "love", "like", "happy", "awesome", "fantastic"}
	neg = {"bad", "terrible", "awful", "hate", "dislike", "sad", "poor", "horrible", "worse"}
	words = [w.strip(".,!?;:").lower() for w in text.split()]
	score = sum(1 for w in words if w in pos) - sum(1 for w in words if w in neg)
	label = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
	# Normalize score into [-1, 1] by a simple squash
	norm = max(1.0, len(words) / 10.0)
	val = float(score / norm)
	# Clamp to [-1, 1]
	val = max(-1.0, min(1.0, val))
	return (label, val)


with gr.Blocks(title="Complex Multi-step Workflows", theme=gr.themes.Soft()) as demo:
	gr.Markdown("""
	# Complex Apps with Gradio Blocks
	Multi-step workflows across image and text pipelines. Each step updates state and UI.
	""")

	with gr.Tabs():
		# ---------------- Image pipeline tab ----------------
		with gr.TabItem("Image Pipeline"):
			with gr.Row():
				with gr.Column(scale=1):
					image_in = gr.Image(label="Upload Image", type="pil")
					strength = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Edge Strength")
					# Removed Demo Delay slider
					with gr.Row():
						btn_pre = gr.Button("Step 1: Preprocess")
						btn_edge = gr.Button("Step 2: Detect Edges")
						btn_enh = gr.Button("Step 3: Enhance")
					with gr.Row():
						btn_run_all = gr.Button("Run All", variant="primary")
						btn_reset_img = gr.Button("Reset")

					# Internal states to pass between steps
					st_pre = gr.State()
					st_edge = gr.State()

				with gr.Column(scale=1):
					out_pre = gr.Image(label="Preprocessed", interactive=False)
					out_edge = gr.Image(label="Edges", interactive=False)
					out_enh = gr.Image(label="Enhanced", interactive=False)

			# Wiring events for image pipeline
			def _preprocess_and_store(img, progress=gr.Progress(track_tqdm=True)):
				p = preprocess_image(img, progress=progress)
				return p, p

			btn_pre.click(_preprocess_and_store, inputs=[image_in], outputs=[out_pre, st_pre])

			def _edge_and_store(img_pre, k, progress=gr.Progress(track_tqdm=True)):
				if img_pre is None:
					gr.Warning("Run Step 1 first.")
					return None, None
				e = detect_edges(img_pre, strength=k, progress=progress)
				return e, e

			btn_edge.click(_edge_and_store, inputs=[st_pre, strength], outputs=[out_edge, st_edge])

			def _enhance(img_edge, progress=gr.Progress(track_tqdm=True)):
				if img_edge is None:
					gr.Warning("Run Step 2 first.")
					return None
				return enhance_image(img_edge, progress=progress)

			btn_enh.click(_enhance, inputs=[st_edge], outputs=out_enh)

			def _run_all(img, k, progress=gr.Progress(track_tqdm=True)):
				p, e, h = run_all_image(img, k, progress=progress)
				# Also store states for continuity
				return p, e, h, p, e

			btn_run_all.click(_run_all, inputs=[image_in, strength], outputs=[out_pre, out_edge, out_enh, st_pre, st_edge])

			def _reset_img():
				return None, None, None, None, None

			btn_reset_img.click(_reset_img, outputs=[image_in, out_pre, out_edge, out_enh, st_pre])

		# ---------------- Text pipeline tab ----------------
		with gr.TabItem("Text Pipeline"):
			with gr.Row():
				with gr.Column(scale=1):
					text_in = gr.Textbox(label="Input Text", lines=8, placeholder="Paste or type some text…")
					with gr.Accordion("Options", open=False):
						max_sents = gr.Slider(1, 5, value=2, step=1, label="Summary Sentences")
					with gr.Row():
						btn_clean = gr.Button("Step 1: Clean")
						btn_sum = gr.Button("Step 2: Summarize")
						btn_sent = gr.Button("Step 3: Sentiment")
					with gr.Row():
						btn_run_all_txt = gr.Button("Run All", variant="primary")
						btn_reset_txt = gr.Button("Reset")

					st_clean = gr.State()
					st_sum = gr.State()

				with gr.Column(scale=1):
					out_clean = gr.Textbox(label="Cleaned Text", lines=8)
					out_sum = gr.Textbox(label="Summary", lines=6)
					out_sent = gr.Label(label="Sentiment")

			# Wiring events for text pipeline
			def _clean_and_store(t):
				c = clean_text(t)
				return c, c

			btn_clean.click(_clean_and_store, inputs=text_in, outputs=[out_clean, st_clean])

			def _summarize_and_store(c, n):
				if not c:
					gr.Warning("Run Step 1 first.")
					return "", ""
				s = summarize_text(c, int(n))
				return s, s

			btn_sum.click(_summarize_and_store, inputs=[st_clean, max_sents], outputs=[out_sum, st_sum])

			def _sentiment(s):
				if not s:
					gr.Warning("Run Step 2 first.")
					return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
				label, score = sentiment(s)
				# Map score in [-1,1] to a 3-class distribution
				p_pos = max(0.0, score)
				p_neg = max(0.0, -score)
				p_neu = 1.0 - abs(score)
				return {"positive": round(p_pos, 3), "neutral": round(p_neu, 3), "negative": round(p_neg, 3)}

			btn_sent.click(_sentiment, inputs=st_sum, outputs=out_sent)

			def _run_all_txt(t, n):
				c = clean_text(t)
				s = summarize_text(c, int(n))
				label, score = sentiment(s)
				p_pos = max(0.0, score)
				p_neg = max(0.0, -score)
				p_neu = 1.0 - abs(score)
				return c, s, {"positive": round(p_pos, 3), "neutral": round(p_neu, 3), "negative": round(p_neg, 3)}, c, s

			btn_run_all_txt.click(_run_all_txt, inputs=[text_in, max_sents], outputs=[out_clean, out_sum, out_sent, st_clean, st_sum])

			def _reset_txt():
				return "", "", None, "", ""

			btn_reset_txt.click(_reset_txt, outputs=[text_in, out_sum, out_sent, st_clean, st_sum])

demo.queue().launch()