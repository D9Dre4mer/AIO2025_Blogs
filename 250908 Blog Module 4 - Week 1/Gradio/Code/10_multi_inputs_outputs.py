import re
import math
from typing import Dict, List, Tuple

import gradio as gr


POSITIVE = {"good", "great", "excellent", "love", "awesome", "happy", "wonderful", "amazing", "fantastic"}
NEGATIVE = {"bad", "terrible", "awful", "hate", "sad", "angry", "horrible", "poor", "worse"}


def tokenize(text: str) -> List[str]:
	return re.findall(r"\w+", text or "", flags=re.UNICODE)


def multi_analyze(text: str, threshold: float, verbose: bool) -> Tuple[Dict[str, float], Dict, str]:
	tokens = [t.lower() for t in tokenize(text)]
	n = len(tokens)
	if n == 0:
		label = {"neutral": 1.0}
		js = {"summary": {"num_tokens": 0, "pos": 0, "neg": 0, "pos_ratio": 0.0, "neg_ratio": 0.0}, "params": {"threshold": threshold, "verbose": verbose}}
		txt = "No text provided."
		return label, js, txt

	pos = sum(1 for t in tokens if t in POSITIVE)
	neg = sum(1 for t in tokens if t in NEGATIVE)
	pos_ratio = pos / n
	neg_ratio = neg / n

	# Confidence via softmax over [pos, neutral_prior, neg]
	neutral_prior = max(0.2, 1.0 - (pos_ratio + neg_ratio))
	exps = [math.exp(pos_ratio), math.exp(neutral_prior), math.exp(neg_ratio)]
	s = sum(exps)
	confidences = {"positive": exps[0] / s, "neutral": exps[1] / s, "negative": exps[2] / s}

	# Apply threshold to choose class (for description only)
	if pos_ratio >= threshold and pos_ratio >= neg_ratio:
		pred = "positive"
	elif neg_ratio >= threshold and neg_ratio > pos_ratio:
		pred = "negative"
	else:
		pred = "neutral"

	js = {
		"summary": {"num_tokens": n, "pos": pos, "neg": neg, "pos_ratio": round(pos_ratio, 3), "neg_ratio": round(neg_ratio, 3)},
		"params": {"threshold": threshold, "verbose": verbose},
	}

	if verbose:
		js["tokens_sample"] = tokens[:20]

	txt = f"Predicted: {pred}\nPos ratio: {pos_ratio:.2f}, Neg ratio: {neg_ratio:.2f}, Threshold: {threshold:.2f}"

	return confidences, js, txt


demo = gr.Interface(
	fn=multi_analyze,
	inputs=[
		gr.Textbox(label="Text", placeholder="Type a sentence...", lines=4),
		gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.4),
		gr.Checkbox(label="Verbose JSON", value=True),
	],
	outputs=[
		gr.Label(label="Sentiment"),
		gr.JSON(label="JSON Summary"),
		gr.Textbox(label="Text Summary", lines=3),
	],
	title="Multi-Input / Multi-Output Demo",
	description="Enter text, set a threshold, toggle verbosity. " \
                "Returns a label, JSON summary, and a short text summary.",
	examples=[
		["I love this amazing course but some parts are bad.", 0.4, True],
		["The service was awful and the food was horrible.", 0.5, True],
		["Just a regular day.", 0.3, False],
	],
)
demo.launch()
