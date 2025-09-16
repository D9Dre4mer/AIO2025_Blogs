import collections
import re
from typing import Dict, List

import gradio as gr


def tokenize(text: str) -> List[str]:
	return re.findall(r"\w+", text or "", flags=re.UNICODE)


def analyze(text: str) -> Dict:
	tokens = [t.lower() for t in tokenize(text)]
	n_tokens = len(tokens)
	unique = sorted(set(tokens))
	freq = collections.Counter(tokens)
	top5 = freq.most_common(5)

	# simple sentence split (very rough):
	sentences = [s.strip() for s in re.split(r"[.!?]+", text or "") if s.strip()]

	return {
		"summary": {
			"num_tokens": n_tokens,
			"num_unique": len(unique),
			"num_sentences": len(sentences),
		},
		"top_5_tokens": [{"token": t, "count": c} for t, c in top5],
		"unique_tokens_sample": unique[:20],
		"sentences": sentences,
		"meta": {
			"note": "This is a simple, rule-based analysis for demo purposes.",
			"version": 1,
		},
	}


demo = gr.Interface(
	fn=analyze,
	inputs=gr.Textbox(
		label="Input text",
		placeholder="e.g. Gradio is great for prototyping ML apps!",
		lines=6,
	),
	outputs=gr.JSON(label="Analysis JSON"),
	title="JSON Output Demo",
	description="Type some text and get a structured JSON summary.",
	examples=[
		"Hello world. Hello Gradio. This is a simple JSON output demo.",
		"One two two three three three.",
		"Gradio makes it easy to build demos. JSON shows structured data.",
	],
)

demo.launch()