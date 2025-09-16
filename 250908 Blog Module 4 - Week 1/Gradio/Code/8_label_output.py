import math
import re
from typing import Dict
import gradio as gr

POSITIVE = {"good", "great", "excellent", "love", "awesome", "happy", "wonderful", "amazing", "fantastic"}
NEGATIVE = {"bad", "terrible", "awful", "hate", "sad", "angry", "horrible", "poor", "worse"}


def tokenize(text: str):
	return re.findall(r"\w+|[^\w\s]", text or "", flags=re.UNICODE)


def classify_sentiment(text: str) -> Dict[str, float]:
	tokens = [t.lower() for t in tokenize(text)]
	if not tokens:
		return {"neutral": 1.0}

	pos = sum(1 for t in tokens if t in POSITIVE)
	neg = sum(1 for t in tokens if t in NEGATIVE)

	# Convert counts into softmax-like probabilities with a neutral prior
	# Use small smoothing so empty classes don't become 0 exactly
	z_pos = pos + 0.5
	z_neg = neg + 0.5
	z_neu = max(len(tokens) - (pos + neg), 0) * 0.2 + 0.5

	exps = [math.exp(z_pos), math.exp(z_neu), math.exp(z_neg)]
	s = sum(exps)
	probs = [v / s for v in exps]
	return {"positive": probs[0], "neutral": probs[1], "negative": probs[2]}


demo = gr.Interface(
	fn=classify_sentiment,
	inputs=gr.Textbox(
		label="Input text",
		placeholder="e.g. I love this course, it is amazing!",
		lines=4,
	),
	outputs=gr.Label(label="Predicted sentiment"),
	title="Label Output Demo",
	description="gr.Label — simple sentiment classifier demo. " \
                "Type text to see predicted label and confidences.",
	examples=[
		"I love this amazing course but the pacing is bad.",
		"The movie was good but the ending was terrible.",
		"What a wonderful day, I'm so happy.",
		"The service was bad and the food was awful.",
	],
)
demo.launch()