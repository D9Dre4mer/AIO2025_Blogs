import re
from typing import List, Optional, Tuple
import gradio as gr

POSITIVE = {
	"good",
	"great",
	"excellent",
	"love",
	"awesome",
	"happy",
	"wonderful",
	"amazing",
	"fantastic",
}
NEGATIVE = {
	"bad",
	"terrible",
	"awful",
	"hate",
	"sad",
	"angry",
	"horrible",
	"poor",
	"worse",
}


def tokenize(text: str) -> List[str]:
	"""Split text into simple word/punctuation tokens.

	This is intentionally simple for an entry-level demo.
	"""
	if not text:
		return []
	# Words or single punctuation characters
	return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def highlight_sentiment(text: str) -> List[Tuple[str, Optional[str]]]:
	"""Return tokens with optional sentiment labels for HighlightedText.

	Output shape: List[(token: str, label: Optional[str])]
	"""
	tokens = tokenize(text)
	highlighted: List[Tuple[str, Optional[str]]] = []
	for tok in tokens:
		label: Optional[str]
		low = tok.lower()
		if low in POSITIVE:
			label = "positive"
		elif low in NEGATIVE:
			label = "negative"
		else:
			label = None
		highlighted.append((tok, label))
	return highlighted


demo = gr.Interface(
	fn=highlight_sentiment,
	inputs=gr.Textbox(
		label="Input text",
		placeholder="I love this amazing course.",
		lines=4,
	),
	outputs=gr.HighlightedText(
		label="Token highlights",
		color_map={
			"positive": "#10b981",
			"negative": "#ef4444",
		},
	),
	title="HighlightedText Demo",
	description="Tiny sentiment demo: " \
                "highlights a few positive/negative words.",
	examples=[
		"I love this course, it is amazing!",
		"The movie was good but the ending was terrible.",
		"What a wonderful day, I'm so happy.",
		"The service was bad and the food was awful.",
	],
)
demo.launch()