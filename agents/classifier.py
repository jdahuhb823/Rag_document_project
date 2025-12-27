

# Classifier removed — simplified app focuses on summarization and Q&A using loaded documents.
# This stub remains to avoid import errors in older scripts. Do not use in new code.

DEFAULT_LABELS = []


def _build_prompt(labels: Sequence[str], document: str) -> str:
	labels_block = "\n".join(f"- {l}" for l in labels)
	prompt = (
		"Classify the document by choosing exactly ONE label from the list below.\n"
		"Return ONLY the single label (exactly as shown in the list), with no extra text.\n"
		"If multiple labels apply, select the most specific one.\n\n"
		"ALLOWED LABELS:\n"
		f"{labels_block}\n\n"
		"DOCUMENT:\n"
		f"{document}\n\n"
		"Return only the chosen label:" 
	)
	return prompt


def classify_document(
	document: str,
	labels: Sequence[str],
	llm_predict: Optional[Callable[[str], str]] = None,
	temperature: float = 0.0,
) -> str:
	if not labels:
		raise ValueError("labels must be a non-empty sequence")
	if not document:
		raise ValueError("document must be a non-empty string")

	prompt = _build_prompt(labels, document)

	if llm_predict is not None:
		raw = llm_predict(prompt)
	else:
		default = get_llm_predict(temperature=temperature)
		raw = default(prompt)

	if raw is None:
		raise RuntimeError("LLM produced no output")

	candidate = raw.strip().splitlines()[0].strip()

	if candidate in labels:
		return candidate

	lower_map = {l.lower(): l for l in labels}
	if candidate.lower() in lower_map:
		return lower_map[candidate.lower()] 

	close = get_close_matches(candidate, labels, n=1, cutoff=0.6)
	if close:
		return close[0]

	doc_lower = document.lower()
	best_label = max(labels, key=lambda l: doc_lower.count(l.lower()))
	return best_label


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Classify a document into one label using an LLM.")
	parser.add_argument("--file", type=str, help="Path to a text file to classify")
	parser.add_argument("--labels", type=str, nargs="+", required=True, help="Allowed labels")
	args = parser.parse_args()

	if not args.file:
		pass
	else:
		with open(args.file, "r", encoding="utf-8") as fh:
			text = fh.read()
		try:
			label = classify_document(text, args.labels)
			pass
		except Exception as e:
			pass
