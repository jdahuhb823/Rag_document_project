
# Findings extractor removed — simplified app uses concise summary and direct retrieval for Q&A.
# This stub remains to avoid import errors in older scripts. Do not use in new code.

__all__ = []


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = max(0, end - overlap)

    return chunks


def _build_prompt(passages: Sequence[str], max_findings: int = 6) -> str:
    passages_block = "\n\n".join(f"PASSAGE {i+1}:\n{p.strip()}" for i, p in enumerate(passages))
    prompt = (
        "Extract key findings from the passages below and return them as bullet points.\n"
        "Use only the text in the PASSAGE blocks; do not use external knowledge.\n"
        "Return only bullet points (one per line) describing key findings. Each bullet must be prefixed with '- '.\n"
        "Return at most "
        f"{max_findings} bullets. Each bullet should be 1-2 short factual sentences.\n"
        "If there are no extractable findings in the passages, reply exactly: No key findings found.\n\n"
        "When processing legal documents, prioritize clauses, obligations, parties, and dates.\n"
        "When processing transactional or invoice-like documents, prioritize amounts, parties, dates, and identifiers (invoice numbers, transaction IDs).\n\n"
        "PASSAGES:\n\n"
        f"{passages_block}\n\n"
        "Now list the key findings as described above."
    )
    return prompt


def _parse_bullets(raw: str) -> List[str]:
    if not raw:
        return []

    text = raw.strip()
    if text.strip() == "No key findings found.":
        return []

    lines = text.splitlines()
    bullets: List[str] = []
    bullet_re = re.compile(r"^\s*[-*•]\s+(.*)$")
    numbered_re = re.compile(r"^\s*\d+[.)]\s+(.*)$")

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        m = bullet_re.match(ln)
        if m:
            bullets.append(m.group(1).strip())
            continue
        m2 = numbered_re.match(ln)
        if m2:
            bullets.append(m2.group(1).strip())
            continue
        
        if len(ln) < 240:
            bullets.append(ln)

    seen = set()
    cleaned: List[str] = []
    for b in bullets:
        if not b:
            continue
        if b in seen:
            continue
        seen.add(b)
        cleaned.append(b)
    return cleaned


def extract_findings_from_text(
    text: str,
    llm_predict: Optional[Callable[[str], str]] = None,
    max_findings: int = 6,
    top_k: int = 8,
    temperature: float = 0.0,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[str]:
    from rag.loader import get_text_splitter

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    if not chunks:
        return []

    store = FaissVectorStore()
    metadatas = [{"text": c} for c in chunks]
    store.add_documents([m["text"] for m in metadatas], metadatas)
    query = "Extract key findings from the document."
    results = store.search(query, k=top_k)
    passages = [md.get("text", "") for md, _ in results if md.get("text")]
    if not passages:
        return []

    prompt = _build_prompt(passages, max_findings=max_findings)
    if llm_predict is not None:
        raw = llm_predict(prompt)
    else:
        default = get_llm_predict(temperature=temperature)
        raw = default(prompt)

    bullets = _parse_bullets(raw)
    return bullets[:max_findings]


def extract_findings_from_document(
    path: str,
    llm_predict: Optional[Callable[[str], str]] = None,
    max_findings: int = 6,
    top_k: int = 8,
    temperature: float = 0.0,
) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    from rag.loader import load_and_split

    docs = load_and_split(str(p), chunk_size=800, chunk_overlap=100)
    parts = []
    for d in docs:
        parts.append(getattr(d, "page_content", None) or getattr(d, "text", None) or str(d))
    text = "\n\n".join(parts)

    return extract_findings_from_text(
        text,
        llm_predict=llm_predict,
        max_findings=max_findings,
        top_k=top_k,
        temperature=temperature,
    )


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Extract key findings as bullet points")
	parser.add_argument("path", help="Path to file or '-' to read from stdin")
	parser.add_argument("--max", type=int, default=6, help="Maximum number of findings to return")
	args = parser.parse_args()

	if args.path == "-":
		import sys

		text = sys.stdin.read()
		bullets = extract_findings_from_text(text, max_findings=args.max)
	else:
		bullets = extract_findings_from_document(args.path, max_findings=args.max)

	if not bullets:
		pass
	else:
		for b in bullets:
			pass
