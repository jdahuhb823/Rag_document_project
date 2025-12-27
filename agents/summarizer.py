# Summarizer removed — simplified app provides a single short summary (5-8 bullets) via `app.py` using the project's `llm.py`.
# This stub remains to avoid import errors in older scripts. Do not use in new code.

__all__ = []


def summarize_text(
    text: str,
    llm_predict: Optional[Callable[[str], str]] = None,
    depth: str = "medium",
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    temperature: float = 0.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_output_words: int = 200,
) -> Dict[str, Any]:
    try:
        from rag.loader import get_text_splitter

        splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)

        if not chunks:
            return {"error": "Insufficient information to produce a reliable summary.", "status": "failed"}

        llm = _ensure_llm(llm_predict, temperature=temperature)

        chunk_summaries: List[str] = []
        total = len(chunks)
        for idx, c in enumerate(chunks):
            s = _summarize_chunk(c, llm, depth=depth, temperature=temperature, max_output_words=max_output_words)
            chunk_summaries.append(s.strip())

            if progress_callback is not None:
                try:
                    progress_callback(idx + 1, total)
                except Exception:
                    pass

        num_chunks = max(1, len(chunk_summaries))
        if depth == "short":
            target_sections = min(6, max(1, num_chunks // 8))
        elif depth == "detailed":
            target_sections = min(12, max(2, num_chunks // 4))
        else:
            target_sections = min(8, max(1, num_chunks // 6))
        target_sections = max(1, target_sections)

        group_size = max(1, -(-num_chunks // target_sections))
        section_summaries: List[str] = []
        for i in range(0, num_chunks, group_size):
            group = chunk_summaries[i : i + group_size]
            sec = _synthesize_section(group, llm, depth=depth, temperature=temperature)
            section_summaries.append(sec.strip())

        final = _final_synthesis(section_summaries, llm, depth=depth, temperature=temperature)

        return {"final_summary": final, "chunk_summaries": chunk_summaries, "status": "ok"}

    except Exception as e:
        return {"error": str(e), "status": "failed"}


def summarize_document(
    path: str,
    llm_predict: Optional[Callable[[str], str]] = None,
    depth: str = "medium",
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    temperature: float = 0.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    try:
        docs = load_and_split(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        parts: List[str] = []
        chunks: List[str] = []
        for d in docs:
            content = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
            if content:
                chunks.append(content)

        if not chunks:
            return {"error": "Insufficient information to produce a reliable summary.", "status": "failed"}

        combined_text = "\n\n".join(chunks)

        return summarize_text(
            combined_text,
            llm_predict=llm_predict,
            depth=depth,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temperature=temperature,
            progress_callback=progress_callback,
        )
    except Exception as e:
        return {"error": str(e), "status": "failed"}





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize a document using RAG retrieval")
    parser.add_argument("path", help="Path to file or '-' to read from stdin")
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    if args.path == "-":
        import sys

        text = sys.stdin.read()
        summarize_text(text, max_output_words=args.max_words)
    else:
        summarize_document(args.path, max_output_words=args.max_words)
