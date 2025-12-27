from __future__ import annotations

import streamlit as st
from pathlib import Path
import tempfile
from typing import List

from rag.loader import load_and_split
from rag.vectorstore import from_langchain_documents, FaissVectorStore
from llm import get_llm_predict


st.set_page_config(page_title="Simple RAG — Summarize & Q&A", layout="wide")


def save_uploaded_file(uploaded) -> str:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        return tmp.name


def build_store_from_docs(docs, model_name: str = "all-MiniLM-L6-v2") -> FaissVectorStore:
    # convert langchain Document objects to a Faiss store
    return from_langchain_documents(docs, model_name=model_name)


def _parse_bullets_from_text(raw: str, max_bullets: int = 8) -> str:
    if not raw or not raw.strip():
        return ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    bullets = []
    for ln in lines:
        if ln.startswith("-") or ln.startswith("•") or ln[0].isdigit():
            # strip leading bullet markers/numbers
            cleaned = ln.lstrip("-•0123456789. )\t").strip()
            if cleaned:
                bullets.append(f"- {cleaned}")
        elif "." in ln:
            # split into sentences and take the first sentence
            sent = ln.split(".")
            first = sent[0].strip()
            if first:
                bullets.append(f"- {first}")
        else:
            bullets.append(f"- {ln}")
        if len(bullets) >= max_bullets:
            break
    # ensure at least one bullet
    if not bullets and raw:
        bullets = [f"- {s.strip()}" for s in raw.split("\n") if s.strip()][:max_bullets]
    return "\n".join(bullets[:max_bullets])


def make_summary_from_chunks(chunks: List[str], llm_predict, max_passages: int = 10) -> str:
    # merge all chunks into a single corpus and re-chunk for summary passages
    if not chunks:
        return "Insufficient information to produce a reliable summary."

    from rag.loader import get_text_splitter

    combined_text = "\n\n".join(chunks)
    splitter = get_text_splitter(chunk_size=1200, chunk_overlap=200)
    passages = splitter.split_text(combined_text) if hasattr(splitter, 'split_text') else []
    # fallback: use original chunks if splitter didn't produce
    if not passages:
        passages = [p for p in chunks if p and p.strip()][:max_passages]

    # pick top passages by length
    passages = [p for p in sorted(passages, key=lambda x: len(x), reverse=True)[:max_passages] if p and p.strip()]

    if not passages:
        return "Insufficient information to produce a reliable summary."

    # Limit total prompt size to avoid model timeouts; keep the longest passages until max_chars reached
    max_input_chars = 3500
    chosen = []
    total = 0
    for p in passages:
        piece = p.strip()
        add_len = min(len(piece), 1000)
        if total + add_len > max_input_chars and chosen:
            break
        chosen.append(piece[:1000])
        total += add_len

    passages_block = "\n\n".join(f"PASSAGE {i+1}:\n{p}" for i, p in enumerate(chosen))
    prompt = (
        "Using ONLY the passages below, produce a concise executive summary as 5 to 8 short bullet points."
        " Do NOT introduce external facts — summarize the provided text. Each bullet should be 1-2 short sentences."
        " If fewer than 5 distinct points are present, make a best-effort concise list up to 8 bullets summarizing the content.\n\n"
        "PASSAGES:\n\n"
        f"{passages_block}\n\n"
        "Provide the summary now as 5–8 short bullet points (each bullet starting with '- '):"
    )

    raw = llm_predict(prompt).strip()
    bullets = _parse_bullets_from_text(raw, max_bullets=8)
    if not bullets:
        # fallback: try to synthesize short bullets from raw text
        fallback = "\n".join([f"- {s.strip()}" for s in raw.split("\n") if s.strip()])
        bullets = fallback or ("- Summary could not be generated from provided passages.")

    return bullets


def answer_question_from_store(question: str, store: FaissVectorStore, llm_predict, top_k: int = 5) -> dict:
    hits = store.search(question, k=top_k)
    passages = []
    for i, (md, score) in enumerate(hits, 1):
        text = md.get("text") or md.get("content") or ""
        src = md.get("source")
        label = f"(source: {src})" if src else ""
        if text and text.strip():
            passages.append((f"PASSAGE {i}: {text}", label, score))

    if not passages:
        return {"answer": "No relevant passages found for this question.", "hits": []}

    # Trim and limit passages to avoid overly long prompts
    max_input_chars = 3000
    chosen = []
    total = 0
    for p, l, s in passages:
        text = p
        piece = text.strip()[:800]
        if total + len(piece) > max_input_chars and chosen:
            break
        chosen.append((piece, l, s))
        total += len(piece)

    passages_block = "\n\n".join(p for p, l, s in chosen)

    prompt = (
        "Answer the question using ONLY the passages below. Cite sources when possible in parentheses."
        " If the answer is not contained in the passages, reply: 'I don't know — the documents do not contain sufficient information.'"
        " Keep the answer concise (1-3 sentences).\n\n"
        f"QUESTION: {question}\n\n"
        "PASSAGES:\n\n"
        f"{passages_block}\n\n"
        "Provide the answer now."
    )

    try:
        answer = llm_predict(prompt)
    except Exception as e:
        return {"answer": f"LLM error while answering: {e}", "hits": passages}

    if not answer or not answer.strip():
        return {"answer": "LLM returned an empty answer.", "hits": passages}

    return {"answer": answer.strip(), "hits": passages}


def main():
    st.title("Simple RAG — Summarize & Q&A")

    with st.sidebar:
        st.markdown("### Upload Documents")
        uploaded_files = st.file_uploader("Upload files (PDF, DOCX, TXT, PPTX, HTML, CSV, XLSX)", accept_multiple_files=True)
        raw_text = st.text_area("Or paste text here (optional)", height=150)
        model_name = st.text_input("Embedding model", value="all-MiniLM-L6-v2")
        summarize_btn = st.button("Generate Summary")

    st.markdown("---")

    llm_predict = None
    try:
        llm_predict = get_llm_predict()
    except Exception as e:
        st.warning(f"No LLM available: {e}. LLM features will be disabled.")

    # session state to hold store, chunks and summary
    if "store" not in st.session_state:
        st.session_state.store = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "summary_meta" not in st.session_state:
        st.session_state.summary_meta = {}

    # prepare input files
    paths = []
    if uploaded_files:
        for f in uploaded_files:
            suffix = Path(f.name).suffix.lower()
            supported = {".pdf", ".txt", ".docx", ".pptx", ".xml", ".json", ".html", ".htm", ".csv", ".xlsx"}
            if suffix not in supported:
                st.error(f"Unsupported file type: {suffix}")
                return
            p = save_uploaded_file(f)
            paths.append(p)

    if raw_text and not paths:
        # if only raw text provided, treat it like a single document
        paths.append(raw_text)

    # clear summary immediately if uploaded files or pasted text changed since last summary
    current_uploaded_names = tuple(sorted([f.name for f in uploaded_files])) if uploaded_files else None
    current_raw_present = bool(raw_text and raw_text.strip())
    if st.session_state.summary is not None:
        prev_uploaded = st.session_state.summary_meta.get("uploaded_names")
        prev_raw = st.session_state.summary_meta.get("raw_text_present")
        if prev_uploaded != current_uploaded_names or prev_raw != current_raw_present:
            st.session_state.summary = None
            st.session_state.summary_meta = {}

    if summarize_btn:
        if not paths:
            st.warning("Please upload files or paste text to summarize.")
        else:
            with st.spinner("Loading and chunking documents..."):
                docs = load_and_split(paths)
                # filter out empty/whitespace-only chunks
                chunks = []
                for d in docs:
                    content = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
                    if content and content.strip():
                        chunks.append(content.strip())
                st.session_state.chunks = chunks

                # show diagnostics: number of extracted chunks and sources
                sources = [((getattr(d, "metadata", {}) or {}).get("source") or "<unknown>") for d in docs]
                st.info(f"Extracted {len(chunks)} chunks from {len(set(sources))} source files.")

                if not chunks:
                    # show helpful diagnostics and avoid calling the LLM
                    st.error("No text could be extracted from the uploaded documents. Try pasting text or uploading plain-text / DOCX files. If you uploaded PDFs, ensure a PDF loader is available (install pypdf or unstructured).")
                    st.markdown("**Detected document sources (for debugging):**")
                    st.write(sources)
                    st.session_state.store = None
                else:
                    with st.spinner("Building vector store (this may take a few seconds)..."):
                        try:
                            st.session_state.store = build_store_from_docs(docs, model_name=model_name)
                        except Exception as e:
                            st.error(f"Failed to build vector store: {e}")
                            st.session_state.store = None

                    # compute summary fingerprint/meta to avoid re-generating for same uploads
                    combined_text = "\n\n".join(st.session_state.chunks)
                    current_meta = {
                        "sources": tuple(sorted(set(sources))) ,
                        "combined_len": len(combined_text),
                        "uploaded_names": tuple(sorted([f.name for f in uploaded_files])) if uploaded_files else None,
                        "raw_text_present": bool(raw_text and raw_text.strip()),
                    }

                    # clear stored summary if the documents changed
                    if st.session_state.summary is not None and st.session_state.summary_meta != current_meta:
                        st.session_state.summary = None
                        st.session_state.summary_meta = {}

                    # Generate summary only if not already present for these docs
                    if st.session_state.summary is None:
                        if not llm_predict:
                            st.error("LLM is not available to generate summary.")
                        else:
                            try:
                                with st.spinner("Generating short summary..."):
                                    summary = make_summary_from_chunks(st.session_state.chunks, llm_predict)
                                if not summary or not summary.strip():
                                    st.error("LLM returned an empty summary. Try a different model or check document content.")
                                else:
                                    st.session_state.summary = summary
                                    st.session_state.summary_meta = current_meta
                            except Exception as e:
                                st.error(f"Summary generation failed: {e}")
                    else:
                        st.info("Summary already generated for current documents. Use the existing summary or change uploaded files to regenerate.")

    st.markdown("---")

    st.markdown("### Summary")
    if st.session_state.summary:
        st.write(st.session_state.summary)
    else:
        st.info("No summary generated yet. Click 'Generate Summary' to create a short (5–8 bullets) summary for uploaded documents.")

    st.markdown("---")

    st.markdown("### Ask a question")
    question = st.text_input("Ask a question about the uploaded documents")
    if st.button("Get answer"):
        if not question.strip():
            st.warning("Please type a question first.")
        elif st.session_state.store is None:
            st.warning("Please generate a summary (build the knowledge base) first by clicking 'Generate Summary'.")
        elif not llm_predict:
            st.error("LLM is not available to generate answers.")
        else:
            with st.spinner("Retrieving relevant passages and generating answer..."):
                result = answer_question_from_store(question, st.session_state.store, llm_predict)
            st.markdown("**Answer:**")
            st.write(result["answer"])

            with st.expander("Show retrieved passages"):
                for i, (text_label, src_label, score) in enumerate(result["hits"], 1):
                    st.markdown(f"**Passage {i}** {src_label} — score: {score:.3f}")
                    st.write(text_label.replace(f"PASSAGE {i}:", ""))


if __name__ == "__main__":
    main()
