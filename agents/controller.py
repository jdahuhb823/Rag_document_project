
# Controller removed — simplified app uses `rag.loader`, `rag.vectorstore`, and `llm.py` directly.
# This stub remains to avoid import errors in older scripts. Do not use in new code.

__all__ = []

	def _read_file_text(self, path: str) -> str:
		p = Path(path)
		if not p.exists():
			raise FileNotFoundError(f"File not found: {p}")

		from rag.loader import load_and_split

		docs = load_and_split(str(p), chunk_size=800, chunk_overlap=100)
		parts: List[str] = []
		for d in docs:
			parts.append(getattr(d, "page_content", None) or getattr(d, "text", None) or str(d))
		return "\n\n".join(parts)

	def process_file(
		self,
		path: str,
		labels: Optional[Sequence[str]] = None,
		run_classify: bool = True,
		run_summarize: bool = True,
		run_findings: bool = True,
		summarize_max_words: int = 200,
		summary_depth: str = "medium",
	) -> Dict[str, Any]:
		text = self._read_file_text(path)
		return self.process_text(
			text,
			labels=labels,
			run_classify=run_classify,
			run_summarize=run_summarize,
			run_findings=run_findings,
			summarize_max_words=summarize_max_words,
			summary_depth=summary_depth,
		)

	def process_text(
		self,
		text: str,
		labels: Optional[Sequence[str]] = None,
		run_classify: bool = True,
		run_summarize: bool = True,
		run_findings: bool = True,
		summarize_max_words: int = 200,
		summary_depth: str = "medium",
		source_path: Optional[str] = None,
	) -> Dict[str, Any]:
		results: Dict[str, Any] = {"label": None, "summary": None, "findings": [], "summary_details": None}

		effective_labels = list(labels) if labels is not None else self.labels
		if run_classify:
			if not effective_labels:
				raise ValueError("No labels provided for classification")
			label = classify_document(
				text,
				effective_labels,
				llm_predict=self.llm_predict,
				temperature=self.temperature,
			)
			results["label"] = label

		if run_summarize:
			try:
				progress_cb = None
				try:
					import streamlit as st
					progress_bar = st.progress(0)
					status = st.empty()
					def _progress(current: int, total: int) -> None:
						pct = int((current / total) * 100)
						progress_bar.progress(pct)
						status.text(f"Summarizing chunk {current}/{total} ({pct}%)")
					progress_cb = _progress
				except Exception:
					progress_cb = None

				if source_path:
					summary = summarize_document(
						source_path,
						llm_predict=self.llm_predict,
						depth=summary_depth,
						temperature=self.temperature,
						chunk_size=4000,
						progress_callback=progress_cb,
					)
				else:
					from agents.summarizer import summarize_text
					summary = summarize_text(
						text,
						llm_predict=self.llm_predict,
						depth=summary_depth,
						chunk_size=4000,
						progress_callback=progress_cb,
						temperature=self.temperature,
					)

				if isinstance(summary, dict):
					if summary.get("status") == "ok":
						results["summary"] = summary.get("final_summary")
						results["summary_details"] = summary
					else:
						results["summary"] = None
						results["summary_details"] = summary
				else:
					results["summary"] = str(summary)
					results["summary_details"] = {"status": "ok", "final_summary": str(summary)}

				try:
					if 'progress_bar' in locals():
						progress_bar.progress(100)
						status.text("Summarization complete.")
				except Exception:
					pass
			except Exception as e:
				results["summary"] = None
				results["summary_details"] = {"status": "failed", "error": str(e)}
		if run_findings:
			try:
				findings = extract_findings_from_document(
					text,
					llm_predict=self.llm_predict,
					temperature=self.temperature,
				)
				if findings is None:
					findings = []
				results["findings"] = list(findings)
			except Exception as e:
				results["findings"] = []

		return results