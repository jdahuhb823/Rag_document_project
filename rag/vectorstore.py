

from __future__ import annotations

from typing import List, Dict, Any, Optional, Sequence, Tuple, Union
from pathlib import Path
import pickle

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None



class FaissVectorStore:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence_transformers not installed. Install with: pip install sentence-transformers"
            )
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss (faiss-cpu) to use this module.")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadatas: List[Dict[str, Any]] = []

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        embs = self.model.encode(list(texts), convert_to_numpy=True)
        embs = embs.astype("float32")
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embs = embs / norms
        return embs

    def add_documents(self, texts: Sequence[str], metadatas: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts")

        embs = self._embed(texts)
        self.index.add(embs)
        if metadatas is None:
            for t in texts:
                self.metadatas.append({"text": t})
        else:
            for md in metadatas:
                self.metadatas.append(dict(md))

    def search(self, query: str, k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        q_emb = self._embed([query])
        D, I = self.index.search(q_emb, k)
        D = D[0]
        I = I[0]

        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, score in zip(I, D):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            results.append((self.metadatas[idx], float(score)))
        return results

    def save(self, folder: Union[str, Path]) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        with open(folder / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadatas, f)
        with open(folder / "store_info.pkl", "wb") as f:
            pickle.dump({"model_name": self.model_name, "dim": self.dim}, f)

    @classmethod
    def load(cls, folder: Union[str, Path]) -> "FaissVectorStore":
        folder = Path(folder)
        if not (folder / "index.faiss").exists():
            raise FileNotFoundError(f"No FAISS index found in {folder}")

        with open(folder / "store_info.pkl", "rb") as f:
            info = pickle.load(f)

        obj = cls(model_name=info.get("model_name", "all-MiniLM-L6-v2"))
        obj.index = faiss.read_index(str(folder / "index.faiss"))
        with open(folder / "metadata.pkl", "rb") as f:
            obj.metadatas = pickle.load(f)
        return obj


def from_documents(
    documents: Sequence[Union[str, Dict[str, Any]]],
    model_name: str = "all-MiniLM-L6-v2",
) -> FaissVectorStore:
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for d in documents:
        if isinstance(d, str):
            texts.append(d)
            metadatas.append({"text": d})
        elif isinstance(d, dict):
            text = d.get("text")
            if not text:
                raise ValueError("Document dicts must contain a 'text' key")
            texts.append(text)
            md = dict(d)
            metadatas.append(md)
        else:
            raise TypeError("documents must be str or dict with 'text' key")

    store = FaissVectorStore(model_name=model_name)
    store.add_documents(texts, metadatas)
    return store


def from_langchain_documents(
    documents: Sequence[Any],
    model_name: str = "all-MiniLM-L6-v2",
) -> FaissVectorStore:
    """Build a FaissVectorStore from langchain Document-like objects.

    Each document should have `page_content` (or `text`) and optional `metadata` dict.
    """
    docs: List[Dict[str, Any]] = []
    for d in documents:
        text = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
        md = dict(getattr(d, "metadata", {}) or {})
        docs.append({"text": text, **md})
    return from_documents(docs, model_name=model_name)


__all__ = ["FaissVectorStore", "from_documents"]
