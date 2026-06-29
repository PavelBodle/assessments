"""Retrieval layer — chunk the corpus, embed with HuggingFace, index in FAISS.

The index is built over the *corpus* only (resolution notes + KB articles).
Incoming/held-out tickets are never indexed, so retrieval quality is measured
honestly. Two logical collections share one FAISS store and are separated by
the ``type`` metadata field:
  * ``resolution_note`` -> similar past tickets / prior resolutions
  * ``kb_article``      -> knowledge-base articles
"""
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from helpmate import config
from helpmate.llm import get_embeddings

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_VECTORSTORE = None  # process-level cache


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def _load_corpus_documents() -> list[Document]:
    """Read every markdown file under data/corpus and split into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    docs: list[Document] = []
    for path in sorted(config.CORPUS_DIR.rglob("*.md")):
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        meta.setdefault("doc_id", path.stem)
        meta.setdefault("type", "kb_article" if "kb" in path.parts else "resolution_note")
        meta["source"] = str(path.relative_to(config.ROOT))
        for chunk in splitter.split_text(body):
            docs.append(Document(page_content=chunk.strip(), metadata=dict(meta)))
    return docs


def build_index(persist: bool = True):
    """Embed the corpus and build (and optionally persist) the FAISS index."""
    from langchain_community.vectorstores import FAISS

    documents = _load_corpus_documents()
    if not documents:
        raise RuntimeError(
            "No corpus documents found. Run `python data/generate_data.py` first."
        )
    store = FAISS.from_documents(documents, get_embeddings())
    if persist:
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        store.save_local(str(config.INDEX_DIR))
    global _VECTORSTORE
    _VECTORSTORE = store
    return store


def load_index():
    """Load the persisted FAISS index, building it on first use if missing."""
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    from langchain_community.vectorstores import FAISS

    faiss_file = config.INDEX_DIR / "index.faiss"
    if faiss_file.exists():
        _VECTORSTORE = FAISS.load_local(
            str(config.INDEX_DIR), get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        return _VECTORSTORE
    return build_index(persist=True)


def _search(query: str, doc_type: str, k: int) -> list[dict]:
    store = load_index()
    results = store.similarity_search_with_relevance_scores(
        query, k=k * 3, filter={"type": doc_type}
    )
    hits = []
    for doc, score in results:
        if score < config.RETRIEVAL_MIN_SCORE:
            continue
        hits.append({
            "doc_id": doc.metadata.get("doc_id"),
            "type": doc.metadata.get("type"),
            "ticket_id": doc.metadata.get("ticket_id"),
            "case_id": doc.metadata.get("case_id"),
            "title": doc.metadata.get("title", doc.metadata.get("subject", "")),
            "score": round(float(score), 3),
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
        })
        if len(hits) >= k:
            break
    return hits


def search_similar_tickets(query: str, k: int = 3) -> list[dict]:
    """Vector search over prior resolution / RCA notes."""
    return _search(query, "resolution_note", k)


def search_kb(query: str, k: int = 3) -> list[dict]:
    """Vector search over KB articles."""
    return _search(query, "kb_article", k)


if __name__ == "__main__":
    store = build_index()
    print(f"Indexed corpus into FAISS at {config.INDEX_DIR}")
    for q in ["VPN keeps disconnecting", "account locked out repeatedly"]:
        print(f"\nQuery: {q}")
        for h in search_similar_tickets(q, k=2):
            print(f"  [{h['score']}] {h['doc_id']} ({h['ticket_id']}) {h['title']}")
        for h in search_kb(q, k=1):
            print(f"  KB [{h['score']}] {h['doc_id']} {h['title']}")
