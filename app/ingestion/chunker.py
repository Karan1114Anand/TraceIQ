"""
app/ingestion/chunker.py

Document chunking with three strategies:
  1. Agentic (proposition-based via Ollama) — default for shorter docs
  2. Section-aware (header splitting)
  3. Fixed-size with overlap (fallback)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional

import ollama
from loguru import logger

from app.config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)


def _chat(prompt: str, system: str = "", model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> str:
    """Call Ollama SDK directly — no aiohttp dependency."""
    try:
        client = ollama.Client(host=base_url)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat(model=model, messages=messages)
        return response["message"]["content"].strip()
    except Exception as exc:
        logger.error(f"Ollama call failed: {exc}")
        return ""



# ---------------------------------------------------------------------------
# Proposition extractor
# ---------------------------------------------------------------------------

class PropositionExtractor:
    """Extracts atomic propositions from a paragraph using a local Ollama LLM."""

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url

    def get_propositions(self, text: str) -> List[str]:
        """Return a list of proposition strings from the given text."""
        if not text.strip():
            return []
        try:
            raw = _chat(
                prompt=f"Extract propositions from:\n\n{text}",
                system=(
                    "You are an expert at breaking text into atomic propositions. "
                    "Each proposition is a single, self-contained factual statement. "
                    "Return ONLY a numbered list of propositions, nothing else."
                ),
                model=self.model_name,
                base_url=self.base_url,
            )
            lines = raw.strip().split("\n")
            propositions = []
            for line in lines:
                cleaned = line.strip()
                if not cleaned or len(cleaned) < 10:
                    continue
                for prefix in ["- ", "* ", "• "]:
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):]
                if cleaned and cleaned[0].isdigit() and ". " in cleaned[:4]:
                    cleaned = cleaned.split(". ", 1)[1]
                if cleaned:
                    propositions.append(cleaned.strip())
            return propositions
        except Exception as exc:
            logger.error(f"Proposition extraction failed: {exc}")
            return [s.strip() + "." for s in text.split(".") if s.strip()]



# ---------------------------------------------------------------------------
# Agentic chunker
# ---------------------------------------------------------------------------

class AgenticChunker:
    """Groups propositions into semantically coherent chunks via LLM."""

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.chunks: Dict[str, Dict] = {}
        self.id_truncate_limit = 5
        self.generate_new_metadata_ind = True
        self.model_name = model_name
        self.base_url = base_url

    # ------------------------------------------------------------------

    def add_propositions(self, propositions: List[str]) -> None:
        for prop in propositions:
            self.add_proposition(prop)

    def add_proposition(self, proposition: str) -> None:
        if not self.chunks:
            self._create_new_chunk(proposition)
            return
        chunk_id = self._find_relevant_chunk(proposition)
        if chunk_id:
            self._add_to_chunk(chunk_id, proposition)
        else:
            self._create_new_chunk(proposition)

    def _add_to_chunk(self, chunk_id: str, proposition: str) -> None:
        self.chunks[chunk_id]["propositions"].append(proposition)
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]["summary"] = self._update_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]["title"] = self._update_title(self.chunks[chunk_id])

    def _create_new_chunk(self, proposition: str) -> None:
        new_id = str(uuid.uuid4())[: self.id_truncate_limit]
        summary = self._get_summary(proposition)
        title = self._get_title(summary)
        self.chunks[new_id] = {
            "chunk_id": new_id,
            "propositions": [proposition],
            "title": title,
            "summary": summary,
            "chunk_index": len(self.chunks),
        }

    def _find_relevant_chunk(self, proposition: str) -> Optional[str]:
        outline = self._get_outline()
        result = _chat(
            prompt=f"Chunks:\n{outline}\n\nProposition: {proposition}",
            system=(
                "Determine if the proposition belongs to an existing chunk. "
                "Return the chunk_id if yes, or 'No chunks' if no. "
                "Return ONLY the chunk_id or 'No chunks'."
            ),
            model=self.model_name,
            base_url=self.base_url,
        ).strip()
        return result if len(result) == self.id_truncate_limit else None

    def _get_outline(self) -> str:
        return "\n".join(
            f"ID: {c['chunk_id']} | Title: {c['title']} | Summary: {c['summary']}"
            for c in self.chunks.values()
        )

    def _get_summary(self, proposition: str) -> str:
        return _chat(
            prompt=proposition,
            system="Write a 1-sentence summary for a chunk containing this proposition.",
            model=self.model_name,
            base_url=self.base_url,
        ) or proposition[:100]

    def _update_summary(self, chunk: Dict) -> str:
        props = "\n".join(chunk["propositions"])
        return _chat(
            prompt=f"Current summary: {chunk['summary']}\nPropositions:\n{props}",
            system="Update the 1-sentence summary for this chunk given new propositions.",
            model=self.model_name,
            base_url=self.base_url,
        ) or chunk["summary"]

    def _get_title(self, summary: str) -> str:
        return _chat(
            prompt=summary,
            system="Write a short 2-5 word title for this chunk summary.",
            model=self.model_name,
            base_url=self.base_url,
        ) or summary[:30]

    def _update_title(self, chunk: Dict) -> str:
        return self._get_title(chunk["summary"])

    def get_chunks(self, get_type: str = "dict"):
        if get_type == "dict":
            return self.chunks
        return [" ".join(c["propositions"]) for c in self.chunks.values()]


    # ------------------------------------------------------------------

    def add_propositions(self, propositions: List[str]) -> None:
        for prop in propositions:
            self.add_proposition(prop)

    def add_proposition(self, proposition: str) -> None:
        if not self.chunks:
            self._create_new_chunk(proposition)
            return
        chunk_id = self._find_relevant_chunk(proposition)
        if chunk_id:
            self._add_to_chunk(chunk_id, proposition)
        else:
            self._create_new_chunk(proposition)

    def _add_to_chunk(self, chunk_id: str, proposition: str) -> None:
        self.chunks[chunk_id]["propositions"].append(proposition)
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]["summary"] = self._update_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]["title"] = self._update_title(self.chunks[chunk_id])

    def _create_new_chunk(self, proposition: str) -> None:
        new_id = str(uuid.uuid4())[: self.id_truncate_limit]
        summary = self._get_summary(proposition)
        title = self._get_title(summary)
        self.chunks[new_id] = {
            "chunk_id": new_id,
            "propositions": [proposition],
            "title": title,
            "summary": summary,
            "chunk_index": len(self.chunks),
        }

    def _find_relevant_chunk(self, proposition: str) -> Optional[str]:
        outline = self._get_outline()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Determine if the proposition belongs to an existing chunk.\n"
                    "Return the chunk_id if yes, or 'No chunks' if no.\n"
                    "Return ONLY the chunk_id or 'No chunks'.",
                ),
                ("user", f"Chunks:\n{outline}\n\nProposition: {proposition}"),
            ]
        )
        try:
            result = (prompt | self.llm).invoke({}).strip()
            return result if len(result) == self.id_truncate_limit else None
        except Exception:
            return None

    def _get_outline(self) -> str:
        return "\n".join(
            f"ID: {c['chunk_id']} | Title: {c['title']} | Summary: {c['summary']}"
            for c in self.chunks.values()
        )

    def _get_summary(self, proposition: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Write a 1-sentence summary for a chunk containing this proposition."),
                ("user", proposition),
            ]
        )
        try:
            return (prompt | self.llm).invoke({}).strip()
        except Exception:
            return proposition[:100]

    def _update_summary(self, chunk: Dict) -> str:
        propositions_text = "\n".join(chunk["propositions"])
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Update the 1-sentence summary for this chunk given new propositions.",
                ),
                (
                    "user",
                    f"Current summary: {chunk['summary']}\nPropositions:\n{propositions_text}",
                ),
            ]
        )
        try:
            return (prompt | self.llm).invoke({}).strip()
        except Exception:
            return chunk["summary"]

    def _get_title(self, summary: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Write a short 2-5 word title for this chunk summary."),
                ("user", summary),
            ]
        )
        try:
            return (prompt | self.llm).invoke({}).strip()
        except Exception:
            return summary[:30]

    def _update_title(self, chunk: Dict) -> str:
        return self._get_title(chunk["summary"])

    def get_chunks(self, get_type: str = "dict"):
        if get_type == "dict":
            return self.chunks
        return [
            " ".join(c["propositions"]) for c in self.chunks.values()
        ]


# ---------------------------------------------------------------------------
# Document chunker (main interface)
# ---------------------------------------------------------------------------

class DocumentChunker:
    """
    Orchestrates three chunking strategies based on document size and structure.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        model_name: str = OLLAMA_MODEL,
        use_agentic: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_agentic = use_agentic
        if use_agentic:
            self.proposition_extractor = PropositionExtractor(model_name=model_name)
            self.agentic_chunker = AgenticChunker(model_name=model_name)
            logger.info(f"Agentic chunker ready with model: {model_name}")

    def chunk_document(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        strategy: str = "auto",
    ) -> List[Dict]:
        if not text.strip():
            return []
        metadata = metadata or {}

        if strategy == "auto":
            if len(text) < 50_000 and self.use_agentic:
                strategy = "agentic"
            elif self._has_sections(text):
                strategy = "section"
            else:
                strategy = "fixed"

        logger.info(f"Chunking with strategy: {strategy}")
        if strategy == "agentic" and self.use_agentic:
            return self._chunk_agentic(text, metadata)
        if strategy == "section":
            return self._chunk_by_section(text, metadata)
        return self._chunk_fixed(text, metadata)

    # ------------------------------------------------------------------

    def _chunk_agentic(self, text: str, metadata: Dict) -> List[Dict]:
        try:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            all_propositions: List[str] = []
            for para in paragraphs:
                all_propositions.extend(self.proposition_extractor.get_propositions(para))
            self.agentic_chunker.add_propositions(all_propositions)
            chunk_dict = self.agentic_chunker.get_chunks(get_type="dict")
            chunks = []
            for cid, cdata in chunk_dict.items():
                chunks.append(
                    {
                        "text": " ".join(cdata["propositions"]),
                        "chunk_id": cid,
                        "title": cdata.get("title", ""),
                        "summary": cdata.get("summary", ""),
                        "chunk_index": cdata.get("chunk_index", 0),
                        "metadata": {
                            **metadata,
                            "chunking_strategy": "agentic",
                            "proposition_count": len(cdata["propositions"]),
                        },
                    }
                )
            self.agentic_chunker.chunks = {}  # reset for next document
            return chunks
        except Exception as exc:
            logger.error(f"Agentic chunking failed: {exc}. Falling back to fixed.")
            return self._chunk_fixed(text, metadata)

    def _chunk_by_section(self, text: str, metadata: Dict) -> List[Dict]:
        sections = self._split_sections(text)
        chunks: List[Dict] = []
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            if len(section) > self.chunk_size:
                sub = self._chunk_fixed(section, {**metadata, "section_index": i})
                chunks.extend(sub)
            else:
                chunks.append(
                    {
                        "text": section.strip(),
                        "chunk_id": f"sec_{i}",
                        "chunk_index": i,
                        "metadata": {**metadata, "chunking_strategy": "section", "section_index": i},
                    }
                )
        return chunks

    def _chunk_fixed(self, text: str, metadata: Dict) -> List[Dict]:
        chunks: List[Dict] = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            if end < len(text):
                bp = max(chunk_text.rfind("."), chunk_text.rfind("\n"))
                if bp > self.chunk_size * 0.5:
                    chunk_text = chunk_text[: bp + 1]
                    end = start + bp + 1
            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "chunk_id": f"chunk_{idx}",
                    "chunk_index": idx,
                    "metadata": {**metadata, "chunking_strategy": "fixed", "start_char": start, "end_char": end},
                }
            )
            start = end - self.chunk_overlap
            idx += 1
        return chunks

    def _has_sections(self, text: str) -> bool:
        lines = text.split("\n")
        count = sum(
            1
            for ln in lines
            if ln.strip().startswith("#")
            or (ln.strip() and ln.strip()[0].isdigit() and ". " in ln[:5])
        )
        return count > 3

    def _split_sections(self, text: str) -> List[str]:
        sections: List[str] = []
        current: List[str] = []
        for line in text.split("\n"):
            is_header = line.strip().startswith("#") or (
                line.strip() and line.strip()[0].isdigit() and ". " in line[:5]
            )
            if is_header and current:
                sections.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append("\n".join(current))
        return sections
