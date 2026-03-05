"""
app/agents/synthesizer_agent.py

Generates a structured research report using ONLY the retrieved context.
All claims must cite [CIT:source_name:chunk_id:page] labels as provided
by the ContextBuilder.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

import ollama
from loguru import logger

from app.config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL
from app.config.prompts import SYNTHESIZER_PROMPT


def _chat(prompt: str, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> str:
    client = ollama.Client(host=base_url)
    response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()



class SynthesizerAgent:
    """
    Constructs a grounded research report from reranked context.

    Receives the context_package produced by ContextBuilder:
        {
            "context_str":  str   (labeled chunks),
            "citation_map": Dict  (chunk_id → {source, page, file_name})
        }
    """

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        logger.info(f"SynthesizerAgent ready (model: {model_name}).")


    # ------------------------------------------------------------------

    def synthesize(
        self,
        sub_questions: List[Dict],
        context_package: Dict,
    ) -> Dict:
        """
        Generate a structured report.

        Args:
            sub_questions: List of {id, type, question, status} dicts.
            context_package: {context_str, citation_map} from ContextBuilder.

        Returns:
            Report dict with sections, claims, unanswered_questions,
            confidence_score, and resolved citation_map.
        """
        context_str = context_package.get("context_str", "")
        citation_map = context_package.get("citation_map", {})

        if not context_str.strip():
            logger.warning("SynthesizerAgent: empty context provided.")
            return self._empty_report(sub_questions)

        q_texts = json.dumps(
            [{"id": q["id"], "question": q["question"]} for q in sub_questions],
            indent=2,
        )
        prompt = SYNTHESIZER_PROMPT.format(
            sub_questions=q_texts,
            context=context_str,
        )

        try:
            raw = _chat(prompt, model=self.model_name, base_url=self.base_url)
            report = self._parse_report(raw)
            report["citation_map"] = citation_map
            logger.success("SynthesizerAgent: report generated.")
            return report
        except Exception as exc:
            logger.error(f"SynthesizerAgent failed: {exc}")
            return self._empty_report(sub_questions, error=str(exc))


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_report(raw: str) -> Dict:
        """Extract JSON report from LLM response."""
        raw = raw.strip()
        try:
            data = json.loads(raw)
            return data.get("report", data)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("report", data)
            except json.JSONDecodeError:
                pass
        # Return raw text wrapped in a minimal structure
        return {
            "title": "Research Report",
            "sections": [{"heading": "Summary", "content": raw, "claims": []}],
            "unanswered_questions": [],
            "confidence_score": 0.0,
        }

    @staticmethod
    def _empty_report(sub_questions: List[Dict], error: str = "") -> Dict:
        questions = [q["question"] for q in sub_questions]
        return {
            "title": "Research Report",
            "sections": [],
            "unanswered_questions": questions,
            "confidence_score": 0.0,
            "error": error,
            "citation_map": {},
        }
