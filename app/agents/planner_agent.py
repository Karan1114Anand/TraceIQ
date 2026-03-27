"""
app/agents/planner_agent.py

Decomposes a research topic into structured sub-questions using Ollama.
Output: List[{id, type, question, status}]
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

import ollama
from loguru import logger

from app.config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL
from app.config.prompts import PLANNER_PROMPT

_CATEGORIES = ["foundations", "applications", "frontiers"]


def _chat(prompt: str, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> str:
    client = ollama.Client(host=base_url)
    response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()



class PlannerAgent:
    """
    Breaks a research topic into 7 focused academic sub-questions,
    one per category: definitional, causal, comparative, quantitative,
    contrarian, procedural, gap_seeking.

    Calls Ollama with PLANNER_PROMPT and parses the JSON response.
    Falls back to heuristic sub-questions if the LLM response cannot
    be parsed.
    """

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        logger.info(f"PlannerAgent ready (model: {model_name}).")

    # ------------------------------------------------------------------

    def plan(self, topic: str) -> List[Dict]:
        """
        Generate a structured research plan for the given topic.

        Returns:
            List of sub-question dicts:
            [{id, type, question, status}]
        """
        logger.info(f"Planning sub-questions for: '{topic}'")
        prompt = PLANNER_PROMPT.format(topic=topic)

        try:
            raw = _chat(prompt, model=self.model_name, base_url=self.base_url)
            questions = self._parse_json(raw)
            if questions:
                logger.success(f"PlannerAgent: generated {len(questions)} sub-questions.")
                return questions
        except Exception as exc:
            logger.error(f"PlannerAgent LLM call failed: {exc}")

        # Heuristic fallback
        logger.warning("PlannerAgent: using heuristic fallback sub-questions.")
        return self._heuristic_plan(topic)


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> List[Dict]:
        """Extract JSON array from LLM response."""
        raw = raw.strip()
        # Try direct parse
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        # Try extracting first [...] block
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        return []

    @staticmethod
    def _heuristic_plan(topic: str) -> List[Dict]:
        """Generate broad sub-questions when LLM parsing fails."""
        templates = {
            "foundations":  f"What is {topic}, how does it work, and what are its core principles and mechanisms?",
            "applications": f"What are the proven uses, benefits, and limitations of {topic}, and how does it compare to alternatives?",
            "frontiers":    f"What are the current challenges, open problems, and future directions in {topic}?",
        }
        return [
            {"id": f"q{i+1}", "type": cat, "question": q, "status": "pending"}
            for i, (cat, q) in enumerate(templates.items())
        ]
