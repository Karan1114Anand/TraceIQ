"""
app/agents/gap_analysis_agent.py

Evaluates the draft report on a per-sub-question basis.
Returns whether to continue researching (has_gaps) and targeted
follow-up queries for unanswered sub-questions.

Enforces MAX_RESEARCH_ITERATIONS to prevent infinite loops.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

import ollama
from loguru import logger

from app.config.settings import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    MAX_RESEARCH_ITERATIONS,
)
from app.config.prompts import GAP_ANALYSIS_PROMPT


def _chat(prompt: str, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> str:
    client = ollama.Client(host=base_url)
    response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()



class GapAnalysisAgent:
    """
    Checks research completeness per sub-question and decides whether
    to trigger another retrieval loop or finalize.

    Output schema:
    {
        "has_gaps": bool,
        "gap_details": [
            {"q_id": "q2", "status": "unanswered", "follow_up_query": "..."}
        ],
        "recommendation": "refine_questions" | "finalize"
    }
    """

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        max_iterations: int = MAX_RESEARCH_ITERATIONS,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.max_iterations = max_iterations
        logger.info(f"GapAnalysisAgent ready (max_iterations={max_iterations}).")


    # ------------------------------------------------------------------

    def analyze(
        self,
        report: Dict,
        sub_questions: List[Dict],
        iteration: int,
        retrieval_scores: Dict | None = None,
    ) -> Dict:
        """
        Analyze the draft report for per-sub-question coverage gaps.

        Args:
            report:           Draft report dict from SynthesizerAgent.
            sub_questions:    Original list of {id, type, question, status}.
            iteration:        Current pipeline iteration (1-indexed).
            retrieval_scores: Optional dict of avg scores per sub-question.

        Returns:
            {has_gaps, gap_details, recommendation}
        """
        # Enforce max-iterations guard — always finalize if limit reached
        if iteration >= self.max_iterations:
            logger.info(
                f"GapAnalysisAgent: iteration {iteration}/{self.max_iterations} — forcing finalize."
            )
            return {
                "has_gaps": False,
                "gap_details": [],
                "recommendation": "finalize",
            }

        report_str = json.dumps(report, indent=2)[:4000]   # truncate for context
        q_str = json.dumps(
            [{"id": q["id"], "question": q["question"]} for q in sub_questions],
            indent=2,
        )
        scores_str = json.dumps(retrieval_scores or {}, indent=2)

        prompt = GAP_ANALYSIS_PROMPT.format(
            report=report_str,
            sub_questions=q_str,
            retrieval_scores=scores_str,
            iteration=iteration,
            max_iterations=self.max_iterations,
            min_confidence=0.7,
        )

        try:
            raw = _chat(prompt, model=self.model_name, base_url=self.base_url)
            result = self._parse_gap(raw)
            if iteration + 1 >= self.max_iterations:
                result["recommendation"] = "finalize"
            logger.info(
                f"GapAnalysis iteration {iteration}: has_gaps={result['has_gaps']}, "
                f"recommendation={result['recommendation']}"
            )
            return result
        except Exception as exc:
            logger.error(f"GapAnalysisAgent failed: {exc}")
            return {"has_gaps": False, "gap_details": [], "recommendation": "finalize"}


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gap(raw: str) -> Dict:
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"has_gaps": False, "gap_details": [], "recommendation": "finalize"}
