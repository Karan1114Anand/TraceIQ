"""
app/config/prompts.py

Centralised LLM prompt templates.
Citation format enforced end-to-end: [CIT:source_name:chunk_id:page]
"""

# ---------------------------------------------------------------------------
# Planner — decomposes user topic into sub-questions
# One focused question per category (7 total) to keep retrieval fast.
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """\
You are an expert research analyst.

Given a research topic, generate exactly 3 broad, comprehensive sub-questions
that together provide complete coverage of the topic.

Use these 3 angles:
- foundations:    What is it, how does it work, and what are its core principles
                  and mechanisms?
- applications:   What are its proven uses, benefits, limitations, and how does
                  it compare to alternatives?
- frontiers:      What challenges, open problems, and future directions exist?

Topic: {topic}

Rules:
- Generate exactly 3 questions (one per angle above)
- Each question should be broad enough to elicit a detailed multi-paragraph answer
- Questions must be directly answerable from research documents
- Do not repeat or overlap questions

Return ONLY valid JSON, no preamble, no explanation:
[
  {{"id": "q1", "type": "foundations",   "question": "...", "status": "pending"}},
  {{"id": "q2", "type": "applications",  "question": "...", "status": "pending"}},
  {{"id": "q3", "type": "frontiers",     "question": "...", "status": "pending"}}
]
"""

# ---------------------------------------------------------------------------
# Synthesizer — generates grounded report using ONLY retrieved context
# Citation format: [CIT:source_name:chunk_id:page]
# ---------------------------------------------------------------------------
SYNTHESIZER_PROMPT = """\
You are a knowledgeable research assistant. Using ONLY the provided context,
write a detailed, thorough answer to each sub-question.

RULES:
1. Use ONLY information from the provided context. Never add external knowledge.
2. Cite sources inline immediately after the relevant sentence using the exact
   label format from the context: [CIT:source_name:chunk_id:page]
3. Write 2-4 detailed paragraphs per section. Be thorough and explain concepts
   clearly, as if explaining to someone genuinely curious about the topic.
4. If a question cannot be answered from the available context, write one natural
   sentence acknowledging this, e.g. "The available documents do not cover this
   aspect in detail."
5. Use clear, natural headings — not academic category labels.
6. Output must be valid JSON exactly as shown. No preamble, no markdown fences.

Sub-questions to address:
{sub_questions}

Context (each chunk starts with its label [CIT:source_name:chunk_id:page]):
{context}

Output (valid JSON only):
{{
  "report": {{
    "title": "<concise descriptive title for this topic>",
    "sections": [
      {{
        "heading": "<clear natural section heading>",
        "content": "<2-4 detailed paragraphs with inline [CIT:...] citations after each supported sentence>",
        "claims": [
          {{
            "claim": "<one specific verifiable claim from this section>",
            "source_chunks": ["chunk_id_1"]
          }}
        ]
      }}
    ],
    "unanswered_questions": [],
    "confidence_score": 0.85
  }}
}}
"""

# ---------------------------------------------------------------------------
# Gap Analysis — evaluates per-sub-question coverage
# Skips looping if overall confidence is already high enough.
# ---------------------------------------------------------------------------
GAP_ANALYSIS_PROMPT = """\
You are a peer-review quality auditor for academic research syntheses.

Your job is to assess whether each sub-question has been adequately addressed
in the synthesis report, and decide if another retrieval pass is warranted.

Report: {report}
Original sub-questions: {sub_questions}
Retrieval scores summary: {retrieval_scores}
Current iteration: {iteration} / {max_iterations}
Minimum acceptable confidence: {min_confidence}

For each sub-question classify its coverage as one of:
- answered:   substantive evidence is present with inline citations
- partial:    some relevant content found but key aspects are missing
- unanswered: no useful evidence found in the retrieved context

Decision rules:
1. If the report's overall confidence_score >= {min_confidence}, set
   has_gaps=false and recommendation="finalize" — do not loop further.
2. If iteration >= {max_iterations}, set has_gaps=false and
   recommendation="finalize" regardless of coverage.
3. Only set has_gaps=true if at least one sub-question is "unanswered" AND
   confidence_score < {min_confidence} AND iteration < {max_iterations}.
4. For each gap, provide a precise academic follow_up_query targeting the
   missing aspect — not a restatement of the original question.

Return ONLY valid JSON, no preamble:
{{
  "has_gaps": false,
  "gap_details": [
    {{
      "q_id": "q2",
      "status": "unanswered",
      "follow_up_query": "<specific retrieval query to fill this gap>"
    }}
  ],
  "recommendation": "finalize"
}}
"""

# ---------------------------------------------------------------------------
# HyDE — Hypothetical Document Embedding for query expansion
# ---------------------------------------------------------------------------
HYDE_PROMPT = """\
Given a research question, write a hypothetical passage that would appear in
an academic paper or report perfectly answering this question. This passage
will be used to improve document retrieval — not shown to the user.

Question: {question}

Write 2-3 paragraphs that:
- Directly and specifically answer the question
- Use precise academic and technical terminology likely to appear in the
  relevant literature
- Read like an excerpt from a peer-reviewed paper or research report

Hypothetical passage:
"""

# ---------------------------------------------------------------------------
# LLM Reranker — scores chunk relevance (0-1) for a query
# ---------------------------------------------------------------------------
RERANKING_PROMPT = """\
Rate the relevance of the following document chunk to the research query on a
scale of 0.0 (completely irrelevant) to 1.0 (directly answers the query).

Query: {query}
Chunk: {chunk}

Scoring criteria:
- Does the chunk directly address the query topic or sub-question?
- Does it contain specific evidence, data, or findings relevant to the query?
- Is the terminology aligned with the research domain of the query?

Return ONLY a single decimal number between 0.0 and 1.0, nothing else:
"""
