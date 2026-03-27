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
You are an expert academic research analyst.

Given a research topic, generate exactly ONE focused sub-question for each of
these 7 categories:

- definitional:  How is the core concept defined, classified, or characterised
                 in the literature?
- causal:        What mechanisms, variables, or processes cause or influence X?
- comparative:   How do findings, methods, or outcomes differ across studies,
                 populations, or conditions?
- quantitative:  What empirical data, statistics, or measurable results are
                 reported for X?
- contrarian:    What contradictory evidence, limitations, or dissenting views
                 exist in the literature?
- procedural:    What methodologies, protocols, or experimental designs are
                 used to study X?
- gap_seeking:   What aspects of X remain under-investigated, contested, or
                 unresolved in current research?

Topic: {topic}

Rules:
- Generate exactly 1 question per category (7 questions total)
- Each question must be specific and directly answerable from research documents
- Use precise academic language; avoid business or market framing
- Do not repeat or overlap questions across categories

Return ONLY valid JSON, no preamble, no explanation:
[
  {{"id": "q1", "type": "definitional", "question": "...", "status": "pending"}},
  {{"id": "q2", "type": "causal",       "question": "...", "status": "pending"}},
  {{"id": "q3", "type": "comparative",  "question": "...", "status": "pending"}},
  {{"id": "q4", "type": "quantitative", "question": "...", "status": "pending"}},
  {{"id": "q5", "type": "contrarian",   "question": "...", "status": "pending"}},
  {{"id": "q6", "type": "procedural",   "question": "...", "status": "pending"}},
  {{"id": "q7", "type": "gap_seeking",  "question": "...", "status": "pending"}}
]
"""

# ---------------------------------------------------------------------------
# Synthesizer — generates grounded report using ONLY retrieved context
# Citation format: [CIT:source_name:chunk_id:page]
# ---------------------------------------------------------------------------
SYNTHESIZER_PROMPT = """\
You are a rigorous academic research analyst synthesising findings from \
retrieved literature.

STRICT RULES:
1. Use ONLY the provided context below. Never introduce external knowledge.
2. Every factual claim MUST be supported by a citation label in exactly this
   format: [CIT:source_name:chunk_id:page]
   Use the exact labels that appear at the start of each context chunk.
3. If a sub-question cannot be answered from the available context, write:
   "Insufficient evidence: <question_text>"
4. Write in clear, concise academic prose. Avoid speculative or hedging language
   unless directly supported by the source.
5. Output must be exactly the JSON structure below — no preamble, no markdown.

Sub-questions to address:
{sub_questions}

Context (each chunk is labelled [CIT:source_name:chunk_id:page]):
{context}

Output (valid JSON only, no preamble):
{{
  "report": {{
    "title": "Research Synthesis: <topic>",
    "sections": [
      {{
        "heading": "<section_heading_matching_a_sub_question_type>",
        "content": "<grounded academic prose with inline [CIT:...] labels>",
        "claims": [
          {{
            "claim": "<one specific, verifiable claim>",
            "source_chunks": ["chunk_id_1", "chunk_id_2"]
          }}
        ]
      }}
    ],
    "unanswered_questions": ["<question text for any sub-question with no evidence>"],
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
