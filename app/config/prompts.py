"""
app/config/prompts.py

Centralised LLM prompt templates.
Citation format enforced end-to-end: [CIT:source_name:chunk_id:page]
"""

# ---------------------------------------------------------------------------
# Planner — decomposes user topic into sub-questions
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """\
You are a senior business intelligence research strategist.

Given a research topic, generate sub-questions across ALL of these categories:
- definitional: What is X? How is X defined or measured in the industry?
- causal: Why does X happen? What are the root drivers of X?
- comparative: How does X differ across competitors, time periods, or markets?
- quantitative: What are the key metrics, figures, or trends for X?
- contrarian: What challenges the mainstream view of X?
- procedural: How is X implemented, evaluated, or managed?
- gap_seeking: What is unknown, contested, or under-researched about X?

Topic: {topic}

Rules:
- Generate 2-3 questions per category (14-21 questions total)
- Make each question specific and answerable from documents
- Avoid vague or overlapping questions

Return ONLY valid JSON, no preamble:
[
  {{"id": "q1", "type": "definitional", "question": "...", "status": "pending"}}
]
"""

# ---------------------------------------------------------------------------
# Synthesizer — generates grounded report using ONLY retrieved context
# Citation format: [CIT:source_name:chunk_id:page]
# ---------------------------------------------------------------------------
SYNTHESIZER_PROMPT = """\
You are a business intelligence analyst writing a structured research report.

STRICT RULES:
1. Use ONLY the provided context below. Never add external knowledge.
2. Every factual claim MUST include a citation label in exactly this format:
   [CIT:source_name:chunk_id:page]
   Use the exact labels that appear at the start of each context chunk.
3. If a sub-question cannot be answered from context, explicitly state:
   "Insufficient data: <question_text>"
4. Output format must be exactly as specified below.

Sub-questions to address:
{sub_questions}

Context (each chunk is labelled [CIT:source_name:chunk_id:page]):
{context}

Output (valid JSON only, no preamble):
{{
  "report": {{
    "title": "Research Report: <topic>",
    "sections": [
      {{
        "heading": "<section_name>",
        "content": "<grounded text with [CIT:...] labels inline>",
        "claims": [
          {{
            "claim": "<specific claim>",
            "source_chunks": ["chunk_id_1", "chunk_id_2"]
          }}
        ]
      }}
    ],
    "unanswered_questions": ["<question_text>"],
    "confidence_score": 0.85
  }}
}}
"""

# ---------------------------------------------------------------------------
# Gap Analysis — evaluates per-sub-question coverage
# ---------------------------------------------------------------------------
GAP_ANALYSIS_PROMPT = """\
You are a research quality auditor.

Review the synthesis report and identify which sub-questions are adequately
answered, partially answered, or unanswered.

Report: {report}
Original sub-questions: {sub_questions}
Retrieval scores summary: {retrieval_scores}
Current iteration: {iteration} / {max_iterations}

For each sub-question, assess:
- answered: sufficient evidence with citations
- partial: some evidence but incomplete
- unanswered: no useful evidence found

Return ONLY valid JSON:
{{
  "has_gaps": true,
  "gap_details": [
    {{
      "q_id": "q2",
      "status": "unanswered",
      "follow_up_query": "<targeted retrieval query to fill this gap>"
    }}
  ],
  "recommendation": "refine_questions | finalize"
}}

IMPORTANT: If iteration >= {max_iterations}, set recommendation to "finalize".
"""

# ---------------------------------------------------------------------------
# HyDE — Hypothetical Document Embedding for query expansion
# ---------------------------------------------------------------------------
HYDE_PROMPT = """\
Given a research question, generate a hypothetical ideal answer that would
perfectly address this question. This will be used to improve retrieval.

Question: {question}

Generate a detailed, specific hypothetical answer (2-3 paragraphs) that:
- Directly addresses the question
- Includes specific terminology likely to appear in relevant documents
- Maintains a professional, analytical tone

Hypothetical Answer:
"""

# ---------------------------------------------------------------------------
# LLM Reranker — scores chunk relevance (0-1) for a query
# ---------------------------------------------------------------------------
RERANKING_PROMPT = """\
Given a query and a document chunk, rate the relevance of the chunk to the
query on a scale of 0.0 to 1.0.

Query: {query}
Chunk: {chunk}

Consider:
- Direct relevance to the query topic
- Presence of key terms and concepts
- Quality and specificity of information

Return ONLY a single decimal number between 0.0 and 1.0:
"""
