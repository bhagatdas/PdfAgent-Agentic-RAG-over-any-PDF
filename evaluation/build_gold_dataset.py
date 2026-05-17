"""
Build a gold dataset for retrieval evaluation by LLM-synthesizing questions
whose answer lives in a specific chunk.

Strategy:
  1. Pull all child chunks from the FAISS vector store.
  2. Sample N of them.
  3. For each chunk, ask the LLM to write a SPECIFIC, ANSWERABLE question
     whose answer is contained in the chunk's text.
  4. Drop generic / non-specific questions via a structured `is_specific` flag.
  5. Save to evaluation/gold_retrieval.jsonl, one record per line:
       {
         "qid": "q_001",
         "question": "...",
         "relevant_chunk_ids": ["..."],
         "relevant_pages": [{"document": "...", "page": 0}],
         "ground_truth_excerpt": "...",
         "source_chunk_type": "child" | "table_repr" | "image_caption"
       }

Run:
    python -m evaluation.build_gold_dataset --n 50
    python -m evaluation.build_gold_dataset --n 100 --out evaluation/gold_retrieval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from pydantic import BaseModel, Field

from retrieval.vector_store import vector_store
from utils.llm import get_structured_llm
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("evaluation/gold_retrieval.jsonl")
DEFAULT_N = 50
RANDOM_SEED = 42
MIN_CHUNK_CHARS = 250  # avoid stubby chunks that produce bad questions


class GeneratedQuestion(BaseModel):
    """Structured output for the question-generation LLM call."""

    question: str = Field(
        description="A specific, factual question whose answer is contained in the chunk. "
                    "Must reference concrete entities, numbers, or events from the chunk — "
                    "not generic prompts like 'what does this section discuss'."
    )
    is_specific: bool = Field(
        description="True if the question references concrete details (entities, numbers, "
                    "dates, named programs) that uniquely point to THIS chunk. "
                    "False for vague questions like 'what is mentioned about ESG'."
    )
    ground_truth_excerpt: str = Field(
        description="The 1-2 sentence excerpt from the chunk that directly answers the question.",
        default="",
    )


GEN_PROMPT = """You are building an evaluation dataset for a retrieval system over ESG/sustainability reports.

Given the chunk below, write ONE question whose answer is contained in this chunk. The question should be:
- SPECIFIC: references concrete entities, numbers, dates, or program names from the chunk
- ANSWERABLE: a careful reader of this chunk could write the answer
- UNIQUE: ideally the question would only be answered by THIS chunk, not generic chunks

If the chunk is too vague, fragmentary, or generic to produce such a question, set is_specific=false and the question can be a best effort.

CHUNK (from document='{doc}', page={page}):
{chunk_text}

Return the structured object."""


def _sample_chunks(n: int) -> list[dict]:
    """Sample n child / table / image chunks from the vector store."""
    all_docs = vector_store.get_all_documents()
    candidates = [
        d for d in all_docs
        if d["metadata"].get("chunk_type") in ("child", "table_repr", "image_caption")
        and len(d["content"]) >= MIN_CHUNK_CHARS
    ]

    if not candidates:
        raise RuntimeError(
            "No usable chunks in vector store. Run `python preprocessing.py --clear` first."
        )

    if n >= len(candidates):
        return candidates

    rng = random.Random(RANDOM_SEED)
    return rng.sample(candidates, n)


def build_gold_dataset(n: int = DEFAULT_N, out_path: Path = DEFAULT_OUT) -> dict:
    """
    Build the gold dataset. Returns a summary dict with counts.
    Overwrites out_path.
    """
    chunks = _sample_chunks(n)
    logger.info("Sampled %d chunks (min %d chars each)", len(chunks), MIN_CHUNK_CHARS)

    structured_llm = get_structured_llm(GeneratedQuestion, task_type="light")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    rejected = 0
    failed = 0

    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        prompt = GEN_PROMPT.format(
            doc=meta.get("document_name", "?"),
            page=meta.get("page_number", "?"),
            chunk_text=chunk["content"][:1500],
        )

        try:
            result: GeneratedQuestion = structured_llm.invoke(prompt)
        except Exception as e:
            failed += 1
            logger.warning("[%d/%d] LLM call failed: %s", i + 1, len(chunks), e)
            continue

        if not result.is_specific or len(result.question.strip()) < 10:
            rejected += 1
            logger.debug("[%d/%d] rejected (not specific): %s", i + 1, len(chunks), result.question[:80])
            continue

        record = {
            "qid": f"q_{len(records) + 1:03d}",
            "question": result.question.strip(),
            "relevant_chunk_ids": [chunk["id"]],
            "relevant_pages": [{
                "document": meta.get("document_name", ""),
                "page": meta.get("page_number", 0),
            }],
            "ground_truth_excerpt": result.ground_truth_excerpt.strip(),
            "source_chunk_type": meta.get("chunk_type", "child"),
        }
        records.append(record)
        logger.info("[%d/%d] %s -> %s", i + 1, len(chunks), record["qid"], record["question"][:80])

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "sampled": len(chunks),
        "accepted": len(records),
        "rejected_not_specific": rejected,
        "llm_failures": failed,
        "out_path": str(out_path),
    }
    logger.info("Gold dataset built: %s", summary)
    return summary


def _cli():
    parser = argparse.ArgumentParser(description="Build a gold retrieval-eval dataset.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of chunks to sample")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSONL path")
    args = parser.parse_args()

    setup_logging(level="INFO")
    summary = build_gold_dataset(n=args.n, out_path=args.out)
    print("\n=== GOLD DATASET BUILD ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
