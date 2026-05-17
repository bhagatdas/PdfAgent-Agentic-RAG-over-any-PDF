"""
RAGAS evaluation — faithfulness, answer relevancy, context precision.
Uses the RAGAS framework with LLM-as-judge approach.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def evaluate_with_ragas(
    test_data: list[dict],
    output_path: Optional[str] = None,
) -> dict:
    """
    Evaluate RAG pipeline using RAGAS metrics.

    Args:
        test_data: List of dicts with keys: question, answer, contexts, ground_truth
        output_path: Optional path to save results JSON

    Returns:
        Dict with metric scores
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        # Prepare dataset
        dataset_dict = {
            "question": [d["question"] for d in test_data],
            "answer": [d["answer"] for d in test_data],
            "contexts": [d.get("contexts", []) for d in test_data],
            "ground_truth": [d.get("ground_truth", "") for d in test_data],
        }
        dataset = Dataset.from_dict(dataset_dict)

        # Run evaluation
        logger.info("Running RAGAS evaluation on %d samples...", len(test_data))
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )

        scores = dict(results)
        logger.info("RAGAS Results: %s", scores)

        # Save results
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "num_samples": len(test_data),
                    "scores": scores,
                }, f, indent=2)

        return scores

    except ImportError:
        logger.warning("RAGAS not installed. Run: pip install ragas datasets")
        return {"error": "RAGAS not installed"}
    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        return {"error": str(e)}
