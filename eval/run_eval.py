"""
Runs the golden dataset (eval/golden_dataset.json) against the live API and
reports faithfulness, answer relevance, retrieval precision@k, latency, and
cost. Results are written to eval/results.json and a summary printed to stdout
so you can paste it straight into the README.

Usage:
    uvicorn app.main:app &   # make sure the API is running first
    python eval/run_eval.py
"""
import json
import statistics
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import answer_relevance, faithfulness, retrieval_precision_at_k  # noqa: E402

API_URL = "http://localhost:8000"
GOLDEN_SET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def run() -> None:
    golden_set = json.loads(GOLDEN_SET_PATH.read_text())

    per_question_results = []
    for item in golden_set:
        response = requests.post(f"{API_URL}/query", json={"query": item["question"]}, timeout=60)
        response.raise_for_status()
        data = response.json()

        retrieved_titles = [c["document_title"] for c in data["citations"]]
        context_chunks = [c["snippet"] for c in data["citations"]]

        per_question_results.append(
            {
                "question": item["question"],
                "answer": data["answer"],
                "retrieval_precision": retrieval_precision_at_k(
                    retrieved_titles, item.get("relevant_document_titles", [])
                ),
                "answer_relevance": answer_relevance(
                    data["answer"], item.get("expected_answer_contains", [])
                ),
                "faithfulness": faithfulness(data["answer"], context_chunks),
                "latency_ms": data["latency_ms"],
                "cost_usd": data["estimated_cost_usd"],
            }
        )

    summary = {
        "n_questions": len(per_question_results),
        "avg_retrieval_precision": statistics.mean(r["retrieval_precision"] for r in per_question_results),
        "avg_answer_relevance": statistics.mean(r["answer_relevance"] for r in per_question_results),
        "avg_faithfulness": statistics.mean(r["faithfulness"] for r in per_question_results),
        "avg_latency_ms": statistics.mean(r["latency_ms"] for r in per_question_results),
        "avg_cost_usd": statistics.mean(r["cost_usd"] for r in per_question_results),
        "per_question": per_question_results,
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    print(f"Evaluated {summary['n_questions']} questions.\n")
    print(f"  Retrieval precision@k : {summary['avg_retrieval_precision']:.3f}")
    print(f"  Answer relevance      : {summary['avg_answer_relevance']:.3f}")
    print(f"  Faithfulness          : {summary['avg_faithfulness']:.3f}")
    print(f"  Avg latency           : {summary['avg_latency_ms']:.0f} ms")
    print(f"  Avg cost/query        : ${summary['avg_cost_usd']:.5f}")
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    run()
