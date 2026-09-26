"""
Runs the golden dataset (eval/golden_dataset.json) against the RAG system and
reports faithfulness, answer relevance, retrieval precision@k, latency, and
cost. Results are written to eval/results.json and a markdown summary is printed.

Supports testing against a live HTTP server or direct in-process FastAPI TestClient.

Usage:
    python eval/run_eval.py [--api-url http://localhost:8000]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import answer_relevance, faithfulness, retrieval_precision_at_k  # noqa: E402

GOLDEN_SET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def query_api(url: str | None, payload: dict) -> dict:
    if url:
        import requests

        response = requests.post(f"{url.rstrip('/')}/query", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    else:
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.post("/query", json=payload)
        response.raise_for_status()
        return response.json()


def run(api_url: str | None = None) -> dict:
    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found at {GOLDEN_SET_PATH}")

    golden_set = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    per_question_results = []

    print(f"\n🚀 Running DocuMind Evaluation Harness on {len(golden_set)} benchmark cases...\n")

    for idx, item in enumerate(golden_set, 1):
        q = item["question"]
        try:
            data = query_api(api_url, {"query": q})
            retrieved_titles = [c["document_title"] for c in data.get("citations", [])]
            context_chunks = [c["snippet"] for c in data.get("citations", [])]

            prec = retrieval_precision_at_k(retrieved_titles, item.get("relevant_document_titles", []))
            rel = answer_relevance(data.get("answer", ""), item.get("expected_answer_contains", []))
            faith = faithfulness(data.get("answer", ""), context_chunks)
            lat = data.get("latency_ms", 0)
            cost = data.get("estimated_cost_usd", 0.0)

            per_question_results.append(
                {
                    "id": idx,
                    "question": q,
                    "answer": data.get("answer", ""),
                    "is_grounded": data.get("is_grounded", True),
                    "retrieval_precision": prec,
                    "answer_relevance": rel,
                    "faithfulness": faith,
                    "latency_ms": lat,
                    "cost_usd": cost,
                }
            )
            print(f"[{idx:02d}/{len(golden_set):02d}] P@k: {prec:.2f} | Rel: {rel:.2f} | Faith: {faith:.2f} | {lat}ms | {q[:50]}...")
        except Exception as e:
            print(f"[{idx:02d}/{len(golden_set):02d}] ❌ Error evaluating query: {e}")

    if not per_question_results:
        print("No evaluation results collected.")
        return {}

    summary = {
        "n_questions": len(per_question_results),
        "avg_retrieval_precision": statistics.mean(r["retrieval_precision"] for r in per_question_results),
        "avg_answer_relevance": statistics.mean(r["answer_relevance"] for r in per_question_results),
        "avg_faithfulness": statistics.mean(r["faithfulness"] for r in per_question_results),
        "avg_latency_ms": statistics.mean(r["latency_ms"] for r in per_question_results),
        "avg_cost_usd": statistics.mean(r["cost_usd"] for r in per_question_results),
        "per_question": per_question_results,
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("📊 DOCUMIND QUANTITATIVE BENCHMARK RESULTS")
    print("=" * 60)
    print("| Metric | Score |")
    print("|---|---|")
    print(f"| Retrieval precision@5 | {summary['avg_retrieval_precision']:.3f} |")
    print(f"| Answer relevance      | {summary['avg_answer_relevance']:.3f} |")
    print(f"| Faithfulness          | {summary['avg_faithfulness']:.3f} |")
    print(f"| Avg. latency (ms)     | {summary['avg_latency_ms']:.1f} ms |")
    print(f"| Avg. cost per query   | ${summary['avg_cost_usd']:.6f} |")
    print("=" * 60)
    print(f"Full results saved to: {RESULTS_PATH}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DocuMind RAG Evaluation Harness.")
    parser.add_argument("--api-url", default=None, help="Target API URL (optional, runs in-process if omitted)")
    args = parser.parse_args()
    run(args.api_url)
