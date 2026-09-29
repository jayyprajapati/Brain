"""Retrieval + faithfulness eval harness.

Runs a per-app "golden set" of queries against the live retrieval pipeline
(and, optionally, generation) and scores it on:

  - Retrieval recall@k   — did the expected doc_id(s) actually come back?
  - Retrieval MRR        — how high did the first expected doc_id rank?
  - Keyword coverage     — cheap fallback check when a case has no known
                            doc_id yet, just terms the answer should surface.
  - Citation faithfulness — for cases that opt in, generate an answer the
                            same way /v1/chat would (same system-prompt
                            assembly) and have an LLM judge whether every
                            claim in it is actually backed by the retrieved
                            chunks.

This exists so a chunking, reranking, prompt, or model-routing change can be
checked against a standing baseline instead of eyeballing a few queries by
hand. It talks to Brain's pipeline in-process (not over HTTP) so it needs the
same .env Brain itself uses (QDRANT_URL, QDRANT_API_KEY, OLLAMA_API_KEY, ...).

Run: python scripts/eval.py [--goldens-dir eval/goldens] [--app portfolio]
                             [--top-k N] [--min-recall 0.7]
                             [--min-faithfulness 0.9] [--skip-faithfulness]
                             [--judge-model MODEL]

Golden set format: see eval/goldens/README.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm, pipeline  # noqa: E402
from app.prompts import build_chat_system, build_context_block  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

JUDGE_SYSTEM_PROMPT = """\
You are a strict fact-checking judge for a RAG system. You will be given SOURCE \
NOTES (the only ground truth) and an ANSWER a chatbot produced from them. \
Decide whether every factual claim in the ANSWER — names, numbers, dates, \
links, and specific statements — is directly supported by the SOURCE NOTES. \
Paraphrasing and reasonable inference are fine; anything invented, guessed, or \
not traceable to the notes is NOT fine.

Respond with ONLY a JSON object of this exact shape, nothing else:
{"faithful": true|false, "unsupported_claims": ["...", ...]}
"""


@dataclass
class CaseResult:
    case_id: str
    query: str
    recall: float | None = None
    mrr: float | None = None
    keyword_coverage: float | None = None
    faithful: bool | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    error: str | None = None


def _score_doc_ids(expected: list[str], retrieved: list[str]) -> tuple[float, float]:
    if not expected:
        return 1.0, 1.0
    expected_set = set(expected)
    found = expected_set & set(retrieved)
    recall = len(found) / len(expected_set)
    rank = next((i for i, d in enumerate(retrieved, start=1) if d in expected_set), None)
    mrr = (1.0 / rank) if rank else 0.0
    return recall, mrr


def _score_keywords(expected_keywords: list[str], chunks) -> float:
    if not expected_keywords:
        return 1.0
    blob = " ".join(c.text for c in chunks).lower()
    hit = sum(1 for kw in expected_keywords if kw.lower() in blob)
    return hit / len(expected_keywords)


async def _judge_faithfulness(client_prompt: str, chunks, query: str, judge_model: str | None) -> tuple[bool, list[str]]:
    system = build_chat_system(client_prompt, chunks)
    answer = await llm.generate(system, query, max_tokens=400, temperature=0.2)

    judge_user = f"SOURCE NOTES:\n{build_context_block(chunks)}\n\nANSWER:\n{answer}"
    verdict_text = await llm.generate(
        JUDGE_SYSTEM_PROMPT,
        judge_user,
        model=judge_model,
        response_format="json",
        temperature=0.0,
        max_tokens=400,
    )
    verdict = llm.parse_json(verdict_text)
    return bool(verdict.get("faithful")), list(verdict.get("unsupported_claims") or [])


async def run_case(app_name: str, client_prompt: str | None, case: dict, top_k: int | None, judge_model: str | None, skip_faithfulness: bool) -> CaseResult:
    case_id = case.get("id", case["query"][:40])
    result = CaseResult(case_id=case_id, query=case["query"])
    try:
        chunks = pipeline.retrieve(
            app_name,
            case["query"],
            case.get("doc_ids"),
            namespace=case.get("namespace"),
            top_k=top_k,
        )
        result.retrieved_doc_ids = [c.doc_id for c in chunks]
        result.recall, result.mrr = _score_doc_ids(case.get("expected_doc_ids") or [], result.retrieved_doc_ids)
        result.keyword_coverage = _score_keywords(case.get("expected_keywords") or [], chunks)

        if case.get("check_faithfulness") and not skip_faithfulness:
            if not client_prompt:
                raise ValueError("case requests check_faithfulness but golden file has no client_prompt")
            if not chunks:
                result.faithful = False
                result.unsupported_claims = ["no chunks retrieved to ground an answer in"]
            else:
                result.faithful, result.unsupported_claims = await _judge_faithfulness(
                    client_prompt, chunks, case["query"], judge_model
                )
    except Exception as exc:  # noqa: BLE001 — surface as a failed case, don't abort the run
        result.error = str(exc)
    return result


def _load_goldens(goldens_dir: Path, only_app: str | None) -> list[dict]:
    files = sorted(p for p in goldens_dir.glob("*.json") if ".example" not in p.suffixes and not p.stem.endswith(".example"))
    goldens = []
    for path in files:
        data = json.loads(path.read_text())
        if only_app and data.get("app_name") != only_app:
            continue
        goldens.append(data)
    return goldens


def _print_report(app_name: str, results: list[CaseResult]) -> None:
    print(f"\n=== {app_name} ({len(results)} cases) ===")
    for r in results:
        if r.error:
            print(f"  [ERROR] {r.case_id}: {r.error}")
            continue
        bits = [f"recall={r.recall:.2f}", f"mrr={r.mrr:.2f}", f"kw={r.keyword_coverage:.2f}"]
        if r.faithful is not None:
            bits.append(f"faithful={'yes' if r.faithful else 'NO'}")
        print(f"  {r.case_id}: {' '.join(bits)}")
        if r.unsupported_claims:
            for claim in r.unsupported_claims:
                print(f"      unsupported: {claim}")


def _aggregate(results: list[CaseResult]) -> dict:
    scored = [r for r in results if r.error is None]
    faith_checked = [r for r in scored if r.faithful is not None]
    return {
        "cases": len(results),
        "errors": len(results) - len(scored),
        "avg_recall": round(sum(r.recall for r in scored) / len(scored), 4) if scored else 0.0,
        "avg_mrr": round(sum(r.mrr for r in scored) / len(scored), 4) if scored else 0.0,
        "avg_keyword_coverage": round(sum(r.keyword_coverage for r in scored) / len(scored), 4) if scored else 0.0,
        "faithfulness_pass_rate": round(sum(1 for r in faith_checked if r.faithful) / len(faith_checked), 4) if faith_checked else None,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goldens-dir", default=str(REPO_ROOT / "eval" / "goldens"))
    parser.add_argument("--report-dir", default=str(REPO_ROOT / "eval" / "reports"))
    parser.add_argument("--app", default=None, help="only run the golden file for this app_name")
    parser.add_argument("--top-k", type=int, default=None, help="override RERANK_TOP_N-scoped retrieval count")
    parser.add_argument("--judge-model", default=None, help="override the model used for the faithfulness judge")
    parser.add_argument("--skip-faithfulness", action="store_true", help="skip LLM-judged faithfulness checks (fast, free)")
    parser.add_argument("--min-recall", type=float, default=0.7)
    parser.add_argument("--min-faithfulness", type=float, default=0.9)
    args = parser.parse_args()

    goldens_dir = Path(args.goldens_dir)
    goldens = _load_goldens(goldens_dir, args.app)
    if not goldens:
        print(f"No golden files found in {goldens_dir} (see eval/goldens/README.md to add one).")
        return 1

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    overall_ok = True
    full_report: dict = {"generated_at": timestamp, "apps": {}}

    for golden in goldens:
        app_name = golden["app_name"]
        client_prompt = golden.get("client_prompt")
        results = [
            await run_case(app_name, client_prompt, case, args.top_k, args.judge_model, args.skip_faithfulness)
            for case in golden.get("cases", [])
        ]
        _print_report(app_name, results)
        agg = _aggregate(results)
        print(f"  -> avg_recall={agg['avg_recall']:.2f} avg_mrr={agg['avg_mrr']:.2f} "
              f"avg_keyword_coverage={agg['avg_keyword_coverage']:.2f} "
              f"faithfulness_pass_rate={agg['faithfulness_pass_rate']} errors={agg['errors']}")

        if agg["errors"]:
            overall_ok = False
        if agg["avg_recall"] < args.min_recall:
            overall_ok = False
            print(f"  FAIL: avg_recall {agg['avg_recall']:.2f} < min_recall {args.min_recall}")
        if agg["faithfulness_pass_rate"] is not None and agg["faithfulness_pass_rate"] < args.min_faithfulness:
            overall_ok = False
            print(f"  FAIL: faithfulness_pass_rate {agg['faithfulness_pass_rate']:.2f} < min_faithfulness {args.min_faithfulness}")

        full_report["apps"][app_name] = {
            "aggregate": agg,
            "cases": [
                {
                    "id": r.case_id,
                    "query": r.query,
                    "recall": r.recall,
                    "mrr": r.mrr,
                    "keyword_coverage": r.keyword_coverage,
                    "faithful": r.faithful,
                    "unsupported_claims": r.unsupported_claims,
                    "retrieved_doc_ids": r.retrieved_doc_ids,
                    "error": r.error,
                }
                for r in results
            ],
        }

    report_path = report_dir / f"eval_{timestamp}.json"
    report_path.write_text(json.dumps(full_report, indent=2))
    print(f"\nReport written to {report_path}")
    print("PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
