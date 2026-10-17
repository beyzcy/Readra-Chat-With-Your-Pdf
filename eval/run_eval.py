"""Evaluate Readra's retrieval and (optionally) answer quality on labelled question sets.

Run from the project root:
    python eval/run_eval.py                       # retrieval metrics only, no API key needed
    GROQ_API_KEY=gsk_... python eval/run_eval.py  # + answer correctness, wrong-value, grounding checks
    python eval/run_eval.py --model llama-3.3-70b-versatile
    READRA_RERANKER=off python eval/run_eval.py   # measure the pipeline without the reranker

Writes eval/RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastembed import TextEmbedding  # noqa: E402
from groq import Groq  # noqa: E402

import rag  # noqa: E402
from rag import (  # noqa: E402
    EMBED_MODEL,
    TOP_K,
    EvidenceSplitter,
    PDFIndex,
    build_messages,
    judge_answer,
    stream_answer,
    verify_answer,
)

DOCS = {"handbook_eval.pdf": ROOT / "eval" / "handbook_eval.pdf", "ticket_tr.pdf": ROOT / "eval" / "ticket_tr.pdf"}
DEFAULT_DOC = "handbook_eval.pdf"
DATASET = ROOT / "eval" / "dataset.json"
OUT = ROOT / "eval" / "RESULTS.md"

REFUSAL_HINTS = (
    "not contain", "does not", "doesn't", "no information", "not mention", "not specified",
    "not provided", "not in the", "cannot find", "can't find", "not available", "not stated",
    "içermiyor", "bulunmuyor", "yer almıyor", "yer almamaktadır", "belirtilmemiş", "bilgi yok",
    "bulunamadı", "bulamadım", "mevcut değil", "yok",
)


def pick_model(client: Groq) -> str:
    ids = [m.id for m in client.models.list().data]
    llamas = [i for i in ids if "llama" in i.lower() and not any(t in i.lower() for t in ("guard", "whisper"))]
    llamas.sort(key=lambda i: (0 if "70b" in i else 1, i))
    return (llamas or ids)[0]


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Groq model id (default: best Llama available to your key)")
    args = ap.parse_args()

    items = json.loads(DATASET.read_text(encoding="utf-8"))
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    indexes: dict[str, PDFIndex] = {}
    for name, path in DOCS.items():
        idx = PDFIndex(embedder)
        idx.add_pdf(path.read_bytes(), name)
        indexes[name] = idx
    reranker_on = rag.get_reranker() is not None
    print(f"reranker: {'on (' + rag.RERANKER_MODEL + ')' if reranker_on else 'off'}")
    for name, idx in indexes.items():
        print(f"{name}: {idx.n_chunks} chunks")

    # ---- retrieval ----
    answerable = [it for it in items if it.get("answerable", True) is not False]
    page_ranks, evid_ranks = [], []
    cache: dict[tuple[str, str], list] = {}  # one retrieval per question; the reranker is the slow part
    for n, it in enumerate(answerable, 1):
        name = it.get("doc", DEFAULT_DOC)
        idx = indexes[name]
        chunks = cache[(name, it["q"])] = idx.search_chunks(it["q"])
        pages = [s.page for s in idx.pages_from_chunks(chunks)]
        page_ranks.append((it, pages.index(it["page"]) + 1 if it["page"] in pages else None))
        if "evidence" in it:
            hit = next((r for r, c in enumerate(chunks, 1) if it["evidence"].lower() in c.text.lower()), None)
            evid_ranks.append((it, hit))
        print(f"\rretrieval {n}/{len(answerable)}", end="", flush=True)
    print()

    def rate(ranks, k, sub=lambda it: True):
        sel = [(it, r) for it, r in ranks if sub(it)]
        return sum(1 for _, r in sel if r and r <= k) / max(1, len(sel))

    non_en = lambda it: it.get("lang", "en") != "en"  # noqa: E731
    retrieval = {
        "Page Hit@1": rate(page_ranks, 1),
        f"Page Hit@{TOP_K}": rate(page_ranks, TOP_K),
        "Page MRR": sum(1 / r for _, r in page_ranks if r) / len(page_ranks),
        "Page Hit@1 (non-English questions)": rate(page_ranks, 1, non_en),
        "Evidence-chunk Hit@1": rate(evid_ranks, 1),
        "Evidence-chunk Hit@3": rate(evid_ranks, 3),
    }
    for k, v in retrieval.items():
        print(f"{k:42s} {v:.2f}")
    misses = [(it["q"], it.get("page"), r) for it, r in evid_ranks if r != 1]

    # ---- answers ----
    answer = None
    key = os.getenv("GROQ_API_KEY")
    if key:
        client = Groq(api_key=key)
        model = args.model or pick_model(client)
        print(f"\nGenerating answers with {model} …")
        tot = {"correct": 0, "forbidden": 0, "verified": 0, "judge_ok": 0, "unsupported": 0, "answerable": 0}
        refused_ok = refused_n = 0
        failures = []
        for n, it in enumerate(items, 1):
            name = it.get("doc", DEFAULT_DOC)
            idx = indexes[name]
            chunks = cache.get((name, it["q"])) or idx.search_chunks(it["q"])
            sources = idx.pages_from_chunks(chunks)
            print(f"\ranswers {n}/{len(items)}", end="", flush=True)
            split = EvidenceSplitter()
            text = ""
            for tok in stream_answer(client, model, build_messages(it["q"], sources, [])):
                text += split.feed(tok)
            text += split.finish()
            text = text.strip()
            ev = verify_answer(text, split.evidence, sources)
            low = text.lower()
            if it.get("answerable", True) is False:
                refused_n += 1
                ok = any(h in low for h in REFUSAL_HINTS)
                refused_ok += ok
                if not ok:
                    failures.append((it["q"], "should have said it is not in the document", text[:160]))
            else:
                tot["answerable"] += 1
                ok = any(k.lower() in low for k in it["any"])
                tot["correct"] += ok
                wrong = [f for f in it.get("forbid", []) if f.lower() in low]
                tot["forbidden"] += bool(wrong)
                tot["verified"] += ev["status"] == "verified"
                tot["unsupported"] += bool(ev["unsupported"])
                if ev["items"]:
                    tot["judge_ok"] += judge_answer(client, model, it["q"], text, ev["items"])["ok"]
                if not ok or wrong:
                    why = f"expected one of {it['any']}" + (f", but it contains {wrong}" if wrong else "")
                    failures.append((it["q"], why, text[:160]))
            time.sleep(1.5)  # stay well inside free-tier rate limits
        n = max(1, tot["answerable"])
        answer = {
            "model": model,
            "Answer correctness": tot["correct"] / n,
            "Wrong value given (forbidden value present)": tot["forbidden"] / n,
            "Evidence quotes verified in the document": tot["verified"] / n,
            "Answers with a value not found in the cited text": tot["unsupported"] / n,
            "Correct refusals (unanswerable)": refused_ok / max(1, refused_n),
            "failures": failures,
        }
        for k, v in answer.items():
            if k not in ("model", "failures"):
                print(f"{k:50s} {v:.2f}")
    else:
        print("\nGROQ_API_KEY not set: skipping answer evaluation.")

    # ---- report ----
    n_docs = len(DOCS)
    md = [
        "# Evaluation results",
        "",
        f"{len(answerable)} answerable + {len(items) - len(answerable)} unanswerable questions over {n_docs} documents "
        f"(a 12-page handbook and a one-page Turkish flight booking whose labels and values are stored in separate "
        f"column streams) · English, Turkish, German · embeddings `{EMBED_MODEL.split('/')[-1]}` + BM25 · "
        f"reranker: {'`' + rag.RERANKER_MODEL + '`' if reranker_on else 'off'} · top {TOP_K} pages",
        "",
        "## Retrieval",
        "",
        "| Metric | Score |", "|---|---|",
        *[f"| {k} | {f'{v:.2f}' if 'MRR' in k else pct(v)} |" for k, v in retrieval.items()],
        "",
        "_Evidence-chunk Hit@k: the passage containing the exact answer is among the top k retrieved chunks._",
        "",
    ]
    if misses:
        md += ["Not ranked first:", ""]
        md += [f"- “{q}” → evidence found at rank {r or 'not in candidates'}" for q, _, r in misses]
        md.append("")
    if answer:
        md += [
            f"## Answers (`{answer['model']}`, temperature 0)", "",
            "| Metric | Score |", "|---|---|",
            f"| Answer correctness (expected fact present) | {pct(answer['Answer correctness'])} |",
            f"| **Wrong value given** (e.g. payment time as flight time) | {pct(answer['Wrong value given (forbidden value present)'])} |",
            f"| Evidence quotes verified verbatim in the document | {pct(answer['Evidence quotes verified in the document'])} |",
            f"| Answers containing a number/time/date absent from the cited text | {pct(answer['Answers with a value not found in the cited text'])} |",
            f"| Correct refusals on unanswerable questions | {pct(answer['Correct refusals (unanswerable)'])} |",
            "",
        ]
        if answer["failures"]:
            md += ["Failures:", ""]
            md += [f"- “{q}” — {why}. Got: _{got}…_" for q, why, got in answer["failures"]]
            md.append("")
        md.append("_Correctness, wrong-value and refusal checks are keyword heuristics, not human grading._")
    else:
        md.append("_Answer-level metrics need `GROQ_API_KEY`; run `python eval/run_eval.py` with a key to add them._")
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
