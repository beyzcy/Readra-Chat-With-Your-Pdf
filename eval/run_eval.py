"""Evaluate Readra's retrieval and (optionally) answer quality.

Two question sets, deliberately kept apart:
  * held-out test set (eval/dataset_heldout.json, eval/heldout/*.pdf): written once, never used to tune
    the pipeline. This is the number to quote.
  * development set (eval/dataset.json): written while fixing known failures, so it is optimistic.

Run from the project root:
    python eval/run_eval.py                         # retrieval only, no API key needed
    GROQ_API_KEY=gsk_... python eval/run_eval.py    # + answer correctness, wrong values, grounding
    python eval/run_eval.py --only heldout --model llama-3.3-70b-versatile
    READRA_RERANKER=off python eval/run_eval.py     # measure without the reranker

Writes eval/RESULTS.md. Every rate is shown with its count and a 95% Wilson confidence interval.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
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

EVAL = ROOT / "eval"
OUT = EVAL / "RESULTS.md"
DEV_DOCS = {"handbook_eval.pdf": EVAL / "handbook_eval.pdf", "ticket_tr.pdf": EVAL / "ticket_tr.pdf"}
HELDOUT_DIR = EVAL / "heldout"
SCAN = "invoice_scan.pdf"  # same content as invoice_en.pdf, so it gets its own index to keep retrieval unambiguous

REFUSAL_HINTS = (
    "not contain", "does not", "doesn't", "no information", "not mention", "not specified", "not provided",
    "not in the", "cannot find", "can't find", "not available", "not stated", "not given", "no mention",
    "içermiyor", "bulunmuyor", "yer almıyor", "yer almamaktadır", "belirtilmemiş", "bilgi yok", "bulunamadı",
    "bulamadım", "mevcut değil", "bulunmamaktadır", "not state", "not specify", "not say",
)
NEGATED_FACT = re.compile(r"\bno\b[^.]{0,40}\b(mentioned|stated|specified|given|provided)\b")


def norm(s: str) -> str:
    """Compare answers by content, not typography: model output uses non-breaking spaces, non-ASCII hyphens, bold markers."""
    s = unicodedata.normalize("NFKC", s).lower().replace("*", "")
    s = re.sub(r"[‐-―−]", "-", s)
    return re.sub(r"\s+", " ", s)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}% ({k}/{n}; 95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"


def pick_model(client: Groq) -> str:
    """Strongest chat model the key can use (Groq retires model ids, so ask instead of hard-coding)."""
    skip = ("whisper", "guard", "tts", "playai", "embed", "orpheus", "safeguard")
    ids = [m.id for m in client.models.list().data if not any(t in m.id.lower() for t in skip)]
    ids.sort(key=lambda i: (0 if "llama" in i.lower() else 1, 0 if ("70b" in i or "120b" in i) else 1, i))
    if not ids:
        raise SystemExit("This key can use no chat model.")
    return ids[0]


def build_index(embedder, paths: dict[str, Path]) -> PDFIndex:
    idx = PDFIndex(embedder)
    for name, path in paths.items():
        idx.add_pdf(path.read_bytes(), name)
    return idx


class Suite:
    """One question set with the indexes its questions are asked against."""

    def __init__(self, label: str, items: list[dict], resolve):
        self.label, self.items, self.resolve = label, items, resolve  # resolve(item) -> (index, scope or None)
        self.cache: dict[int, list] = {}
        self.retrieval: dict = {}
        self.answers: dict | None = None


def retrieval_metrics(s: Suite) -> None:
    answerable = [it for it in s.items if it.get("answerable", True) is not False]
    page_ranks, evid_ranks, non_en = [], [], []
    for n, it in enumerate(answerable, 1):
        idx, _ = s.resolve(it)
        chunks = s.cache[id(it)] = idx.search_chunks(it["q"])   # all documents of the pool: no scope help
        found = [(src.file, src.page) for src in idx.pages_from_chunks(chunks)]
        want = (it["doc"], it["page"])
        page_ranks.append(found.index(want) + 1 if want in found else None)
        if "evidence" in it:
            hit = next((r for r, c in enumerate(chunks, 1)
                        if c.file == it["doc"] and it["evidence"].lower() in c.text.lower()), None)
            evid_ranks.append(hit)
        non_en.append(it.get("lang", "en") != "en")
        print(f"\r  {s.label}: retrieval {n}/{len(answerable)}", end="", flush=True)
    print()
    at = lambda ranks, k: sum(1 for r in ranks if r and r <= k)  # noqa: E731
    s.retrieval = {
        "n": len(answerable), "page1": at(page_ranks, 1), "pageK": at(page_ranks, TOP_K),
        "mrr": sum(1 / r for r in page_ranks if r) / max(1, len(page_ranks)),
        "n_evid": len(evid_ranks), "evid1": at(evid_ranks, 1), "evid3": at(evid_ranks, 3),
        "n_foreign": sum(non_en),
        "foreign1": sum(1 for r, f in zip(page_ranks, non_en) if f and r == 1),
        "misses": [(it["q"], it["doc"], r) for it, r in zip(answerable, page_ranks) if r != 1],
    }


def answer_metrics(s: Suite, client: Groq, model: str) -> None:
    tot = dict(n=0, correct=0, wrong=0, verified=0, mixup=0, unsupported=0, refused_ok=0, refused_n=0)
    by_kind: dict[str, list[int]] = {}
    failures = []
    for n, it in enumerate(s.items, 1):
        idx, scope = s.resolve(it)
        chunks = s.cache.get(id(it)) or idx.search_chunks(it["q"], files=scope)
        if scope:  # the user asked inside one document: search only there
            chunks = idx.search_chunks(it["q"], files=scope)
        sources = idx.pages_from_chunks(chunks)
        print(f"\r  {s.label}: answers {n}/{len(s.items)}", end="", flush=True)
        split, text = EvidenceSplitter(), ""
        try:
            for tok in stream_answer(client, model, build_messages(it["q"], sources, [])):
                text += split.feed(tok)
        except Exception as e:
            failures.append((it["q"], "the model call failed", type(e).__name__))
            time.sleep(4)
            continue
        text = (text + split.finish()).strip()
        low = norm(text)
        ev = verify_answer(text, split.evidence, sources)
        kind = it.get("kind", "other")
        if it.get("answerable", True) is False:
            tot["refused_n"] += 1
            ok = any(h in low for h in REFUSAL_HINTS) or bool(NEGATED_FACT.search(low))
            tot["refused_ok"] += ok
            by_kind.setdefault(kind, []).append(int(ok))
            if not ok:
                failures.append((it["q"], "should have said the document does not contain it", text[:150]))
        else:
            tot["n"] += 1
            ok = any(norm(k) in low for k in it["any"])
            wrong = [f for f in it.get("forbid", []) if norm(f) in low]
            tot["correct"] += ok
            tot["wrong"] += bool(wrong)
            tot["verified"] += ev["status"] == "verified"
            tot["unsupported"] += bool(ev["unsupported"])
            if ev["items"]:
                tot["mixup"] += not judge_answer(client, model, it["q"], text, ev["items"])["ok"]
            by_kind.setdefault(kind, []).append(int(ok and not wrong))
            if not ok or wrong:
                why = f"expected one of {it['any']}" + (f"; contains forbidden {wrong}" if wrong else "")
                failures.append((it["q"], why, text[:150]))
        time.sleep(1.5)  # stay inside free-tier rate limits
    print()
    s.answers = {"model": model, "tot": tot, "by_kind": by_kind, "failures": failures}


def report(suites: list[Suite], reranker_on: bool, model: str | None) -> str:
    md = [
        "# Evaluation results", "",
        f"Embeddings `{EMBED_MODEL.split('/')[-1]}` + BM25 · reranker: "
        f"{'`' + rag.RERANKER_MODEL + '`' if reranker_on else 'off'} · top {TOP_K} pages. "
        "Every rate shows its count and a 95% Wilson confidence interval: with few questions the interval is "
        "wide, and that is the honest size of the claim.", "",
    ]
    for s in suites:
        r = s.retrieval
        md += [f"## {s.label}", ""]
        if s.label.startswith("Held-out"):
            md += ["Written once, not used to tune the pipeline. Two extraction bugs (a label and its value in "
                   "different fonts were ordered wrongly; two text columns were read interleaved) were found by "
                   "inspecting the extracted text of these documents and fixed *before any question was run*.", ""]
        else:
            md += ["Written while fixing known failures, so this set is optimistic by construction.", ""]
        md += ["### Retrieval", "", "| Metric | Result |", "|---|---|",
               f"| Correct page ranked first | {rate(r['page1'], r['n'])} |",
               f"| Correct page in the top {TOP_K} | {rate(r['pageK'], r['n'])} |",
               f"| Mean reciprocal rank | {r['mrr']:.2f} |",
               f"| Non-English questions, page first | {rate(r['foreign1'], r['n_foreign'])} |",
               f"| Evidence passage is the top chunk | {rate(r['evid1'], r['n_evid'])} |",
               f"| Evidence passage in the top 3 chunks | {rate(r['evid3'], r['n_evid'])} |", ""]
        if r["misses"]:
            md += ["Not ranked first:", ""] + [f"- “{q}” ({d}): {'rank ' + str(k) if k else 'not found'}" for q, d, k in r["misses"]] + [""]
        a = s.answers
        if a:
            t = a["tot"]
            md += [f"### Answers (`{a['model']}`, temperature 0)", "", "| Metric | Result |", "|---|---|",
                   f"| Expected fact present | {rate(t['correct'], t['n'])} |",
                   f"| **A forbidden look-alike value appears** | {rate(t['wrong'], t['n'])} |",
                   f"| Evidence quotes verified verbatim | {rate(t['verified'], t['n'])} |",
                   f"| Judge flagged a possible mix-up | {rate(t['mixup'], t['n'])} |",
                   f"| A value absent from the cited text | {rate(t['unsupported'], t['n'])} |",
                   f"| Unanswerable questions refused | {rate(t['refused_ok'], t['refused_n'])} |", "",
                   "By kind of question (correct and no forbidden value; for unanswerable: refused):", "",
                   "| Kind | Result |", "|---|---|"]
            md += [f"| {k} | {rate(sum(v), len(v))} |" for k, v in sorted(a["by_kind"].items())] + [""]
            if a["failures"]:
                md += ["Failures:", ""] + [f"- “{q}”: {why}. Got: _{got}…_" for q, why, got in a["failures"]] + [""]
    if not any(s.answers for s in suites):
        md += ["_Answer-level results need `GROQ_API_KEY`. Run `python eval/run_eval.py` with a key to add them._", ""]
    md.append("_Correctness, forbidden-value and refusal checks are keyword heuristics, not human grading._")
    return "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Groq model id (default: best Llama available to your key)")
    ap.add_argument("--only", choices=["heldout", "dev"], help="run just one of the two sets")
    args = ap.parse_args()

    embedder = TextEmbedding(model_name=EMBED_MODEL)
    reranker_on = rag.get_reranker() is not None
    print("reranker:", rag.RERANKER_MODEL if reranker_on else "off")
    suites: list[Suite] = []

    if args.only != "heldout":
        items = json.loads((EVAL / "dataset.json").read_text(encoding="utf-8"))
        for it in items:
            it.setdefault("doc", "handbook_eval.pdf")  # older questions omit it
        per_doc = {name: build_index(embedder, {name: path}) for name, path in DEV_DOCS.items()}
        suites.append(Suite("Development set", items, lambda it: (per_doc[it.get("doc", "handbook_eval.pdf")], None)))

    if args.only != "dev":
        items = json.loads((EVAL / "dataset_heldout.json").read_text(encoding="utf-8"))
        paths = {p.name: p for p in sorted(HELDOUT_DIR.glob("*.pdf"))}
        pooled = build_index(embedder, {n: p for n, p in paths.items() if n != SCAN})  # like "All documents"
        scan = build_index(embedder, {SCAN: paths[SCAN]})
        print(f"held-out pool: {pooled.files} ({pooled.n_chunks} chunks) + {SCAN}")

        def resolve(it, pooled=pooled, scan=scan):
            return (scan, None) if it["doc"] == SCAN else (pooled, [it["doc"]])

        # retrieval is measured against the whole pool, answers inside the question's own document
        class Heldout(Suite):
            pass

        suites.append(Heldout("Held-out test set", items, lambda it: (resolve(it)[0], None)))
        suites[-1].scope_for = lambda it: resolve(it)[1]

    for s in suites:
        retrieval_metrics(s)
        r = s.retrieval
        print(f"{s.label}: page@1 {rate(r['page1'], r['n'])} | evidence top chunk {rate(r['evid1'], r['n_evid'])}")

    model = None
    key = os.getenv("GROQ_API_KEY")
    if key:
        client = Groq(api_key=key)
        model = args.model or pick_model(client)
        print(f"answers with {model} ...")
        for s in suites:
            if hasattr(s, "scope_for"):
                inner = s.resolve
                s.resolve = lambda it, inner=inner, f=s.scope_for: (inner(it)[0], f(it))
            answer_metrics(s, client, model)
            t = s.answers["tot"]
            print(f"{s.label}: correct {rate(t['correct'], t['n'])} | forbidden value {rate(t['wrong'], t['n'])}")
    else:
        print("GROQ_API_KEY not set: skipping answer evaluation.")

    OUT.write_text(report(suites, reranker_on, model), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
