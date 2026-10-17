"""Core RAG pipeline.

PDF (+OCR) -> row-aware text -> chunks -> embeddings (ChromaDB) + BM25
-> reciprocal-rank fusion -> cross-encoder rerank -> page-level context -> Groq LLM
-> answer with verbatim evidence quotes, checked against the source passages.
"""

from __future__ import annotations

import difflib
import math
import os
import re
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator

import chromadb
import numpy as np
import pymupdf
from fastembed import TextEmbedding
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Multilingual model: works for English, Turkish, German and 50+ other languages.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Multilingual cross-encoder (int8, ~0.6 GB). Set READRA_RERANKER=off to disable.
RERANKER_MODEL = os.getenv("READRA_RERANKER", "BAAI/bge-reranker-v2-m3-int8")

CHUNK_SIZE = 600        # small chunks keep a label and its value together
CHUNK_OVERLAP = 100
TOP_K = 5               # pages handed to the LLM
CANDIDATES = 8          # chunks that survive fusion and go to the reranker (latency grows with this)
PAGE_FULL_MAX = 3500    # pages shorter than this are given to the LLM in full

OCR_DPI = 200
MAX_OCR_PAGES = 25      # per file; OCR is CPU-heavy
MIN_TEXT_CHARS = 20     # a page with less text than this (but with images) is treated as scanned
SPARSE_TEXT_CHARS = 400  # below this a page may keep its real content outside the text layer: OCR it too
IMAGE_HEAVY = 0.25      # share of the page covered by pictures that also triggers OCR

EVIDENCE_MARK = "###EVIDENCE"

SYSTEM_PROMPT = f"""You are a careful, literal document assistant. Answer ONLY from the numbered passages.

Rules
1. Cite every fact with its passage number, e.g. [1] or [2][3].
2. Documents contain many look-alike values: booking date vs payment date vs travel date, issue date vs due date, net vs gross amounts, departure vs arrival times. Before you answer, find the LABEL printed next to the value in the passage and use the value only if that label matches exactly what the user asked for. Never use one kind of date, time, amount or number in place of another.
3. Text such as `Label  |  Value` comes from a table or form row: the value on the right belongs to the label on the left of the same row.
4. If two or more values could fit the question and you cannot tell which one is meant, say so and list each candidate with its label.
5. If the passages do not contain the answer, say so clearly. Never guess, infer or fill gaps from general knowledge.
6. Answer in the language of the question. Be brief; use bullets for lists.

Output format
- First the answer.
- Then a line containing exactly {EVIDENCE_MARK}
- Then one line per fact you used: [n] "words copied verbatim from passage n, including the label"
Keep quotes short (one row or one sentence). Do not paraphrase inside quotes."""


@dataclass
class Source:
    n: int
    file: str
    page: int
    text: str


@dataclass
class Chunk:
    id: str
    file: str
    page: int
    pos: int
    text: str
    tf: Counter = field(repr=False, default_factory=Counter)
    length: int = 0


# ---------- lexical helpers ----------
_WORD = re.compile(r"\w+", re.UNICODE)


def _terms(text: str) -> list[str]:
    """Lower-cased words plus 4-letter prefixes: a crude stemmer that helps inflected languages (Turkish...)."""
    out: list[str] = []
    for w in _WORD.findall(text.casefold()):
        out.append(w)
        if len(w) > 4:
            out.append(w[:4])
    return out


def _norm(word: str) -> str:
    return re.sub(r"\W+", "", word.lower())


def _squash(text: str) -> str:
    """Whitespace/punctuation-insensitive form used to compare quotes with passages."""
    return " ".join(_WORD.findall(text.casefold()))


# ---------- row-aware text extraction ----------
def _rows_text(items: list[tuple[float, float, float, float, str]], page_width: float) -> str:
    """Join short text lines that sit on the same visual row with ' | '.

    Printed forms and tables are often stored column by column, so a label and its value end up
    far apart in a plain text dump. Merging by visual row keeps `Departure | 07:15` together.
    Long lines (paragraphs, two-column papers) are never merged.
    """
    rows: list[list[tuple[float, float, float, str, bool]]] = []
    for x0, y0, x1, y1, text in items:
        text = text.replace(" ", " ").strip()
        if not text:
            continue
        short = len(text) <= 40 and (x1 - x0) <= 0.42 * page_width
        placed = False
        if short:
            for row in reversed(rows[-6:]):
                if not all(c[4] for c in row):
                    continue
                ry0, ry1 = min(c[1] for c in row), max(c[2] for c in row)
                overlap = min(y1, ry1) - max(y0, ry0)
                if overlap >= 0.5 * min(y1 - y0, ry1 - ry0) and x0 > max(c[0] for c in row):
                    row.append((x0, y0, y1, text, True))
                    placed = True
                    break
        if not placed:
            rows.append([(x0, y0, y1, text, short)])
    return "\n".join("  |  ".join(c[3] for c in sorted(r, key=lambda c: c[0])) for r in rows)


def _page_items(page: pymupdf.Page) -> list[tuple[float, float, float, float, str]]:
    d = page.get_text("dict", sort=True)
    items = []
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            t = "".join(s["text"] for s in ln["spans"])
            if t.strip():
                items.append((*ln["bbox"], t))
    return items


def _page_text(page: pymupdf.Page) -> str:
    return _rows_text(_page_items(page), page.rect.width)


def _image_coverage(page: pymupdf.Page) -> float:
    area = page.rect.width * page.rect.height
    try:
        boxes = [pymupdf.Rect(i["bbox"]) & page.rect for i in page.get_image_info()]
    except Exception:
        return 0.0
    return min(1.0, sum(b.get_area() for b in boxes) / area) if area else 0.0


def _needs_ocr(page: pymupdf.Page, native_chars: int) -> bool:
    """Is part of what a reader sees on this page missing from its text layer?

    Pages printed from browsers or apps often keep only a header and footer as text while the
    content (times, prices, tables) is stored as images or outlined shapes.
    """
    if native_chars < MIN_TEXT_CHARS:  # scanned page
        return bool(page.get_images() or page.get_drawings())
    if native_chars < SPARSE_TEXT_CHARS:  # little text: look for content outside the text layer
        return bool(page.get_images() or len(page.get_drawings()) > 20)
    return _image_coverage(page) >= IMAGE_HEAVY  # plenty of text, but also large pictures


def _ocr_only(native_items, ocr_lines):
    """OCR lines that add something: not on top of native text and not already spelled out there."""
    boxes = [pymupdf.Rect(*it[:4]) for it in native_items]
    known = {t for it in native_items for t in _terms(it[4])}
    keep = []
    for rect, txt in ocr_lines:
        toks = _terms(txt)
        if not toks:
            continue
        r = pymupdf.Rect(*rect)
        area = r.get_area() or 1.0
        covered = any((r & b).get_area() >= 0.5 * area for b in boxes)
        spelled = sum(t in known for t in toks) / len(toks) >= 0.8
        if not covered and not spelled:
            keep.append((rect, txt))
    return keep


# ---------- OCR ----------
_ocr_engine = None
_ocr_lock = threading.Lock()


def _ocr_page(page: pymupdf.Page) -> list[tuple[tuple[float, float, float, float], str]]:
    """OCR one page. Returns [(rect in PDF points, text)] per detected text line."""
    global _ocr_engine
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        img = np.repeat(img, 3, axis=2)
    elif pix.n == 4:
        img = img[:, :, :3]
    with _ocr_lock:  # the ONNX session is shared across requests
        if _ocr_engine is None:
            from rapidocr import RapidOCR
            _ocr_engine = RapidOCR()
        result = _ocr_engine(np.ascontiguousarray(img), use_cls=False)  # document pages are upright; the cls model flips lines
    if result is None or result.boxes is None or result.txts is None:
        return []
    scale = 72 / OCR_DPI
    lines = []
    for box, txt in zip(result.boxes, result.txts):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        rect = (min(xs) * scale, min(ys) * scale, max(xs) * scale, max(ys) * scale)
        lines.append((rect, str(txt)))
    lines.sort(key=lambda it: (round(it[0][1] / 4), it[0][0]))
    return lines


# ---------- reranker (lazy, optional) ----------
_reranker = None
_reranker_failed = False
_reranker_lock = threading.Lock()


def get_reranker():
    global _reranker, _reranker_failed
    if RERANKER_MODEL.lower() in ("", "off", "none", "0"):
        return None
    with _reranker_lock:
        if _reranker is None and not _reranker_failed:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                _reranker = TextCrossEncoder(model_name=RERANKER_MODEL)
            except Exception as e:  # offline, disk full...: retrieval still works without it
                print(f"[rag] reranker unavailable ({e}); continuing without it")
                _reranker_failed = True
        return _reranker


class PDFIndex:
    """Holds the vector index (and the PDFs themselves) for one user session."""

    def __init__(self, embedder: TextEmbedding):
        self.embedder = embedder
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection(
            name=f"docs_{uuid.uuid4().hex[:12]}",
            metadata={"hnsw:space": "cosine"},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self.files: list[str] = []
        self.n_chunks = 0
        self.ocr_pages = 0
        self.file_chunks: dict[str, int] = {}
        self.sizes: dict[str, list[tuple[float, float]]] = {}  # page sizes in PDF points, per file
        self.chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self._page_text: dict[tuple[str, int], str] = {}
        self._pdfs: dict[str, bytes] = {}
        self._docs: dict[str, pymupdf.Document] = {}
        self._ocr_lines: dict[tuple[str, int], list] = {}
        self._render_lock = threading.Lock()  # PyMuPDF documents are not thread-safe

    # ----- lifecycle -----
    def total_bytes(self) -> int:
        return sum(len(b) for b in self._pdfs.values())

    def close(self) -> None:
        """Free this session's vectors. ChromaDB's in-process store would otherwise keep them forever."""
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass
        self._docs.clear()
        self._pdfs.clear()
        self.chunks.clear()
        self._by_id.clear()
        self._page_text.clear()

    # ----- indexing -----
    def add_pdf(self, data: bytes, filename: str) -> int:
        """Extract text page by page (OCR for scanned pages), chunk, embed, store. Returns chunk count."""
        doc = pymupdf.open(stream=data, filetype="pdf")
        if doc.needs_pass:
            raise ValueError("PDF is password protected")
        sizes = [(round(p.rect.width, 1), round(p.rect.height, 1)) for p in doc]
        new: list[Chunk] = []
        page_texts: dict[tuple[str, int], str] = {}
        ocr_budget = MAX_OCR_PAGES
        for page_no, page in enumerate(doc, start=1):
            native_items = _page_items(page)
            text = _rows_text(native_items, page.rect.width).strip()
            if ocr_budget > 0 and _needs_ocr(page, len(text)):
                ocr_budget -= 1
                extra = _ocr_only(native_items, _ocr_page(page))
                if extra:
                    # Keep what the text layer has, then add what only the picture shows.
                    self._ocr_lines[(filename, page_no)] = extra
                    ocr_text = _rows_text([(*r, t) for r, t in extra], page.rect.width)
                    text = f"{text}\n{ocr_text}".strip()
                    self.ocr_pages += 1
            if not text.strip():
                continue
            page_texts[(filename, page_no)] = text
            for pos, chunk in enumerate(self.splitter.split_text(text)):
                ch = Chunk(uuid.uuid4().hex, filename, page_no, pos, chunk)
                ch.tf = Counter(_terms(chunk))
                ch.length = sum(ch.tf.values())
                new.append(ch)
        if not new:
            return 0
        vectors = [v.tolist() for v in self.embedder.embed([c.text for c in new])]
        self.collection.add(
            ids=[c.id for c in new],
            documents=[c.text for c in new],
            metadatas=[{"file": c.file, "page": c.page} for c in new],
            embeddings=vectors,
        )
        self.chunks += new
        self._by_id.update({c.id: c for c in new})
        self._page_text.update(page_texts)
        self._pdfs[filename] = data
        self.sizes[filename] = sizes
        if filename not in self.files:
            self.files.append(filename)
        self.n_chunks += len(new)
        self.file_chunks[filename] = self.file_chunks.get(filename, 0) + len(new)
        return len(new)

    # ----- retrieval -----
    def _bm25(self, query: str, pool: list[Chunk], limit: int) -> list[Chunk]:
        q = set(_terms(query))
        if not q or not pool:
            return []
        n = len(pool)
        avg = sum(c.length for c in pool) / n or 1.0
        df = Counter(t for c in pool for t in c.tf.keys() if t in q)
        scored = []
        for c in pool:
            s = 0.0
            for t in q:
                f = c.tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * c.length / avg))
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:limit]]

    def _fused(self, query: str, files: list[str] | None, limit: int) -> list[Chunk]:
        """Vectors and BM25 combined with reciprocal-rank fusion, best first."""
        pool = [c for c in self.chunks if not files or c.file in files]
        if not pool:
            return []
        wide = min(max(limit * 2, 20), len(pool))
        qvec = next(iter(self.embedder.query_embed(query))).tolist()
        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=wide,
            where={"file": {"$in": list(files)}} if files else None,
        )
        dense = [self._by_id[i] for i in res["ids"][0] if i in self._by_id]
        lexical = self._bm25(query, pool, wide)

        fused: dict[str, float] = {}
        for ranking in (dense, lexical):
            for rank, c in enumerate(ranking):
                fused[c.id] = fused.get(c.id, 0.0) + 1.0 / (60 + rank)
        ranked = sorted(fused, key=fused.get, reverse=True)[:limit]
        return [self._by_id[i] for i in ranked]

    def search_chunks(self, query: str, files: list[str] | None = None, candidates: int = CANDIDATES) -> list[Chunk]:
        """Hybrid retrieval: vectors + BM25, fused with reciprocal-rank fusion, then reranked."""
        cands = self._fused(query, files, candidates)
        reranker = get_reranker()
        if reranker and len(cands) > 1:
            try:
                scores = list(reranker.rerank(query, [c.text[:900] for c in cands]))
                cands = [c for _, c in sorted(zip(scores, cands), key=lambda p: -p[0])]
            except Exception as e:
                print(f"[rag] rerank failed ({e}); using fused order")
        return cands

    def search(self, query: str, k: int = TOP_K, files: list[str] | None = None) -> list[Source]:
        """Top-k *pages*. Short pages are returned whole so labels keep their neighbours.

        When several documents are searched together, each one is represented by at least its best
        page. Otherwise a longer or more similar document can crowd the others out completely,
        and "all documents" would quietly mean "the dominant document".
        """
        cands = self.search_chunks(query, files)
        scope = list(files) if files else list(self.files)
        must: list[Chunk] = []
        if len(scope) > 1:
            present = {f for f, _ in self._page_order(cands, k)}
            for f in scope:
                if f not in present:
                    best = self._fused(query, [f], 3)
                    if best:
                        must.append(best[0])
        return self.pages_from_chunks(cands, k, must)

    @staticmethod
    def _page_order(cands: list[Chunk], k: int) -> list[tuple[str, int]]:
        order: list[tuple[str, int]] = []
        for c in cands:
            key = (c.file, c.page)
            if key not in order:
                order.append(key)
            if len(order) >= k:
                break
        return order

    def pages_from_chunks(self, cands: list[Chunk], k: int = TOP_K, must: list[Chunk] | tuple = ()) -> list[Source]:
        """Group ranked chunks into pages (see `search`). `must` pages are kept even if they rank low."""
        forced = list(dict.fromkeys((c.file, c.page) for c in must))
        order = self._page_order(cands, max(1, k - len(forced)))
        order += [key for key in forced if key not in order]
        pool = list(cands) + list(must)
        out = []
        for n, (file, page) in enumerate(order, start=1):
            full = self._page_text.get((file, page), "")
            if full and len(full) <= PAGE_FULL_MAX:
                text = full
            else:
                parts = sorted((c for c in pool if (c.file, c.page) == (file, page)), key=lambda c: c.pos)
                text = "\n…\n".join(c.text for c in parts)
            out.append(Source(n=n, file=file, page=page, text=text))
        return out

    # ----- page viewer -----
    def _doc(self, filename: str) -> pymupdf.Document:
        doc = self._docs.get(filename)
        if doc is None:
            doc = self._docs[filename] = pymupdf.open(stream=self._pdfs[filename], filetype="pdf")
        return doc

    def render_plain(self, filename: str, page_no: int, dpi: int = 110) -> bytes:
        """PNG of one PDF page."""
        with self._render_lock:
            doc = self._doc(filename)
            if not 1 <= page_no <= len(doc):
                raise IndexError("page out of range")
            return doc[page_no - 1].get_pixmap(dpi=dpi).tobytes("png")

    def _focus(self, passage: str, query: str) -> str:
        """Narrow a retrieved passage to the 1-2 sentences that best match the question/answer."""
        # Short lines without closing punctuation are headings: keep them out of the neighbouring sentence.
        parts: list[str] = []
        buf: list[str] = []
        for line in (ln.strip() for ln in passage.split("\n")):
            if not line:
                continue
            if len(line) < 45 and not re.search(r"[.!?;:,]$", line):
                if buf:
                    parts.append(" ".join(buf))
                    buf = []
                parts.append(line)
            else:
                buf.append(line)
        if buf:
            parts.append(" ".join(buf))
        sents = [s.strip() for p in parts for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
        if len(sents) <= 1 or not query.strip():
            return passage
        try:
            sv = np.array(list(self.embedder.embed(sents)))
            qv = np.array(next(iter(self.embedder.query_embed(query))))
        except Exception:
            return passage
        sims = sv @ qv / (np.linalg.norm(sv, axis=1) * np.linalg.norm(qv) + 1e-9)
        order = np.argsort(-sims)
        keep = [int(order[0])] + [int(i) for i in order[1:2] if sims[i] >= 0.9 * sims[order[0]]]
        return " ".join(sents[i] for i in sorted(keep))

    def highlight(
        self, filename: str, page_no: int, passage: str, query: str = ""
    ) -> list[tuple[float, float, float, float]]:
        """Boxes (PDF points) for the part of the passage that answers the query, for the viewer to overlay."""
        passage = self._focus(passage, query)
        with self._render_lock:
            doc = self._doc(filename)
            if not 1 <= page_no <= len(doc):
                raise IndexError("page out of range")
            rects = self._highlight_rects(doc[page_no - 1], filename, page_no, passage)
        return [(r.x0, r.y0, r.x1, r.y1) for r in rects]

    def _highlight_rects(self, page, filename: str, page_no: int, passage: str) -> list[pymupdf.Rect]:
        chunk = [t for t in (_norm(w) for w in passage.split()) if t]
        if not chunk:
            return []

        rects: list[pymupdf.Rect] = []
        # Lines read by OCR (scanned pages, or content that only exists as a picture): match whole lines.
        for rect, txt in self._ocr_lines.get((filename, page_no), []):
            line = [t for t in (_norm(w) for w in txt.split()) if t]
            if not line:
                continue
            if len(line) <= 2:
                # a time, a price or a code on its own line: require every word to be in the passage
                if all(t in chunk for t in line) and any(any(c.isdigit() for c in t) or len(t) >= 3 for t in line):
                    rects.append(pymupdf.Rect(*rect))
                continue
            # a line counts if most of its words appear (in order) inside the passage
            blocks = difflib.SequenceMatcher(None, line, chunk, autojunk=False).get_matching_blocks()
            matched = sum(b.size for b in blocks if b.size >= 2)
            if matched >= 3 and matched / len(line) >= 0.6:
                rects.append(pymupdf.Rect(*rect))

        words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
        pos = [i for i, w in enumerate(words) if _norm(w[4])]
        page_tokens = [_norm(words[i][4]) for i in pos]
        matcher = difflib.SequenceMatcher(None, page_tokens, chunk, autojunk=False)
        runs: dict[tuple[int, int], pymupdf.Rect] = {}  # merge words of one line into one box
        for a, _, size in matcher.get_matching_blocks():
            if size < 2 and len(chunk) >= 4:
                continue  # ignore stray single-word matches
            for k in range(a, a + size):
                w = words[pos[k]]
                key = (w[5], w[6])
                r = pymupdf.Rect(w[:4])
                runs[key] = runs[key] | r if key in runs else r
        return rects + list(runs.values())


# ---------- prompting ----------
def build_messages(question: str, sources: list[Source], history: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[{s.n}] (file: {s.file}, page {s.page})\n{s.text}" for s in sources
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Keep the last few turns so follow-up questions ("and the second one?") work.
    messages += history[-4:]
    messages.append(
        {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"}
    )
    return messages


def condense_question(client: Groq, model: str, question: str, history: list[dict]) -> str:
    """Turn a short follow-up ('and the return flight?') into a standalone search query."""
    if not history or len(question.split()) > 12:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    try:
        r = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=80,
            messages=[
                {"role": "system", "content": "Rewrite the user's last question as one standalone question, "
                 "keeping its language and every concrete term from the conversation it refers to. "
                 "Output only the question."},
                {"role": "user", "content": f"{convo}\nuser: {question}"},
            ],
        )
        out = (r.choices[0].message.content or "").strip().strip('"')
        return out if 3 <= len(out) <= 300 else question
    except Exception:
        return question


def stream_answer(client: Groq, model: str, messages: list[dict]) -> Iterator[str]:
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=0, stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------- grounding: split off and verify the evidence ----------
_MARK_RE = re.compile(r"#{2,4}\s*EVIDENCE\s*:?", re.IGNORECASE)
# every proper beginning of the marker, so a half-received marker is not shown to the user
_MARK_PREFIX = re.compile(r"#{1,4}\s*(?:E(?:V(?:I(?:D(?:E(?:N(?:C(?:E)?)?)?)?)?)?)?)?", re.IGNORECASE)


class EvidenceSplitter:
    """Streams the visible answer and sets aside everything after the evidence marker."""

    def __init__(self) -> None:
        self._buf = ""
        self.evidence = ""
        self.in_evidence = False

    def feed(self, token: str) -> str:
        if self.in_evidence:
            self.evidence += token
            return ""
        self._buf += token
        m = _MARK_RE.search(self._buf)
        if m:
            visible, self.evidence = self._buf[: m.start()], self._buf[m.end():]
            self.in_evidence, self._buf = True, ""
            return visible
        # Hold back the tail only if it could still grow into the marker ('#', '##', '###EVI', ...).
        for cut in (i for i, ch in enumerate(self._buf) if ch == "#"):
            if _MARK_PREFIX.fullmatch(self._buf[cut:]):
                visible, self._buf = self._buf[:cut], self._buf[cut:]
                return visible
        visible, self._buf = self._buf, ""
        return visible

    def finish(self) -> str:
        rest, self._buf = ("" if self.in_evidence else self._buf), ""
        return rest


def split_evidence_text(full: str) -> tuple[str, str]:
    """Split a complete (non-streamed) reply into (answer, evidence block)."""
    m = _MARK_RE.search(full)
    if not m:
        return full.strip(), ""
    return full[: m.start()].strip(), full[m.end():]


JUDGE_PROMPT = (
    "You audit answers that were drawn from documents. You get a question, an answer, and the verbatim "
    "evidence quotes behind it. Decide whether the evidence really answers THE QUESTION ASKED. "
    "Look closely at the label inside each quote: a payment date does not answer 'when is the flight', "
    "an arrival time does not answer 'what time does it depart', a net amount does not answer 'gross amount', "
    "the return leg does not answer the outbound leg. "
    "Reply with exactly `OK`, or `MISMATCH: <one short sentence saying what the evidence actually is>`. "
    "Reply in the language of the question."
)


def judge_answer(client: Groq, model: str, question: str, answer: str, items: list[dict]) -> dict:
    """Second opinion on whether the quoted evidence matches what was asked. Fails open (ok=True) on errors."""
    quotes = "\n".join(f"- [{i['n']}] \"{i['quote']}\"" for i in items) or "(none)"
    try:
        r = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=80,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"Question: {question}\nAnswer: {answer}\nEvidence:\n{quotes}"},
            ],
        )
        out = (r.choices[0].message.content or "").strip()
    except Exception:
        return {"ok": True, "reason": "", "checked": False}
    if out.upper().startswith("MISMATCH"):
        return {"ok": False, "reason": out.split(":", 1)[-1].strip() or "The evidence does not match the question.", "checked": True}
    return {"ok": True, "reason": "", "checked": True}


_QUOTE_RE = re.compile(r"\[(\d{1,2})\]\s*[\"“”'«](.+?)[\"“”'»]\s*$", re.MULTILINE)


def _quote_in_passage(quote: str, passage: str) -> bool:
    q, p = _squash(quote), _squash(passage)
    if not q:
        return False
    if q in p:
        return True
    # tolerate small differences (hyphenation, OCR noise): most quote words in order
    qt, pt = q.split(), p.split()
    if len(qt) < 3:
        return False
    blocks = difflib.SequenceMatcher(None, pt, qt, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(qt) >= 0.9


# one maximal number-like token: 07:15, 13.10.2026, 2.450,00, 4821
_VALUE_RE = re.compile(r"\d(?:[\d.,:/-]*\d)?")


def unsupported_values(answer: str, passages: list[str]) -> list[str]:
    """Numbers, times and dates in the answer that appear in none of the cited passages."""
    haystack = " ".join(passages)
    known = {re.sub(r"\D", "", t) for t in _VALUE_RE.findall(haystack)}
    seen, bad = set(), []
    for m in _VALUE_RE.findall(re.sub(r"\[\d{1,2}\]", " ", answer)):
        if m in seen:
            continue
        seen.add(m)
        d = re.sub(r"\D", "", m)
        # a bare 1-2 digit number could be a count or list index; only check values with structure or 3+ digits
        if len(d) < 3 and not re.search(r"[:./,-]", m):
            continue
        if m in haystack or d in known:
            continue
        bad.append(m)
    return bad


def verify_answer(answer: str, evidence_text: str, sources: list[Source]) -> dict:
    """Check the model's quotes against the passages it was given.

    status: 'verified' (all quotes found), 'partial' (some), 'unverified' (none) or 'missing' (no quotes).
    """
    by_n = {s.n: s for s in sources}
    items = []
    for n_str, quote in _QUOTE_RE.findall(evidence_text):
        n = int(n_str)
        src = by_n.get(n)
        ok = bool(src and _quote_in_passage(quote.strip(), src.text))
        items.append({
            "n": n,
            "file": src.file if src else None,
            "page": src.page if src else None,
            "quote": quote.strip(),
            "verified": ok,
        })
    cited = {int(x) for x in re.findall(r"\[(\d{1,2})\]", answer)}
    cited_text = [by_n[n].text for n in cited if n in by_n] or [s.text for s in sources]
    bad = unsupported_values(answer, cited_text)

    if not items:
        status = "missing"
    elif all(i["verified"] for i in items):
        status = "verified"
    elif any(i["verified"] for i in items):
        status = "partial"
    else:
        status = "unverified"
    return {"status": status, "items": items, "unsupported": bad}
