"""Readra API: FastAPI backend serving the RAG pipeline and the web UI.

Run locally:  uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastembed import TextEmbedding
from groq import APIConnectionError, AuthenticationError, Groq, RateLimitError
from pydantic import BaseModel

from rag import (
    EMBED_MODEL,
    EVIDENCE_MARK,
    EvidenceSplitter,
    PDFIndex,
    build_messages,
    condense_question,
    get_reranker,
    judge_answer,
    split_evidence_text,
    stream_answer,
    verify_answer,
)

log = logging.getLogger("readra")

BASE_DIR = Path(__file__).parent
SAMPLES = {
    "sample_handbook.pdf": "Try the sample handbook",
    "sample_scanned.pdf": "Try a scanned PDF (OCR)",
}

MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B — best quality",
    "llama-3.1-8b-instant": "Llama 3.1 8B — fastest",
}
MAX_FILES = 5
MAX_FILE_MB = 20
MAX_WORKSPACE_MB = 40      # all documents of one session together
MAX_SESSIONS = int(os.getenv("READRA_MAX_SESSIONS", "150"))
SESSION_TTL_S = 60 * 60    # drop idle sessions after an hour
RATE = float(os.getenv("READRA_RATE", "1"))  # multiplier for every rate limit below; 0 switches them off

_embedder: TextEmbedding | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> TextEmbedding:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = TextEmbedding(model_name=EMBED_MODEL)
        return _embedder


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the models in the background so the first visitor does not wait for downloads."""
    def work() -> None:
        get_embedder()
        get_reranker()
    threading.Thread(target=work, daemon=True).start()
    yield


app = FastAPI(title="Readra", version="1.1.0", lifespan=lifespan)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    # Hugging Face shows a Space inside an iframe on huggingface.co, so those origins must be allowed to embed us.
    response.headers.setdefault(
        "Content-Security-Policy",
        "frame-ancestors 'self' https://huggingface.co https://*.huggingface.co https://*.hf.space",
    )
    return response


# ---------- abuse protection: a public demo spends the owner's API quota ----------
def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


class RateLimiter:
    """Sliding-window limiter, in memory (one process)."""

    def __init__(self) -> None:
        self._hits: dict[tuple, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: tuple, limit: int, window: float) -> None:
        if RATE <= 0:
            return
        limit = max(1, int(limit * RATE))
        now = time.time()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                wait = int(window - (now - q[0])) + 1
                raise HTTPException(429, f"Too many requests. Please try again in about {wait} s.")
            q.append(now)
            if len(self._hits) > 5000:  # forget idle clients
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > window]:
                    self._hits.pop(k, None)


limiter = RateLimiter()


def limited(bucket: str, limit: int, window: float):
    def dep(request: Request) -> None:
        limiter.check((bucket, client_ip(request)), limit, window)
    return Depends(dep)


# ---------- sessions ----------
class Session:
    def __init__(self) -> None:
        self.index: PDFIndex | None = None
        self.history: list[dict] = []
        self.scope_key: tuple | None = None  # which documents the running conversation is about
        self.touched = time.time()

    def set_index(self, index: PDFIndex | None) -> None:
        if self.index is not None and self.index is not index:
            self.index.close()
        self.index = index


_sessions: dict[str, Session] = {}


def _drop(sid: str) -> None:
    s = _sessions.pop(sid, None)
    if s:
        s.set_index(None)


def get_session(sid: str | None) -> Session:
    now = time.time()
    for key in [k for k, s in _sessions.items() if now - s.touched > SESSION_TTL_S]:
        _drop(key)
    if not sid or sid not in _sessions:
        raise HTTPException(404, "Session expired. Please reload the page.")
    s = _sessions[sid]
    s.touched = now
    return s


def state(s: Session) -> dict:
    idx = s.index
    return {
        "files": idx.files if idx else [],
        "docs": [{"name": f, "sizes": idx.sizes.get(f, [])} for f in idx.files] if idx else [],
        "chunks": idx.n_chunks if idx else 0,
        "ocr_pages": idx.ocr_pages if idx else 0,
    }


def server_key() -> str | None:
    return os.getenv("GROQ_API_KEY")


def clean_key(raw: str | None) -> str | None:
    """Copy-paste leaves invisible debris on keys: spaces, line breaks, zero-width characters, quotes, 'Bearer '."""
    if not raw:
        return None
    k = re.sub(r"[\s\u200b-\u200f\u2060\ufeff\"'\u201c\u201d\u2018\u2019`]", "", raw)
    if k.lower().startswith("bearer"):
        k = k[6:]
    return k or None


def key_problem(k: str) -> str | None:
    """A reason to refuse a key before it is sent anywhere, or None."""
    if not k.isascii():
        return "The key contains characters that are not allowed (it should only have letters, digits and _). Copy it again from console.groq.com/keys."
    if len(k) < 20:
        return f"The key looks cut off ({len(k)} characters). Copy the whole key from console.groq.com/keys."
    return None


def error_status(e: Exception) -> int:
    if isinstance(e, AuthenticationError):
        return 401
    if isinstance(e, RateLimitError):
        return 429
    if isinstance(e, APIConnectionError):
        return 502
    return 500


def friendly_error(e: Exception) -> str:
    """What the visitor sees when the language model call fails. Details go to the log, not the UI."""
    if isinstance(e, RateLimitError):
        return "The language-model provider's rate limit was reached. Wait a minute and try again, or pick the faster model."
    if isinstance(e, AuthenticationError):
        return "Groq rejected this API key as invalid. Check that it is the complete secret key (starts with gsk_) and that it was not deleted or replaced on console.groq.com."
    if isinstance(e, APIConnectionError):
        return "Could not reach the language-model provider. Check the connection and try again."
    log.warning("LLM call failed: %s", type(e).__name__)
    return "The language model returned an error. Please try again."


# ---------- API ----------
@app.get("/healthz")
def healthz() -> dict:
    """For load balancers and uptime checks."""
    return {"ok": True, "sessions": len(_sessions), "embedder_ready": _embedder is not None}


@app.get("/api/config")
def config() -> dict:
    return {
        "server_key": bool(server_key()),
        "models": [{"id": k, "label": v} for k, v in MODELS.items()],
        "samples": [{"name": n, "label": l} for n, l in SAMPLES.items() if (BASE_DIR / n).exists()],
        "max_files": MAX_FILES,
        "max_file_mb": MAX_FILE_MB,
    }


@app.get("/api/state")
def current_state(x_session: str | None = Header(None)) -> dict:
    """What this session currently holds, so the app can be reopened after visiting the home page."""
    return state(get_session(x_session))


@app.post("/api/session", dependencies=[limited("session", 40, 600)])
def new_session() -> dict:
    # Keep memory bounded: when the table is full the least recently used session goes.
    while len(_sessions) >= MAX_SESSIONS:
        _drop(min(_sessions, key=lambda k: _sessions[k].touched))
    sid = uuid.uuid4().hex
    _sessions[sid] = Session()
    return {"session": sid}


@app.delete("/api/session")
def delete_session(x_session: str | None = Header(None)) -> dict:
    """'Remove all documents': free the memory right away instead of waiting for the timeout."""
    if x_session:
        _drop(x_session)
    return {"ok": True}


@app.post("/api/upload", dependencies=[limited("upload", 12, 600)])
def upload(files: list[UploadFile] = File(...), x_session: str | None = Header(None)) -> dict:
    s = get_session(x_session)
    index = s.index or PDFIndex(get_embedder())  # uploads add to the current workspace
    if len(index.files) + len(files) > MAX_FILES:
        raise HTTPException(400, f"A workspace holds at most {MAX_FILES} documents.")
    chunks_before = index.n_chunks
    skipped: list[str] = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            skipped.append(f"{f.filename} (not a PDF)")
            continue
        if f.filename in index.files:
            skipped.append(f"{f.filename} (already added)")
            continue
        data = f.file.read(MAX_FILE_MB * 1024 * 1024 + 1)
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            skipped.append(f"{f.filename} (over {MAX_FILE_MB} MB)")
            continue
        if b"%PDF" not in data[:1024]:
            skipped.append(f"{f.filename} (not a valid PDF)")
            continue
        if index.total_bytes() + len(data) > MAX_WORKSPACE_MB * 1024 * 1024:
            skipped.append(f"{f.filename} (workspace limit of {MAX_WORKSPACE_MB} MB reached)")
            continue
        try:
            n = index.add_pdf(data, f.filename)
        except Exception:
            n = 0
        if n == 0:
            skipped.append(f"{f.filename} (no readable text)")
    if index.n_chunks == chunks_before:
        if s.index is None:
            index.close()
        raise HTTPException(422, "Nothing was added: " + "; ".join(skipped))
    s.index = index
    return {**state(s), "skipped": skipped}


@app.delete("/api/document")
def remove_document(file: str, x_session: str | None = Header(None)) -> dict:
    """Close one document; the rest of the workspace stays."""
    s = _doc_session(x_session, file)
    s.index.remove_file(file)
    if not s.index.files:
        s.set_index(None)
    s.history = []   # earlier answers may have quoted the document that is now gone
    s.scope_key = None
    return state(s)


@app.post("/api/sample", dependencies=[limited("upload", 12, 600)])
def load_sample(name: str = "sample_handbook.pdf", x_session: str | None = Header(None)) -> dict:
    s = get_session(x_session)
    if name not in SAMPLES or not (BASE_DIR / name).exists():
        raise HTTPException(404, "Sample document is not available.")
    index = PDFIndex(get_embedder())
    index.add_pdf((BASE_DIR / name).read_bytes(), name)
    s.set_index(index)
    s.history = []
    return {**state(s), "skipped": []}


def _doc_session(sid: str | None, file: str) -> Session:
    s = get_session(sid)
    if not s.index or file not in s.index.files:
        raise HTTPException(404, "Document not found in this session.")
    return s


@app.get("/api/render", dependencies=[limited("render", 900, 60)])
def render_page(file: str, page: int, session: str, dpi: int = 110) -> Response:
    """One page as PNG. The session id is a query param so a plain <img src> can load it."""
    s = _doc_session(session, file)
    try:
        png = s.index.render_plain(file, page, dpi=max(40, min(dpi, 220)))
    except IndexError:
        raise HTTPException(404, "Page not found.")
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


class HighlightRequest(BaseModel):
    file: str
    page: int
    text: str
    query: str = ""  # question + the answer sentence citing this source; narrows the highlight


@app.post("/api/highlight", dependencies=[limited("render", 900, 60)])
def highlight(req: HighlightRequest, x_session: str | None = Header(None)) -> dict:
    """Where the retrieved passage sits on the page (PDF points)."""
    s = _doc_session(x_session, req.file)
    try:
        rects = s.index.highlight(req.file, req.page, req.text, req.query)
    except IndexError:
        raise HTTPException(404, "Page not found.")
    return {"rects": rects}


_NON_CHAT = ("whisper", "guard", "tts", "playai", "embed", "orpheus", "safeguard")


@app.get("/api/models", dependencies=[limited("models", 20, 60)])
def list_models(x_groq_key: str | None = Header(None)) -> dict:
    """Ask Groq which chat models this key can use, so model ids never go stale."""
    api_key = server_key() or clean_key(x_groq_key)
    if not api_key:
        return {"models": [{"id": k, "label": v} for k, v in MODELS.items()], "live": False}
    if (problem := key_problem(api_key)) and not server_key():
        raise HTTPException(400, problem)
    try:
        ids = [m.id for m in Groq(api_key=api_key).models.list().data]
    except Exception as e:
        raise HTTPException(error_status(e), friendly_error(e))
    chat = sorted(i for i in ids if not any(t in i.lower() for t in _NON_CHAT))
    # Show the strongest-looking Llama models first.
    chat.sort(key=lambda i: (0 if "llama" in i.lower() else 1, 0 if "70b" in i or "120b" in i else 1, i))
    return {"models": [{"id": i, "label": i} for i in chat], "live": True}


class ChatRequest(BaseModel):
    question: str
    model: str = "llama-3.3-70b-versatile"
    files: list[str] | None = None  # restrict the search to these documents; None = all


def _line(kind: str, data) -> str:
    return json.dumps({"type": kind, "data": data}, ensure_ascii=False) + "\n"


@app.post("/api/chat", dependencies=[limited("chat", 20, 60)])
def chat(req: ChatRequest, x_session: str | None = Header(None), x_groq_key: str | None = Header(None)):
    s = get_session(x_session)
    if not s.index:
        raise HTTPException(400, "Upload a document first.")
    api_key = server_key() or clean_key(x_groq_key)
    if not api_key:
        raise HTTPException(401, "Add a Groq API key in Settings.")
    if (problem := key_problem(api_key)) and not server_key():
        raise HTTPException(400, problem)
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Empty question.")
    if len(question) > 2000:
        raise HTTPException(400, "Please keep the question under 2000 characters.")

    scope = [f for f in (req.files or []) if f in s.index.files] or None
    # A conversation belongs to one set of documents. If the user switches to another document, or adds
    # one, the earlier turns would pull the new question back to the old topic, so they are dropped.
    scope_key = (tuple(sorted(scope)) if scope else ("*",), tuple(s.index.files))
    if s.scope_key != scope_key:
        s.history = []
        s.scope_key = scope_key
    client = Groq(api_key=api_key)
    # A follow-up like "and the return flight?" is rewritten so retrieval sees the full question.
    search_query = condense_question(client, req.model, question, s.history)
    sources = s.index.search(search_query, files=scope)

    def events():
        yield _line("sources", [vars(src) for src in sources])
        splitter, answer = EvidenceSplitter(), ""
        try:
            messages = build_messages(question, sources, s.history)
            for token in stream_answer(client, req.model, messages):
                visible = splitter.feed(token)
                if visible:
                    answer += visible
                    yield _line("token", visible)
            tail = splitter.finish()
            if tail:
                answer += tail
                yield _line("token", tail)
        except Exception as e:  # provider errors (bad key, rate limit) become a readable message
            yield _line("error", friendly_error(e))
            return
        answer = answer.strip()
        # 1) are the quoted words really in the passages? 2) does the quoted label match the question?
        ev = verify_answer(answer, splitter.evidence, sources)
        try:
            if ev["items"]:
                ev["judge"] = judge_answer(client, req.model, question, answer, ev["items"])
                if not ev["judge"]["ok"]:
                    # One self-correction attempt before the user ever sees a mixed-up value.
                    retry = messages + [
                        {"role": "assistant", "content": f"{answer}\n{EVIDENCE_MARK}\n{splitter.evidence}"},
                        {"role": "user", "content": (
                            f"Check: {ev['judge']['reason']} Answer the original question again. Use only a value "
                            "whose printed label matches what was asked, or say that you cannot tell. "
                            "Same format, including the evidence lines.")},
                    ]
                    r = client.chat.completions.create(model=req.model, messages=retry, temperature=0)
                    new_answer, new_evidence = split_evidence_text(r.choices[0].message.content or "")
                    if new_answer:
                        ev2 = verify_answer(new_answer, new_evidence, sources)
                        ev2["judge"] = (
                            judge_answer(client, req.model, question, new_answer, ev2["items"])
                            if ev2["items"] else {"ok": True, "reason": "", "checked": False}
                        )
                        ev2["corrected"] = True
                        answer, ev = new_answer, ev2
                        yield _line("replace", answer)
        except Exception:
            pass  # the first answer and its verification are still valid
        yield _line("evidence", ev)
        s.history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        yield _line("done", None)

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/api/leaf-images")
def leaf_images() -> dict:
    """Pictures the landing page uses for its falling leaves: whatever sits in static/leaves/."""
    folder = BASE_DIR / "static" / "leaves"
    names = sorted(p.name for p in folder.glob("*") if p.suffix.lower() in {".png", ".webp", ".svg"}) if folder.is_dir() else []
    return {"images": [f"/static/leaves/{n}" for n in names]}


@app.post("/api/reset")
def reset(x_session: str | None = Header(None)) -> dict:
    s = get_session(x_session)
    s.history = []
    return {"ok": True}


# ---------- Web UI ----------
@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request) -> HTMLResponse:
    # Social-preview tags need absolute URLs, so the page learns its own address here.
    html = (BASE_DIR / "static" / "landing.html").read_text(encoding="utf-8")
    base = str(request.base_url).rstrip("/")
    return HTMLResponse(html.replace("{{BASE}}", base))


@app.get("/app")
def app_page() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
