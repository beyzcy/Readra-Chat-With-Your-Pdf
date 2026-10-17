"""Tests for the server's safety nets: rate limits, memory limits, cleanup and input checks.
No API key and no network needed.

Run from the project root:  python eval/test_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

passed = failed = 0


def check(name: str, cond: bool) -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")


def collections() -> int:
    return len(chromadb.EphemeralClient().list_collections())


c = TestClient(server.app)
PDF = (ROOT / "eval" / "handbook_eval.pdf").read_bytes()


def new_session() -> dict:
    sid = c.post("/api/session").json()["session"]
    return {"X-Session": sid}


print("health and pages")
check("/healthz answers", c.get("/healthz").json()["ok"] is True)
page = c.get("/", headers={"host": "readra.example"}).text
check("landing page learns its own address for link previews", "http://readra.example/static/og-image.png" in page and "{{BASE}}" not in page)
check("security headers are set", c.get("/healthz").headers.get("x-content-type-options") == "nosniff")
csp = c.get("/healthz").headers.get("content-security-policy", "")
check("the app may be embedded by Hugging Face, and by no other site", "https://huggingface.co" in csp and "https://*.hf.space" in csp and "*;" not in csp)

print("input checks")
H = new_session()
bad = c.post("/api/upload", files=[("files", ("fake.pdf", b"this is not a pdf", "application/pdf"))], headers=H)
check("a file named .pdf that is not a PDF is refused", bad.status_code == 422 and "not a valid PDF" in bad.json()["detail"])
big = c.post("/api/upload", files=[("files", ("big.pdf", b"%PDF-" + b"0" * (server.MAX_FILE_MB * 1024 * 1024 + 10), "application/pdf"))], headers=H)
check("an oversized file is refused", big.status_code == 422 and "over" in big.json()["detail"])
check("an over-long question is refused", c.post("/api/chat", json={"question": "x" * 2500}, headers={**H, "X-Groq-Key": "k"}).status_code in (400,))

print("memory is freed")
base = collections()
H1 = new_session()
c.post("/api/upload", files=[("files", ("a.pdf", PDF, "application/pdf"))], headers=H1)
check("an index adds a collection", collections() == base + 1)
c.post("/api/sample", params={"name": "sample_handbook.pdf"}, headers=H1)
check("replacing the index frees the old one", collections() == base + 1)
c.delete("/api/session", headers=H1)
check("deleting the session frees its vectors", collections() == base)
check("a deleted session is gone", c.get("/api/state", headers=H1).status_code == 404)

print("closing one document")
HD = new_session()
up = lambda n: c.post("/api/upload", files=[("files", (n, PDF, "application/pdf"))], headers=HD)
up("one.pdf"); up("two.pdf")
idx = server._sessions[HD["X-Session"]].index
total = idx.n_chunks
left = c.delete("/api/document", params={"file": "one.pdf"}, headers=HD).json()
check("closing a document keeps the other", [d["name"] for d in left["docs"]] == ["two.pdf"])
check("its passages are gone from the index", idx.n_chunks == total // 2 and all(ch.file == "two.pdf" for ch in idx.chunks) and idx.collection.count() == idx.n_chunks)
check("its pages can no longer be rendered", c.get("/api/render", params={"file": "one.pdf", "page": 1, "session": HD["X-Session"]}).status_code == 404)
check("the remaining document still renders", c.get("/api/render", params={"file": "two.pdf", "page": 1, "session": HD["X-Session"]}).status_code == 200)
check("a file can be added again after closing it", len(up("one.pdf").json()["docs"]) == 2)
c.delete("/api/document", params={"file": "one.pdf"}, headers=HD)
last = c.delete("/api/document", params={"file": "two.pdf"}, headers=HD).json()
check("closing the last document empties the workspace", last["docs"] == [] and server._sessions[HD["X-Session"]].index is None)
check("closing an unknown document is a 404", c.delete("/api/document", params={"file": "nope.pdf"}, headers=HD).status_code == 404)
c.delete("/api/session", headers=HD)

print("session cap")
server.MAX_SESSIONS = 5
for _ in range(12):
    c.post("/api/session")
check("the session table never exceeds its cap", len(server._sessions) <= 5)
server.MAX_SESSIONS = 150

print("rate limits")
server._sessions.clear()
server.limiter._hits.clear()
codes = [c.post("/api/session").status_code for _ in range(45)]
check("session creation is limited to ~40 / 10 min per client", codes.count(429) >= 4 and codes[0] == 200)
msg = c.post("/api/session").json()["detail"]
check("the limit message says when to retry", "try again in about" in msg)
other = c.post("/api/session", headers={"x-forwarded-for": "203.0.113.7"})
check("another client is not affected", other.status_code == 200)
server.RATE = 0
check("READRA_RATE=0 switches limits off", c.post("/api/session").status_code == 200)
server.RATE = 1

print("friendly errors")
import httpx  # noqa: E402
from groq import APIConnectionError, AuthenticationError, RateLimitError  # noqa: E402

req = httpx.Request("POST", "https://api.groq.com")
check("rate-limit error is explained", "rate limit" in server.friendly_error(RateLimitError("x", response=httpx.Response(429, request=req), body=None)))
check("bad key is explained", "rejected this API key" in server.friendly_error(AuthenticationError("x", response=httpx.Response(401, request=req), body=None)))
check("connection error is explained", "Could not reach" in server.friendly_error(APIConnectionError(request=req)))
check("unknown errors do not leak details", "Traceback" not in server.friendly_error(RuntimeError("secret internals")) and "secret" not in server.friendly_error(RuntimeError("secret internals")))

print("API key handling")
check("spaces, line breaks, quotes and 'Bearer ' are stripped", server.clean_key('  "Bearer gsk_abc123\n" ') == "gsk_abc123")
check("zero-width characters are stripped", server.clean_key("gsk_ab​c12﻿3") == "gsk_abc123")
check("an empty key is None", server.clean_key("  \n ") is None)
check("a non-ASCII key is refused with a clear reason", "not allowed" in (server.key_problem("gsk_çğ" + "a" * 30) or ""))
check("a truncated key is refused with a clear reason", "cut off" in (server.key_problem("gsk_abc") or ""))
check("a normal-looking key passes the local check", server.key_problem("gsk_" + "a" * 52) is None)
check("auth errors map to 401, rate limits to 429, network to 502",
      (server.error_status(AuthenticationError("x", response=httpx.Response(401, request=req), body=None)),
       server.error_status(RateLimitError("x", response=httpx.Response(429, request=req), body=None)),
       server.error_status(APIConnectionError(request=req))) == (401, 429, 502))

seen: dict = {}


class FakeGroq:
    """Stands in for the Groq client and records the key it was given."""

    def __init__(self, api_key=None):
        seen["key"] = api_key
        item = type("Model", (), {"id": "llama-3.3-70b-versatile"})()
        listing = type("Listing", (), {"data": [item]})()
        self.models = type("Models", (), {"list": staticmethod(lambda: listing)})()


real_groq, server.Groq = server.Groq, FakeGroq
server.limiter._hits.clear()
r = c.get("/api/models", headers={"X-Groq-Key": '  "gsk_' + "b" * 52 + '"'})
check("a pasted key with debris reaches Groq cleaned", r.status_code == 200 and seen["key"] == "gsk_" + "b" * 52)
r = c.get("/api/models", headers={"X-Groq-Key": "gsk_abc"})
check("a cut-off key is stopped before any network call", r.status_code == 400 and "cut off" in r.json()["detail"])
server.Groq = real_groq

print("several documents")
from types import SimpleNamespace  # noqa: E402

from fastembed import TextEmbedding  # noqa: E402

import rag  # noqa: E402

idx = rag.PDFIndex(TextEmbedding(model_name=rag.EMBED_MODEL))
idx.add_pdf((ROOT / "eval" / "handbook_eval.pdf").read_bytes(), "handbook_eval.pdf")
idx.add_pdf((ROOT / "eval" / "ticket_tr.pdf").read_bytes(), "ticket_tr.pdf")
for q in ("Summarize the key points", "How many days of annual leave do employees get?", "Uçuş saati kaç?"):
    files = {s.file for s in idx.search(q)}
    check(f"'All documents' always includes every document: {q[:34]}…", files == {"handbook_eval.pdf", "ticket_tr.pdf"})
only = idx.search("How many days of annual leave do employees get?", files=["handbook_eval.pdf"])
check("a single-document search stays in that document", {s.file for s in only} == {"handbook_eval.pdf"})
check("the best page of the question still ranks first", idx.search("How many days of annual leave do employees get?")[0].page == 3)
check("at most TOP_K pages are returned", len(idx.search("Summarize the key points")) <= rag.TOP_K)
idx.close()

print("conversation follows the documents")
sizes: list[int] = []


class ChatFake:
    def __init__(self, api_key=None):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @staticmethod
    def _create(**kw):
        if kw.get("stream"):
            sizes.append(len(kw["messages"]))  # system + earlier turns + the new question
            tok = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Answer [1]."))])
            return iter([tok])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])


server.Groq = ChatFake
server.limiter._hits.clear()
HC = new_session()
HC["X-Groq-Key"] = "gsk_" + "t" * 48
for name in ("handbook_eval.pdf", "ticket_tr.pdf"):
    data = (ROOT / "eval" / name).read_bytes()
    c.post("/api/upload", files=[("files", (name, data, "application/pdf"))], headers=HC)


def ask(files):
    body = {"question": "What are the working hours?", "model": "x"}
    if files is not None:
        body["files"] = files
    with c.stream("POST", "/api/chat", json=body, headers=HC) as r:
        for _ in r.iter_lines():
            pass


ask(["handbook_eval.pdf"]); ask(["handbook_eval.pdf"]); ask(["ticket_tr.pdf"]); ask(None); ask(None)
check("same documents keep the conversation; switching documents starts fresh", sizes == [2, 4, 2, 2, 4])
server.Groq = real_groq

print("visitors bring their own key")
import logging  # noqa: E402
import os  # noqa: E402

saved_key = os.environ.pop("GROQ_API_KEY", None)
server.limiter._hits.clear()
try:
    check("without GROQ_API_KEY the app asks every visitor for a key", c.get("/api/config").json()["server_key"] is False)
    HV = new_session()
    c.post("/api/sample", params={"name": "sample_handbook.pdf"}, headers=HV)
    r = c.post("/api/chat", json={"question": "What are the working hours?", "model": "x"}, headers=HV)
    check("chatting without a key is refused with a clear message", r.status_code == 401 and "API key" in r.json()["detail"])

    SECRET = "gsk_" + "S" * 48
    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture()
    logging.getLogger().addHandler(handler)
    logging.getLogger("readra").setLevel(logging.DEBUG)

    class Boom:
        def __init__(self, api_key=None):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        @staticmethod
        def _create(**kw):
            if kw.get("stream"):
                raise RuntimeError(f"upstream failed for key {SECRET}")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

    server.Groq = Boom
    with c.stream("POST", "/api/chat", json={"question": "What are the working hours?", "model": "x"},
                  headers={**HV, "X-Groq-Key": SECRET}) as r:
        body = "".join(r.iter_lines())
    server.Groq = real_groq
    logging.getLogger().removeHandler(handler)
    check("a failing call never writes the visitor's key to the log", not any(SECRET in m for m in records))
    check("nor sends it back to the browser", SECRET not in body)
finally:
    if saved_key is not None:
        os.environ["GROQ_API_KEY"] = saved_key

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
