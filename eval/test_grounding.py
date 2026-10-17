"""Tests for the answer-grounding layer. They need no API key: the LLM is replaced by a script.

Run from the project root:  python eval/test_grounding.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag  # noqa: E402
from rag import EvidenceSplitter, Source, unsupported_values, verify_answer  # noqa: E402

PASSAGE = (
    "Rezervasyon tarihi  |  13 Ekim 2026, 14:32\n"
    "Ödeme tarihi  |  13 Ekim 2026, 14:35\n"
    "Kalkış  |  İzmir Adnan Menderes (ADB) - 07:15\n"
    "Varış  |  Berlin Brandenburg (BER) - 10:05"
)
SOURCES = [Source(1, "ticket_tr.pdf", 1, PASSAGE)]

passed = failed = 0


def check(name: str, cond: bool) -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")


# ---------------------------------------------------------------- splitter
print("EvidenceSplitter")
full = 'Uçuş 07:15 [1]\n###EVIDENCE\n[1] "Kalkış | İzmir (ADB) - 07:15"'
for size in (1, 2, 3, 7, 50):
    sp, shown = EvidenceSplitter(), ""
    for i in range(0, len(full), size):
        shown += sp.feed(full[i:i + size])
    shown += sp.finish()
    check(f"marker split across {size}-char tokens", shown.strip() == "Uçuş 07:15 [1]" and "Kalkış" in sp.evidence)

sp, shown = EvidenceSplitter(), ""
for tok in ["Fiyat ", "# 5 ", "TL ", "[1]"]:
    shown += sp.feed(tok)
shown += sp.finish()
check("a lone '#' in the answer is not swallowed", shown == "Fiyat # 5 TL [1]")

# ---------------------------------------------------------------- verification
print("verify_answer / unsupported_values")
good = verify_answer("Uçuş 07:15'te kalkıyor [1].", '[1] "Kalkış  |  İzmir Adnan Menderes (ADB) - 07:15"', SOURCES)
check("verbatim quote is verified", good["status"] == "verified" and not good["unsupported"])

fake = verify_answer("Uçuş 09:00'da [1].", '[1] "Kalkış | İzmir (ADB) - 09:00"', SOURCES)
check("invented quote is not verified", fake["status"] == "unverified")
check("invented time is flagged as unsupported", "09:00" in fake["unsupported"])

part = verify_answer("x [1]", '[1] "Kalkış | İzmir Adnan Menderes (ADB) - 07:15"\n[1] "Koltuk 12A"', SOURCES)
check("one good + one invented quote -> partial", part["status"] == "partial")

none = verify_answer("Uçuş 07:15 [1]", "", SOURCES)
check("no evidence given -> 'missing'", none["status"] == "missing")

check("amount from the passage passes", unsupported_values("14:35 [1]", [PASSAGE]) == [])
check("number absent from the passage is caught", unsupported_values("Toplam 3.100,00 TL", [PASSAGE]) == ["3.100,00"])
check("small counts are not flagged", unsupported_values("Sayfada 2 değer var [1]", [PASSAGE]) == [])

# ---------------------------------------------------------------- end-to-end with a scripted model
print("chat flow with a scripted model (mix-up -> judge -> self-correction)")
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def message(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class ScriptedClient:
    """Plays a model that first confuses payment time with flight time, then corrects itself."""

    def __init__(self, scenario: str):
        self.scenario = scenario
        self.judge_calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        sys_prompt = kw["messages"][0]["content"]
        if kw.get("stream"):
            if self.scenario == "mixup":
                text = 'Uçuş saati 14:35 [1].\n###EVIDENCE\n[1] "Ödeme tarihi  |  13 Ekim 2026, 14:35"'
            else:
                text = 'Uçuş 07:15\'te kalkıyor [1].\n###EVIDENCE\n[1] "Kalkış  |  İzmir Adnan Menderes (ADB) - 07:15"'
            return iter([chunk(text[i:i + 9]) for i in range(0, len(text), 9)])
        if "audit answers" in sys_prompt:
            self.judge_calls += 1
            quote = kw["messages"][1]["content"]
            return message("MISMATCH: Alıntı ödeme tarihidir, uçuş saati değil." if "Ödeme tarihi" in quote else "OK")
        # the self-correction request
        return message('Uçuş 07:15\'te kalkıyor [1].\n###EVIDENCE\n[1] "Kalkış  |  İzmir Adnan Menderes (ADB) - 07:15"')


def run(scenario: str):
    scripted = ScriptedClient(scenario)
    server.Groq = lambda api_key=None: scripted
    c = TestClient(server.app)
    sid = c.post("/api/session").json()["session"]
    H = {"X-Session": sid, "X-Groq-Key": "gsk_" + "t" * 48}
    c.post("/api/sample", params={"name": "sample_handbook.pdf"}, headers=H)
    with open(ROOT / "eval" / "ticket_tr.pdf", "rb") as f:
        c.post("/api/upload", files=[("files", ("ticket_tr.pdf", f, "application/pdf"))], headers=H)
    events = []
    with c.stream("POST", "/api/chat", json={"question": "Uçuş saati kaç?", "model": "x", "files": ["ticket_tr.pdf"]}, headers=H) as r:
        for line in r.iter_lines():
            events.append(json.loads(line))
    return events, scripted


events, scripted = run("mixup")
types = [e["type"] for e in events]
streamed = "".join(e["data"] for e in events if e["type"] == "token")
check("streamed text hides the evidence block", "###" not in streamed and "Ödeme tarihi" not in streamed)
check("a corrected answer replaces the wrong one", "replace" in types and "07:15" in next(e["data"] for e in events if e["type"] == "replace"))
ev = next(e["data"] for e in events if e["type"] == "evidence")
check("final evidence is verified and marked corrected", ev["status"] == "verified" and ev.get("corrected") is True)
check("final judge verdict is OK", ev["judge"]["ok"] is True)
check("judge ran for the first answer and the corrected one", scripted.judge_calls == 2)
check("stream ends with 'done'", types[-1] == "done")

events, scripted = run("correct")
types = [e["type"] for e in events]
ev = next(e["data"] for e in events if e["type"] == "evidence")
check("a correct answer is left alone (no replace)", "replace" not in types)
check("a correct answer is verified", ev["status"] == "verified" and ev["judge"]["ok"] is True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
