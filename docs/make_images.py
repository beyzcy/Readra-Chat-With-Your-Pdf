"""Builds the project's cover images from the real app.

    python docs/make_images.py

Output (in docs/):
    app-viewer.png      a real screenshot of the app: PDF viewer with an evidence row highlighted
    social-preview.png  1280x640, for GitHub (Settings > Social preview)
    upwork-cover.png    1600x900, for an Upwork portfolio project
    poster.html         the editable source of both covers

Needs the dev environment (playwright with Edge or Chromium) and the app's dependencies.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PORT = 8040
URL = f"http://localhost:{PORT}"

# the real leaf pictures, so the cover matches the website
LEAVES = {
    "maple": ("leaf-01-maple-orange.png", 512, 474), "oak": ("leaf-02-oak-brown.png", 291, 512),
    "beech": ("leaf-03-beech-red.png", 320, 512), "aspen": ("leaf-04-aspen-yellow.png", 461, 512),
    "ginkgo": ("leaf-05-ginkgo-gold.png", 512, 464), "willow": ("leaf-06-willow-darkred.png", 137, 512),
    "chestnut": ("leaf-07-chestnut-orange.png", 201, 512),
}
TREE_PLACE = [(17, 33, -55, "ginkgo"), (31, 23, -25, "beech"), (40, 15, 0, "aspen"), (50, 18, 28, "chestnut"),
              (63, 29, 55, "maple"), (20, 47, -80, "oak"), (60, 44, 80, "willow")]


def tree_svg(cls: str = "") -> str:
    s = (f'<svg viewBox="0 0 80 80" class="{cls}" aria-hidden="true"><g fill="none" stroke="#ece8dc" stroke-width="3.4" '
         'stroke-linecap="round" stroke-linejoin="round"><path d="M40 74V40M28 74h24M40 58L27 44M40 52L54 40M40 42V24'
         'M27 44L20 36M54 40L60 32M40 34L32 26M40 28L48 21"/></g>')
    for x, y, r, key in TREE_PLACE:
        f, w, h = LEAVES[key]
        k = 22 / max(w, h)
        s += (f'<g transform="translate({x} {y}) rotate({r})"><image href="../static/leaves/{f}" '
              f'x="{-w * k / 2:.2f}" y="{-h * k / 2:.2f}" width="{w * k:.2f}" height="{h * k:.2f}"/></g>')
    return s + "</svg>"


def leaf(key: str, style: str) -> str:
    return f'<img class="leaf" src="../static/leaves/{LEAVES[key][0]}" style="{style}" alt="">'


def icon(path: str) -> str:
    return (f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
            f'stroke-linejoin="round">{path}</svg>')


FEATURES = [
    ("Verified answers", "Evidence quotes are checked against the source, and a judge catches mixed-up values",
     icon('<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z"/><path d="M9 12l2 2 4-4"/>')),
    ("Highlighted on the page", "Click a citation and the viewer marks the exact sentence or table row",
     icon('<path d="M4 20h16"/><path d="M6 16l9-9 3 3-9 9H6z"/>')),
    ("Hybrid retrieval and reranking", "Embeddings, BM25 and a multilingual cross-encoder, in any language",
     icon('<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>')),
    ("Tables, scans and OCR", "Row-aware extraction keeps every label with its value",
     icon('<path d="M4 8V5a1 1 0 011-1h3M16 4h3a1 1 0 011 1v3M20 16v3a1 1 0 01-1 1h-3M8 20H5a1 1 0 01-1-1v-3"/><path d="M8 12h8"/>')),
]
PILLS = ["FastAPI", "ChromaDB", "BM25", "BGE reranker", "PyMuPDF", "RapidOCR", "Groq", "Docker"]


def poster_html() -> str:
    feats = "".join(
        f'<li class="feat"><span class="ico">{ic}</span><span><b>{t}</b><i>{d}</i></span></li>'
        for t, d, ic in FEATURES)
    pills = "".join(f"<span>{p}</span>" for p in PILLS)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Readra cover</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{ --bg:#17150f; --ink:#ece8dc; --muted:#b3ab97; --soft:#968c74; --line:#3a3626; --amber:#bf8440; --amber-hi:#d9a35f; }}
  * {{ box-sizing: border-box; margin: 0; }}
  html, body {{ width: 1600px; background: var(--bg); }}
  body {{ font-family: "Inter", "Segoe UI", system-ui, sans-serif; color: var(--ink); -webkit-font-smoothing: antialiased; }}
  .canvas {{ position: relative; width: 1600px; height: var(--h, 900px); overflow: hidden; padding: 56px 72px 48px;
    background: radial-gradient(1200px 700px at 78% 40%, #231e14 0%, transparent 70%),
                linear-gradient(180deg, #1c1710 0%, #17150f 45%, #141a11 100%); display: grid;
    grid-template-columns: 660px 1fr; column-gap: 64px; grid-template-rows: 1fr auto; }}
  .canvas::after {{ content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .05;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>"); }}
  .leaf {{ position: absolute; z-index: 0; opacity: .9; }}
  .left {{ position: relative; z-index: 2; display: flex; flex-direction: column; justify-content: center; }}
  .brand {{ display: flex; align-items: center; gap: 14px; font-size: 40px; font-weight: 650; letter-spacing: -.03em; margin-bottom: 34px; }}
  .brand svg {{ width: 66px; height: 66px; overflow: visible; }}
  h1 {{ font-size: 64px; line-height: 1.04; font-weight: 700; letter-spacing: -.04em; }}
  h1 .hl {{ background: linear-gradient(var(--amber), var(--amber)) no-repeat 0 90% / 100% .26em; }}
  .sub {{ margin-top: 22px; font-size: 25px; line-height: 1.4; color: var(--muted); max-width: 600px; }}
  .feats {{ list-style: none; padding: 0; margin-top: 34px; display: grid; gap: 18px; }}
  .feat {{ display: grid; grid-template-columns: 50px 1fr; column-gap: 18px; align-items: start; }}
  .ico {{ width: 50px; height: 50px; border-radius: 14px; background: rgba(191,132,64,.14); color: var(--amber-hi); display: grid; place-items: center; }}
  .ico svg {{ width: 26px; height: 26px; }}
  .feat b {{ display: block; font-size: 23px; font-weight: 600; letter-spacing: -.015em; }}
  .feat i {{ display: block; font-style: normal; font-size: 18px; line-height: 1.35; color: var(--muted); margin-top: 2px; }}
  .right {{ position: relative; z-index: 2; display: flex; align-items: center; }}
  .frame {{ position: relative; width: 100%; border-radius: 20px; background: #221d19; border: 1px solid var(--line);
    box-shadow: 0 40px 90px rgba(0,0,0,.55); overflow: hidden; }}
  .bar {{ height: 40px; display: flex; align-items: center; gap: 8px; padding: 0 16px; border-bottom: 1px solid var(--line); background: #1d1a13; }}
  .bar i {{ width: 11px; height: 11px; border-radius: 50%; background: #3a3626; }}
  .bar span {{ margin-left: 14px; font-size: 14px; color: var(--soft); }}
  .shot {{ height: var(--shot-h, 470px); overflow: hidden; }}
  .frame img {{ display: block; width: 150%; max-width: none; }}
  .evidence {{ position: absolute; z-index: 3; left: -48px; bottom: var(--ev-bottom, 56px); width: 470px; background: #27231a; border: 1px solid #4a4333;
    border-radius: 16px; padding: 16px 18px 12px; box-shadow: 0 24px 60px rgba(0,0,0,.6); }}
  .evidence .head {{ display: flex; align-items: center; gap: 9px; font-size: 15px; font-weight: 600; color: var(--muted); margin-bottom: 8px; }}
  .evidence .dot {{ width: 10px; height: 10px; border-radius: 50%; background: #6fbf8f; }}
  .evidence .row {{ display: grid; grid-template-columns: 20px 1fr auto; gap: 10px; align-items: baseline; border-top: 1px solid var(--line); padding: 9px 0 4px; font-size: 16px; }}
  .evidence .row b {{ color: #6fbf8f; }} .evidence .row em {{ font-style: normal; color: var(--amber-hi); font-size: 14px; }}
  .strip {{ grid-column: 1 / -1; position: relative; z-index: 2; display: flex; gap: 10px; align-items: center; padding-top: 22px; }}
  .strip small {{ font-size: 15px; color: var(--soft); margin-right: 8px; }}
  .strip span {{ border: 1px solid var(--line); border-radius: 999px; padding: 7px 16px; font-size: 16px; color: var(--muted); background: rgba(255,255,255,.02); }}
  .strip .by {{ margin-left: auto; margin-right: 150px; border: 0; padding: 0; background: none; color: var(--soft); font-size: 16px; }}
  /* the 2:1 social card is 100 px shorter than the 16:9 cover */
  body.social .canvas {{ --h: 800px; padding-top: 44px; padding-bottom: 38px; }}
  body.social .brand {{ margin-bottom: 22px; }} body.social .brand svg {{ width: 58px; height: 58px; }}
  body.social h1 {{ font-size: 60px; }} body.social .sub {{ font-size: 22px; margin-top: 16px; }}
  body.social .feats {{ margin-top: 24px; gap: 13px; }} body.social .feat i {{ display: none; }}
  body.social .ico {{ width: 44px; height: 44px; }} body.social .feat {{ grid-template-columns: 44px 1fr; align-items: center; }}
  body.social {{ --shot-h: 440px; --ev-bottom: 56px; }}
</style></head>
<body class="MODE">
<div class="canvas">
  {leaf("maple", "left:-34px;top:-40px;width:150px;transform:rotate(34deg)")}
  {leaf("ginkgo", "right:420px;top:-34px;width:116px;transform:rotate(-24deg)")}
  {leaf("beech", "right:-34px;top:120px;width:104px;transform:rotate(58deg);opacity:.8")}
  {leaf("oak", "right:-44px;bottom:-64px;width:132px;transform:rotate(-40deg)")}
  {leaf("aspen", "right:150px;bottom:-60px;width:100px;transform:rotate(20deg);opacity:.85")}
  <div class="left">
    <div class="brand">{tree_svg()}Readra</div>
    <h1>Chat with your PDFs.<br><span class="hl">Every answer verified.</span></h1>
    <p class="sub">Answers are checked against the source and highlighted on the exact page.</p>
    <ul class="feats">{feats}</ul>
  </div>
  <div class="right">
    <div class="frame">
      <div class="bar"><i></i><i></i><i></i><span>Readra · ticket_tr.pdf</span></div>
      <div class="shot"><img src="app-viewer.png" alt=""></div>
    </div>
    <div class="evidence">
      <div class="head"><span class="dot"></span>Verified against the document</div>
      <div class="row"><b>✓</b><span>Kalkış | İzmir Adnan Menderes (ADB) - 07:15</span><em>p. 1</em></div>
    </div>
  </div>
  <div class="strip"><small>Built with</small>{pills}<span class="by">Hatice Beyza Ceyhan</span></div>
</div>
</body></html>"""


def main() -> None:
    env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
    env.update({"READRA_RERANKER": "off", "READRA_RATE": "0"})
    venv = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = str(venv) if venv.exists() else sys.executable  # the app's packages live in the project venv
    srv = subprocess.Popen([python, "-m", "uvicorn", "server:app", "--port", str(PORT)],
                           cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(URL + "/healthz")
                break
            except Exception:
                time.sleep(0.5)
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="msedge")
            except Exception:
                browser = p.chromium.launch()

            # 1) a real screenshot: the viewer with an evidence row highlighted
            page = browser.new_page(viewport={"width": 1440, "height": 860})
            # any key value hides the "add your key" notice; no request to Groq is made from this page
            page.add_init_script("sessionStorage.setItem('groq_key', 'gsk_' + 'x'.repeat(52))")
            page.goto(URL + "/app", wait_until="load")
            page.add_style_tag(content="#toast { display: none !important; }")
            page.set_input_files("#file", str(ROOT / "eval" / "ticket_tr.pdf"))
            try:
                page.wait_for_selector(".pg img.ready", state="attached", timeout=120000)
            except Exception:
                shown = page.evaluate("document.getElementById('toast').textContent")
                raise SystemExit(f"The PDF page never appeared. The app said: {shown!r}")
            page.wait_for_timeout(500)
            page.evaluate("() => jumpTo('ticket_tr.pdf', 1, 'Kalkış  |  İzmir Adnan Menderes (ADB) - 07:15', '')")
            page.wait_for_selector(".hl", timeout=20000)
            page.wait_for_timeout(1600)
            page.screenshot(path=str(DOCS / "app-viewer.png"))
            page.close()

            # 2) the covers
            for mode, size, scale, name in (("social", (1600, 800), 0.8, "social-preview.png"),
                                            ("cover", (1600, 900), 1.0, "upwork-cover.png")):
                (DOCS / "poster.html").write_text(poster_html().replace("MODE", mode), encoding="utf-8")
                pg = browser.new_page(viewport={"width": size[0], "height": size[1]}, device_scale_factor=scale)
                pg.goto((DOCS / "poster.html").as_uri(), wait_until="networkidle")
                pg.wait_for_timeout(600)
                pg.screenshot(path=str(DOCS / name))
                pg.close()
            (DOCS / "poster.html").write_text(poster_html().replace("MODE", "cover"), encoding="utf-8")
            browser.close()
    finally:
        srv.terminate()
    for n in ("app-viewer.png", "social-preview.png", "upwork-cover.png"):
        print(n, round((DOCS / n).stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
