/* Readra — autumn leaves.
 *
 * 1. One leaf lets go of the hero tree after the page has loaded and drifts down.
 * 2. Scrolling releases more leaves: the further you scroll, the more fall (no constant loop).
 * 3. A pile of leaves grows above the footer as you approach the end of the page.
 *    Moving the pointer over the pile scatters a few leaves; they settle again.
 *
 * Two canvases: a fixed one for falling leaves (behind the text, never intercepts input)
 * and a normal-flow one for the pile. Everything runs in one requestAnimationFrame loop that
 * sleeps when nothing is moving.
 */
(() => {
  "use strict";

  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const PALETTE = ["#bf8440", "#c8622a", "#a2432a", "#d6a92b", "#8a6a2f", "#7d8a3c"]; // amber, burnt orange, rust, mustard, bronze, olive
  const rand = (a, b) => a + Math.random() * (b - a);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const isSmall = () => innerWidth < 760;
  const DPR = () => Math.min(window.devicePixelRatio || 1, 2);

  /* ---------- sprites: each leaf is drawn once, then stamped ---------- */
  function makeSprite(color, variant) {
    const s = 64, c = document.createElement("canvas");
    c.width = c.height = s;
    const g = c.getContext("2d");
    g.translate(s / 2, s / 2 - 1);
    g.scale(variant === 0 ? 1 : 0.86, 1);      // a slimmer second variety
    g.lineJoin = "round";
    g.beginPath();
    g.moveTo(0, -30);
    g.lineTo(5, -17);
    g.lineTo(11, -21);
    g.lineTo(10, -9);
    g.lineTo(25, -18);
    g.lineTo(18, -3);
    g.lineTo(28, 3);
    g.lineTo(16, 7);
    g.lineTo(18, 17);
    g.lineTo(6, 11);
    g.lineTo(2, 19);
    g.lineTo(0, 22);
    g.lineTo(-2, 19);
    g.lineTo(-6, 11);
    g.lineTo(-18, 17);
    g.lineTo(-16, 7);
    g.lineTo(-28, 3);
    g.lineTo(-18, -3);
    g.lineTo(-25, -18);
    g.lineTo(-10, -9);
    g.lineTo(-11, -21);
    g.lineTo(-5, -17);
    g.closePath();
    g.fillStyle = color; g.fill();
    g.save(); g.clip(); g.fillStyle = "rgba(0,0,0,.2)"; g.fillRect(0, -34, 34, 70); g.restore();
    // veins and stem
    g.strokeStyle = "rgba(0,0,0,.4)"; g.lineWidth = 1.5; g.lineCap = "round";
    g.beginPath();
    g.moveTo(0, 24); g.lineTo(0, -22);
    g.moveTo(0, 8); g.lineTo(-20, -14); g.moveTo(0, 8); g.lineTo(20, -14);
    g.moveTo(0, 12); g.lineTo(-22, 4); g.moveTo(0, 12); g.lineTo(22, 4);
    g.stroke();
    g.lineWidth = 2.2;
    g.beginPath(); g.moveTo(0, 22); g.lineTo(0, 31); g.stroke();
    return c;
  }
  const SPRITES = PALETTE.map((col) => [makeSprite(col, 0), makeSprite(col, 1)]);
  /* Your own leaf pictures. Drop transparent PNG/WebP/SVG files into static/leaves/ and they replace
   * the drawn leaves everywhere (falling leaves, the pile, the leaf that leaves the tree). */
  let PHOTOS = [];
  const spriteFor = (ci, v) =>
    PHOTOS.length ? PHOTOS[ci % PHOTOS.length] : SPRITES[ci % SPRITES.length][v];

  function photoSprite(img) {
    const max = 160, k = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(img.naturalWidth * k));
    c.height = Math.max(1, Math.round(img.naturalHeight * k));
    const g = c.getContext("2d");
    g.drawImage(img, 0, 0, c.width, c.height);
    // How much of the picture is actually leaf? A slim willow leaf and a round aspen leaf with the
    // same longest side differ a lot in visual weight, which would let the broad leaves dominate.
    let solid = 0;
    const px = g.getImageData(0, 0, c.width, c.height).data;
    for (let i = 3; i < px.length; i += 4) if (px[i] > 24) solid++;
    c.solid = Math.max(1, solid);
    return c;
  }

  /* Give every picture roughly the same visible area (clamped so nothing becomes huge or tiny). */
  function balance(sprites) {
    const areas = sprites.map((s) => s.solid).sort((a, b) => a - b);
    const mid = areas[Math.floor(areas.length / 2)];
    sprites.forEach((s) => { s.weight = Math.min(1.45, Math.max(0.8, Math.sqrt(mid / s.solid))); });
    return sprites;
  }

  function loadPhotos() {
    return fetch("/api/leaf-images")
      .then((r) => r.json())
      .then(({ images }) => Promise.all(images.map((src) => new Promise((done) => {
        const img = new Image();
        img.onload = () => done(img);
        img.onerror = () => done(null);
        img.src = src;
      }))))
      .then((list) => {
        const ok = list.filter((i) => i && i.naturalWidth);
        if (ok.length) { PHOTOS = balance(ok.map(photoSprite)); buildPile(); setPileTarget(reduce ? 1 : progress); wake(); }
      })
      .catch(() => {}); // no folder, no pictures: the drawn leaves stay
  }

  /* `size` is the longest side of the leaf in CSS pixels */
  function stamp(ctx, sprite, x, y, size, rot, squash = 1, alpha = 1) {
    const w = sprite.width, h = sprite.height, k = (size * (sprite.weight || 1)) / Math.max(w, h);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.translate(x, y);
    ctx.rotate(rot);
    ctx.scale(k * squash, k);
    ctx.drawImage(sprite, -w / 2, -h / 2);
    ctx.restore();
  }

  /* ---------- falling leaves (fixed canvas, behind the content) ---------- */
  const sky = document.createElement("canvas");
  sky.id = "leaves";
  sky.setAttribute("aria-hidden", "true");
  Object.assign(sky.style, {
    position: "fixed", inset: "0", width: "100%", height: "100%",
    zIndex: "0", pointerEvents: "none",
  });
  document.body.prepend(sky);
  const skyCtx = sky.getContext("2d");
  let skyW = 0, skyH = 0;

  function sizeSky() {
    const d = DPR();
    skyW = innerWidth; skyH = innerHeight;
    sky.width = Math.round(skyW * d); sky.height = Math.round(skyH * d);
    skyCtx.setTransform(d, 0, 0, d, 0, 0);
  }

  const falling = [];
  const maxFalling = () => (isSmall() ? 7 : 15);
  const spacing = () => (isSmall() ? 420 : 340); // px of scrolling per released leaf: few, so the page stays calm

  function sideBias(x) {
    if (Math.random() > 0.65) return x;                       // some leaves still wander anywhere
    const band = skyW * 0.2, left = Math.random() < 0.5;
    return left ? rand(0, band) : rand(skyW - band, skyW);
  }

  function spawn(x, y, opts = {}) {
    if (falling.length >= maxFalling() && !opts.force) return;
    const small = isSmall();
    falling.push({
      x0: opts.force ? x : sideBias(x), y,
      t: rand(0, 6),
      vy: opts.vy ?? rand(small ? 20 : 26, small ? 44 : 58),
      amp: opts.amp ?? rand(16, 46),
      w: opts.w ?? rand(0.4, 1.0),
      ph: rand(0, 6.28),
      drift: rand(-8, 8),
      rot: rand(0, 6.28),
      spin: rand(-0.3, 0.3),
      flip: rand(0.8, 1.8),
      size: opts.size ?? rand(small ? 18 : 20, small ? 28 : 34) * (PHOTOS.length ? 1.3 : 1),
      ci: opts.ci ?? Math.floor(rand(0, 42)),
      v: Math.random() < 0.55 ? 0 : 1,
      alpha: 1,
      vis: 0.92,   // eased visibility: drops while the leaf is over text
      fade: false,
    });
  }

  /* ---------- the pile ---------- */
  const pile = document.getElementById("pile");
  const pileCtx = pile ? pile.getContext("2d") : null;
  let pileW = 0, pileH = 0;
  let pileLeaves = [];
  let shown = 0;           // how many pile leaves are currently visible
  let pileDirty = true;
  let pileActive = false;  // some leaf is still moving or fading in
  let drawOrder = [], orderFor = -1;  // leaves sorted by height, rebuilt only when the count changes

  function profile(x) {    // mound height, 0..1 of the canvas, for x in 0..1
    return 0.40 + 0.30 * (0.5 + 0.5 * Math.sin(x * 6.0 + 1.0)) + 0.17 * (0.5 + 0.5 * Math.sin(x * 15 + 2.0));
  }

  function buildPile() {
    if (!pile) return;
    const d = DPR();
    const r = pile.getBoundingClientRect();
    pileW = r.width; pileH = r.height;
    pile.width = Math.round(pileW * d); pile.height = Math.round(pileH * d);
    pileCtx.setTransform(d, 0, 0, d, 0, 0);

    // Enough leaves to cover the mound about one and a half times: it should look piled, not scattered.
    const count = PHOTOS.length ? (isSmall() ? 260 : 620) : (isSmall() ? 330 : 820);
    const grow = PHOTOS.length ? 1.35 : 1;
    const prev = shown / Math.max(1, pileLeaves.length);
    // The same seed on every build keeps the pile identical across resizes.
    let seed = 20261015;
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    pileLeaves = Array.from({ length: count }, () => {
      const x = rnd();
      const h = profile(x);
      const rel = rnd() < 0.4 ? 0.6 + 0.4 * rnd() : rnd(); // extra leaves along the surface
      const size = ((isSmall() ? 22 : 30) + rnd() * (isSmall() ? 12 : 18)) * grow;
      const hx = x * pileW;
      const hy = pileH - rel * h * pileH * 0.9 - size * 0.25;
      return {
        x: hx, y: hy, hx, hy,
        vx: 0, vy: 0, spin: 0,
        rot: rnd() * 6.28,
        size, ci: Math.floor(rnd() * 42), v: rnd() < 0.5 ? 0 : 1,
        order: rel + rnd() * 0.18,   // low leaves first, so the mound builds upward
        born: -1,
      };
    }).sort((a, b) => a.order - b.order);
    shown = Math.round(prev * pileLeaves.length);
    orderFor = -1;
    pileDirty = true;
  }

  function drawPile(now) {
    if (!pileCtx) return;
    pileCtx.clearRect(0, 0, pileW, pileH);
    // The ground under the leaves, so gaps read as shadow rather than empty page. It rises with the pile.
    const rise = shown / Math.max(1, pileLeaves.length);
    const base = pileCtx.createLinearGradient(0, pileH * (1 - 0.8 * rise), 0, pileH);
    base.addColorStop(0, "rgba(58,44,24,0)");
    base.addColorStop(1, "rgba(58,44,24,.9)");
    pileCtx.fillStyle = base;
    pileCtx.beginPath();
    pileCtx.moveTo(0, pileH);
    for (let i = 0; i <= 48; i++) {
      const x = i / 48;
      pileCtx.lineTo(x * pileW, pileH - profile(x) * pileH * 0.82 * rise);
    }
    pileCtx.lineTo(pileW, pileH);
    pileCtx.closePath();
    pileCtx.fill();
    // draw low leaves first so higher ones lie on top
    if (orderFor !== shown) {
      drawOrder = pileLeaves.slice(0, shown).sort((a, b) => b.hy - a.hy);
      orderFor = shown;
    }
    const list = drawOrder;
    let animating = false;
    for (const l of list) {
      let a = 1, dy = 0;
      if (l.born >= 0) {
        const k = clamp((now - l.born) / 650, 0, 1);
        const e = 1 - Math.pow(1 - k, 3);
        a = e; dy = -(1 - e) * 46;
        if (k < 1) animating = true; else l.born = -1;
      }
      stamp(pileCtx, spriteFor(l.ci, l.v), l.x, l.y + dy, l.size, l.rot, 1, a);
    }
    pileActive = animating;
    for (let i = 0; i < shown && !pileActive; i++) if (pileLeaves[i].vx || pileLeaves[i].vy) pileActive = true;
    pileDirty = false;
  }

  function updatePile(dt) {
    let moving = false;
    for (let i = 0; i < shown; i++) {
      const l = pileLeaves[i];
      if (!l.vx && !l.vy) continue;
      l.vy += 900 * dt;
      l.x = clamp(l.x + l.vx * dt, 0, pileW);
      l.y += l.vy * dt;
      l.rot += l.spin * dt;
      if (l.y >= l.hy && l.vy > 0) { l.y = l.hy; l.vx = l.vy = l.spin = 0; }
      else moving = true;
    }
    if (moving) pileDirty = true;
  }

  function setPileTarget(progress) {
    if (!pileLeaves.length) return;
    // nothing before ~20% of the page, full pile at the very end
    const p = clamp((progress - 0.2) / 0.8, 0, 1);
    const target = Math.round(pileLeaves.length * (p * p * (3 - 2 * p)));
    if (target > shown) {
      const now = performance.now();
      for (let i = shown; i < target; i++) pileLeaves[i].born = reduce ? -1 : now + (i - shown) * 14;
      shown = target; pileDirty = true;
    } else if (target < shown) { // scrolled back up
      shown = target; pileDirty = true;
    }
  }

  /* ---------- scroll ---------- */
  let lastY = scrollY, accum = 0, progress = 0;
  function readProgress() {
    const max = document.documentElement.scrollHeight - innerHeight;
    progress = max > 0 ? clamp(scrollY / max, 0, 1) : 0;
  }
  function onScroll() {
    const y = scrollY, dy = y - lastY;
    lastY = y;
    readProgress();
    setPileTarget(progress);
    if (!reduce && dy > 0) {
      accum += dy;
      let n = 0;
      while (accum >= spacing() && n < 3) {
        accum -= spacing(); n++;
        spawn(rand(0, skyW), rand(-40, -10));
      }
      if (accum > spacing() * 4) accum = 0;
    }
    wake();
  }

  /* ---------- pointer scatter ---------- */
  let lastScatter = 0;
  function onMove(e) {
    if (!pile || reduce || !shown) return;
    const now = performance.now();
    if (now - lastScatter < 60) return;
    const r = pile.getBoundingClientRect();
    if (e.clientY < r.top - 10 || e.clientY > r.bottom || e.clientX < r.left || e.clientX > r.right) return;
    const px = e.clientX - r.left, py = e.clientY - r.top;
    let hit = 0;
    for (let i = 0; i < shown && hit < 6; i++) {
      const l = pileLeaves[i];
      if (l.vx || l.vy) continue;
      const dx = l.x - px, dy = l.y - py;
      if (dx * dx + dy * dy < 56 * 56) {
        l.vy = -rand(140, 300);
        l.vx = (dx >= 0 ? 1 : -1) * rand(30, 150);
        l.spin = rand(-5, 5);
        hit++;
      }
    }
    if (hit) { lastScatter = now; pileActive = true; wake(); }
  }

  /* ---------- keep leaves off the text ---------- */
  const TEXT_SEL = "h1,h2,h3,h4,p,.mono,.l,.foot-note,.btn,.brand,footer span,.num .v,.chip,.chat .u,.chat .a";
  let textRects = [];
  let rectFrame = 0;
  const range = document.createRange();

  function refreshTextRects() {
    const out = [];
    for (const el of document.querySelectorAll(TEXT_SEL)) {
      const box = el.getBoundingClientRect();
      if (box.bottom < -30 || box.top > skyH + 30 || box.width < 4) continue;
      range.selectNodeContents(el);                    // the real line boxes, not the whole block
      for (const r of range.getClientRects()) {
        if (r.width > 3 && r.height > 3 && r.bottom > -30 && r.top < skyH + 30) out.push(r);
      }
    }
    textRects = out;
  }

  function overText(x, y, radius) {
    const pad = radius + 6;
    for (const r of textRects) {
      if (x > r.left - pad && x < r.right + pad && y > r.top - pad && y < r.bottom + pad) return true;
    }
    return false;
  }

  /* ---------- loop ---------- */
  let raf = 0, last = 0;
  function wake() { if (!raf) { last = performance.now(); raf = requestAnimationFrame(tick); } }

  function tick(now) {
    raf = 0;
    const dt = clamp((now - last) / 1000, 0, 0.05);
    last = now;

    // surface of the pile in viewport coordinates, if it is on screen
    let surface = Infinity;
    if (pile) {
      const r = pile.getBoundingClientRect();
      if (r.top < skyH) surface = r.top + r.height * 0.42;
    }

    skyCtx.clearRect(0, 0, skyW, skyH);
    if (falling.length && (rectFrame++ & 1) === 0) refreshTextRects(); // every other frame is plenty
    for (let i = falling.length - 1; i >= 0; i--) {
      const l = falling[i];
      l.t += dt;
      l.y += l.vy * dt;
      l.rot += l.spin * dt;
      const x = l.x0 + l.drift * l.t + l.amp * Math.sin(l.w * l.t + l.ph);
      const tilt = l.rot + 0.55 * Math.sin(l.w * l.t + l.ph + 1.2); // rocks while swaying
      const squash = 0.3 + 0.7 * Math.abs(Math.cos(l.flip * l.t + l.ph));
      if (l.y > surface) l.fade = true;
      if (l.fade) l.alpha -= dt * 2.2;
      if (l.alpha <= 0 || l.y > skyH + 40) { falling.splice(i, 1); continue; }
      const goal = overText(x, l.y, l.size * 0.5) ? 0.02 : 0.9;
      l.vis += (goal - l.vis) * Math.min(1, dt * 7);
      stamp(skyCtx, spriteFor(l.ci, l.v), x, l.y, l.size, tilt, squash, l.alpha * l.vis);
    }

    updatePile(dt);
    if (pileDirty || pileActive) drawPile(now);

    if (falling.length || pileActive || pileDirty) raf = requestAnimationFrame(tick);
  }

  /* ---------- the leaf that leaves the hero tree ---------- */
  function releaseHeroLeaf() {
    const el = document.getElementById("heroLeaf");
    if (!el || reduce) return;
    const r = el.getBoundingClientRect();
    if (r.bottom < 0 || r.top > innerHeight) return; // scrolled away: leave it on the tree
    const color = el.getAttribute("data-color") || PALETTE[0];
    const ci = PHOTOS.length ? 0 : Math.max(0, PALETTE.indexOf(color)); // 0 = first picture (the maple)
    el.classList.add("gone");
    spawn(r.left + r.width / 2, r.top + r.height / 2, {
      force: true, ci, size: PHOTOS.length ? 34 : Math.max(16, Math.max(r.width, r.height) * 1.5),
      vy: 15, amp: 40, w: 0.55,
    });
    wake();
  }

  /* ---------- start ---------- */
  function init() {
    sizeSky();
    buildPile();
    readProgress();
    setPileTarget(reduce ? 1 : progress);
    if (reduce) {
      // Static: the finished pile, no motion.
      pileLeaves.forEach((l) => (l.born = -1));
      drawPile(performance.now());
    }
    addEventListener("scroll", onScroll, { passive: true });
    addEventListener("pointermove", onMove, { passive: true });
    addEventListener("resize", () => {
      sizeSky(); buildPile(); readProgress();
      setPileTarget(reduce ? 1 : progress);
      if (reduce) drawPile(performance.now()); else wake();
    });
    document.addEventListener("visibilitychange", () => { if (!document.hidden) { last = performance.now(); wake(); } });
    loadPhotos();
    if (!reduce) setTimeout(releaseHeroLeaf, 2600); // after the tree has finished drawing
    wake();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
