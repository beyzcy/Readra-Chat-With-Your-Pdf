FROM python:3.12-slim

# Small runtime libraries that PyMuPDF, OpenCV and ONNX Runtime can expect on a slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as user 1000, so build as that user too.
RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed \
    PORT=7860
WORKDIR /home/app/readra

# requirements.txt lists what the app needs; constraints.txt pins every package those pull in, so a
# rebuild next month installs exactly what was tested today.
COPY --chown=app:app requirements.txt constraints.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt -c constraints.txt

COPY --chown=app:app . .

# Models are fetched at build time (about 0.9 GB), not on the first visitor's request.
RUN python prefetch_models.py

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','7860'))"

# --proxy-headers: behind Hugging Face / Render / Fly the real client address arrives in X-Forwarded-For.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
