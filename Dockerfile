# ============================================================
# FILE: Dockerfile
# ============================================================
# Single-worker by construction. See the CMD note at the bottom —
# --workers 1 is a correctness requirement of this application, not
# a resource-saving default. Every store in this system is in-memory
# and single-process; a second worker gets its own budget ledger and
# its own audit chain, sees none of the first worker's writes, and the
# spend ceilings silently stop binding globally. Scale with a bigger
# machine, never with more workers, until the stores are backed by
# something with server-side atomic check-and-commit.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, as their own layer — application code changes far
# more often than requirements.txt, so this keeps rebuilds off the network.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Drop root. Nothing in this app writes to disk (every store is in
# memory), so the image needs no writable paths beyond what Python
# itself uses, and PYTHONDONTWRITEBYTECODE keeps it from trying.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ.get('PORT','8000')+'/api/health').status==200 else 1)"

# Shell form (not exec form) so ${PORT} expands — Render, Railway, Fly and
# Cloud Run all inject the port to bind as an env var rather than letting
# the process choose. `exec` replaces the shell so uvicorn becomes PID 1
# and receives SIGTERM directly, giving it a clean shutdown on redeploy
# instead of being killed after the platform's grace period.
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
