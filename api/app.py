"""
api/app.py

FastAPI web server for the Symbaroum GM Assistant.

Run with:
    uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Or via start.sh.
"""

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api import pipelines
from api.models import QueryRequest


# ---------------------------------------------------------------------------
# Lifespan — load indexes once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipelines.initialise()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Symbaroum GM Assistant",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def sse(event: str, data: str) -> str:
    """Format a single SSE event as a wire-format string."""
    # Escape any newlines in data so they don't break the SSE framing
    safe_data = data.replace("\n", "\\n")
    return f"event: {event}\ndata: {safe_data}\n\n"


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

@app.post("/query")
async def query(request: QueryRequest):
    """
    Stream a query response as server-sent events.

    Events emitted:
        routing  — {"pipeline": "hybrid"|"lightrag"}
        token    — answer text chunk (newlines escaped as \\n)
        done     — {"pipeline": ..., "timings": {...}, "total": float}
        error    — {"message": str}
    """
    async def event_stream():
        total_start = time.time()
        try:
            # Route
            pipeline = await pipelines.route(request.query)
            yield sse("routing", json.dumps({"pipeline": pipeline}))

            # Stream answer
            timings = {}
            gen = (
                pipelines.lightrag_pipeline(request.query)
                if pipeline == "lightrag"
                else pipelines.hybrid_pipeline(request.query)
            )

            async for chunk in gen:
                if chunk.startswith("__timings__"):
                    timings = json.loads(chunk[len("__timings__"):])
                else:
                    yield sse("token", chunk)

            # Done
            yield sse("done", json.dumps({
                "pipeline": pipeline,
                "timings": timings,
                "total": time.time() - total_start,
            }))

        except Exception as e:
            yield sse("error", json.dumps({"message": str(e)}))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )