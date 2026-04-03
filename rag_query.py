"""
Symbaroum GM Assistant — CLI entry point

Imports pipeline logic from api/pipelines.py, which is shared with the
FastAPI web server (api/app.py). Run the web UI instead with:

    uvicorn api.app:app --reload

Usage:
    uv run rag_query.py

Debug mode (shows timings, reranked chunks, rewritten queries):
    SYMBAROUM_DEBUG=1 uv run rag_query.py
"""

import asyncio
import json
import os
import time

from api import pipelines

DEBUG = os.environ.get("SYMBAROUM_DEBUG", "").lower() in ("1", "true", "yes")


async def main() -> None:
    await pipelines.initialise()

    print("Symbaroum RAG ready. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        total_start = time.time()

        # Route
        t = time.time()
        pipeline = await pipelines.route(query)
        route_time = time.time() - t
        print(f"\nRouted to: {pipeline} ({route_time:.1f}s)")
        print("\nThinking...\n")

        # Run pipeline, collect answer and timings
        answer_parts = []
        timings = {}

        if pipeline == "lightrag":
            async for chunk in pipelines.lightrag_pipeline(query):
                if chunk.startswith("__timings__"):
                    timings = json.loads(chunk[len("__timings__"):])
                else:
                    answer_parts.append(chunk)
                    print(chunk, end="", flush=True)
        else:
            async for chunk in pipelines.hybrid_pipeline(query):
                if chunk.startswith("__timings__"):
                    timings = json.loads(chunk[len("__timings__"):])
                else:
                    answer_parts.append(chunk)
                    print(chunk, end="", flush=True)

        print()  # newline after streamed answer
        total = time.time() - total_start

        print(f"\nAnswer [{pipeline}] complete.")

        if DEBUG:
            print(f"\nTimings:")
            print(f"  {'route':20s}: {route_time:.1f}s")
            for stage, t in timings.items():
                print(f"  {stage:20s}: {t:.1f}s")
            print(f"  {'TOTAL':20s}: {total:.1f}s")
        else:
            print(f"[{pipeline} | {total:.1f}s]")

        print("-" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())