"""HTTP in, `AskService` out.

There is no retrieval logic in this file on purpose. It parses a request, calls the
service, and turns a `ServiceError` into a status code — so the browser and `ask.py`
reach identical arms through identical rules, and a bug in one is a bug in both.

Run it:

    make ui                                    # http://127.0.0.1:8000
    uvicorn rpsg.web.app:app --reload          # same thing, with reload

FastAPI and uvicorn are an optional extra (`pip install -e ".[web]"`) because CI installs
neither them nor the `vector` extras, and importing this module is not required to run the
pipeline, the eval harness or the tests.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rpsg.config import get_settings
from rpsg.logging import get_logger
from rpsg.retrieval.build import SYSTEMS
from rpsg.web.service import AskService, ServiceError

log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

#: 409 rather than 400: the request is well-formed, the server is not in a state to serve
#: it — no index built, no API key, an arm whose graph is missing. All of those are fixed
#: by doing something outside the browser, which is what the message says.
REFUSED = 409


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    system: str = "vector_fulltext"
    #: Vector arms only. None keeps the arm's own measured breadth — see `service`.
    top_k: int | None = Field(default=None, ge=1, le=200)
    retrieval_only: bool = False
    show_evidence: bool = False
    #: Agentic arms only; ignored by the others, as on the CLI.
    max_retrievals: int | None = Field(default=None, ge=1, le=50)
    graph_hints: bool = True
    anchor: bool = True


def create_app(service: AskService | None = None) -> FastAPI:
    """Build the app. `service` is injectable so a test can supply a stub instead of
    loading a sentence-transformer and a 34MB index."""
    app = FastAPI(
        title="RPSG — ask the corpus",
        description="A browser front end for the arms `scripts/ask.py` asks.",
        version="0.1.0",
    )
    # One service for the process, so the embedder and the built arms survive between
    # requests. Created lazily: constructing it must not load anything, and it doesn't.
    # Closed over rather than injected with `Depends`, because `create_app` already is
    # the injection point and a request-scoped dependency would only re-find this object.
    svc = service if service is not None else AskService()

    @app.get("/api/arms")
    def arms() -> dict[str, Any]:
        """The arm picker's contents, including which arms cannot run and why."""
        settings = get_settings()
        return {
            "arms": [asdict(a) for a in svc.arms()],
            "default": "vector_fulltext",
            "synthesis_model": settings.models.synthesis_model,
        }

    # Deliberately `def`, not `async def`: `ask` blocks for as long as retrieval and
    # synthesis take. Starlette runs a sync endpoint in a threadpool, so a slow question
    # does not stall the event loop and the page can still fetch its static assets.
    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        try:
            result = svc.ask(
                req.question,
                system=req.system,
                top_k=req.top_k,
                retrieval_only=req.retrieval_only,
                show_evidence=req.show_evidence,
                max_retrievals=req.max_retrievals,
                graph_hints=req.graph_hints,
                anchor=req.anchor,
            )
        except ServiceError as exc:
            raise HTTPException(status_code=REFUSED, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - logged, then reported as a 500
            # The message is not forwarded: an arbitrary exception from a provider SDK
            # can carry request internals, and the log is where that belongs.
            log.exception("ask failed: system=%s", req.system)
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__} — see the server log."
            ) from exc
        return asdict(result)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "arms": list(SYSTEMS)}

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
