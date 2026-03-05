"""
api/main.py
-----------
FastAPI backend for the econ-intelligence-agent.

Endpoints:
  POST /chat                — send a message, get a response (streaming SSE)
  POST /sessions            — create a new session
  GET  /sessions/{id}       — get session history
  DELETE /sessions/{id}     — clear a session
  GET  /health              — liveness probe
  GET  /agent/tools         — list available tools
"""

from __future__ import annotations

import os
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger

from agent.agent import EconAgent, AgentSession

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Econ Intelligence Agent",
    description="AI assistant for macroeconomic analysis — RAG + ML forecasts + live data",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (use Redis in production)
_sessions: dict[str, AgentSession] = {}
_agent    = EconAgent()
_start_ts = time.time()


# ── Request / response schemas ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:    str
    session_id: str | None = None
    stream:     bool        = True

class SessionResponse(BaseModel):
    session_id: str
    created_at: float
    turn_count: int
    history:    list[dict]


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_or_create_session(session_id: str | None) -> AgentSession:
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    sid     = session_id or str(uuid.uuid4())
    session = AgentSession(session_id=sid)
    _sessions[sid] = session
    return session


async def sse_stream(session: AgentSession, message: str) -> AsyncIterator[str]:
    """Wrap agent output as Server-Sent Events."""
    try:
        async for chunk in _agent.run(message, session, stream=True):
            # SSE format: data: <payload>\n\n
            safe = chunk.replace("\n", "\\n")
            yield f"data: {safe}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.exception(f"Agent error: {e}")
        yield f"data: ERROR: {e}\n\n"
        yield "data: [DONE]\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":      "ok",
        "uptime_s":    round(time.time() - _start_ts, 1),
        "sessions":    len(_sessions),
        "model":       os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
        "vector_db":   os.environ.get("VECTOR_DB", "chroma"),
    }


@app.get("/agent/tools")
def list_tools():
    from agent.agent import TOOLS
    return {
        "tools": [
            {"name": t["name"], "description": t["description"][:120] + "..."}
            for t in TOOLS
        ]
    }


@app.post("/sessions")
def create_session():
    session = AgentSession(session_id=str(uuid.uuid4()))
    _sessions[session.session_id] = session
    return {"session_id": session.session_id}


@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        turn_count=session.turn_count,
        history=session.history,
    )


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
    return {"status": "cleared"}


@app.post("/chat")
async def chat(req: ChatRequest):
    session = get_or_create_session(req.session_id)
    logger.info(f"Chat: session={session.session_id} turn={session.turn_count + 1}")

    if req.stream:
        return StreamingResponse(
            sse_stream(session, req.message),
            media_type="text/event-stream",
            headers={
                "X-Session-ID": session.session_id,
                "Cache-Control": "no-cache",
                "Connection":    "keep-alive",
            },
        )
    else:
        # Non-streaming: collect full response
        full_response = ""
        async for chunk in _agent.run(req.message, session, stream=False):
            full_response += chunk
        return {
            "response":   full_response,
            "session_id": session.session_id,
            "turn":       session.turn_count,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8001)),
        reload=os.environ.get("ENV") == "development",
    )
