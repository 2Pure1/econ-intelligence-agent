"""
agent.py
--------
The core intelligence layer. Uses Anthropic's Claude with native tool use
to answer economic questions by orchestrating five tools:

  1. query_macro_data     — SQL against fct_macro_indicators_monthly
  2. get_forecast         — POST to econ-ml-platform /predict endpoints
  3. search_documents     — RAG over Fed/IMF/FOMC corpus (Chroma/pgvector/Pinecone)
  4. calculate            — Python REPL for statistics and derived metrics
  5. get_recent_releases  — Latest data from streaming.releases_enriched

Architecture:
  - Agentic loop: Claude decides which tools to call, we execute them,
    feed results back, Claude decides if it needs more or can answer.
  - Conversation memory: full turn history in-context (Claude's 200k window).
  - Streaming: yields text chunks as Claude generates them.
  - Observability: every tool call + result is logged with latency.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import anthropic
from loguru import logger

from agent.tools.macro_db    import query_macro_data
from agent.tools.forecast    import get_forecast
from agent.tools.rag_search  import search_documents
from agent.tools.calculator  import calculate
from agent.tools.releases    import get_recent_releases

# ── Tool definitions (Anthropic tool-use schema) ──────────────────────────────
TOOLS: list[dict] = [
    {
        "name":        "query_macro_data",
        "description": (
            "Query the fct_macro_indicators_monthly PostgreSQL mart for historical "
            "US macroeconomic data. Use this for questions about GDP, unemployment, "
            "CPI, Fed Funds Rate, M2, payrolls, PCE, and other indicators. "
            "Returns up to 200 rows of time-series data. Always filter by date range "
            "and limit columns to what's needed. Do NOT write UPDATE/DELETE/INSERT."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type":        "string",
                    "description": "A read-only PostgreSQL SELECT query against marts_marts.fct_macro_indicators_monthly or streaming.releases_enriched. Use LIMIT.",
                },
                "description": {
                    "type":        "string",
                    "description": "Human-readable description of what this query retrieves.",
                },
            },
            "required": ["sql", "description"],
        },
    },
    {
        "name":        "get_forecast",
        "description": (
            "Get a live ML forecast from the econ-ml-platform for one of three targets: "
            "'gdp_growth' (QoQ %), 'unemployment_rate' (monthly level), or "
            "'fed_funds_direction' (UP/FLAT/DOWN classification). "
            "Pass current macro indicator values as features for best accuracy. "
            "Use this when asked about future/predicted values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type":        "string",
                    "enum":        ["gdp_growth", "unemployment_rate", "fed_funds_direction"],
                    "description": "Which economic indicator to forecast.",
                },
                "features": {
                    "type":        "object",
                    "description": "Current macro values as input features. Include unemployment_rate, fed_funds_rate, cpi_yoy_pct, gdp_billions_usd if known.",
                    "properties": {
                        "unemployment_rate":            {"type": "number"},
                        "fed_funds_rate":               {"type": "number"},
                        "cpi_yoy_pct":                  {"type": "number"},
                        "core_pce_yoy_pct":             {"type": "number"},
                        "gdp_billions_usd":             {"type": "number"},
                        "nonfarm_payrolls_mom_change":  {"type": "number"},
                        "observation_month":            {"type": "string"},
                    },
                },
                "horizon": {
                    "type":        "integer",
                    "description": "Forecast horizon (months or quarters ahead). Default: 1.",
                },
            },
            "required": ["target"],
        },
    },
    {
        "name":        "search_documents",
        "description": (
            "Search the RAG corpus of Federal Reserve documents, IMF Working Papers, "
            "FOMC meeting minutes, and economic research papers. Use this for: "
            "policy context, historical Fed reasoning, academic economic theory, "
            "IMF country assessments, and any question requiring document evidence. "
            "Returns the top-k most relevant passages with source citations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "Semantic search query — write it as a natural language question or topic.",
                },
                "k": {
                    "type":        "integer",
                    "description": "Number of document chunks to retrieve (default 5, max 10).",
                },
                "source_filter": {
                    "type":        "string",
                    "enum":        ["fed", "imf", "fomc", "all"],
                    "description": "Filter results by document source. Default: 'all'.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name":        "calculate",
        "description": (
            "Execute a Python expression or short script to compute statistics, "
            "growth rates, correlations, or derived metrics. Has access to numpy, "
            "pandas, and scipy. Use this for: YoY/MoM calculations, recession probability "
            "estimates, correlation analysis, or any arithmetic the other tools can't do inline. "
            "Do NOT use for file I/O or network requests."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type":        "string",
                    "description": "Python code to execute. Print the final result. Has numpy as np, pandas as pd, scipy.stats as stats.",
                },
                "description": {
                    "type":        "string",
                    "description": "What this calculation computes.",
                },
            },
            "required": ["code", "description"],
        },
    },
    {
        "name":        "get_recent_releases",
        "description": (
            "Get the most recent economic data releases processed by the streaming pipeline. "
            "Returns BEA, FRED, and BLS releases with actual values, prior values, "
            "surprise percentages, and market impact classifications. "
            "Use this when asked about recent data, this week's releases, or latest figures."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type":        "integer",
                    "description": "Number of recent releases to return (default 10, max 50).",
                },
                "source": {
                    "type":        "string",
                    "enum":        ["BEA", "FRED", "BLS", "all"],
                    "description": "Filter by data source. Default: 'all'.",
                },
                "high_impact_only": {
                    "type":        "boolean",
                    "description": "If true, return only HIGH market impact releases.",
                },
            },
        },
    },
]

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert macroeconomic analyst and AI assistant for the Econ Intelligence Platform. You have access to:

1. **Live US macro data** — a PostgreSQL mart (fct_macro_indicators_monthly) with monthly US indicators from 1994–present including GDP, unemployment, CPI, Fed Funds Rate, M2, payrolls, PCE, industrial production, and housing starts.

2. **ML forecasts** — trained models (XGBoost, LSTM, Prophet, ARIMA) that forecast GDP growth, unemployment rate, and Fed Funds Rate direction. Always cite model version and latency.

3. **Document corpus** — Federal Reserve publications, IMF Working Papers, and FOMC meeting minutes. Use this for policy context, historical reasoning, and academic grounding.

4. **Streaming release feed** — real-time economic data releases with surprise scores and market impact classifications.

5. **Python calculator** — for statistical computations, growth rate calculations, and derived metrics.

## How to answer:

- **Always use tools first** before answering quantitative questions. Never make up numbers.
- **Cite sources** — when using SQL data, mention the table and date range. When using documents, cite the source and date. When using forecasts, cite the model name and version.
- **Chain tools** when needed: query historical data → calculate statistics → search documents for context → get forecast → synthesise.
- **Be precise** — use exact figures from tool results, not approximations.
- **Flag uncertainty** — distinguish between historical facts (certain), model forecasts (uncertain), and document interpretations (your analysis).
- **Structure long answers** with headers. For short factual questions, be concise.

## Tone:
Professional economist. Rigorous but accessible. No financial advice — analytical commentary only."""


# ── Message types ──────────────────────────────────────────────────────────────
@dataclass
class ConversationTurn:
    role:    str   # "user" | "assistant"
    content: str | list  # str for simple, list for tool-use blocks


@dataclass
class AgentSession:
    session_id:  str
    history:     list[dict] = field(default_factory=list)
    created_at:  float      = field(default_factory=time.time)
    turn_count:  int        = 0


# ── Tool dispatcher ────────────────────────────────────────────────────────────
TOOL_FUNCTIONS = {
    "query_macro_data":   query_macro_data,
    "get_forecast":       get_forecast,
    "search_documents":   search_documents,
    "calculate":          calculate,
    "get_recent_releases":get_recent_releases,
}

async def dispatch_tool(name: str, inputs: dict) -> str:
    """Execute a tool and return its result as a string."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"ERROR: Unknown tool '{name}'"

    t0 = time.perf_counter()
    try:
        if asyncio.iscoroutinefunction(fn):
            result = await fn(**inputs)
        else:
            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: fn(**inputs))
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"Tool {name} completed in {elapsed:.0f}ms")
        return str(result)
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return f"ERROR executing {name}: {e}"


# ── Agent ──────────────────────────────────────────────────────────────────────
class EconAgent:
    def __init__(self):
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model  = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    async def run(
        self,
        user_message: str,
        session: AgentSession,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """
        Run the agentic loop for a single user turn.
        Yields text chunks if stream=True, otherwise yields the full response.

        Agentic loop:
          1. Append user message to history
          2. Call Claude with tools
          3. If Claude calls tools → execute them → feed results back → repeat
          4. When Claude returns text with no tool calls → yield and finish
        """
        session.history.append({"role": "user", "content": user_message})
        session.turn_count += 1

        max_iterations = 8   # prevent infinite loops
        iterations     = 0

        while iterations < max_iterations:
            iterations += 1
            logger.info(f"Agent iteration {iterations} | session={session.session_id}")

            if stream:
                # Use the streaming helper for the first generation OR any subsequent 
                # generation after tool results are added.
                full_text = ""
                async for chunk in self._stream_turn(session):
                    if not chunk.startswith("\n\n_Executing"):
                        full_text += chunk
                    yield chunk
                
                # Check the last message in history (which was added by _stream_turn)
                # to see if we need to continue the loop (i.e. if it contains tool_calls)
                last_msg = session.history[-1]
                tool_uses = [b for b in last_msg["content"] if hasattr(b, "type") and b.type == "tool_use"]
                if not tool_uses:
                    break
                # If there were tools, _stream_turn already executed them and added 
                # tool_results to history. The loop continues to process those results.
            else:
                # Non-streaming implementation (as before)
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=session.history,
                )
                session.history.append({"role": "assistant", "content": response.content})
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                if not tool_uses:
                    text = " ".join(b.text for b in response.content if hasattr(b, "text"))
                    yield text
                    break

                tool_results = await asyncio.gather(*[
                    dispatch_tool(tu.name, tu.input) for tu in tool_uses
                ])
                session.history.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tu.id, "content": result}
                        for tu, result in zip(tool_uses, tool_results)
                    ],
                })

    async def _stream_turn(self, session: AgentSession) -> AsyncIterator[str]:
        """Stream the Claude response, handling tool calls mid-stream."""
        accumulated_content = []
        current_text        = ""

        with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=session.history,
        ) as stream:
            for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_start":
                        if hasattr(event.content_block, "text"):
                            pass  # text block starting
                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            chunk = event.delta.text
                            current_text += chunk
                            yield chunk

            # Get the final message for tool-use handling
            final = stream.get_final_message()

        session.history.append({
            "role":    "assistant",
            "content": final.content,
        })

        tool_uses = [b for b in final.content if b.type == "tool_use"]
        if tool_uses:
            # Signal that we're executing tools
            yield "\n\n_Executing tools..._\n\n"

            tool_results = await asyncio.gather(*[
                dispatch_tool(tu.name, tu.input) for tu in tool_uses
            ])

            session.history.append({
                "role": "user",
                "content": [
                    {
                        "type":        "tool_result",
                        "tool_use_id": tu.id,
                        "content":     result,
                    }
                    for tu, result in zip(tool_uses, tool_results)
                ],
            })

            # Continue the agentic loop (non-streaming for subsequent turns)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=session.history,
            )
            session.history.append({
                "role":    "assistant",
                "content": response.content,
            })
            text = " ".join(
                b.text for b in response.content if hasattr(b, "text")
            )
            yield text
