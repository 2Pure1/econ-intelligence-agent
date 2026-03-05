#!/usr/bin/env python3
"""
cli/chat.py
-----------
Terminal chat client for the Econ Intelligence Agent.

Features:
  - Streaming output with Rich formatting
  - Slash commands: /new, /history, /tools, /clear, /help, /export
  - Tool-call visibility (shows which tools are running)
  - Session persistence across invocations (saves session_id to ~/.econ_agent_session)
  - Offline mode: hits the API directly and renders Markdown

Usage:
    python cli/chat.py
    python cli/chat.py --api http://localhost:8001
    python cli/chat.py --no-stream    # non-streaming mode
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich import print as rprint

API_URL         = os.environ.get("AGENT_API_URL", "http://localhost:8001")
SESSION_FILE    = Path.home() / ".econ_agent_session"
console         = Console()


# ── Session management ─────────────────────────────────────────────────────────
def load_session_id() -> str | None:
    if SESSION_FILE.exists():
        return SESSION_FILE.read_text().strip() or None
    return None


def save_session_id(session_id: str) -> None:
    SESSION_FILE.write_text(session_id)


def new_session(api_url: str) -> str:
    resp = httpx.post(f"{api_url}/sessions", timeout=10)
    resp.raise_for_status()
    session_id = resp.json()["session_id"]
    save_session_id(session_id)
    return session_id


# ── Display helpers ────────────────────────────────────────────────────────────
WELCOME = """
╔══════════════════════════════════════════════════════════════╗
║          ECON INTELLIGENCE AGENT  v1.0                      ║
║          Powered by Claude + RAG + ML Forecasting           ║
╚══════════════════════════════════════════════════════════════╝

Type a question or use a slash command:
  /new      — start a new conversation
  /history  — show conversation history
  /tools    — list available tools
  /export   — save conversation to file
  /clear    — clear the screen
  /help     — show this message
  /quit     — exit

Example questions:
  • "What is the current unemployment trend over the last 12 months?"
  • "Forecast GDP growth for next quarter"
  • "What did the Fed say about inflation in the last FOMC meeting?"
  • "Compare current CPI to the 2008 financial crisis period"
  • "What's the probability the Fed cuts rates in the next 2 months?"
"""

COMMANDS = {
    "/new":     "Start a new conversation session",
    "/history": "Show conversation history for this session",
    "/tools":   "List the agent's available tools",
    "/export":  "Export conversation to a Markdown file",
    "/clear":   "Clear the screen",
    "/help":    "Show help",
    "/quit":    "Exit",
    "/exit":    "Exit",
}


def show_welcome():
    console.print(Panel(WELCOME, border_style="cyan", expand=False))


def show_tools(api_url: str):
    try:
        resp  = httpx.get(f"{api_url}/agent/tools", timeout=5)
        tools = resp.json()["tools"]
        console.print("\n[bold cyan]Available tools:[/bold cyan]")
        for t in tools:
            console.print(f"  [green]•[/green] [bold]{t['name']}[/bold]: {t['description']}")
        console.print()
    except Exception as e:
        console.print(f"[red]Could not fetch tools: {e}[/red]")


def show_history(api_url: str, session_id: str):
    try:
        resp    = httpx.get(f"{api_url}/sessions/{session_id}", timeout=5)
        history = resp.json().get("history", [])
        turns   = resp.json().get("turn_count", 0)
        console.print(f"\n[bold cyan]Session {session_id[:8]}... | {turns} turns[/bold cyan]\n")
        for msg in history:
            role = msg["role"]
            if isinstance(msg["content"], str):
                if role == "user":
                    console.print(f"[bold yellow]You:[/bold yellow] {msg['content']}")
                else:
                    console.print(Markdown(msg["content"]))
            console.print()
    except Exception as e:
        console.print(f"[red]Could not fetch history: {e}[/red]")


def export_conversation(api_url: str, session_id: str):
    try:
        resp    = httpx.get(f"{api_url}/sessions/{session_id}", timeout=5)
        history = resp.json().get("history", [])
        lines   = [f"# Econ Intelligence Agent — Session {session_id[:8]}\n",
                   f"_Exported {time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n"]
        for msg in history:
            role = msg["role"]
            content = msg["content"] if isinstance(msg["content"], str) else "[tool interaction]"
            if role == "user":
                lines.append(f"**You:** {content}\n\n")
            else:
                lines.append(f"**Agent:**\n\n{content}\n\n---\n\n")
        fname = f"econ_agent_{session_id[:8]}_{int(time.time())}.md"
        Path(fname).write_text("".join(lines))
        console.print(f"[green]Exported to {fname}[/green]")
    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")


# ── Streaming chat ─────────────────────────────────────────────────────────────
def chat_streaming(api_url: str, session_id: str, message: str) -> str:
    """Send message and stream response, rendering Markdown progressively."""
    payload = {"message": message, "session_id": session_id, "stream": True}

    console.print(f"\n[bold cyan]Agent:[/bold cyan] ", end="")

    buffer   = ""
    in_tools = False

    with httpx.stream(
        "POST", f"{api_url}/chat",
        json=payload, timeout=120.0,
        headers={"Accept": "text/event-stream"},
    ) as response:
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break

            # Unescape newlines from SSE encoding
            chunk = data.replace("\\n", "\n")

            if "_Executing tools..._" in chunk:
                in_tools = True
                console.print("\n[dim italic]⚙  Running tools...[/dim italic]", end="")
                continue
            if in_tools and chunk.strip():
                in_tools = False
                console.print()   # newline after tool indicator

            buffer += chunk
            # Print chunk directly (no progressive Markdown rendering for streaming)
            console.print(chunk, end="", markup=False)

    console.print()  # final newline
    console.print()

    return buffer


def chat_blocking(api_url: str, session_id: str, message: str) -> str:
    """Non-streaming fallback."""
    payload = {"message": message, "session_id": session_id, "stream": False}
    resp    = httpx.post(f"{api_url}/chat", json=payload, timeout=120.0)
    resp.raise_for_status()
    response_text = resp.json()["response"]
    console.print(f"\n[bold cyan]Agent:[/bold cyan]")
    console.print(Markdown(response_text))
    console.print()
    return response_text


# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Econ Intelligence Agent CLI")
    parser.add_argument("--api",       default=API_URL,  help="API base URL")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--new",       action="store_true", help="Start fresh session")
    args = parser.parse_args()

    api_url = args.api

    # Verify API is running
    try:
        health = httpx.get(f"{api_url}/health", timeout=5).json()
        console.print(f"[dim]Connected to {api_url} | model={health['model']} | vector_db={health['vector_db']}[/dim]")
    except Exception:
        console.print(f"[red]Cannot connect to agent API at {api_url}[/red]")
        console.print("[yellow]Start the API first: uvicorn api.main:app --port 8001[/yellow]")
        sys.exit(1)

    # Session
    session_id = None if args.new else load_session_id()
    if not session_id:
        session_id = new_session(api_url)
        console.print(f"[dim]New session: {session_id[:8]}...[/dim]")
    else:
        console.print(f"[dim]Resuming session: {session_id[:8]}...[/dim]")

    show_welcome()

    # Main loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold yellow]You[/bold yellow]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        # Slash commands
        cmd = user_input.lower().split()[0] if user_input.startswith("/") else ""

        if cmd in ("/quit", "/exit"):
            console.print("[dim]Goodbye.[/dim]")
            break
        elif cmd == "/new":
            session_id = new_session(api_url)
            console.print(f"[green]New session started: {session_id[:8]}...[/green]")
        elif cmd == "/history":
            show_history(api_url, session_id)
        elif cmd == "/tools":
            show_tools(api_url)
        elif cmd == "/export":
            export_conversation(api_url, session_id)
        elif cmd == "/clear":
            console.clear()
        elif cmd == "/help":
            show_welcome()
        elif cmd in COMMANDS:
            console.print(f"[dim]{COMMANDS[cmd]}[/dim]")
        else:
            # Send to agent
            try:
                if args.no_stream:
                    chat_blocking(api_url, session_id, user_input)
                else:
                    chat_streaming(api_url, session_id, user_input)
            except httpx.ReadTimeout:
                console.print("[red]Request timed out. The agent may be processing a complex query.[/red]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
