"""Streamlit chat UI for the Enterprise RAG + Agentic Assistant.

Features:
  - streaming chat (SSE from the FastAPI backend)
  - retrieved-source citations panel per answer
  - human-in-the-loop approval panel when the graph interrupts for ticket creation

This module is a *presentation* layer only: all backend calls, streaming, and
state handling are untouched. The dark product look comes from
`.streamlit/config.toml` (base palette) plus the single `CUSTOM_CSS` block
below (components native theming can't reach: bubbles, badges, input).
"""
from __future__ import annotations

import html
import json
import os
from typing import Iterator

import httpx
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise RAG + Agent", page_icon="🤖", layout="wide")


# --- theming ---------------------------------------------------------------
# One maintainable CSS block. Colors mirror .streamlit/config.toml so the app
# looks identical whether or not the config file is picked up.
CUSTOM_CSS = """
<style>
:root {
  --bg: #0e0e10;
  --panel: #16161a;
  --panel-2: #1c1c22;
  --border: #26262e;
  --text: #e7e7ea;
  --muted: #8b8b95;
  --primary: #6366f1;
  --primary-soft: rgba(99, 102, 241, 0.14);
  --ok: #34d399;
  --ok-soft: rgba(52, 211, 153, 0.13);
  --warn: #f59e0b;
  --warn-soft: rgba(245, 158, 11, 0.14);
}

/* Remove default Streamlit chrome for a cleaner product surface. */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; }

/* Constrain the reading column and give the header room. */
[data-testid="stAppViewContainer"] .block-container {
  max-width: 880px;
  padding-top: 2.2rem;
  padding-bottom: 6rem;
}

/* ---- product header ---- */
.app-header { margin: 0 0 1.4rem 0; }
.app-title {
  font-size: 1.9rem;
  font-weight: 750;
  letter-spacing: -0.02em;
  background: linear-gradient(90deg, #c7d2fe 0%, #8b93ff 55%, #6366f1 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}
.app-sub { color: var(--muted); font-size: 0.95rem; margin-top: 0.3rem; line-height: 1.45; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] { background: #121216; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.side-title {
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--muted);
  margin: 1.1rem 0 0.5rem 0;
}
.side-kv { color: var(--muted); font-size: 0.82rem; margin-bottom: 0.6rem; }
.side-kv code { color: var(--text); background: var(--panel-2); padding: 1px 6px; border-radius: 6px; }

/* status badge */
.status-badge {
  display: inline-block;
  font-size: 0.82rem;
  font-weight: 600;
  padding: 0.32rem 0.7rem;
  border-radius: 999px;
  border: 1px solid transparent;
  width: 100%;
  box-sizing: border-box;
}
.status-ok  { color: var(--ok);   background: var(--ok-soft);   border-color: rgba(52,211,153,.3); }
.status-err { color: var(--warn); background: var(--warn-soft); border-color: rgba(245,158,11,.3); }

/* buttons (sidebar actions + example suggestions) */
[data-testid="stSidebar"] .stButton > button {
  background: var(--panel-2);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 10px;
  text-align: left;
  font-weight: 500;
  white-space: normal;      /* wrap long conversation previews */
  line-height: 1.3;
  transition: border-color .15s ease, background .15s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
  border-color: var(--primary);
  background: var(--primary-soft);
}
/* Active conversation (rendered as a primary button). */
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
  border-color: var(--primary) !important;
  background: var(--primary-soft) !important;
  color: #c7d2fe !important;
}

/* ---- chat bubbles ---- */
[data-testid="stChatMessage"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.55rem 1.05rem;
  margin-bottom: 0.85rem;
}
/* User turns get a subtle indigo tint so roles read distinctly. */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: #191a24;
  border-color: rgba(99, 102, 241, 0.28);
}

/* ---- metadata badge row (under assistant answers) ---- */
.meta-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0 0.15rem 0; }
.badge {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--muted);
  background: var(--panel-2);
  border: 1px solid var(--border);
  padding: 0.16rem 0.55rem;
  border-radius: 999px;
}
.badge-ok   { color: var(--ok);   background: var(--ok-soft);   border-color: rgba(52,211,153,.28); }
.badge-warn { color: var(--warn); background: var(--warn-soft); border-color: rgba(245,158,11,.28); }

/* ---- retrieved sources ---- */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  background: var(--panel);
}
.src-row {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  padding: 0.3rem 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.86rem;
}
.src-row:last-child { border-bottom: none; }
.src-idx  { color: var(--primary); font-weight: 700; }
.src-file { color: var(--text); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.src-meta { color: var(--muted); margin-left: auto; font-size: 0.78rem; }

/* ---- chat input ---- */
[data-testid="stChatInput"] {
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--panel);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--primary); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- SSE client ------------------------------------------------------------
def sse_post(path: str, payload: dict) -> Iterator[tuple[str, str]]:
    """POST to an SSE endpoint and yield (event_name, data_str) frames."""
    url = f"{BACKEND_URL}{path}"
    with httpx.stream("POST", url, json=payload, timeout=None) as resp:
        resp.raise_for_status()
        event = None
        data_lines: list[str] = []
        for line in resp.iter_lines():
            if line == "":
                if data_lines:
                    yield event or "message", "\n".join(data_lines)
                event, data_lines = None, []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip())
        if data_lines:
            yield event or "message", "\n".join(data_lines)


# --- session state ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {role, content, sources, meta}
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pending" not in st.session_state:
    st.session_state.pending = None          # {thread_id, request, message}


# --- rendering helpers -----------------------------------------------------
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Retrieved sources ({len(sources)})"):
        rows = []
        for i, s in enumerate(sources, start=1):
            src = html.escape(str(s.get("source", "?")))
            chunk = html.escape(str(s.get("chunk_index", "?")))
            try:
                score = f"{float(s.get('score', 0)):.4f}"
            except (TypeError, ValueError):
                score = html.escape(str(s.get("score", "?")))
            rows.append(
                f'<div class="src-row">'
                f'<span class="src-idx">[{i}]</span>'
                f'<span class="src-file">{src}</span>'
                f'<span class="src-meta">chunk {chunk} · score {score}</span>'
                f"</div>"
            )
        st.markdown("".join(rows), unsafe_allow_html=True)


def render_meta(meta: dict) -> None:
    """Render intent / injection score / self-critique as a subtle badge row."""
    badges: list[str] = []
    if meta.get("intent"):
        badges.append(f'<span class="badge">🎯 intent: {html.escape(str(meta["intent"]))}</span>')
    sec = meta.get("security") or {}
    if sec:
        score = sec.get("score", 0)
        cls = "badge badge-warn" if isinstance(score, (int, float)) and score > 0 else "badge"
        badges.append(f'<span class="{cls}">🛡️ injection: {html.escape(str(score))}</span>')
    crit = meta.get("critique") or {}
    if crit and "supported" in crit:
        ok = bool(crit.get("supported"))
        cls = "badge badge-ok" if ok else "badge badge-warn"
        label = "supported" if ok else "unsupported"
        badges.append(f'<span class="{cls}">✓ self-critique: {label}</span>')
    if meta.get("sql_result", {}).get("sql"):
        badges.append('<span class="badge">🗄️ SQL executed</span>')
    if badges:
        st.markdown(f'<div class="meta-row">{"".join(badges)}</div>', unsafe_allow_html=True)


def _truncate(text: str, limit: int = 40) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _relative_time(ts: float) -> str:
    import time

    try:
        delta = time.time() - float(ts)
    except (TypeError, ValueError):
        return ""
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def load_thread(thread_id: str) -> None:
    """Replace the chat window with a past thread's history and make it active."""
    resp = httpx.get(f"{BACKEND_URL}/threads/{thread_id}/messages", timeout=10)
    resp.raise_for_status()
    st.session_state.messages = resp.json().get("messages", [])
    st.session_state.thread_id = thread_id
    st.session_state.pending = None


def stream_turn(path: str, payload: dict) -> None:
    """Drive an SSE turn, updating the UI live. Sets pending state on interrupt."""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        sources: list[dict] = []
        meta: dict = {}
        interrupted = False

        for event, data in sse_post(path, payload):
            if event == "thread":
                st.session_state.thread_id = json.loads(data).get("thread_id")
            elif event == "status":
                obj = json.loads(data)
                meta.update({k: obj[k] for k in ("intent", "security") if k in obj})
                node = obj.get("node")
                if node:
                    placeholder.markdown(f"_…{node}_")
            elif event == "sources":
                sources = json.loads(data).get("data", [])
            elif event == "token":
                answer += data
                placeholder.markdown(answer + "▌")
            elif event == "interrupt":
                obj = json.loads(data)
                st.session_state.pending = {
                    "thread_id": obj.get("thread_id") or st.session_state.thread_id,
                    "payload": obj.get("data", {}),
                }
                interrupted = True
                placeholder.markdown("⏸️ **Awaiting your approval below.**")
            elif event == "final":
                obj = json.loads(data).get("data", {})
                answer = obj.get("answer", answer)
                sources = obj.get("citations", sources)
                meta.update(
                    {
                        "intent": obj.get("intent"),
                        "security": obj.get("security"),
                        "critique": obj.get("critique"),
                        "sql_result": obj.get("sql_result") or {},
                    }
                )

        if not interrupted:
            placeholder.markdown(answer or "_(no answer)_")
            render_sources(sources)
            render_meta(meta)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources, "meta": meta}
            )


# --- sidebar ---------------------------------------------------------------
EXAMPLES = [
    ("What is the refund policy for hardware?", "💬  Refund policy for hardware  ·  Q&A"),
    ("How many delivered orders were in EMEA?", "📊  Delivered orders in EMEA  ·  SQL"),
    ("Create an urgent ticket: VPN is down", "🎫  Urgent ticket: VPN is down  ·  action"),
]

with st.sidebar:
    st.markdown('<div class="side-title">Connection</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="side-kv">Backend<br><code>{html.escape(BACKEND_URL)}</code></div>',
        unsafe_allow_html=True,
    )
    try:
        h = httpx.get(f"{BACKEND_URL}/health", timeout=5).json()
        st.markdown(
            f'<div class="status-badge status-ok">● LLM: {html.escape(str(h["llm_provider"]))}'
            f' · DB: {html.escape(str(h["database"]))}</div>',
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.markdown(
            '<div class="status-badge status-err">● Backend unreachable</div>',
            unsafe_allow_html=True,
        )
        st.caption(str(exc))

    st.markdown('<div class="side-title">Session</div>', unsafe_allow_html=True)
    if st.button("🔄  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.session_state.pending = None
        st.rerun()

    if st.button("🎫  Show tickets", use_container_width=True):
        try:
            data = httpx.get(f"{BACKEND_URL}/tickets", timeout=5).json()
            st.json(data)
        except Exception as exc:
            st.error(str(exc))

    # --- conversation history ---
    st.markdown('<div class="side-title">Conversations</div>', unsafe_allow_html=True)
    try:
        past_threads = httpx.get(f"{BACKEND_URL}/threads", timeout=5).json().get("threads", [])
    except Exception:
        past_threads = []

    if not past_threads:
        st.caption("No past conversations yet.")
    else:
        active_tid = st.session_state.get("thread_id")
        for t in past_threads:
            tid = t.get("thread_id")
            is_active = tid == active_tid
            rel = _relative_time(t.get("updated_at"))
            marker = "🟢 " if is_active else "💬 "
            label = f"{marker}{_truncate(t.get('preview', ''))}"
            if rel:
                label += f"  ·  {rel}"
            if st.button(
                label,
                key=f"thread_{tid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                load_thread(tid)
                st.rerun()

    st.markdown('<div class="side-title">Try an example</div>', unsafe_allow_html=True)
    for i, (question, label) in enumerate(EXAMPLES):
        if st.button(label, key=f"example_{i}", use_container_width=True):
            st.session_state.queued_prompt = question
            st.rerun()


# --- header ----------------------------------------------------------------
st.markdown(
    '<div class="app-header">'
    '<div class="app-title">Enterprise RAG + Agentic Assistant</div>'
    '<div class="app-sub">Hybrid retrieval over your documents, NL→SQL over your business '
    "data, and human-approved actions — grounded, cited, and streamed in real time.</div>"
    "</div>",
    unsafe_allow_html=True,
)

# --- history ---------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))
            render_meta(msg.get("meta", {}))

# --- pending approval panel ------------------------------------------------
if st.session_state.pending:
    payload = st.session_state.pending["payload"]
    request = payload.get("request", {})
    st.warning("🔔 **Human approval required**")
    st.markdown(payload.get("message", "Approve this action?"))
    with st.form("approval_form"):
        title = st.text_input("Title", value=request.get("title", ""))
        priority = st.selectbox(
            "Priority",
            ["low", "normal", "high", "urgent"],
            index=["low", "normal", "high", "urgent"].index(request.get("priority", "normal")),
        )
        col1, col2 = st.columns(2)
        approve = col1.form_submit_button("✅ Approve", use_container_width=True)
        reject = col2.form_submit_button("❌ Reject", use_container_width=True)

    if approve or reject:
        thread_id = st.session_state.pending["thread_id"]
        edits = {"title": title, "priority": priority} if approve else None
        st.session_state.pending = None
        stream_turn(
            "/approve/stream",
            {"thread_id": thread_id, "approved": bool(approve), "edits": edits},
        )
        st.rerun()

# --- chat input ------------------------------------------------------------
# A typed message OR a clicked example suggestion both drive the same turn.
prompt = st.chat_input("Ask a question, query data, or request an action…")
if not prompt:
    prompt = st.session_state.pop("queued_prompt", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    stream_turn("/chat/stream", {"message": prompt, "thread_id": st.session_state.thread_id})
    st.rerun()
