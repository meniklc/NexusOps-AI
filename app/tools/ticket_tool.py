"""Mock 'create_ticket' tool.

Simulates an external ticketing API by persisting to a local JSON file. This is
the *action* the graph gates behind human approval (see the LangGraph interrupt
in app/graph). Kept dependency-free (stdlib only) so it is unit-testable offline.

In a real system `create_ticket` would POST to Jira/ServiceNow/Zendesk; the
human-in-the-loop approval flow around it would be identical.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

_DB_PATH = Path(os.environ.get("TICKET_DB_PATH", Path(__file__).with_name("_ticket_db.json")))


@dataclass
class TicketRequest:
    title: str
    description: str = ""
    priority: str = "normal"      # 'low' | 'normal' | 'high' | 'urgent'
    requester: str = "anonymous"
    tags: list[str] = field(default_factory=list)

    def normalized(self) -> "TicketRequest":
        pr = self.priority.lower().strip()
        if pr not in {"low", "normal", "high", "urgent"}:
            pr = "normal"
        return TicketRequest(
            title=self.title.strip(),
            description=self.description.strip(),
            priority=pr,
            requester=self.requester.strip() or "anonymous",
            tags=[t.strip() for t in self.tags if t.strip()],
        )


def _load() -> list[dict]:
    if not _DB_PATH.exists():
        return []
    try:
        return json.loads(_DB_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(tickets: list[dict]) -> None:
    _DB_PATH.write_text(json.dumps(tickets, indent=2), encoding="utf-8")


def create_ticket(request: TicketRequest) -> dict:
    """Simulate an external API call that creates a ticket. Returns the record."""
    req = request.normalized()
    if not req.title:
        raise ValueError("ticket title is required")

    tickets = _load()
    seq = 1 + max((t.get("_seq", 0) for t in tickets), default=0)
    ticket = {
        "_seq": seq,
        "id": f"TCK-{seq:04d}",
        "status": "open",
        **asdict(req),
    }
    tickets.append(ticket)
    _save(tickets)
    # Don't leak the internal sequence counter to callers.
    return {k: v for k, v in ticket.items() if k != "_seq"}


def list_tickets() -> list[dict]:
    return [{k: v for k, v in t.items() if k != "_seq"} for t in _load()]
