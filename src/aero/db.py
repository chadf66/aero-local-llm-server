"""The web UI's conversation store — durable, searchable chat history in SQLite.

This is the *only* server-side state aero keeps, and it is deliberately fenced off
from inference. The engine never touches it: the UI generates by calling the
stateless ``/v1/chat/completions`` (exactly as an agent would), then persists the
resulting turns here. So history is a UI convenience layered on top of the API, not
a thing the model path depends on — drop ``aero.db`` and inference is unaffected.

SQLite (stdlib ``sqlite3``) is the right size for a single-user local box: one file
under ``~/.aero``, no server, full-text-ish search with ``LIKE`` (FTS is a later
win). uvicorn runs single-process here, so one shared connection behind a lock
mirrors the engine's single-lock simplicity rather than a per-request pool.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# One shared connection + lock for the process (see module docstring). Installed by
# connect() at server startup; tests point it at a temp home.
_conn: Optional[sqlite3.Connection] = None
_lock = threading.RLock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id       TEXT PRIMARY KEY,
    title    TEXT NOT NULL,
    model    TEXT,
    system   TEXT,
    created  REAL NOT NULL,
    updated  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT,
    tool_calls_json TEXT,
    created         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created);
"""


def connect(home: Path) -> None:
    """Open (and create if needed) the history database under ``home``."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
        home.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: uvicorn's threadpool may touch the connection from
        # different threads; _lock serializes every access so that's safe here.
        _conn = sqlite3.connect(str(home / "aero.db"), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA foreign_keys = ON")
        _conn.executescript(_SCHEMA)
        _conn.commit()


def _db() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("db.connect() has not been called")
    return _conn


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #


def create_conversation(
    title: str, *, model: Optional[str] = None, system: Optional[str] = None
) -> dict:
    """Create an empty conversation and return it (without messages)."""
    cid = _new_id("conv")
    now = _now()
    with _lock:
        _db().execute(
            "INSERT INTO conversations (id, title, model, system, created, updated)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (cid, title, model, system, now, now),
        )
        _db().commit()
    return {"id": cid, "title": title, "model": model, "system": system,
            "created": now, "updated": now}


def list_conversations() -> list[dict]:
    """All conversations, most-recently-updated first (no message bodies)."""
    with _lock:
        rows = _db().execute(
            "SELECT id, title, model, system, created, updated"
            " FROM conversations ORDER BY updated DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(cid: str) -> Optional[dict]:
    """One conversation with its messages in order, or None if it doesn't exist."""
    with _lock:
        row = _db().execute(
            "SELECT id, title, model, system, created, updated FROM conversations WHERE id = ?",
            (cid,),
        ).fetchone()
        if row is None:
            return None
        msgs = _db().execute(
            "SELECT id, role, content, tool_calls_json, created"
            " FROM messages WHERE conversation_id = ? ORDER BY created, rowid",
            (cid,),
        ).fetchall()
    conv = dict(row)
    conv["messages"] = [_message_row(m) for m in msgs]
    return conv


def update_conversation(
    cid: str, *, title: Optional[str] = None, model: Optional[str] = None,
    system: Optional[str] = None,
) -> bool:
    """Patch a conversation's metadata. Returns False if it doesn't exist."""
    sets, params = [], []
    for field, value in (("title", title), ("model", model), ("system", system)):
        if value is not None:
            sets.append(f"{field} = ?")
            params.append(value)
    with _lock:
        if not sets:  # still bump updated so "touch" works
            cur = _db().execute(
                "UPDATE conversations SET updated = ? WHERE id = ?", (_now(), cid)
            )
        else:
            params += [_now(), cid]
            cur = _db().execute(
                f"UPDATE conversations SET {', '.join(sets)}, updated = ? WHERE id = ?",
                params,
            )
        _db().commit()
        return cur.rowcount > 0


def delete_conversation(cid: str) -> bool:
    """Delete a conversation and its messages. Returns False if it didn't exist."""
    with _lock:
        cur = _db().execute("DELETE FROM conversations WHERE id = ?", (cid,))
        _db().commit()
        return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


def add_message(
    cid: str, role: str, content: Optional[str], *, tool_calls: Optional[Any] = None
) -> Optional[dict]:
    """Append a message to a conversation and bump its ``updated`` time.

    ``tool_calls`` (the OpenAI-shaped list, if any) is stored as JSON. Returns the
    stored message, or None if the conversation doesn't exist.
    """
    with _lock:
        exists = _db().execute(
            "SELECT 1 FROM conversations WHERE id = ?", (cid,)
        ).fetchone()
        if exists is None:
            return None
        mid = _new_id("msg")
        now = _now()
        tc_json = json.dumps(tool_calls) if tool_calls else None
        _db().execute(
            "INSERT INTO messages (id, conversation_id, role, content, tool_calls_json, created)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (mid, cid, role, content, tc_json, now),
        )
        _db().execute("UPDATE conversations SET updated = ? WHERE id = ?", (now, cid))
        _db().commit()
    return {"id": mid, "role": role, "content": content, "tool_calls": tool_calls,
            "created": now}


def delete_last_message(cid: str) -> Optional[dict]:
    """Drop the most recent message in a conversation (used by 'regenerate').

    Returns the deleted message, or None if the conversation has no messages.
    """
    with _lock:
        row = _db().execute(
            "SELECT id, role, content, tool_calls_json, created FROM messages"
            " WHERE conversation_id = ? ORDER BY created DESC, rowid DESC LIMIT 1",
            (cid,),
        ).fetchone()
        if row is None:
            return None
        _db().execute("DELETE FROM messages WHERE id = ?", (row["id"],))
        _db().execute("UPDATE conversations SET updated = ? WHERE id = ?", (_now(), cid))
        _db().commit()
    return _message_row(row)


def search(query: str) -> list[dict]:
    """Conversations whose title or any message content matches ``query`` (LIKE).

    A pragmatic substring search — good enough for a single user's history. SQLite
    FTS5 would rank and scale better and is the natural upgrade later.
    """
    like = f"%{query}%"
    with _lock:
        rows = _db().execute(
            "SELECT DISTINCT c.id, c.title, c.model, c.system, c.created, c.updated"
            " FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id"
            " WHERE c.title LIKE ? OR m.content LIKE ?"
            " ORDER BY c.updated DESC",
            (like, like),
        ).fetchall()
    return [dict(r) for r in rows]


def _message_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    tc = d.pop("tool_calls_json", None)
    d["tool_calls"] = json.loads(tc) if tc else None
    return d
