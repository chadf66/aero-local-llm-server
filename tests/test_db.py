"""Conversation-store tests (db.py) against a temp aero home — no model needed."""

from __future__ import annotations

import pytest

from aero import db


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    db.connect(tmp_path)
    yield


def test_create_and_get_conversation():
    conv = db.create_conversation("Greetings", model="model-a", system="be terse")
    assert conv["id"].startswith("conv_")
    fetched = db.get_conversation(conv["id"])
    assert fetched["title"] == "Greetings"
    assert fetched["model"] == "model-a"
    assert fetched["messages"] == []


def test_get_missing_conversation_is_none():
    assert db.get_conversation("conv_nope") is None


def test_add_messages_and_order():
    conv = db.create_conversation("chat")
    db.add_message(conv["id"], "user", "hello")
    db.add_message(conv["id"], "assistant", "hi there")
    msgs = db.get_conversation(conv["id"])["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "hi there"


def test_add_message_to_missing_conversation_returns_none():
    assert db.add_message("conv_nope", "user", "hi") is None


def test_tool_calls_round_trip_as_json():
    conv = db.create_conversation("tools")
    calls = [{"id": "call_0", "type": "function",
              "function": {"name": "get_weather", "arguments": "{}"}}]
    db.add_message(conv["id"], "assistant", None, tool_calls=calls)
    stored = db.get_conversation(conv["id"])["messages"][0]
    assert stored["content"] is None
    assert stored["tool_calls"][0]["function"]["name"] == "get_weather"


def test_update_and_delete_conversation():
    conv = db.create_conversation("old title")
    assert db.update_conversation(conv["id"], title="new title") is True
    assert db.get_conversation(conv["id"])["title"] == "new title"
    assert db.delete_conversation(conv["id"]) is True
    assert db.get_conversation(conv["id"]) is None
    assert db.update_conversation("conv_nope", title="x") is False
    assert db.delete_conversation("conv_nope") is False


def test_delete_last_message():
    conv = db.create_conversation("chat")
    db.add_message(conv["id"], "user", "q")
    db.add_message(conv["id"], "assistant", "a")
    removed = db.delete_last_message(conv["id"])
    assert removed["role"] == "assistant"
    assert [m["role"] for m in db.get_conversation(conv["id"])["messages"]] == ["user"]


def test_list_orders_by_recent_update():
    a = db.create_conversation("first")
    b = db.create_conversation("second")
    db.add_message(a["id"], "user", "bump")  # touches a -> most recent
    listed = [c["id"] for c in db.list_conversations()]
    assert listed[0] == a["id"] and b["id"] in listed


def test_search_matches_title_and_content():
    a = db.create_conversation("about pelicans")
    b = db.create_conversation("unrelated")
    db.add_message(b["id"], "user", "tell me about quantization")
    by_title = {c["id"] for c in db.search("pelican")}
    by_content = {c["id"] for c in db.search("quantization")}
    assert a["id"] in by_title and b["id"] not in by_title
    assert b["id"] in by_content
