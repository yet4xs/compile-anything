"""Regression: xLAM tools-as-JSON-string must be parsed, not iterated by char.

Pre-fix behavior (Phase 6A audit): every character hit map_tool fallback and
ALL 19,700 xLAM records collapsed to the constant ['EXEC_ACTION'].
Post-fix: semantic_capabilities reflects the actual tool list.
"""
import json
import pytest

from src.dataset.adapters.tooluse import semantic_capabilities


def test_json_string_tools_are_parsed():
    tools = [{"name": "get_account_info"}, {"name": "search_ticker"}]
    raw = json.dumps(tools)  # xLAM raw field is a JSON string
    caps = semantic_capabilities(raw)
    assert caps != ["EXEC_ACTION"], "regressed to constant EXEC_ACTION marker"
    assert "FETCH" in caps and "QUERY_DB" in caps


def test_parsed_list_still_works():
    tools = [{"name": "get_account_info"}, {"name": "search_ticker"}]
    caps = semantic_capabilities(tools)
    assert "FETCH" in caps and "QUERY_DB" in caps


def test_empty_and_garbage_do_not_crash():
    assert semantic_capabilities(None) == []
    assert semantic_capabilities("") == []
    assert semantic_capabilities("not json") == []


def test_dedup_across_tools():
    tools = [{"name": "get_a"}, {"name": "get_b"}]  # both -> FETCH
    assert semantic_capabilities(tools) == ["FETCH"]
