"""#48: the server context carries the case summary, clinical limits and skill text in advance, per workflow.yml
`function_groups.cualign.context_preload`, so the agent does not spend a model round-trip on load_case, clinical_limits,
get_constraints and load_skill before planning. Through the real NAT app with the local fake planner (no NVIDIA call)."""
import json

import pytest

from rails_fakes import PlanningLLM
from test_rails_middleware import ask, serve, store  # noqa: F401  (store is a fixture)
from cualign.agent.register import ContextPreload, context_preload, limits_view
from cualign.core import skills as S

PLAN = [{"role": "user", "content": "moderate 케이스로 발치 없이 계획 짜줘."}]


MARK = "cuAlign server context: "


def _context(llm) -> dict:
    """The one `cuAlign server context` message of the first model request, parsed. NAT folds the chat history into
    its own prompt, so the marker is searched in every message, as test_rails_middleware does."""
    text = json.dumps(llm.requests[0]["messages"], ensure_ascii=False)
    assert text.count(MARK) == 1
    raw = json.loads(text)  # back to the message list, then find the content that carries the marker
    content = next(str(m.get("content", "")) for m in raw if MARK in str(m.get("content", "")))
    obj, _ = json.JSONDecoder().raw_decode(content[content.index(MARK) + len(MARK):])
    return obj


def _set(pre: dict):
    def edit(cfg):
        cfg["function_groups"]["cualign"]["context_preload"] = pre
    return edit


def test_preload_on_puts_case_limits_and_skill_in_the_context(tmp_path, monkeypatch, store):
    with PlanningLLM() as llm, serve(tmp_path, monkeypatch, llm) as client:   # workflow.yml as committed: all on
        ask(client, "/chat/stream", PLAN, cualign={"case_id": "moderate"})
    ctx = _context(llm)
    assert ctx["case_id"] == "moderate" and ctx["constraints"]["allow_extraction"] is False  # the form's default
    case = ctx["case"]
    assert case["n_teeth"] == len(case["teeth"]) > 0 and isinstance(case["crowding_mm"], (int, float))
    assert case["unsupported"] == []                     # moderate is a supported synthetic case
    assert ctx["limits"] == limits_view()                # exactly what the clinical_limits tool returns
    assert ctx["skill"] == S.read_skill("cualign-clinical-rules")   # exactly what load_skill returns
    assert "Clinical limits" in ctx["skill"]["instructions"]


def test_preload_off_keeps_the_context_as_before(tmp_path, monkeypatch, store):
    with PlanningLLM() as llm, serve(tmp_path, monkeypatch, llm, edit=_set({"case": False, "limits": False, "skill": None})) as client:
        ask(client, "/chat/stream", PLAN, cualign={"case_id": "moderate"})
    ctx = _context(llm)
    assert set(ctx) == {"case_id", "base_plan_id", "constraints", "teeth_fdi", "conditions_ko", "step", "step_ko"}


def test_absent_block_means_nothing_extra(tmp_path, monkeypatch, store):
    def drop(cfg):
        cfg["function_groups"]["cualign"].pop("context_preload")
    with PlanningLLM() as llm, serve(tmp_path, monkeypatch, llm, edit=drop) as client:
        ask(client, "/chat/stream", PLAN, cualign={"case_id": "moderate"})
    assert set(_context(llm)) == {"case_id", "base_plan_id", "constraints", "teeth_fdi", "conditions_ko", "step", "step_ko"}


def test_failing_view_is_left_out_not_a_500(tmp_path, monkeypatch, store, caplog):
    """A view that raises on some case must not turn the request into an unhandled 500: the item is skipped with a
    warning and the rest of the context stays, so the agent reads that item with the tool as before."""
    from cualign.agent import register

    def boom(case):
        raise IndexError("no teeth")
    monkeypatch.setattr(register, "case_view", boom)
    with PlanningLLM() as llm, serve(tmp_path, monkeypatch, llm) as client:
        with caplog.at_level("WARNING", logger="cualign.agent.register"):
            ask(client, "/chat/stream", PLAN, cualign={"case_id": "moderate"})   # asserts HTTP 200
    ctx = _context(llm)
    assert "case" not in ctx and ctx["limits"] == limits_view() and "skill" in ctx
    assert any("context preload of case skipped" in r.message and "IndexError" in r.message for r in caplog.records)


def test_unknown_skill_fails_at_startup_not_per_request():
    with pytest.raises(ValueError, match="unknown skill"):
        context_preload(ContextPreload(skill="no-such-skill"))
    assert context_preload(ContextPreload()) is None
