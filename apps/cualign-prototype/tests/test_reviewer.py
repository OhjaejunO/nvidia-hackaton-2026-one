"""No paid calls: inject empty replies, 503, malformed responses and timeouts."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from cualign.agent.context import CURRENT_RUN, PlanRun
from cualign.agent.reviewer import review_plan
from cualign.core.constraints import Constraints
from cualign.core.service import PlanningService
from cualign.core import store as store_module


@pytest.mark.parametrize("failure,error", [("", "empty_response"), (RuntimeError("503 overloaded"), "upstream_503"),
                                          (Exception("[429] Too Many Requests"), "upstream_429"),
                                          ("Thought: unfinished", "invalid_response"),
                                          (SimpleNamespace(content="We need to produce a short Korean review memo",
                                                           response_metadata={"finish_reason": "length"}), "invalid_response")])
def test_review_budget_and_cached_failure(tmp_path, monkeypatch, failure, error):
    monkeypatch.setattr(store_module, "OUT_DIR", tmp_path)
    s = store_module.Store()
    svc = PlanningService(s)
    pid = svc.stages(svc.target("moderate", "expansion_ipr", Constraints()))
    class Fake:
        calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            if isinstance(failure, Exception):
                raise failure
            return failure if isinstance(failure, SimpleNamespace) else SimpleNamespace(content=failure)
    async def run():
        fake = Fake()
        context = PlanRun("review-run", "moderate", None, Constraints(), selected_plan_id=pid)
        token = CURRENT_RUN.set(context)
        try:
            first = await review_plan(pid, fake, store=s)
            second = await review_plan(pid, fake, store=s)
            assert first == second
            assert fake.calls == context.review_attempts == 2
            assert first["status"] == "failed" and first["error"] == error
            with pytest.raises(ValueError):
                s.approve(pid, s.lock_fdi(pid))
        finally:
            CURRENT_RUN.reset(token)
    asyncio.run(run())


def test_timeout_and_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "OUT_DIR", tmp_path)
    s = store_module.Store()
    svc = PlanningService(s)
    class Slow:
        async def ainvoke(self, messages):
            await asyncio.sleep(1)
    class Recovers:
        calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            return SimpleNamespace(content="" if self.calls == 1 else "검토 메모 초안")
    async def run():
        for llm, expected in ((Slow(), "failed"), (Recovers(), "passed")):
            pid = svc.stages(svc.target("moderate", "expansion_ipr", Constraints()))
            result = await review_plan(pid, llm, store=s, timeout_seconds=0.01, total_seconds=0.03)
            assert result["status"] == expected
            assert result["attempts"] <= 2
            if expected == "failed":
                assert result["error"] == "timeout"
    asyncio.run(run())

def test_registered_reviewer_function_schema(tmp_path, monkeypatch):
    from cualign.agent import reviewer
    from nat.builder.function import LambdaFunction
    monkeypatch.setattr(store_module, "OUT_DIR", tmp_path)
    # Default STORE is the same object captured by the registered function.
    s = store_module.STORE
    svc = PlanningService(s)
    pid = svc.stages(svc.target("moderate", "expansion_ipr", Constraints()))
    class Builder:
        async def get_llm(self, name, wrapper_type):
            assert str(name) == "nim_review"
            class LLM:
                async def ainvoke(self, messages):
                    return SimpleNamespace(content="계획 데이터 기반 검토 메모 초안")
            return LLM()
    async def run():
        cfg = reviewer.BoundedReviewerConfig(llm_name="nim_review")
        async with reviewer.bounded_reviewer(cfg, Builder()) as info:
            fn = LambdaFunction.from_info(config=cfg, info=info)
            result = await fn.ainvoke(reviewer.ReviewInput(plan_id=pid))
            assert result["status"] == "passed"
    asyncio.run(run())


def test_manual_review_recovers_unreviewed_and_failed_plans(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "OUT_DIR", tmp_path)
    s = store_module.Store()
    svc = PlanningService(s)
    class Reply:
        def __init__(self, text):
            self.text, self.calls = text, 0
        async def ainvoke(self, messages):
            self.calls += 1
            return SimpleNamespace(content=self.text)
    async def run():
        # The agent skipped the reviewer: the plan stays not_requested until the dentist asks.
        skipped = svc.stages(svc.target("moderate", "expansion_ipr", Constraints()))
        assert (await review_plan(skipped, Reply("검토 메모 초안"), store=s, manual=True))["status"] == "passed"
        # The agent path keeps the stored failure; the dentist's request reviews again with a fresh budget.
        failed = svc.stages(svc.target("moderate", "expansion_ipr", Constraints()))
        context = PlanRun("review-run", "moderate", None, Constraints(), selected_plan_id=failed)
        token = CURRENT_RUN.set(context)
        try:
            assert (await review_plan(failed, Reply(""), store=s))["status"] == "failed"
            again = Reply("검토 메모 초안")
            assert (await review_plan(failed, again, store=s))["status"] == "failed" and again.calls == 0
            # A closed or unrelated request context does not block the dentist's request.
            context.closed = True
            result = await review_plan(failed, again, store=s, manual=True)
        finally:
            CURRENT_RUN.reset(token)
        assert result["status"] == "passed" and again.calls == 1
        s.approve(failed, s.lock_fdi(failed))
        # Finished reviews are not run again.
        after = Reply("다른 메모")
        assert (await review_plan(failed, after, store=s, manual=True))["message"] == "검토 메모 초안"
        assert after.calls == 0
    asyncio.run(run())


def test_worker_manual_review_uses_workflow_reviewer(tmp_path, monkeypatch):
    from cualign.server.worker import manual_review
    monkeypatch.setattr(store_module, "OUT_DIR", tmp_path)
    svc = PlanningService(store_module.STORE)
    pid = svc.stages(svc.target("moderate", "expansion_ipr", Constraints()))
    class Builder:
        def get_function_config(self, name):
            from cualign.agent.reviewer import BoundedReviewerConfig
            assert name == "reviewer"
            return BoundedReviewerConfig(llm_name="nim_review", max_attempts=1)
        async def get_llm(self, name, wrapper_type):
            assert str(name) == "nim_review"
            class LLM:
                calls = 0
                async def ainvoke(self, messages):
                    LLM.calls += 1
                    return SimpleNamespace(content="")
            return LLM()
    async def run():
        review = await manual_review(Builder())
        result = await review(pid)
        assert result["status"] == "failed" and result["attempts"] == 1  # the workflow's max_attempts, not the default
    asyncio.run(run())


@pytest.mark.parametrize("exc,label", [(Exception("[429] Too Many Requests"), "upstream_429"),
                                       (Exception("[502] Bad Gateway"), "upstream_502"),
                                       (Exception("[400] Bad Request"), "model_error"),
                                       (RuntimeError("503 overloaded"), "upstream_503"),
                                       (ValueError("parse failed"), "model_error")])
def test_upstream_error_labels(exc, label):
    from cualign.agent.reviewer import upstream_error
    assert upstream_error(exc) == label


def test_reviewer_instructions_come_from_the_workflow_config(tmp_path, monkeypatch):
    """The review model's system message is workflow.yml `functions.reviewer.instructions` (the wording rules live
    there); without the key it is the old built-in text. 2026-09-27 golden set A: memos written with field names
    (per_stage_mm, space_deficit_mm) were copied into the dentist's answer (G-plain-answer)."""
    import yaml
    from pathlib import Path
    from cualign.agent import reviewer
    from nat.builder.function import LambdaFunction
    monkeypatch.setattr(store_module, "OUT_DIR", tmp_path)
    s = store_module.STORE
    svc = PlanningService(s)
    pid = svc.stages(svc.target("moderate", "expansion_ipr", Constraints()))
    class Capture:
        messages = None
        async def ainvoke(self, messages):
            Capture.messages = messages
            return SimpleNamespace(content="검토 메모 초안")
    class Builder:
        async def get_llm(self, name, wrapper_type):
            return Capture()
    async def run(cfg):
        async with reviewer.bounded_reviewer(cfg, Builder()) as info:
            fn = LambdaFunction.from_info(config=cfg, info=info)
            assert (await fn.ainvoke(reviewer.ReviewInput(plan_id=pid)))["status"] == "passed"
        return Capture.messages[0]["content"]
    wf = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "workflow.yml").read_text(encoding="utf-8"))
    configured = wf["functions"]["reviewer"]["instructions"]
    assert "단계당 이동량" in configured and "never write a field name" in configured   # the rule this test exists for
    # 2026-09-27 golden set A: with thinking on, the review reasoned past max_tokens and the cut reasoning came back as
    # the memo (8 of 15 runs). The memo is a transcription of computed fields; it needs no reasoning channel.
    assert wf["llms"][wf["functions"]["reviewer"]["llm_name"]]["chat_template_kwargs"]["enable_thinking"] is False
    assert asyncio.run(run(reviewer.BoundedReviewerConfig(llm_name="nim_review", instructions=configured))) == configured
    s.plans[pid]["review"]["status"] = "not_requested"   # a stored review is returned as-is; ask again
    assert asyncio.run(run(reviewer.BoundedReviewerConfig(llm_name="nim_review"))) == reviewer.DEFAULT_INSTRUCTIONS
    assert "field_notes" in reviewer.DEFAULT_INSTRUCTIONS and "단계당 이동량" not in reviewer.DEFAULT_INSTRUCTIONS


def test_reviewer_gets_what_each_number_means(tmp_path, monkeypatch):
    """A live memo called the total movement of one tooth "per stage" and the tooth-width sum a space need."""
    from cualign.agent.reviewer import FIELD_NOTES
    from cualign.core import limits
    monkeypatch.setattr(store_module, "OUT_DIR", tmp_path)
    s = store_module.Store()
    svc = PlanningService(s)
    pid = svc.stages(svc.target("severe", "extraction", Constraints(extraction=(5, 12))))
    plan = s.plan_json(pid)
    # Every noted field exists in the data the reviewer reads, so a rename cannot leave a note behind.
    assert set(FIELD_NOTES) <= set(plan["info"]) | set(plan["target"])
    assert str(limits.MAX_LINEAR_PER_ALIGNER) in FIELD_NOTES["per_stage_mm"]
    assert "TOTAL" in FIELD_NOTES["max_move_mm"] and "not a space shortage" in FIELD_NOTES["needed_mm"]
    class Capture:
        messages = None
        async def ainvoke(self, messages):
            Capture.messages = messages
            return SimpleNamespace(content="검토 메모 초안")
    assert asyncio.run(review_plan(pid, Capture(), store=s, manual=True))["status"] == "passed"
    system, user = Capture.messages
    assert "field_notes" in system["content"]
    sent = json.loads(user["content"])
    assert sent["field_notes"] == FIELD_NOTES
    assert sent["plan"]["plan_id"] == pid and "stages" not in sent["plan"] and "approval" not in sent["plan"]
    # #113: the reviewer reads FDI numbers, converted by the server (Universal 5, 12 -> FDI 14, 24), not by the model
    assert sent["plan"]["constraints"]["extraction"] == [14, 24] and sent["plan"]["target"]["removed"] == [14, 24]
    assert plan["constraints"]["extraction"] == [5, 12]                       # the stored plan keeps Universal
    assert all(11 <= int(t) <= 28 for t in sent["plan"]["target"]["rotation_deg"])


def test_reviewer_instructions_name_every_field_in_korean(tmp_path, monkeypatch):
    """#106: a live memo wrote `shape_room_mm` to the dentist. The configured instructions must give a Korean name for
    every measured field the reviewer reads (info/target keys ending in _mm/_deg/_teeth) and every violation type,
    and forbid snake_case tokens outright; each field note must start with that Korean name."""
    import re
    import yaml
    from pathlib import Path
    from cualign.agent.reviewer import FIELD_NOTES
    root = Path(__file__).resolve().parents[1]
    wf = yaml.safe_load((root / "configs" / "workflow.yml").read_text(encoding="utf-8"))
    text = wf["functions"]["reviewer"]["instructions"]
    assert "never write a field name" in text and "_mm" in text and "_deg" in text
    named = set(re.findall(r"([a-z][a-z0-9_/]*) -> ", text))
    monkeypatch.setattr(store_module, "OUT_DIR", tmp_path)
    s = store_module.Store()
    svc = PlanningService(s)
    pid = svc.stages(svc.target("severe", "extraction", Constraints(extraction=(5, 12))))
    plan = s.plan_json(pid)
    measured = {k for k in set(plan["info"]) | set(plan["target"]) if k.endswith(("_mm", "_deg", "_teeth"))}
    assert measured, "no measured fields in the plan data"
    assert measured <= named, f"fields without a Korean name in the reviewer instructions: {sorted(measured - named)}"
    source = (root / "src" / "cualign" / "core" / "planner.py").read_text(encoding="utf-8")
    kinds = set(re.findall(r'"type": "([a-z_]+)"', source))
    assert kinds and kinds <= named, f"violation types without a Korean name: {sorted(kinds - named)}"
    assert measured <= set(FIELD_NOTES), f"measured fields without a field note: {sorted(measured - set(FIELD_NOTES))}"
    for key, note in FIELD_NOTES.items():
        assert re.match(r"^[가-힣IPR][^:]*: ", note), f"field note for {key} must start with its Korean name"
