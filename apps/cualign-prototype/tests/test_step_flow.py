"""The agent pushes the plan one step per turn (agent/steps.py, .report/15-step-flow.md).

Through the real NAT app with a local fake planner: a setup turn may only set conditions, a target turn may only make
the target arrangement, a stages turn does the rest and may take the target the previous turn made. A tool past the
step is refused with a normal result (the model reads it and stops), each finished step ends in a `step_done` SSE
event, and Store.flow keeps where the case stands for /activate."""
import json
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rails_fakes import FakeLLM, PlanningLLM
from test_rails_middleware import ask, plain, serve, sse_event, store  # noqa: F401  (store is a fixture)
from cualign.agent import steps
from cualign.agent.context import CURRENT_RUN, PlanRun
from cualign.core import store as store_mod
from cualign.core.constraints import Constraints
from cualign.core.service import PlanningService
from cualign.server import api

PRESCRIPTION = "처방은 제1소구치 14·24 발치입니다. 이 처방으로 단계 계획을 짜 주세요."


def _call(n, name, args):
    return {"id": f"call_{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class SetupLLM(FakeLLM):
    """A setup turn that oversteps: set_constraints, then propose_target (refused), then the question back."""

    def reply(self, req):
        n = len(self.requests)
        if n == 1:
            return {"role": "assistant", "content": "", "tool_calls": [_call(n, "cualign__set_constraints", {"extraction": [14, 24]})]}   # FDI, as said
        if n == 2:
            return {"role": "assistant", "content": "", "tool_calls": [_call(n, "cualign__propose_target", {"strategy": "extraction"})]}
        return {"role": "assistant", "content": "14·24 발치, IPR 없음으로 셋업했습니다. 이대로 목표 배열을 만들까요?"}


class TargetLLM(FakeLLM):
    """A target turn that oversteps: propose_target, then plan_stages (refused), then reviewer (refused), then the question."""

    def reply(self, req):
        n = len(self.requests)
        seen = json.dumps(req.get("messages", []), ensure_ascii=False)
        target = re.findall(r"target_id\W+(t[0-9a-f]+)", seen)
        if n == 1:
            return {"role": "assistant", "content": "", "tool_calls": [_call(n, "cualign__propose_target", {"strategy": "expansion_ipr"})]}
        if n == 2:
            return {"role": "assistant", "content": "", "tool_calls": [_call(n, "cualign__plan_stages", {"target_id": target[-1]})]}
        if n == 3:
            return {"role": "assistant", "content": "", "tool_calls": [_call(n, "reviewer", {"plan_id": "p000000"})]}
        return {"role": "assistant", "content": "목표 배열을 만들었습니다. 확장과 IPR 로 총생을 해소합니다. 단계로 나눌까요?"}


def _refusals(llm, name):
    """The refusal results the fake model received for `name` (NAT's ReAct loop hands tool results back as user text)."""
    return [m["content"] for req in llm.requests for m in req.get("messages", [])
            if m.get("role") == "user" and steps.REFUSED in str(m.get("content")) and f"'rejected': '{name}'" in str(m.get("content"))]


def _context(llm):
    """The cuAlign server context the first request carried (inside the ReAct prompt's conversation history)."""
    for m in llm.requests[0]["messages"]:
        content = str(m.get("content"))
        if "cuAlign server context: " in content:
            return json.loads(content.split("cuAlign server context: ", 1)[1].splitlines()[0])
    raise AssertionError("no server context in the first request")


def test_gating_table():
    assert steps.allowed("setup", "set_constraints") and steps.allowed("setup", "load_case")
    assert not steps.allowed("setup", "propose_target") and not steps.allowed("setup", "reviewer")
    assert steps.allowed("target", "propose_target") and not steps.allowed("target", "plan_stages")
    assert steps.allowed("stages", "reviewer") and steps.allowed(None, "export_stl")   # no step: everything, as before
    note = steps.refusal("setup", "propose_target")
    assert note["rejected"] == "propose_target" and steps.REFUSED in note["note"] and "되물으세요" in note["note"]


def test_setup_turn_sets_conditions_and_stops(tmp_path, monkeypatch, store):
    with SetupLLM() as llm, serve(tmp_path, monkeypatch, llm) as client:
        text = ask(client, "/chat/stream", [{"role": "user", "content": PRESCRIPTION}],
                   cualign={"case_id": "moderate", "request_id": "r-setup"}, step="setup", step_source="chip")
    done = sse_event(text, "step_done")
    assert done["step"] == "setup" and done["request_id"] == "r-setup" and done["case_id"] == "moderate"
    assert done["constraints"]["extraction"] == [5, 12] and done["constraints"]["stage_cap"] is None   # Universal, as plan_context
    assert done["conditions_ko"].startswith("발치 치아 14·24번") or "14" in done["conditions_ko"]
    assert sse_event(text, "plan_context")["constraints"] == {**done["constraints"], "ipr_surfaces": []}   # (12): the event fills the uniform contacts
    assert done["constraints"]["ipr_surfaces"] == []   # an extraction case: the IPR strategies are off, no contact to cut
    assert "plan_selected" not in text and "plan_error" not in text and "이대로 목표 배열을 만들까요?" in text
    # the overstep was refused as a normal result and nothing was computed
    assert _refusals(llm, "propose_target")
    assert not [t for t in store.targets.values() if t["case_id"] == "moderate"] and store.plan_ids_for("moderate") == []
    assert store.flow["moderate"]["step"] == "setup" and store.flow["moderate"]["constraints"].extraction == (5, 12)
    # the setup turn's context told the model its step
    context = _context(llm)
    assert context["step"] == "setup" and "셋업" in context["step_ko"]


def test_target_turn_makes_the_target_and_stops(tmp_path, monkeypatch, store):
    with TargetLLM() as llm, serve(tmp_path, monkeypatch, llm) as client:
        text = ask(client, "/chat/stream", [{"role": "user", "content": "이대로 목표 배열을 만들어줘."}],
                   cualign={"case_id": "moderate", "request_id": "r-target"}, step="target")
        done = sse_event(text, "step_done")
        assert done["step"] == "target" and done["request_id"] == "r-target"
        tid = done["target_id"]
        assert tid in store.targets and store.targets[tid]["case_id"] == "moderate"
        summary = done["summary"]
        assert summary["strategy"] == "expansion_ipr" and summary["crowding_mm"] > 0 and summary["space_mm"] is not None
        assert summary["extraction"] == [] and all(len(p) == 3 and 11 <= p[0] <= 28 for p in summary["ipr"])
        assert "plan_selected" not in text and "plan_error" not in text and "단계로 나눌까요?" in text
        assert store.plan_ids_for("moderate") == [] and store.flow["moderate"] == {"step": "target", "constraints": store.flow["moderate"]["constraints"],
                                                                                    "target_id": tid, "plan_id": None}
        assert _refusals(llm, "plan_stages") and _refusals(llm, "reviewer")
        # the target as the screen draws it: a plan-shaped response with one stage
        t = client.get(f"/api/cases/moderate/targets/{tid}").json()
        assert t["target_id"] == tid and t["strategy"] == "expansion_ipr" and t["violations"] == [] and t["passed"] is True
        assert len(t["stages"]) == 1 and len(t["rotations"]) == 1 and t["pivots"] and t["summary"] == summary
        assert set(t["stages"][0]) and all(len(v) == 3 for v in t["stages"][0].values())
        assert t["target"]["removed"] == [] and "ipr_mm_per_surface" in t["target"] and "ipr_exclude" in t["target"]
        assert t["constraints"]["extraction"] == []
        assert client.get(f"/api/cases/mild/targets/{tid}").status_code == 404 and client.get("/api/cases/moderate/targets/t0").status_code == 404
        # opening the case again restores where it stands
        opened = client.post("/api/cases/moderate/activate").json()
        assert opened["flow"]["step"] == "target" and opened["flow"]["target_id"] == tid and opened["active_plan"] is None


def test_stages_turn_takes_the_target_the_previous_turn_made(tmp_path, monkeypatch, store):
    """After a target turn the stages turn's context carries target_id; the fake planner (like the instructions say)
    calls plan_stages with it, so the plan is built on that very target and no new one is made."""
    cid, case = store.load_case("moderate")
    tid = PlanningService(store).target(cid, "expansion", store.constraints_for(cid))
    store.set_flow(cid, "target", constraints=store.constraints_for(cid), target_id=tid, plan_id=None)
    with PlanningLLM("**확장 전략으로 계획을 만들었습니다.** 이 계획은 초안입니다. 최종 판단은 의사가 합니다.") as llm, \
            serve(tmp_path, monkeypatch, llm) as client:
        text = ask(client, "/chat/stream", [{"role": "user", "content": "단계로 나눠줘."}],
                   cualign={"case_id": "moderate", "request_id": "r-stages"}, step="stages")
    context = _context(llm)
    assert context["step"] == "stages" and context["target_id"] == tid
    selected = sse_event(text, "plan_selected")
    done = sse_event(text, "step_done")
    assert done == {"step": "stages", "request_id": "r-stages", "case_id": "moderate", "plan_id": selected["plan_id"], "target_id": tid}
    assert store.plans[selected["plan_id"]]["target_id"] == tid
    assert [t for t in store.targets if store.targets[t]["case_id"] == cid] == [tid]      # no new target
    assert store.flow[cid] == {"step": "stages", "constraints": store.flow[cid]["constraints"], "target_id": tid, "plan_id": selected["plan_id"]}
    # a turn without a step is a stages turn, as before (the fake makes its own target then)
    with PlanningLLM("**확장 전략으로 계획을 만들었습니다.** 이 계획은 초안입니다. 최종 판단은 의사가 합니다.") as llm, \
            serve(tmp_path, monkeypatch, llm) as client:
        text = ask(client, "/chat/stream", [{"role": "user", "content": "계획 짜줘."}], cualign={"case_id": "mild"})
    assert sse_event(text, "step_done")["step"] == "stages" and sse_event(text, "plan_selected")
    assert _context(llm)["step"] == "stages"


def test_the_intent_words_table():
    """The words that name a step, checked in this order (steps.INTENT_WORDS): stages, then target, then setup."""
    assert [(step, words.pattern) for step, words in steps.INTENT_WORDS] == [
        ("stages", r"단계|장수|\d+\s*개월\s*안에|비교"),
        ("target", r"목표|배열\s*을?\s*다시|target"),
        ("setup", r"조건|처방|발치|IPR\s*한도|셋업"),
    ]


@pytest.mark.parametrize("text, sent, chip, want", [
    ("목표배열 다시 만들어줘", "stages", False, "target"),                 # the screen sent the step after its progress
    ("발치 없이 IPR 한도 0.3mm 로 바꿔줘", "target", False, "setup"),       # condition words only
    ("8개월 안에 끝나게 다시 짜줘", "target", False, "stages"),
    ("좋아요, 그렇게 해 주세요", "target", False, "target"),               # no word: the step sent (the flow's)
    ("이 목표로 단계를 만들어줘.", "target", True, "target"),              # a chip's step stands
])
def test_a_free_sentence_names_its_step(text, sent, chip, want):
    assert steps.turn_step(text, sent, chip) == want


def test_intent_details_and_the_replay_step():
    assert steps.intent("이 목표로 단계를 만들어줘.") == "stages"          # stages words win over the target's
    assert steps.intent("발치로 목표 배열을 다시 만들어줘") == "target"     # a target sentence that changes a condition
    assert steps.intent("배열을 다시 잡아줘") == "target" and steps.intent("make a new Target") == "target"
    assert steps.intent("안녕하세요") is None and steps.turn_step("안녕하세요", None) == steps.DEFAULT_STEP
    # the recording a turn replays (core/recorded.py) comes from the same rule
    from cualign.core import recorded, samples
    assert recorded.replay_step("목표배열 다시 만들어줘", "stages") == "target"
    assert recorded.replay_step("목표배열 다시 만들어줘", "stages", as_sent=True) == "stages"   # what the screen asks for
    assert recorded.replay_step("12개월 안에 끝나게 단계를 만들어줘.", "stages") == "cap"
    assert recorded.replay_step("확장안이랑 IPR안 둘 다 만들어서 비교해줘.", "stages") == "compare"
    # the chips' sentences keep the chip's step: the sample prescription is a setup turn whatever its words
    assert recorded.chip(samples.SAMPLES["poseidon-000097"].request) and not recorded.chip("목표배열 다시 만들어줘")
    assert recorded.replay_step(samples.SAMPLES["poseidon-000097"].request, "setup") == "setup"


def test_a_target_sentence_remakes_the_target_over_a_manual_one(tmp_path, monkeypatch, store):
    """000001 (step-intent): a target on screen — the dentist's own moves, auto-saved — and the free sentence
    「목표배열 다시 만들어줘」, sent by the screen as a stages turn. The server runs it as a target turn: propose_target
    makes a new target, which becomes the flow's; the manual one stays in the list."""
    cid, case = store.load_case("moderate")
    manual = PlanningService(store).target(cid, "expansion", store.constraints_for(cid))
    store.targets[manual]["info"]["source"] = "manual"
    store.set_flow(cid, "target", constraints=store.constraints_for(cid), target_id=manual, plan_id=None)
    with TargetLLM() as llm, serve(tmp_path, monkeypatch, llm) as client:
        res = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "목표배열 다시 만들어줘"}],
                                                 "cualign": {"case_id": cid, "request_id": "r-again"}, "step": "stages"})
    text = plain(res.text)
    assert res.status_code == 200 and res.headers["x-cualign-step"] == "target"   # before any event: 건너뛰기 reads it
    context = _context(llm)
    assert context["step"] == "target" and "target_id" not in context and "target_manual_ko" not in context
    assert sse_event(text, "plan_context")["step"] == "target"
    done = sse_event(text, "step_done")
    assert done["step"] == "target" and done["request_id"] == "r-again" and done["target_id"] != manual
    assert store.flow[cid]["target_id"] == done["target_id"] and manual in store.targets   # the manual target stays listed
    assert "plan_selected" not in text and store.plan_ids_for(cid) == [] and _refusals(llm, "plan_stages")
    assert store.turn_steps[cid] == {"screen": "stages", "turn": "target"}


def test_restart_forgets_the_flow_and_the_conditions_but_keeps_the_plans(store):
    """「처음부터」: after a setup (and a plan), POST /restart drops Store.flow and the case's conditions back to
    Constraints() — not the prescription — and answers like /activate; the plans stay for 지난 계획."""
    app = FastAPI()
    api.add_api_routes(app)
    client = TestClient(app)
    cid = "moderate"
    pid = api.rule_based_plan(cid, extraction=[5, 12])["chosen"]["plan_id"]
    store.set_flow(cid, "setup", constraints=store.constraints_for(cid), target_id=None, plan_id=None)
    assert client.post(f"/api/cases/{cid}/activate").json()["flow"]["step"] == "setup"
    opened = client.post(f"/api/cases/{cid}/restart").json()
    assert opened["case_id"] == cid and opened["flow"] is None and cid not in store.flow
    assert opened["constraints"] == Constraints().prescription().model_dump(mode="json")
    assert not store.constraints_for(cid).extraction and pid in store.plans and pid in store.plan_ids_for(cid)
    assert client.post(f"/api/cases/{cid}/activate").json()["flow"] is None      # opening it again stays at 초기
    assert client.post("/api/cases/nope/restart").status_code == 404


def test_a_bad_step_is_400():
    from cualign.server.plan_events import ChatContext
    with pytest.raises(ValueError):
        ChatContext(case_id="moderate", step="export")


def test_reviewer_is_refused_before_the_stages_step(store):
    import asyncio
    from cualign.agent.reviewer import review_plan
    cid, _ = store.load_case("moderate")
    ids = PlanningService(store).compare(cid, store.constraints_for(cid), ["expansion"])
    run = PlanRun("r", cid, None, store.constraints_for(cid), step="target", selected_plan_id=ids[0])
    token = CURRENT_RUN.set(run)
    try:
        out = asyncio.run(review_plan(ids[0], llm=None, store=store))
    finally:
        CURRENT_RUN.reset(token)
    assert out["rejected"] == "reviewer" and steps.REFUSED in out["note"]
    assert store.plans[ids[0]]["review"]["status"] == "not_requested"


def test_target_summary_reads_the_planner_info():
    info = {"crowding_mm": 4.2, "space_gain_mm": 3.9, "space_deficit_mm": 0.3, "strategy": "ipr", "extraction": [],
            "ipr_applied_teeth": [6, 7, 8, 9, 12], "ipr_mm_per_surface": 0.25, "expansion_mm_per_side": 0.0}
    s = steps.target_summary(info)
    assert s["ipr"] == [[13, 12, 0.25], [12, 11, 0.25], [11, 21, 0.25]] and s["extraction"] == [] and s["space_mm"] == 3.9
    s = steps.target_summary({**info, "strategy": "extraction", "extraction": [5, 12], "ipr_mm_per_surface": 0.0})
    assert s["extraction"] == [14, 24] and s["ipr"] == []


def test_setup_view_fills_the_ipr_surfaces_the_uniform_rule_would_cut(store):
    """Step flow (12): without a prescribed contact list the setup event carries the contacts the planner's uniform
    rule cuts (Universal [[a, b, mm]]), the same ones an IPR target reports; a prescription is passed as given."""
    from cualign.core import planner
    from cualign.server.plan_events import setup_view
    cid, case = store.load_case("moderate")
    c = store.constraints_for(cid)
    view = setup_view(store, cid, c)
    _, info = planner.propose_target(case, "ipr", constraints=c)
    assert view["constraints"]["ipr_surfaces"] == info["ipr_surfaces"] and view["constraints"]["ipr_surfaces"]
    assert all(b == a + 1 and mm > 0 for a, b, mm in view["constraints"]["ipr_surfaces"])
    assert {k: v for k, v in view["constraints"].items() if k != "ipr_surfaces"} == {k: v for k, v in c.model_dump(mode="json").items() if k != "ipr_surfaces"}
    excl = c.patched({"ipr_exclude": [8, 9]})
    assert not [s for s in setup_view(store, cid, excl)["constraints"]["ipr_surfaces"] if s[0] == 8 and s[1] == 9]   # 8|9 excluded on both sides
    given = c.patched({"ipr_surfaces": [[7, 8, 0.3], [8, 9, 0.3]]})
    assert setup_view(store, cid, given)["constraints"]["ipr_surfaces"] == [[7, 8, 0.3], [8, 9, 0.3]]
    assert view["conditions_ko"] == c.describe_ko()



def test_setup_right_click_changes_the_prescription_and_the_setup_cut(store):
    """직접 이동 on 셋업 (manual_api.setup_conditions): 발치 / IPR per tooth face and their 취소 go through
    Constraints.patched into the case's conditions and the flow's setup, the answer is step_done setup's, /setup/cut
    cuts the face as given, and the next turn's plan_context starts from the changed conditions."""
    from cualign.server.plan_events import ChatContext, open_run
    app = FastAPI()
    api.add_api_routes(app)
    client = TestClient(app)
    cid = "poseidon-000097"                                          # 14·24 (Universal 5, 12) prescribed
    edit = lambda **body: client.post(f"/api/cases/{cid}/setup/conditions", json=body)   # noqa: E731
    assert edit(op="extract", tooth=15).status_code == 409           # before the setup there is nothing to change
    store.set_flow(cid, "setup", constraints=store.constraints_for(cid), target_id=None, plan_id=None)
    assert client.get(f"/api/cases/{cid}/setup/cut").json()["teeth_cut"] == {}

    res = edit(op="ipr", tooth=12, face="distal", mm=0.2)            # FDI 12 (U7) distal = the 12-13 contact (U6-7)
    assert res.status_code == 200, res.text
    done = res.json()
    assert done["step"] == "setup" and "conditions_ko" in done
    assert done["constraints"]["ipr_surfaces"] == [[6, 7, 0.2]] and done["constraints"]["ipr_amounts"] == [[7, 6, 0.2]]
    assert store.flow[cid]["constraints"].ipr_amounts == ((7, 6, 0.2),) and store.flow[cid]["target_id"] is None
    cut = client.get(f"/api/cases/{cid}/setup/cut").json()
    assert set(cut["teeth_cut"]) == set(cut["ipr_cut"]) == {"7"} and cut["ipr_cut"]["7"]["mm"] == 0.2   # that face only
    assert 0 < len(cut["ipr_cut"]["7"]["faces"]) < len(cut["teeth_cut"]["7"]["f"])

    both = edit(op="ipr", tooth=13, face="mesial", mm=0.2).json()   # the other face of the same contact, as much: half each
    assert both["constraints"]["ipr_surfaces"] == [[6, 7, 0.4]] and "ipr_amounts" not in both["constraints"]
    assert set(client.get(f"/api/cases/{cid}/setup/cut").json()["teeth_cut"]) == {"6", "7"}
    left = edit(op="unipr", tooth=13).json()                         # 13's face off, 12's stays
    assert left["constraints"]["ipr_surfaces"] == [[6, 7, 0.2]] and left["constraints"]["ipr_amounts"] == [[7, 6, 0.2]]
    assert edit(op="unipr", tooth=12).json()["constraints"]["ipr_surfaces"] == []
    assert client.get(f"/api/cases/{cid}/setup/cut").json()["teeth_cut"] == {}
    assert "IPR 이 없습니다" in edit(op="unipr", tooth=12).json()["detail"]

    assert edit(op="extract", tooth=15).json()["constraints"]["extraction"] == [4, 5, 12]
    assert store.constraints_for(cid).extraction == (4, 5, 12)
    assert "이웃 치아가 없습니다" in edit(op="ipr", tooth=16, face="mesial", mm=0.25).json()["detail"]   # 15 is extracted
    assert edit(op="unextract", tooth=15).json()["constraints"]["extraction"] == [5, 12]
    assert "소구치" in edit(op="extract", tooth=13).json()["detail"]
    assert edit(op="ipr", tooth=12, face="distal", mm=0.3).status_code == 422          # over 0.25 per face
    assert "0.05mm 단위" in edit(op="ipr", tooth=12, face="distal", mm=0.12).json()["detail"]

    edit(op="ipr", tooth=11, face="mesial", mm=0.15)
    run, _ = open_run(ChatContext(case_id=cid, step="target"), store=store)   # the next turn, the form as it was
    assert run.constraints.extraction == (5, 12) and run.constraints.ipr_surfaces == ((8, 9, 0.15),)
    assert run.constraints.model_dump(mode="json")["ipr_amounts"] == [[8, 9, 0.15]]   # as the plan_context event dumps it
