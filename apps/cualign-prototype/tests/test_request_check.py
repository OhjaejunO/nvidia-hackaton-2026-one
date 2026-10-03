"""set_constraints takes the dentist's FDI numbers (a) and every plan is checked against what was asked on its case (b).

A live revision «13번 치아는 움직이지 말고 다시 짜줘.» locked FDI 25 (app tooth 13): the model passed 13 to a tool that took
app numbers, and the rules, the reviewer and the rails all passed the plan. (a) moves the conversion from the model to
the server, at the tool input only: the UI form, the recorded answers and the MCP tools keep ConstraintPatch (app
numbers) and Tooth keeps 2..15. (b) blocks approval unless the case's sentences and the plan's conditions match: three
results (match / mismatch / needs_check, for what was not read), every turn's sentence on the case in order (setup turns
too) and the dentist's hand edits, and approval only with the locked teeth (FDI) confirmed on the screen (round 4,
after Codex re-check 2026-10-03 (3)).
"""
import asyncio
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cualign.agent import register
from cualign.agent.context import CURRENT_RUN, PlanRun
from cualign.core import recorded, request_check
from cualign.core import store as store_module
from cualign.core.constraints import ConstraintPatch, Constraints, ToolConstraintPatch
from cualign.core.request_check import ask, check_case, edit
from cualign.core.service import PlanningService
from cualign.server import api, plan_events
from cualign.server.plan_events import ChatContext, open_run

REVISION = "13번 치아는 움직이지 말고 다시 짜줘."
SAMPLES = ("poseidon-000001", "poseidon-000097", "poseidon-000131")
REVIEW = {"status": "passed", "attempts": 1, "message": "memo", "error": None}
PROPOSE = ("propose_target", {"strategy": "expansion_ipr"})


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    s = store_module.Store()
    monkeypatch.setattr(store_module, "OUT_DIR", tmp_path)
    monkeypatch.setattr(api, "OUT_DIR", tmp_path)
    monkeypatch.setattr(api, "STORE", s)
    monkeypatch.setattr(register, "STORE", s)
    monkeypatch.setattr(plan_events, "STORE", s)
    return s


def client(review=None):
    app = FastAPI()
    api.add_api_routes(app, review=review)
    return TestClient(app)


def run_tools(calls, run):
    """The agent's tools called in order under `run`; an argument may be a function of the results so far."""
    async def scenario():
        async with register.cualign(register.CuAlignToolConfig(), None) as group:
            fns = {k.split("__")[-1]: f for k, f in (await group.get_all_functions()).items()}
            token = CURRENT_RUN.set(run)
            try:
                out = []
                for name, arg in calls:
                    out.append(await fns[name].ainvoke(arg(out) if callable(arg) else arg))
                return out
            finally:
                CURRENT_RUN.reset(token)
    return asyncio.run(scenario())


def C(**changes):
    return Constraints().patched(changes)


def turn(text, calls=(), step=None, form=None):
    """A /chat/stream turn on "moderate" as PlanEventsASGI opens it (open_run applies the form, the case keeps the
    sentence) and the agent's tools under it: their results."""
    ctx = ChatContext(case_id="moderate", constraints=form or {}, **({"step": step} if step else {}))
    run, _ = open_run(ctx, store=register.STORE)
    register.STORE.add_request(run.case_id, text, run.opened)
    return run_tools(list(calls), run)


def turn_target(lock, stage=False):
    """A turn on REVISION that locks `lock` (FDI) and makes one target: its target id, or with `stage` the plan the
    turn stages from it."""
    out = turn(REVISION, [("set_constraints", {"lock": lock}), PROPOSE] +
               [("plan_stages", lambda out: {"target_id": out[1]["target_id"]})] * stage)
    return out[2]["plan_id"] if stage else out[1]["target_id"]


def target_turn(text, calls=()):
    """A target turn that makes a target; the flow keeps it, so the next stages turn may stage it (open_run)."""
    tid = turn(text, [*calls, PROPOSE], step="target")[-1]["target_id"]
    register.STORE.set_flow("moderate", "target", constraints=register.STORE.targets[tid]["constraints"], target_id=tid,
                            plan_id=None)
    return tid


def staged(text, lock=None, step=None):
    """A turn on `text` that sets `lock` (FDI; None: the model leaves it out), makes a target and stages it: the plan."""
    calls = [("set_constraints", {"lock": lock})] * (lock is not None) + [PROPOSE]
    out = turn(text, calls + [("plan_stages", lambda out: {"target_id": out[-1]["target_id"]})], step=step)
    return out[-1]["plan_id"]


def approve(c, pid, lock_fdi="plan"):
    """POST …/approval as the screen sends it: the locked teeth the dentist ticked (the plan's, unless given)."""
    body = {"confirmed": True}
    if lock_fdi is not None:
        body["lock_fdi"] = api.STORE.lock_fdi(pid) if lock_fdi == "plan" else lock_fdi
    return c.post(f"/api/plans/{pid}/approval", json=body)



# ---- (a) the tool takes FDI; everything behind it keeps app numbers -------------------------------------------------

def test_set_constraints_takes_fdi_and_stores_app_numbers(isolated):
    lock, excl, ext = run_tools([("set_constraints", {"lock": [13]}),
                                 ("set_constraints", {"ipr_exclude": [12, 11, 21, 22]}),
                                 ("set_constraints", {"extraction": [14, 24]})], PlanRun("fdi", "moderate", None, Constraints()))
    assert lock["lock"] == [6] and lock["teeth_fdi"]["lock"] == [13] and "고정 치아 13번" in lock["conditions_ko"]
    assert excl["ipr_exclude"] == [7, 8, 9, 10] and excl["lock"] == [6]           # omitted = keep
    assert excl["teeth_fdi"]["ipr_exclude"] == [12, 11, 21, 22]
    assert ext["extraction"] == [5, 12] and ext["teeth_fdi"]["extraction"] == [14, 24]
    assert isolated.case_constraints["moderate"].lock == (6,)


@pytest.mark.parametrize("teeth", [[6], [3], [19]])
def test_a_number_outside_the_fdi_upper_arch_is_refused_at_the_tool(isolated, teeth):
    # app numbers (6, 3) and FDI 19 are not a tooth the dentist reads: refused before the tool runs, nothing stored
    with pytest.raises(ValidationError):
        run_tools([("set_constraints", {"lock": teeth})], PlanRun("fdi", "moderate", None, Constraints()))
    assert "moderate" not in isolated.case_constraints


def test_the_tool_schema_asks_for_fdi():
    props = ToolConstraintPatch.model_json_schema()["properties"]
    for field in ("extraction", "lock", "ipr_exclude", "ipr_surfaces"):
        assert "FDI" in props[field]["description"], field


def test_shared_tooth_range_and_internal_patch_are_unchanged():   # acceptance (c): Tooth keeps 2..15
    assert ConstraintPatch(lock=[13]).lock == [13]                 # app numbers, no conversion
    for bad in (1, 16, 25):
        with pytest.raises(ValidationError):
            ConstraintPatch(lock=[bad])
        with pytest.raises(ValidationError):
            Constraints(lock=(bad,))
    assert Constraints(lock=(2, 15)).lock == (2, 15)
    assert ToolConstraintPatch(lock=[13]).lock == [13]              # kept as sent: a second validation must not reconvert
    assert ToolConstraintPatch(lock=[13]).changes() == {"lock": [6]} and ToolConstraintPatch(lock=[25]).changes() == {"lock": [13]}


def test_the_agent_path_converts_once(isolated):
    # The ReAct agent calls the tool through LangChain's StructuredTool, which validates the arguments with the input
    # schema and hands the validated values to NAT, which validates them again. Calling ainvoke directly validates once
    # and hid that e7fa32b refused every ipr_surfaces prescription on this path (FDI 11 -> 8 -> refused).
    from nat.plugins.langchain.tool_wrapper import langchain_tool_wrapper

    async def scenario():
        async with register.cualign(register.CuAlignToolConfig(), None) as group:
            name, fn = next((k, f) for k, f in (await group.get_all_functions()).items() if k.endswith("set_constraints"))
            tool, out = langchain_tool_wrapper(name, fn, None), []
            for args in ({"lock": [13]}, {"ipr_exclude": [12, 11, 21, 22]}, {"ipr_surfaces": [[11, 21, 0.4]]},
                         {"extraction": [14, 24]}):
                token = CURRENT_RUN.set(PlanRun("agent-path", "moderate", None, Constraints()))
                isolated.case_constraints.pop("moderate", None)
                try:
                    out.append(await tool.ainvoke(args))
                finally:
                    CURRENT_RUN.reset(token)
            return out
    lock, excl, surf, ext = asyncio.run(scenario())
    assert lock["lock"] == [6] and excl["ipr_exclude"] == [7, 8, 9, 10] and ext["extraction"] == [5, 12]
    assert surf["ipr_surfaces"] == [[8, 9, 0.4]] and surf["teeth_fdi"]["ipr_surfaces"] == [[11, 21, 0.4]]


def test_ui_form_and_chat_context_keep_app_numbers(isolated):     # acceptance (b): the UI path
    # state.js turns the form's FDI into app numbers before it sends (#113); the server reads them as they are
    ctx = ChatContext(case_id="moderate", constraints={"lock": [13], "ipr_exclude": [7, 8]})
    assert ctx.constraints.lock == [13]
    run, msg = open_run(ctx, store=isolated)
    context = json.loads(msg["content"].removeprefix("cuAlign server context: "))
    assert context["constraints"]["lock"] == [13] and context["teeth_fdi"]["lock"] == [25]
    assert context["teeth_fdi"]["ipr_exclude"] == [12, 11] and run.opened.lock == (13,)
    with client() as c:
        r = c.post("/api/plan", json={"case_id": "moderate", "lock": [13]}).json()
    assert (r["chosen"] or r["best_failed"])["constraints"]["lock"] == [13]


@pytest.mark.parametrize("case_id", SAMPLES)
def test_shipped_setup_recordings_replay_with_their_app_numbers(isolated, case_id):   # acceptance (a)
    rec = recorded.load(case_id, "setup")
    with client() as c:
        r = c.post(f"/api/cases/{case_id}/replay", json={"step": "setup"})
    assert r.status_code == 200, r.text
    stored = isolated.case_constraints[case_id]
    for field in ("extraction", "lock", "ipr_exclude"):
        assert list(getattr(stored, field)) == rec["constraints"][field], field


@pytest.mark.slow
@pytest.mark.parametrize("case_id", SAMPLES)
def test_every_shipped_recording_replays(isolated, case_id):     # acceptance (a): all 15 files, as the screen walks them
    with client() as c:
        base = None
        for step in recorded.STEPS:
            rec = recorded.load(case_id, step)
            if rec is None:
                continue
            r = c.post(f"/api/cases/{case_id}/replay", json={"step": step, "base_plan_id": base})
            assert r.status_code == 200, (step, r.text)
            sel = r.json().get("plan_selected")
            if sel:
                base = sel["plan_id"]
                got = c.get(f"/api/plans/{base}").json()["constraints"]
                for field in ("extraction", "lock", "ipr_exclude"):
                    assert got[field] == rec["constraints"].get(field, got[field]), (step, field)



# ---- (b) the sentence against the plan's conditions --------------------------------------------------------------

# The plan's four acceptance sentences (기획 ① ⓑ) word for word: «13번도» names no field after it and takes the one
# of the number before it in the same sentence («25번 고정에»).
PLAN_ADD = "기존 25번 고정에 13번도"


@pytest.mark.parametrize("text, before, after", [
    ("13번 고정", C(), C(lock=[6])),                                                   # app 6 added
    (PLAN_ADD, C(lock=[13]), C(lock=[6, 13])),                                        # 25 (app 13) kept, 6 added
    ("13번 고정 풀어줘", C(lock=[6, 13]), C(lock=[13])),                                # 6 released, the rest kept
    ("13번 고정하고 12개월 안에", C(), C(lock=[6], stage_cap=26)),                      # and the cap round(12*30.4/14)
    (REVISION, C(), C(lock=[6])),                                                     # the live revision, applied
    ("13번도 움직이지 마.", C(lock=[13]), C(lock=[6, 13])),                              # a full list keeps the old tooth
    ("14번 고정하고 24번도.", C(), C(lock=[5, 12])),                                     # «도» takes the field before it
    ("13번 고정 풀고 14번도.", C(lock=[5, 6, 13]), C(lock=[13])),                        # ... and its action (release)
    # «도» with a verb of its own takes that verb (Codex re-check 2026-10-03): 빼 releases, 고정 adds
    ("25번 고정하고 13번도 빼줘", C(lock=[13]), C(lock=[13])),                          # 13 released (it was not locked)
    ("25번 고정하고 13번도 빼줘", C(lock=[6, 13]), C(lock=[13])),
    ("25번 고정하고 13번도 고정", C(lock=[13]), C(lock=[6, 13])),
    ("25번 고정하고 13번도 같이 고정해줘", C(lock=[13]), C(lock=[6, 13])),
    ("발치 없이 12개월 안에 끝나는 계획 짜줘. 앞니 총생부터 풀고.", C(), C(stage_cap=26, order="anterior_first")),
    ("IPR 은 앞니 12,11,21,22번 빼고 해줘.", C(), C(ipr_exclude=[7, 8, 9, 10])),
    ("14번과 24번 발치로 계획 짜줘.", C(), C(extraction=[5, 12])),
    ("IPR 11-21·11-12 각 0.4mm 로 해줘.", C(), C()),                                    # contacts, not teeth
    ("단계로 나눠줘.", C(lock=[6]), C(lock=[6])),                                         # nothing asked, nothing changed
])
def test_requests_that_were_applied_are_not_blocked(text, before, after):
    assert request_check.check(text, before, after) == {"status": "match", "reasons_ko": [], "unread_ko": [],
                                                        "unchecked": []}


@pytest.mark.parametrize("text, before, after, reasons, unread", [
    (REVISION, C(), C(lock=[13]),                                                     # the live run: FDI 25 locked
     ["고정 치아에 요청한 13번이 없습니다", "고정 치아에 요청에 없던 25번이 들어갔습니다"], []),
    ("13번 고정 풀어줘", C(lock=[6, 13]), C(lock=[6, 13]), ["고정 치아에 요청에 없던 13번이 들어갔습니다"], []),
    (PLAN_ADD, C(lock=[13]), C(lock=[13]), ["고정 치아에 요청한 13번이 없습니다"], []),  # the addition left out
    (PLAN_ADD, C(lock=[13]), C(lock=[6]), ["고정 치아에 요청한 25번이 없습니다"], []),   # the old tooth dropped
    ("13번 고정하고 12개월 안에", C(), C(lock=[6]), ["요청한 12개월(단계 상한 26단계)과 저장된 단계 상한(없음)이 다릅니다"], []),
    ("14번 고정하고 24번도.", C(), C(lock=[5]), ["고정 치아에 요청한 24번이 없습니다"], []),
    ("25번 고정하고 13번도 빼줘", C(lock=[13]), C(lock=[6, 13]), ["고정 치아에 요청에 없던 13번이 들어갔습니다"], []),
    ("25번 고정하고 13번도 고정", C(lock=[13]), C(lock=[13]), ["고정 치아에 요청한 13번이 없습니다"], []),
    # a number not read adds nothing: a plan that has it anyway differs, and says what was not read
    ("25번 고정하고 13번도 봐줘", C(lock=[13]), C(lock=[6, 13]), ["고정 치아에 요청에 없던 13번이 들어갔습니다"],
     ["«13번도 봐줘»"]),
    ("25번 고정하고 13번도 빼지 마", C(lock=[6, 13]), C(lock=[13]), ["고정 치아에 요청한 13번이 없습니다"],
     ["«13번도 빼지 마»"]),                                                            # not a release, not read
    ("12개월 안에 끝나게 다시 짜줘.", C(stage_cap=26), C(lock=[6], stage_cap=26),
     ["고정 치아가 요청 문장에 없이 바뀌었습니다(없음 → 13번)"], []),
    ("14번과 24번 발치로 계획 짜줘.", C(), C(extraction=[4, 13]),
     ["발치 치아에 요청한 14·24번이 없습니다", "발치 치아에 요청에 없던 15·25번이 들어갔습니다"], []),
])
def test_a_plan_that_differs_from_the_sentence_is_a_mismatch(text, before, after, reasons, unread):
    assert request_check.check(text, before, after) == {"status": "mismatch", "reasons_ko": reasons,
                                                        "unread_ko": unread, "unchecked": []}


@pytest.mark.parametrize("text, before, after, unread", [
    # Codex re-check 2026-10-03 (3) 신규①: «같이 해줘» is no field's verb; 13 was dropped and the plan without it matched
    ("25번 고정하고 13번도 같이 해줘", C(lock=[13]), C(lock=[13]), ["«13번도 같이 해줘»"]),
    ("25번 고정하고 13번도 봐줘", C(lock=[13]), C(lock=[13]), ["«13번도 봐줘»"]),
    ("13번 고정해줘. 14번도 봐줘.", C(), C(lock=[6]), ["«14번도 봐줘»"]),                 # not across a sentence
    ("13번 고정하지 마", C(), C(), ["«13번 고정하지 마»"]),                                # a negated verb
    ("25번 고정하고 13번도 빼지 마", C(lock=[6, 13]), C(lock=[6, 13]), ["«13번도 빼지 마»"]),
    ("35번 고정해줘", C(), C(), ["«35번 고정해줘»"]),                                     # a lower-arch tooth
    ("1년 안에 끝내줘", C(), C(), ["«1년 안에 끝내줘»"]),                                  # a time not in months
    ("12개월 안에, 아니 10개월 안에 끝내줘", C(), C(stage_cap=26), ["«12개월 안에»", "«10개월 안에 끝내줘»"]),
])
def test_what_is_not_read_is_never_a_match(text, before, after, unread):
    assert request_check.check(text, before, after) == {"status": "needs_check", "reasons_ko": [], "unread_ko": unread,
                                                        "unchecked": []}


def test_a_tooth_named_by_words_needs_a_check_not_a_match():
    # «오른쪽 송곳니» has no number to read: the change is listed as unchecked (못 잡는 것), never as checked
    r = request_check.check("오른쪽 송곳니는 움직이지 마.", C(), C(lock=[6]))
    assert r == {"status": "needs_check", "reasons_ko": [], "unread_ko": [], "unchecked": ["lock"]}


@pytest.mark.parametrize("text", ["13번도 해줘", "13번도 빼줘"])
def test_also_with_no_number_before_it_takes_nothing(text):
    asked = request_check.read_request(text)
    assert all(not asked[f]["add"] and not asked[f]["release"] for f in request_check.FIELDS_KO)
    assert [u[0] for u in asked["unread"]] == [6]                    # and is not read: FDI 13 = app 6


# ---- the sentences add up per case (check_case) ------------------------------------------------------------------

UNREAD_ALSO = "25번 고정하고 13번도 같이 해줘"


@pytest.mark.parametrize("entries, after, status", [
    # a later sentence that reads the tooth settles it, both ways
    ([ask(UNREAD_ALSO, C(lock=[13])), ask("13번 고정해줘", C(lock=[13]))], C(lock=[6, 13]), "match"),
    ([ask(UNREAD_ALSO, C(lock=[13])), ask("13번 고정해줘", C(lock=[13]))], C(lock=[13]), "mismatch"),
    ([ask(UNREAD_ALSO, C(lock=[13])), ask("13번 고정 풀어줘", C(lock=[13]))], C(lock=[13]), "match"),
    # so does a hand edit of it
    ([ask(UNREAD_ALSO, C(lock=[13])), edit(C(lock=[13]), C(lock=[6, 13]))], C(lock=[6, 13]), "match"),
    # a sentence about another tooth does not
    ([ask("13번 고정하지 마", C()), ask("14번 고정해줘", C())], C(lock=[5]), "needs_check"),
    # a time: a later limit in months, or the stage cap set by hand
    ([ask("1년 안에 끝내줘", C()), ask("12개월 안에 끝내줘", C())], C(stage_cap=26), "match"),
    ([ask("1년 안에 끝내줘", C()), edit(C(), C(stage_cap=26))], C(stage_cap=26), "match"),
])
def test_what_was_not_read_is_settled_only_by_reading_it_again(entries, after, status):
    assert check_case(entries, after)["status"] == status


@pytest.mark.parametrize("entries, after, status, reasons", [
    # Codex re-check (3) 신규②: the setup turn's sentence, then chips that ask nothing; the lock left out
    ([ask("13번 고정", C()), ask("목표 배열 만들어줘", C()), ask("단계로 나눠줘.", C())], C(), "mismatch",
     ["고정 치아에 요청한 13번이 없습니다"]),
    ([ask("13번 고정", C()), ask("목표 배열 만들어줘", C(lock=[6])), ask("단계로 나눠줘.", C(lock=[6]))], C(lock=[6]),
     "match", []),
    # a later sentence releases what an earlier one asked
    ([ask("13번 고정", C()), ask("13번 고정 풀어줘", C(lock=[6]))], C(), "match", []),
    # a hand edit sets the field it changes, over the sentences before it
    ([ask("13번 고정", C()), edit(C(lock=[6]), C())], C(), "match", []),
    ([ask("13번 고정", C()), edit(C(lock=[6]), C(lock=[6, 5]))], C(lock=[6]), "mismatch",
     ["고정 치아에 요청한 14번이 없습니다"]),
    # ... and the stage cap: the months asked before it no longer bind
    ([ask("13번 고정하고 12개월 안에", C()), edit(C(lock=[6], stage_cap=26), C(lock=[6], stage_cap=30))],
     C(lock=[6], stage_cap=30), "match", []),
    # a field no sentence and no edit named is compared with the conditions the first one opened with
    ([edit(C(), C(extraction=[5])), ask("단계로 나눠줘.", C(extraction=[5]))], C(extraction=[5, 12]), "mismatch",
     ["발치 치아에 요청에 없던 24번이 들어갔습니다"]),
])
def test_every_sentence_of_the_case_counts_in_order(entries, after, status, reasons):
    r = check_case(entries, after)
    assert (r["status"], r["reasons_ko"]) == (status, reasons)


def test_hand_edits_alone_are_nothing_to_compare():
    assert check_case([], C(lock=[6])) is None
    assert check_case([edit(C(), C(lock=[6]))], C(lock=[6])) is None    # a rule-API or MCP-only case


# ---- the turns, as the server keeps them -------------------------------------------------------------------------

def test_the_turn_sentence_is_kept_as_the_turn_opens(isolated):
    """PlanEventsASGI keeps the sentence before the agent runs (a setup turn makes nothing to carry it), after the form
    sent with the turn (an edit)."""
    app = FastAPI()

    @app.post("/chat/stream")
    async def chat():
        async def generate():
            yield 'data: {"value":"done"}\n\n'
        return StreamingResponse(generate(), media_type="text/event-stream")
    app.add_middleware(plan_events.PlanEventsASGI)
    with TestClient(app) as c:
        c.post("/chat/stream", json={"messages": [{"role": "user", "content": "13번 고정"}], "step": "setup",
                                     "cualign": {"case_id": "moderate", "constraints": {"ipr_exclude": [7]}}})
    entries = isolated.requests["moderate"]["entries"]
    assert [e["kind"] for e in entries] == ["edit", "ask"] and entries[1]["text"] == "13번 고정"
    assert entries[0]["after"].ipr_exclude == (7,) and entries[1]["opened"].ipr_exclude == (7,)


@pytest.mark.parametrize("lock, code", [(None, 409), ([13], 200)])
def test_a_setup_turn_sentence_counts_though_the_turn_made_nothing(isolated, lock, code):
    """Codex re-check (3) 신규②: the setup turn «13번 고정» where the model left set_constraints out; the next chips send
    plain sentences, and the target and the plan made after them matched. [13] is the same flow applied right."""
    turn("13번 고정", [("set_constraints", {"lock": lock})] if lock else [], step="setup")
    tid = target_turn("목표 배열 만들어줘")
    pid = turn("단계로 나눠줘.", [("plan_stages", {"target_id": tid})], step="stages")[0]["plan_id"]
    isolated.set_review(pid, REVIEW)
    with client() as c:
        r = approve(c, pid)
    assert r.status_code == code, r.text
    if code == 409:
        assert r.json()["detail"].startswith("요청 문장과 계획의 조건이 다릅니다: 고정 치아에 요청한 13번이 없습니다")


def test_a_later_turn_sentence_reaches_a_target_made_before_it(isolated):
    """Codex re-check (3) 🔴1: «목표 배열 만들어줘» made T with no lock; the next turn «13번 고정하고 단계로 나눠줘» left the
    lock out (mismatch), and T staged again by POST …/targets/T/stages matched: the later sentence was not T's. The
    case keeps every sentence, so T staged again answers both. Then the same turn applied right is approved."""
    tid = target_turn("목표 배열 만들어줘")
    left_out = turn("13번 고정하고 단계로 나눠줘", [("plan_stages", {"target_id": tid})], step="stages")[0]
    assert left_out["request_check"]["status"] == "mismatch"
    with client() as c:
        again = c.post(f"/api/cases/moderate/targets/{tid}/stages", json={}).json()["plan_id"]
        assert isolated.plans[again]["review"]["status"] == "skipped" and not isolated.plans[again]["violations"]
        r = approve(c, again)
        assert r.status_code == 409, r.text
        assert r.json()["detail"].startswith("요청 문장과 계획의 조건이 다릅니다: 고정 치아에 요청한 13번이 없습니다")
        right = staged("13번 고정하고 단계로 나눠줘", lock=[13], step="stages")
        isolated.set_review(right, REVIEW)
        assert approve(c, right).status_code == 200


@pytest.mark.parametrize("lock, first, code", [([25], "needs_check", 409), ([25, 13], "mismatch", 200)])
def test_also_with_a_verb_of_no_field_holds_approval_until_said_again(isolated, lock, first, code):
    """Codex re-check (3) 신규①: «25번 고정하고 13번도 같이 해줘» on FDI 25 locked dropped 13 as unread, and the plan
    that left 13 out matched. Now neither plan is approved until 13 is said with a verb («13번 고정해줘»), which the
    approval reads again: the plan without 13 then differs, the plan with it matches."""
    turn("25번 고정해줘", [("set_constraints", {"lock": [25]})], step="setup")
    pid = staged(UNREAD_ALSO, lock=lock, step="stages")
    isolated.set_review(pid, REVIEW)
    assert isolated.plans[pid]["request_check"]["status"] == first
    with client() as c:
        r = approve(c, pid)
        assert r.status_code == 409, r.text
        if first == "needs_check":
            assert "확인이 필요합니다: «13번도 같이 해줘»" in r.json()["detail"]
        turn("13번 고정해줘", step="setup")                       # said again; the plan is not made again
        r = approve(c, pid)
    assert r.status_code == code, r.text
    assert isolated.plans[pid]["request_check"]["status"] == ("match" if code == 200 else "mismatch")


def test_approval_needs_the_locked_teeth_confirmed(isolated):
    """The dentist ticks the locked teeth (FDI) on the screen; the server compares them with the plan's and records
    them. Without them, or with others, no approval: also on a plan no sentence was asked for."""
    svc = PlanningService(isolated)
    plain = svc.stages(svc.target("moderate", "expansion_ipr", Constraints(lock=(6,))), None)
    isolated.set_review(plain, REVIEW)
    pid = turn_target([13], stage=True)
    isolated.set_review(pid, REVIEW)
    with client() as c:
        for p in (plain, pid):
            none, wrong = approve(c, p, None), approve(c, p, [25])
            assert none.status_code == 409 and none.json()["detail"] == "고정 치아 목록(FDI)을 화면에서 확인한 뒤 승인하세요."
            assert wrong.status_code == 409
            assert wrong.json()["detail"].startswith("확인한 고정 치아(25)와 계획의 고정 치아(13)가 다릅니다")
            ok = approve(c, p, [13])
            assert ok.status_code == 200 and ok.json()["approval"]["confirmed_lock_fdi"] == [13]


@pytest.mark.parametrize("lock, status", [([13], "match"), ([25], "mismatch"), (None, None)])
def test_select_plan_records_the_check_on_the_turn_plan(isolated, lock, status):
    calls = [("set_constraints", {"lock": lock or [13]}), PROPOSE,
             ("plan_stages", lambda out: {"target_id": out[1]["target_id"]}),
             ("select_plan", lambda out: {"plan_id": out[2]["plan_id"]})]
    if status:
        selected = turn(REVISION, calls)[-1]                       # a /chat/stream turn
    else:                                                          # MCP and tests have no sentence
        selected = run_tools(calls, PlanRun("rc", "moderate", None, Constraints()))[-1]
        assert "request_check" not in selected
        return
    assert selected["request_check"]["status"] == status
    saved = json.loads((store_module.OUT_DIR / "plans" / f"{selected['plan_id']}.json").read_text(encoding="utf-8"))
    assert saved["request_check"] == selected["request_check"]


def test_a_mismatch_blocks_approval_and_survives_a_restart(isolated):
    s = isolated
    svc = PlanningService(s)
    pid = svc.stages(svc.target("moderate", "expansion_ipr", Constraints()), None)
    s.set_review(pid, REVIEW)
    assert "request_check" not in s.plans[pid] and s.approve(pid, [])["approval"]   # no sentence: nothing to compare
    wrong = turn_target([25], stage=True)
    s.set_review(wrong, REVIEW)
    with client() as c:
        r = approve(c, wrong)
    assert r.status_code == 409 and "요청 문장과 계획의 조건이 다릅니다" in r.json()["detail"]
    assert "고정 치아에 요청한 13번이 없습니다" in r.json()["detail"]
    with pytest.raises(ValueError, match="요청 문장과 계획의 조건이 다릅니다"):
        store_module.Store().approve(wrong, [25])                 # plans and sentences read back after a restart
    right = turn_target([13], stage=True)
    s.set_review(right, REVIEW)
    assert s.approve(right, [13])["approval"]["status"] == "approved"


def test_the_sentences_are_read_back_after_a_restart(isolated):
    """A sentence asked after a plan was made reaches it at approval, in a new process too."""
    pid = turn_target([13], stage=True)
    isolated.set_review(pid, REVIEW)
    turn("14번도 봐줘", step="setup")                             # «도» with no number before it: not read
    fresh = store_module.Store()
    with pytest.raises(ValueError, match="확인이 필요합니다: «14번도 봐줘»"):
        fresh.approve(pid, [13])
    assert fresh.plans[pid]["request_check"]["status"] == "needs_check"


def test_a_request_file_that_cannot_be_read_holds_approval(isolated):
    pid = turn_target([13], stage=True)
    isolated.set_review(pid, REVIEW)
    (store_module.OUT_DIR / "requests" / "moderate.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="확인이 필요합니다: 요청 기록을 읽지 못했습니다"):
        store_module.Store().approve(pid, [13])


def test_a_plan_whose_sentences_are_gone_is_refused(isolated):
    """fail-closed: a plan checked against sentences that the store no longer has is refused, not let through."""
    pid = turn_target([13], stage=True)
    isolated.set_review(pid, REVIEW)
    assert isolated.plans[pid]["request_check"]["status"] == "match"
    del isolated.requests["moderate"]
    with pytest.raises(ValueError, match="이 계획을 대조한 요청 기록을 찾지 못했습니다"):
        isolated.approve(pid, [13])


def test_starting_over_keeps_each_plan_with_its_own_sentences(isolated):
    """「처음부터」 opens a new segment: the plans made after it answer the new sentences only (the new plan without
    the lock would differ from REVISION), those made before keep being compared with theirs."""
    old = turn_target([13], stage=True)
    with client() as c:
        assert c.post("/api/cases/moderate/restart").status_code == 200
        new = staged("단계로 나눠줘.")                      # no lock: what REVISION asked is the old segment's
        for p in (old, new):
            isolated.set_review(p, REVIEW)
            r = approve(c, p)
            assert r.status_code == 200, (p, r.text)
    assert isolated.plans[new]["request_seg"] == 1 and "request_seg" not in isolated.plans[old]


def test_a_form_plan_answers_the_sentences_it_leaves(isolated):
    """POST /api/plan (조건으로 계산 · 다시 계산): what the form changes the dentist set by hand; what it leaves, the
    sentences still ask (Codex re-check (3) 🟡: the plan made from the form without a sentence)."""
    turn(REVISION, step="setup")                                   # the model left the lock out
    with client() as c:
        def plan(**form):
            r = c.post("/api/plan", json={"case_id": "moderate", **form}).json()
            return isolated.plans[(r["chosen"] or r["best_failed"])["plan_id"]]["request_check"]["status"]
        assert plan() == "mismatch"                                # nothing changed on the form
        assert plan(stage_cap=30) == "mismatch"                    # the cap changed, the lock left
        assert plan(lock=[5, 6]) == "match"                        # the lock set on the form (app numbers): the
                                                                   # form adds FDI 14, which no sentence asked


@pytest.mark.parametrize("lock, status, code", [([25], "mismatch", 409), ([13], "match", 200)])
def test_every_compared_candidate_is_checked_so_switching_to_it_cannot_skip_the_check(isolated, lock, status, code):
    """Codex review 2026-10-03 (1): compare_strategies stores several plans and select_plan picks one; the screen's
    plan menu switches to another with a GET (no select_plan), the dentist asks for its review by hand and approves.
    Every plan of the case is checked. lock [25] is the live mistake (FDI 25 for «13번»); [13] the same path applied
    right, which approval lets through."""
    compared, selected = turn(REVISION, [("set_constraints", {"lock": lock}), ("compare_strategies", {}),
                                         ("select_plan", lambda out: {"plan_id": out[1]["plans"][0]["plan_id"]})])[1:]
    assert len(compared["plans"]) >= 2
    assert [p["request_check"]["status"] for p in compared["plans"]] == [status] * len(compared["plans"])
    other = next(p["plan_id"] for p in compared["plans"][1:] if not isolated.plans[p["plan_id"]]["violations"])

    async def review(pid):                                        # the reviewer passes the memo, as the fake one does
        isolated.set_review(pid, REVIEW)

    with client(review) as c:
        assert c.get(f"/api/plans/{other}").json()["request_check"]["status"] == status     # the menu's switch
        assert c.post(f"/api/plans/{other}/review").status_code == 200
        r = approve(c, other)
    assert r.status_code == code, r.text
    if code == 409:
        assert r.json()["detail"].startswith("요청 문장과 계획의 조건이 다릅니다: 고정 치아에 요청한 13번이 없습니다")


@pytest.mark.parametrize("lock, code", [([25], 409), ([13], 200)])
def test_staging_the_turn_target_again_keeps_the_check(isolated, lock, code):
    """POST …/targets/{id}/stages on the turn's target, outside the turn: the same conditions staged again, review
    «skipped». The plan answers what was asked on the case. [13] is the same path applied right."""
    tid = turn_target(lock)
    with client() as c:
        row = c.post(f"/api/cases/moderate/targets/{tid}/stages", json={}).json()
        p = isolated.plans[row["plan_id"]]
        assert p["review"]["status"] == "skipped" and not p["violations"]
        r = approve(c, row["plan_id"])
    assert r.status_code == code, r.text
    if code == 409:
        assert r.json()["detail"].startswith("요청 문장과 계획의 조건이 다릅니다: 고정 치아에 요청한 13번이 없습니다")


@pytest.mark.parametrize("lock, status", [([25], "mismatch"), ([13], "match")])
def test_a_hand_edit_of_the_turn_target_keeps_the_check(isolated, lock, status):
    """직접 이동 makes a new target from the turn's (parent_target_id): its plan answers the same sentences."""
    tid = turn_target(lock)
    with client() as c:
        view = c.get(f"/api/cases/moderate/targets/{tid}").json()
        d = (np.asarray(view["stages"][0]["8"]) + 0.2 * np.asarray(view["frames"]["8"]["buccal"])).tolist()
        new = c.post(f"/api/cases/moderate/targets/{tid}/manual", json={"teeth": {"8": {"d": d, "yaw": 0.0}}}).json()
        assert new["info"]["parent_target_id"] == tid
        row = c.post(f"/api/cases/moderate/targets/{new['target_id']}/stages", json={}).json()
    assert isolated.plans[row["plan_id"]]["request_check"]["status"] == status


@pytest.mark.parametrize("lock, status", [([25], "mismatch"), ([13], "match")])
def test_a_stages_turn_on_a_target_turn_target_answers_the_target_turn_sentence(isolated, lock, status):
    """The step flow: a target turn reads the sentence and makes the target, the next turn («단계로 나눠줘.») stages it."""
    tid = target_turn(REVISION, [("set_constraints", {"lock": lock})])
    plan = turn("단계로 나눠줘.", [("plan_stages", {"target_id": tid})], step="stages")[0]
    assert plan["request_check"]["status"] == status
