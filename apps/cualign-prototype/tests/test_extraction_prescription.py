"""#56: the teeth to extract are the dentist's prescription; the app plans exactly them and never picks them."""
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cualign.core import Case, store
from cualign.core.constraints import ConstraintPatch, Constraints, ExtractionTeethNeeded
from cualign.core.planner import compare_strategies, plan_stages, propose_target, validate
from cualign.server import api


@pytest.fixture(scope="module")
def moderate():
    return Case.synthetic("moderate")


# ------------------------------------------------------------------------------------------------ the prescription
def test_patch_semantics():
    none, ext = Constraints(), Constraints(extraction=(12, 5))
    assert ext.extraction == (5, 12) and ext.allow_extraction and not none.allow_extraction
    assert ext.patched(ConstraintPatch().changes()) == ext                       # null keeps
    assert ext.patched({"extraction": []}).extraction == ()                      # [] = non-extraction
    assert ext.patched({"extraction": [4, 13]}).extraction == (4, 13)            # a list replaces
    assert ext.patched({"allow_extraction": False}).extraction == ()             # legacy false clears
    assert ext.patched({"allow_extraction": True}).extraction == (5, 12)         # legacy true keeps a prescription
    with pytest.raises(ExtractionTeethNeeded):
        none.patched({"allow_extraction": True})                                 # ... but never picks teeth
    with pytest.raises(ValueError):
        none.patched({"allow_extraction": False, "extraction": [5]})             # contradiction
    # the output keeps allow_extraction for readers, and reads back
    assert Constraints.model_validate(ext.model_dump(mode="json")) == ext


@pytest.mark.parametrize("bad, why", [({"extraction": (7,)}, "소구치"), ({"extraction": (5,), "lock": (5,)}, "고정")])
def test_unsupported_or_contradictory_prescriptions_are_refused(bad, why):
    with pytest.raises(ValueError, match=why):
        Constraints(**bad)


def test_prescribed_tooth_must_be_in_the_case(moderate):
    without_5 = Case({i: m for i, m in moderate.mesh.items() if i != 5}, name="no-5")
    with pytest.raises(ValueError):
        Constraints(extraction=(5, 12)).check_case(without_5.ids)
    with pytest.raises(ValueError, match="케이스에 없습니다"):
        propose_target(without_5, "extraction", extraction=(5, 12))


# ------------------------------------------------------------------------------------------------ the plan
@pytest.mark.parametrize("teeth", [(5, 12), (4, 13)])
def test_only_the_prescribed_teeth_are_extracted(moderate, teeth):
    rows = compare_strategies(moderate, constraints=Constraints(extraction=teeth))
    assert [r["strategy"] for r in rows] == ["extraction"]
    assert rows[0]["removed"] == list(teeth)
    assert not [v for v in rows[0]["_viol"] if v["type"].startswith("extraction")]


def test_non_extraction_removes_nothing_and_the_app_does_not_offer_extraction(moderate):
    rows = compare_strategies(moderate, constraints=Constraints())
    assert "extraction" not in [r["strategy"] for r in rows]
    assert all(r["removed"] == [] for r in rows)
    with pytest.raises(ValueError, match="발치 처방"):
        propose_target(moderate, "extraction")                 # no prescription: no extraction plan at all


def test_a_plan_that_extracts_otherwise_is_a_violation(moderate):
    target, info = propose_target(moderate, "extraction", extraction=(5, 12))
    stages, _ = plan_stages(moderate, target)
    viol = validate(moderate, stages, constraints=Constraints(extraction=(4, 13)))
    bad = [v for v in viol if v["type"] == "extraction_mismatch"]
    assert bad and bad[0]["prescribed"] == [4, 13] and bad[0]["removed"] == [5, 12]
    target, _ = propose_target(moderate, "expansion_ipr")    # a prescribed extraction that is not done
    stages, _ = plan_stages(moderate, target)
    assert any(v["type"] == "extraction_mismatch" for v in validate(moderate, stages, constraints=Constraints(extraction=(5, 12))))


@pytest.mark.parametrize("tooth, moved, kept", [(5, 3, 14), (4, 3, 14), (12, 14, 3), (13, 14, 3)])
def test_one_sided_extraction_closes_on_its_own_side(tooth, moved, kept):
    # mild: a one-sided premolar leaves room to close (moderate's 4.5 mm crowding leaves ~0.7 mm, which the incisors'
    # shape room takes, #76)
    c = Case.synthetic("mild")
    target, info = propose_target(c, "extraction", extraction=(tooth,))
    assert target[tooth] is None and sum(v is None for v in target.values()) == 1
    assert np.linalg.norm(target[moved][:2]) > 1.0             # the extraction side's molars close the space
    assert np.linalg.norm(target[kept][:2]) < 0.05             # the other side's molars stay
    stages, _ = plan_stages(c, target)
    assert not validate(c, stages, space_deficit_mm=info["space_deficit_mm"], constraints=Constraints(extraction=(tooth,)))


# ------------------------------------------------------------------------------------------------ the paths in
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUT_DIR", tmp_path)
    monkeypatch.setattr(store, "OUT_DIR", tmp_path)
    s = store.Store()
    monkeypatch.setattr(store, "STORE", s)
    monkeypatch.setattr(api, "STORE", s)
    app = FastAPI()
    api.add_api_routes(app)
    with TestClient(app) as c:
        yield c


def test_api_plans_the_prescription_and_keeps_it_on_revision(client):
    r = client.post("/api/plan", json={"case_id": "moderate", "extraction": [5, 12]}).json()
    plan = r["chosen"]
    assert plan and plan["strategy"] == "extraction" and [t["strategy"] for t in r["tried"]] == ["extraction"]
    assert plan["constraints"]["extraction"] == [5, 12] and plan["constraints"]["allow_extraction"] is True
    rev = client.post("/api/plan", json={"case_id": "moderate", "parent_plan_id": plan["plan_id"], "stage_cap": 60}).json()
    kept = rev["chosen"] or rev["best_failed"]
    assert kept["constraints"]["extraction"] == [5, 12] and kept["constraints"]["stage_cap"] == 60


def test_api_asks_for_the_teeth_instead_of_picking_them(client):
    r = client.post("/api/plan", json={"case_id": "moderate", "allow_extraction": True})
    assert r.status_code == 400 and "발치할 치아 번호" in r.json()["detail"]
    r = client.post("/api/plan", json={"case_id": "moderate", "extraction": [7]})
    assert r.status_code == 400 and "소구치" in r.json()["detail"]


def test_chat_context_without_teeth_is_refused_with_the_reason():
    from cualign.server.plan_events import ChatContext, open_run
    ctx = ChatContext.model_validate({"request_id": "r1", "case_id": "moderate",
                                      "constraints": {"allow_extraction": True}})
    with pytest.raises(ExtractionTeethNeeded):
        open_run(ctx, store=store.Store())


def test_cli_does_not_plan_extraction_without_teeth(monkeypatch):
    from cualign import cli
    args = type("A", (), {"request": "발치 허용해서 짜줘", "case": "moderate", "export": None})()
    with pytest.raises(SystemExit, match="발치할 치아 번호"):
        cli.cmd_plan(args)


def test_plan_files_from_before_56_read_back():
    # plan files keep constraints; before #56 they said allow_extraction and never which teeth (store reads them, #93)
    old = {"allow_extraction": True, "lock": [], "ipr_exclude": [], "ipr_limit_mm": 0.25, "stage_cap": None,
           "order": "simultaneous"}
    assert Constraints.from_saved(old, removed=[12, 5]).extraction == (5, 12)     # an extraction plan: what it removed
    assert Constraints.from_saved(old, removed=[]).extraction == ()               # only allowed: non-extraction
    assert Constraints.from_saved({**old, "allow_extraction": False}).extraction == ()
    new = Constraints(extraction=(4, 13)).model_dump(mode="json")
    assert Constraints.from_saved(new, removed=[]).extraction == (4, 13)          # new files carry the teeth


# ------------------------------------------------------------------------------------------------ PR #98 review
def test_a_forbidden_extraction_in_an_old_file_stays_forbidden(moderate):
    # an old plan that extracted although extraction was not allowed must not read back as a prescription
    old = {"allow_extraction": False, "lock": [], "ipr_exclude": [], "ipr_limit_mm": 0.25, "stage_cap": None,
           "order": "simultaneous"}
    c = Constraints.from_saved(old, removed=[5, 12])
    assert c.extraction == ()
    target, _ = propose_target(moderate, "extraction", extraction=(5, 12))
    stages, _ = plan_stages(moderate, target)
    assert any(v["type"] == "extraction_forbidden" for v in validate(moderate, stages, constraints=c))


def test_space_behind_a_locked_neighbour_is_reported_not_passed():
    # 13 out, 12 locked: the space lies behind the locked 12, so the incisors cannot use it (mild still needs ~2 mm in
    # front). The molar closes the space; the crowding in front is left and reported as collisions, not passed.
    c = Case.synthetic("mild")
    cons = Constraints(extraction=(13,), lock=(12,))
    target, info = propose_target(c, "extraction", constraints=cons)
    stages, _ = plan_stages(c, target)
    assert _max_neighbour_gap(c, target) < 2.0
    assert any(v["type"] == "collision" for v in validate(c, stages, space_deficit_mm=info["space_deficit_mm"], constraints=cons))


def _max_neighbour_gap(case, target):
    from scipy.spatial import cKDTree
    from cualign.core.planner import yaw_of
    act = [i for i in case.ids if target[i] is not None]
    gaps = []
    for a, b in zip(act, act[1:]):
        A = case.placed(a, target[a], yaw_of(target, a), hull=True)
        B = case.placed(b, target[b], yaw_of(target, b), hull=True)
        gaps.append(float(cKDTree(B.vertices).query(A.vertices)[0].min()))
    return max(gaps)


@pytest.mark.parametrize("tooth, lock", [(5, 13), (12, 4), (5, 4), (12, 13)])
def test_a_locked_tooth_does_not_leave_the_extraction_space_open(tooth, lock):
    # before the review fix any locked tooth switched the closing off: 5 out + 13 locked left 4-6 open by 3.4 mm
    c = Case.synthetic("mild")
    target, info = propose_target(c, "extraction", lock={lock}, extraction=(tooth,))
    assert np.allclose(target[lock], 0)
    assert _max_neighbour_gap(c, target) < 2.0, info["notes"]      # without extraction the widest is ~1.4 mm here
    stages, _ = plan_stages(c, target)
    assert not validate(c, stages, space_deficit_mm=info["space_deficit_mm"],
                        constraints=Constraints(extraction=(tooth,), lock=(lock,)))


@pytest.mark.parametrize("text, teeth", [
    ("14번과 24번 발치, 기간 제한 없이", [5, 12]),                        # "없이" about the time, not extraction
    ("발치 치아 14, 24번으로 계획해줘", [5, 12]),
    ("처방은 비발치, IPR 11-21·11-12·21-22 접촉면에 각 0.4mm입니다.", []),   # IPR numbers are not teeth
])
def test_cli_reads_the_prescribed_teeth(text, teeth):
    from cualign.cli import parse_constraints
    assert parse_constraints(text)["extraction"] == teeth


def test_the_agent_tool_refuses_as_a_normal_result(tmp_path, monkeypatch):
    import asyncio
    from cualign.agent import register
    from cualign.agent.context import CURRENT_RUN, PlanRun
    s = store.Store()
    monkeypatch.setattr(store, "OUT_DIR", tmp_path)
    monkeypatch.setattr(register, "STORE", s)
    s.load_case("mild")

    async def scenario():
        gen = register.cualign(register.CuAlignToolConfig(), None)
        group = await gen.__aenter__()
        tools = await group.get_all_functions()
        tool = next(fn for key, fn in tools.items() if key.split("__")[-1] == "set_constraints")
        token = CURRENT_RUN.set(PlanRun("r1", "mild", None, Constraints()))
        try:
            for patch, why in (({"allow_extraction": True}, "발치할 치아 번호"),
                               ({"extraction": [12]}, "소구치"),   # FDI 12 (app 7) is not a premolar
                               ({"stage_cap": 40, "clear_stage_cap": True}, "")):
                r = await tool.ainvoke(patch)            # never raises: the agent would retry
                assert r["rejected"] and why in r["note"] and r["extraction"] == []
            without_5 = Case({i: m for i, m in s.cases["mild"].mesh.items() if i != 5}, name="no-5")
            s.cases["mild"] = without_5
            r = await tool.ainvoke({"extraction": [14, 24]})   # FDI 14 = app 5, a tooth the case does not have
            assert r["rejected"] and r["extraction"] == []
        finally:
            CURRENT_RUN.reset(token)
            await gen.__aexit__(None, None, None)
    asyncio.run(scenario())


def test_a06_catches_a_claimed_extraction_plan():
    import re
    import yaml
    from pathlib import Path
    spec = yaml.safe_load((Path(__file__).resolve().parents[1] / "evals/golden_a/specs/A06_compare.yaml").read_text())
    pats = next(c for c in spec["checks"] if c["id"] == "A06-no-picked-teeth")["patterns"]
    claim = "5·12번 발치 계획을 만들었습니다. 발치할 치아 번호를 알려 주세요."
    ask = "발치안을 만들려면 발치할 치아 번호가 필요합니다. 어느 치아를 발치할까요? (예: 5번과 12번)"
    assert any(re.search(p, claim) for p in pats) and not any(re.search(p, ask) for p in pats)


# ------------------------------------------------------------------------------------------------ PR #98 re-review
@pytest.mark.parametrize("text", ["5번과 12번 발치 금지", "이전에는 5번과 12번 발치였고 이번엔 비발치로",
                                  "이전 안은 비발치였고 이번엔 5번과 12번 발치"])
def test_cli_does_not_guess_a_prescription_from_a_refusal_or_a_change(text):
    from cualign.cli import parse_constraints
    assert parse_constraints(text)["extraction"] == "ambiguous"


@pytest.mark.parametrize("teeth, lock", [((4,), (13,)), ((4, 13), (8,)), ((4,), (3,)), ((12,), (13,)), ((5,), (2,))])
def test_the_space_goes_to_a_molar_that_can_close_it(teeth, lock):
    # re-review: the room was left in front of the locked tooth (4 out + 13 locked: 3.8 mm at 12-13; 4, 13 out + 8
    # locked: 5.8 mm at 7-8), and a locked molar left 3.9 mm open with no violation
    c = Case.synthetic("mild")
    cons = Constraints(extraction=teeth, lock=lock)
    target, info = propose_target(c, "extraction", constraints=cons)
    assert _max_neighbour_gap(c, target) < 2.0 and info["open_space_mm"] == 0
    stages, _ = plan_stages(c, target)
    assert not validate(c, stages, space_deficit_mm=info["space_deficit_mm"], constraints=cons, target_info=info)


def test_space_no_molar_can_close_is_a_violation():
    c = Case.synthetic("mild")
    cons = Constraints(extraction=(5,), lock=(3, 14))           # both first molars locked
    target, info = propose_target(c, "extraction", constraints=cons)
    stages, _ = plan_stages(c, target)
    viol = validate(c, stages, space_deficit_mm=info["space_deficit_mm"], constraints=cons, target_info=info)
    assert info["open_space_mm"] > 0.5 and any(v["type"] == "extraction_space_open" for v in viol)


def test_runner_keeps_the_stage_cap():
    from evals.real_scans.run import run_case
    r = run_case(Case.synthetic("mild"), extraction=(5, 12), stage_cap=1)
    assert r["outcome"] == "fail" and "stage_cap" in r["tried"][-1]["by_type"]


def test_a_migrated_plan_is_not_shown_as_approved():
    from cualign.core.store import _plan_from_file
    old = {"plan_id": "p1", "case_id": "moderate", "stages": [], "target": {"removed": [5, 12]},
           "constraints": {"allow_extraction": True, "lock": [], "ipr_exclude": [], "ipr_limit_mm": 0.25,
                           "stage_cap": None, "order": "simultaneous"},
           "approval": {"status": "approved", "fingerprint": "old"}}
    p = _plan_from_file(old)
    assert p["constraints"].extraction == (5, 12) and p["approval"] is None


@pytest.mark.parametrize("text, teeth", [
    ("발치 치아 14, 24번 · 단계 상한 52단계", {5, 12}),
    ("발치 14·24 · 단계 상한 52단계", {5, 12}),
    ("발치 치아 15·25번 · IPR 12-13·22-23", {4, 13}),           # the IPR contacts are not extraction teeth
    ("발치 치아 5, 12번 · 단계 상한 52단계", {7}),                # app numbers leaked: 5 is no FDI tooth, 12 reads as FDI 12
    ("발치 치아 14·24번은 처방이 아닙니다.", set()),                 # a negation names no prescription
])
def test_answer_extraction_teeth_are_read_from_the_extraction_phrase(text, teeth):
    from evals.golden_a.checks import stated_extraction_teeth
    assert stated_extraction_teeth(text) == teeth


@pytest.mark.parametrize("answer, bad", [
    ("5·12번 발치 계획을 만들었습니다.", True),
    ("5·12번을 제거한 발치안을 생성했습니다. 발치할 치아 번호를 알려 주세요.", True),
    ("5번과 12번을 뺀 계획을 만들었습니다.", True),
    ("발치할 치아 번호를 알려 주세요. 처방을 받으면 발치 계획을 만들겠습니다.", False),
    ("비발치 계획을 만들었습니다.", False),
])
def test_a06_patterns_catch_claims_not_correct_answers(answer, bad):
    import re
    import yaml
    from pathlib import Path
    spec = yaml.safe_load((Path(__file__).resolve().parents[1] / "evals/golden_a/specs/A06_compare.yaml").read_text())
    pats = next(c for c in spec["checks"] if c["id"] == "A06-no-picked-teeth")["patterns"]
    assert any(re.search(p, answer) for p in pats) is bad
