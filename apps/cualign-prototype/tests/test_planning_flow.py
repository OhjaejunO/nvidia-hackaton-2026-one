"""Constraint, lineage, approval and export regressions using synthetic cases."""
import asyncio
import json
from io import BytesIO
from zipfile import ZipFile
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cualign.core import store as store_module
from cualign.core.constraints import Constraints, ConstraintPatch
from cualign.core.service import PlanningService
from cualign.core import planner
from cualign.server import api
from cualign.agent.context import CURRENT_RUN, PlanRun
from cualign.agent import register


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    s = store_module.Store()
    monkeypatch.setattr(store_module, "OUT_DIR", tmp_path)
    monkeypatch.setattr(api, "OUT_DIR", tmp_path)
    monkeypatch.setattr(api, "STORE", s)
    monkeypatch.setattr(register, "STORE", s)
    return s


def make_plan(s, constraints=None, parent=None):
    svc = PlanningService(s)
    c = constraints or Constraints()
    tid = svc.target("moderate", "expansion_ipr", c)
    return svc.stages(tid, parent)


def test_revision_snapshot_and_same_constraints_in_compare_validate(isolated):
    s = isolated
    c = Constraints(lock=(13,), ipr_exclude=(7, 8, 9, 10), ipr_limit_mm=0.12, stage_cap=50)
    parent = make_plan(s, c)
    before = s.plan_json(parent)
    revised = s.constraints_for("moderate", parent).patched({"lock": [13, 14]})
    ids = PlanningService(s).compare("moderate", revised, parent_plan_id=parent)
    assert len(ids) == 3
    for pid in ids:
        p = PlanningService(s).validate(pid)
        assert p["parent_plan_id"] == parent
        assert p["constraints"] == revised.model_dump(mode="json")
        assert p["target"]["ipr_mm_per_surface"] <= 0.12
        assert not set(p["target"]["ipr_applied_teeth"]) & {7, 8, 9, 10}
        for stage in p["stages"]:
            assert stage["13"] == [0, 0, 0] and stage["14"] == [0, 0, 0]
        saved = json.loads((store_module.OUT_DIR / "plans" / (pid + ".json")).read_text(encoding="utf-8"))
        assert saved["parent_plan_id"] == parent and saved["constraints"] == p["constraints"]
    assert s.plan_json(parent) == before


def test_invalid_parent_and_constraints_rejected_before_writes(isolated):
    s = isolated
    pid = make_plan(s)
    count = len(s.plans)
    for parent in ("missing", pid):
        with pytest.raises(ValueError):
            PlanningService(s).compare("severe", Constraints(), parent_plan_id=parent)
    assert len(s.plans) == count
    with pytest.raises(ValueError):
        Constraints(ipr_limit_mm=0.26)
    with pytest.raises(ValueError):
        Constraints(lock=(99,))
    with pytest.raises(ValueError):
        ConstraintPatch(stage_cap=10, clear_stage_cap=True).changes()
    c = Constraints(lock=(13,), stage_cap=30)
    # null means keep, so a patch full of nulls is a no-op. Clearing is always explicit.
    assert c.patched(ConstraintPatch().changes()) == c
    assert c.patched(ConstraintPatch(lock=None, ipr_exclude=None, stage_cap=None, order=None,
                                     allow_extraction=None, ipr_limit_mm=None).changes()) == c
    assert c.patched(ConstraintPatch(lock=[]).changes()).lock == ()
    assert c.patched(ConstraintPatch(clear_stage_cap=True).changes()).stage_cap is None
    assert c.patched(ConstraintPatch(stage_cap=None).changes()).stage_cap == 30


def test_no_extraction_and_locked_extraction_rejected(isolated):
    svc = PlanningService(isolated)
    with pytest.raises(ValueError):
        svc.target("moderate", "extraction", Constraints())
    with pytest.raises(ValueError):
        svc.target("moderate", "extraction", Constraints(allow_extraction=True, lock=(5,)))


def test_validator_detects_constraint_violations(isolated):
    _, case = isolated.load_case("moderate")
    c = Constraints(lock=(13,), ipr_exclude=(7,), ipr_limit_mm=0.1, stage_cap=1)
    stages = [{i: np.zeros(3) for i in case.ids} for _ in range(2)]
    for st in stages:
        st[13] = np.array([0.1, 0, 0])
        del st[5]
    v = planner.validate(case, stages, constraints=c,
                         target_info={"ipr_mm_per_surface": 0.25, "ipr_applied_teeth": [7]})
    assert {"extraction_forbidden", "locked_tooth", "ipr_limit", "ipr_excluded", "stage_cap"} <= {r["type"] for r in v}


def test_approval_export_revoke_revision_and_file_contents(isolated):
    s = isolated
    app = FastAPI()
    api.add_api_routes(app)
    with TestClient(app) as client:
        result = client.post("/api/plan", json={"case_id": "moderate", "allow_extraction": False}).json()
        pid = result["chosen"]["plan_id"]
        assert client.get(f"/api/plans/{pid}/stl.zip").status_code == 409
        assert client.post(f"/api/plans/{pid}/approval", json={}).status_code == 400
        assert client.post(f"/api/plans/{pid}/approval", json={"confirmed": True, "lock_fdi": api.STORE.lock_fdi(pid)}).status_code == 200
        output = client.get(f"/api/plans/{pid}/stl.zip")
        assert output.status_code == 200
        p = s.plans[pid]
        with ZipFile(BytesIO(output.content)) as z:
            assert len([n for n in z.namelist() if n.startswith("stage_")]) == len(p["stages"]) * 14
            assert output.headers["X-Cualign-Print-Models"].startswith("skipped; files=0")   # no gingiva scan
            _, case = s.load_case("moderate")
            mesh = trimesh.load(BytesIO(z.read(f"stage_{len(p['stages']):02d}/13.stl")), file_type="stl")
            assert np.allclose(mesh.bounds, case.mesh[13].bounds + p["stages"][-1][13], atol=1e-4)
        child = client.post("/api/plan", json={"case_id": "moderate", "parent_plan_id": pid, "lock": [13]}).json()
        child_id = (child["chosen"] or child["best_failed"])["plan_id"]
        assert s.plans[child_id]["parent_plan_id"] == pid
        assert s.plans[child_id]["approval"] is None
        assert any(not np.allclose(p["stages"][-1][i], s.plans[child_id]["stages"][-1][i]) for i in p["stages"][-1])
        assert client.get(f"/api/plans/{child_id}/stl.zip").status_code == 409
        assert client.delete(f"/api/plans/{pid}/approval").status_code == 200
        assert client.get(f"/api/plans/{pid}/stl.zip").status_code == 409  # ZIP already exists
        assert all(x["case_id"] == "moderate" for x in client.get("/api/plans?case_id=moderate").json()["plans"])


def test_approval_invalidated_by_snapshot_change(isolated):
    s = isolated
    pid = make_plan(s)
    s.set_review(pid, {"status": "passed", "attempts": 1, "message": "memo", "error": None})
    s.approve(pid, s.lock_fdi(pid))
    s.plans[pid]["stages"][-1][13][0] += 0.000001
    with pytest.raises(ValueError):
        s.require_approved(pid)


def test_failed_review_or_rules_cannot_approve(isolated):
    s = isolated
    pid = make_plan(s)
    with pytest.raises(ValueError):
        s.approve(pid, s.lock_fdi(pid))
    s.set_review(pid, {"status": "failed", "attempts": 2, "message": "failed", "error": "empty_response"})
    with pytest.raises(ValueError):
        s.approve(pid, s.lock_fdi(pid))
    s.set_review(pid, {"status": "passed", "attempts": 1, "message": "memo", "error": None})
    s.plans[pid]["violations"] = [{"type": "space_deficit"}]
    with pytest.raises(ValueError):
        s.approve(pid, s.lock_fdi(pid))


def test_nat_tools_keep_constraints_select_earlier_candidate_and_deny_export(isolated):
    async def scenario():
        gen = register.cualign(register.CuAlignToolConfig(), None)
        group = await gen.__aenter__()
        tools = await group.get_all_functions()
        # FunctionGroup without an instance prefix uses local names.
        def tool(name):
            return next(fn for key, fn in tools.items() if key.split("__")[-1] == name)
        c = Constraints(lock=(13,), ipr_exclude=(7,), ipr_limit_mm=0.1)
        token = CURRENT_RUN.set(PlanRun("r1", "moderate", None, c))
        try:
            result = await tool("compare_strategies").ainvoke(register.CompareInput())
            plans = result["plans"]
            assert all(p["constraints"] == c.model_dump(mode="json") for p in plans)
            selected = plans[0]["plan_id"]
            await tool("select_plan").ainvoke(register.PlanIdInput(plan_id=selected))
            assert CURRENT_RUN.get().selected_plan_id == selected != plans[-1]["plan_id"]
            # Re-stating the same conditions after targets exist is a no-op, not a hard error:
            # raising here made the ReAct agent retry until its iteration limit.
            # The tool takes FDI (the dentist's numbers): FDI 25 is the app's tooth 13.
            assert await tool("set_constraints").ainvoke({"lock": [25]}) == register.constraints_json(c)
            assert await tool("set_constraints").ainvoke({}) == register.constraints_json(c)
            # A real change after targets exist is refused as a normal result, never an exception:
            # a tool error makes the ReAct agent retry with new arguments until it runs out.
            refused = await tool("set_constraints").ainvoke({"lock": []})
            assert refused["rejected"] == {"lock": []} and refused["note"]
            assert refused["lock"] == [13] and CURRENT_RUN.get().constraints == c
            assert "moderate" not in isolated.case_constraints  # nothing was written
            with pytest.raises(ValueError):
                await tool("export_stl").ainvoke(register.PlanIdInput(plan_id=selected))
            await tool("validate").ainvoke(register.PlanIdInput(plan_id=selected))
            assert isolated.plans[selected]["constraints"] == c
        finally:
            CURRENT_RUN.reset(token)
            await gen.__aexit__(None, None, None)
    asyncio.run(scenario())


def test_cli_export_is_refused_without_a_traceback(isolated, capsys):
    from cualign import cli
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["plan", "발치 없이", "--case", "moderate", "--export", "out/stl"])
    assert "승인" in str(exit_info.value)
    assert capsys.readouterr().out == ""  # nothing is planned or printed before the refusal
    assert cli.main(["plan", "발치 없이", "--case", "moderate"]) == 0


def test_set_constraints_tool_treats_null_as_keep(isolated):
    """A model filling every schema field with null must not error or reset anything.

    The agent emits null for fields it is not changing; making that fatal used to loop the
    ReAct agent until its iteration limit. Passing a built ConstraintPatch hides this.
    """
    async def scenario():
        async with register.cualign(register.CuAlignToolConfig(), None) as group:
            tools = await group.get_all_functions()
            fn = next(f for k, f in tools.items() if k.split("__")[-1] == "set_constraints")
            token = CURRENT_RUN.set(PlanRun("r-null", "moderate", None, Constraints(lock=(13,), stage_cap=30)))
            try:
                all_null = dict.fromkeys(["allow_extraction", "lock", "ipr_exclude",
                                          "ipr_limit_mm", "stage_cap", "order"])
                assert await fn.ainvoke(all_null) == register.constraints_json(Constraints(lock=(13,), stage_cap=30))
                partial = await fn.ainvoke({**all_null, "order": "anterior_first"})
                assert partial["order"] == "anterior_first" and partial["lock"] == [13] and partial["stage_cap"] == 30
                cleared = await fn.ainvoke({**all_null, "clear_stage_cap": True})
                assert cleared["stage_cap"] is None and cleared["lock"] == [13]
            finally:
                CURRENT_RUN.reset(token)
    asyncio.run(scenario())
