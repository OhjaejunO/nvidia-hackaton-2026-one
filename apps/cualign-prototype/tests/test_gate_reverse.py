"""Approval-gate branches the demo flow never walks; each test fails when its branch is removed."""
import asyncio
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cualign.core import store as store_module
from cualign.core.constraints import Constraints
from cualign.core.service import PlanningService
from cualign.server import api
from cualign.agent.context import CURRENT_RUN, PlanRun
from cualign.agent import register

PASSED = {"status": "passed", "attempts": 1, "message": "memo", "error": None}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    # register imports OUT_DIR by name, so the tool export must be redirected there too.
    s = store_module.Store()
    for mod in (store_module, api, register):
        monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    monkeypatch.setattr(api, "STORE", s)
    monkeypatch.setattr(register, "STORE", s)
    return s


def make_plan(s):
    svc = PlanningService(s)
    return svc.stages(svc.target("moderate", "expansion_ipr", Constraints()), None)


def client():
    app = FastAPI()
    api.add_api_routes(app)
    return TestClient(app)


def test_export_tool_allows_after_approval(isolated, tmp_path):
    s = isolated
    pid = make_plan(s)

    async def scenario():
        async with register.cualign(register.CuAlignToolConfig(), None) as group:
            tools = await group.get_all_functions()
            export = next(f for k, f in tools.items() if k.split("__")[-1] == "export_stl")
            token = CURRENT_RUN.set(PlanRun("r-export", "moderate", pid, Constraints()))
            try:
                with pytest.raises(ValueError):
                    await export.ainvoke({"plan_id": pid})
                s.set_review(pid, PASSED)  # clears approval, so approve after it
                s.approve(pid, s.lock_fdi(pid))
                return await export.ainvoke({"plan_id": pid})
            finally:
                CURRENT_RUN.reset(token)

    out = asyncio.run(scenario())
    assert out["download_url"] == f"/api/plans/{pid}/stl.zip"
    assert out["kind"] == "per_tooth" and out["n_files"] == 0 and "잇몸" in out["print_models_note"]   # no gingiva scan
    with ZipFile(tmp_path / "stl" / f"{pid}.zip") as z:
        assert len([n for n in z.namelist() if n.startswith("stage_")]) == len(s.plans[pid]["stages"]) * 14
        assert "print_models/README.txt" in z.namelist()


def test_stl_zip_not_stale(isolated, tmp_path):
    with client() as c:
        plan = c.post("/api/plan", json={"case_id": "moderate", "allow_extraction": False}).json()["chosen"]
        pid = plan["plan_id"]
        assert c.post(f"/api/plans/{pid}/approval", json={"confirmed": True, "lock_fdi": api.STORE.lock_fdi(pid)}).status_code == 200
        stale = tmp_path / "stl" / f"{pid}.zip"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"STALE-MARKER")
        body = c.get(f"/api/plans/{pid}/stl.zip").content
    assert b"STALE-MARKER" not in body
    with ZipFile(BytesIO(body)) as z:
        assert len([n for n in z.namelist() if n.endswith(".stl")]) == plan["n_stages"] * 14


def test_no_approval_tool_names(isolated):
    async def names():
        async with register.cualign(register.CuAlignToolConfig(), None) as group:
            return [k.split("__")[-1] for k in await group.get_all_functions()]

    assert not [n for n in asyncio.run(names()) if re.search("approv|revoke", n)]


def test_agent_code_never_approves():
    agent_dir = Path(register.__file__).parent
    hits = [f"{p.name}: {m}" for p in agent_dir.rglob("*.py")
            for m in re.findall(r'\.approve\(|\.revoke\(|\["approval"\]\s*=', p.read_text(encoding="utf-8"))]
    assert not hits


def test_unreviewed_plan_not_approvable(isolated):
    s = isolated
    pid = make_plan(s)
    assert s.plans[pid]["review"]["status"] == "not_requested"
    with client() as c:
        assert c.post(f"/api/plans/{pid}/approval", json={"confirmed": True, "lock_fdi": api.STORE.lock_fdi(pid)}).status_code == 409
        s.set_review(pid, PASSED)
        assert c.post(f"/api/plans/{pid}/approval", json={"confirmed": True, "lock_fdi": api.STORE.lock_fdi(pid)}).status_code == 200
