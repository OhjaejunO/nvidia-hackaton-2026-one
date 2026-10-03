"""The export zip is built in the background from approval on (server/export_jobs.py, answer-polish (9)).

Approval starts the build; GET /api/plans/{id}/export-status counts the stages done; stl.zip serves the cached file at
once when it is ready, or waits for the running build; a plan is immutable once approved, so the cache is keyed by the
approval's fingerprint and a second download builds nothing."""
import time
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cualign.core import samples
from cualign.core import store as store_mod
from cualign.core.case import Case
from cualign.core.constraints import Constraints
from cualign.core import planner
from cualign.server import api, export_jobs


def _client(monkeypatch, tmp_path):
    monkeypatch.setattr(store_mod, "OUT_DIR", tmp_path)
    monkeypatch.setattr(api, "OUT_DIR", tmp_path)
    monkeypatch.setattr(api, "STORE", store_mod.Store())
    monkeypatch.setattr(export_jobs, "JOBS", {})
    app = FastAPI()
    api.add_api_routes(app)
    return TestClient(app)


def _passing_plan(client, case_id):
    res = client.post("/api/plan", json={"case_id": case_id}).json()
    assert res["chosen"], res
    return res["chosen"]["plan_id"]


def _wait_ready(client, pid, seconds=60):
    deadline = time.time() + seconds
    while time.time() < deadline:
        st = client.get(f"/api/plans/{pid}/export-status").json()
        if st["ready"] or not st["building"]:
            return st
        time.sleep(0.1)
    raise AssertionError("the export did not finish")


def test_approval_starts_the_build_and_the_download_serves_the_cache(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    pid = _passing_plan(client, "moderate")          # a synthetic case: no gingiva, so the print models are skipped, fast
    n = len(client.get(f"/api/plans/{pid}").json()["stages"])
    assert client.get(f"/api/plans/{pid}/export-status").json() == {"building": False, "done": 0, "total": n, "ready": False}
    assert client.get(f"/api/plans/{pid}/stl.zip").status_code == 409     # not approved: nothing is built either
    assert export_jobs.JOBS == {}
    assert client.post(f"/api/plans/{pid}/approval", json={"confirmed": True, "lock_fdi": api.STORE.lock_fdi(pid)}).status_code == 200
    assert pid in export_jobs.JOBS                                          # the build started with the approval
    st = _wait_ready(client, pid)
    assert st == {"building": False, "done": n, "total": n, "ready": True, "stl": n * 14}   # no gingiva: no print models
    job = export_jobs.JOBS[pid]
    r = client.get(f"/api/plans/{pid}/stl.zip")
    assert r.status_code == 200 and export_jobs.JOBS[pid] is job              # served from the cache, no new build
    assert r.headers["X-Cualign-Print-Models"].startswith("skipped")
    with zipfile.ZipFile(tmp_path / "stl" / f"{pid}.zip") as z:
        names = z.namelist()
    assert len([m for m in names if m.startswith("stage_")]) == n * 14 and "print_models/README.txt" in names
    # revoked and approved again: the same snapshot, the same fingerprint, the same cache
    client.delete(f"/api/plans/{pid}/approval")
    assert client.get(f"/api/plans/{pid}/stl.zip").status_code == 409
    client.post(f"/api/plans/{pid}/approval", json={"confirmed": True, "lock_fdi": api.STORE.lock_fdi(pid)})
    assert export_jobs.JOBS[pid] is job and client.get(f"/api/plans/{pid}/stl.zip").status_code == 200


def test_download_waits_for_a_running_build_and_a_lost_file_is_rebuilt(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    pid = _passing_plan(client, "moderate")
    client.post(f"/api/plans/{pid}/approval", json={"confirmed": True, "lock_fdi": api.STORE.lock_fdi(pid)})
    r = client.get(f"/api/plans/{pid}/stl.zip")                              # right after approval: waits for the build
    assert r.status_code == 200 and export_jobs.JOBS[pid].ready
    job = export_jobs.JOBS[pid]
    (tmp_path / "stl" / f"{pid}.zip").unlink()
    assert client.get(f"/api/plans/{pid}/stl.zip").status_code == 200 and export_jobs.JOBS[pid] is not job   # rebuilt
    assert (tmp_path / "stl" / f"{pid}.zip").exists()


@pytest.mark.parametrize("sample", [s for s in samples.SAMPLES.values() if s.available][:1], ids=lambda s: s.case_id)
def test_print_models_are_built_per_stage_in_parallel_with_progress(sample, tmp_path):
    case = Case.from_dir(sample.folder)
    c = Constraints().patched(dict(sample.constraints))
    strategy = "extraction" if c.extraction else "expansion"
    target, info = planner.propose_target(case, strategy, constraints=c)
    stages, _ = planner.plan_stages(case, target, order=c.order)
    stages = stages[:4]
    ticks = []
    report = export_jobs.build_zip(planner.cut_case(case, info), stages, sample.case_id, tmp_path / "x.zip", progress=lambda: ticks.append(1), workers=4)
    assert report["status"] == "ok" and report["n_files"] == 4 and len(ticks) == 4
    with zipfile.ZipFile(tmp_path / "x.zip") as z:
        models = [m for m in z.namelist() if m.startswith("print_models/") and m.endswith(".stl")]
        assert len(models) == 4 and z.read(models[0])[:5] != b""
    # the same report as the serial builder
    planner.export_zip(planner.cut_case(case, info), stages, str(tmp_path / "y.zip"))
    serial = planner.export_print_models(planner.cut_case(case, info), stages, str(tmp_path / "y.zip"), sample.case_id)
    assert serial["status"] == "ok" and serial["n_files"] == 4 and serial["n_stages"] == report["n_stages"]
