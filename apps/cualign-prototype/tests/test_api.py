"""Offline artifact flow and missing-resource handling; no model calls."""
import asyncio
import threading
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cualign.server import api
from cualign.core import planner, store


def test_artifact_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUT_DIR", tmp_path)
    monkeypatch.setattr(store, "OUT_DIR", tmp_path)
    app = FastAPI()
    api.add_api_routes(app)
    with TestClient(app) as client:
        assert client.get("/ui/").status_code == 200
        assert client.get("/ui/app.js").status_code == 200
        for icon in ("favicon.png", "apple-touch-icon.png"):
            assert client.get(f"/ui/{icon}").headers["content-type"] == "image/png"
        result = client.post("/api/plan", json={"case_id": "moderate", "allow_extraction": False}).json()
        plan = result["chosen"] or result["best_failed"]
        pid = plan["plan_id"]
        detail = client.get(f"/api/plans/{pid}")
        assert detail.status_code == 200
        assert client.get(f"/api/plans/{pid}/stl.zip").status_code == 409
        assert client.post(f"/api/plans/{pid}/approval", json={"confirmed": True, "lock_fdi": api.STORE.lock_fdi(pid)}).status_code == 200
        output = client.get(f"/api/plans/{pid}/stl.zip")
        assert output.status_code == 200
        with ZipFile(BytesIO(output.content)) as archive:
            assert len([n for n in archive.namelist() if n.endswith('.stl')]) == plan["n_stages"] * 14


def test_missing_plan_returns_404():
    app = FastAPI()
    api.add_api_routes(app)
    with TestClient(app) as client:
        assert client.get("/api/plans/not-a-plan").status_code == 404
        assert client.get("/api/plans/not-a-plan/stl.zip").status_code == 404


def test_manual_review_route(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUT_DIR", tmp_path)
    monkeypatch.setattr(store, "OUT_DIR", tmp_path)
    from cualign.core.constraints import Constraints
    from cualign.core.service import PlanningService
    svc = PlanningService(store.STORE)
    pid = svc.stages(svc.target("moderate", "expansion_ipr", Constraints()))
    calls = []
    async def review(plan_id):
        calls.append(plan_id)
        return store.STORE.set_review(plan_id, {"status": "passed", "attempts": 1, "message": "메모", "error": None})
    bare = FastAPI()
    api.add_api_routes(bare)
    with TestClient(bare) as client:
        assert client.post(f"/api/plans/{pid}/review").status_code == 503
    app = FastAPI()
    api.add_api_routes(app, review=review)
    with TestClient(app) as client:
        assert client.post("/api/plans/not-a-plan/review").status_code == 404
        store.STORE.set_review(pid, {"status": "failed", "attempts": 2, "message": "검토 실패", "error": "timeout"})
        result = client.post(f"/api/plans/{pid}/review")
        assert result.status_code == 200 and result.json()["review"]["status"] == "passed"
        assert client.post(f"/api/plans/{pid}/review").status_code == 409
        fallback = client.post("/api/plan", json={"case_id": "moderate"}).json()["tried"][0]["plan_id"]
        assert client.post(f"/api/plans/{fallback}/review").status_code == 409
    assert calls == [pid]


def test_upload_without_tooth_files_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUT_DIR", tmp_path)
    app = FastAPI()
    api.add_api_routes(app)
    with TestClient(app) as client:
        r = client.post("/api/cases/upload", files=[("files", ("gingiva.stl", b"solid g\nendsolid g\n", "model/stl"))])
        assert r.status_code == 400


def test_stl_zip_builds_off_the_event_loop(tmp_path, monkeypatch):
    """While a download builds (seconds on a real scan), other requests are answered: the agent shares this
    process (#119). The build is held open until the other request is back."""
    monkeypatch.setattr(api, "OUT_DIR", tmp_path)
    monkeypatch.setattr(store, "OUT_DIR", tmp_path)
    from cualign.server import export_jobs
    monkeypatch.setattr(export_jobs, "JOBS", {})
    started, release, released = threading.Event(), threading.Event(), []
    real = export_jobs.print_models   # the build the approval starts (answer-polish (9)) runs this in its own thread

    def held(*args, **kwargs):
        started.set()
        released.append(release.wait(60))   # on the event loop nothing else runs, so this only times out
        return real(*args, **kwargs)

    def tiny_zip(case, stages, zip_path):    # the tooth files are not the point: no geometry work, so the timing below
        Path(zip_path).parent.mkdir(parents=True, exist_ok=True)   # measures only whether the loop is free while the build is held
        ZipFile(zip_path, "w").close()
        return zip_path

    monkeypatch.setattr(export_jobs, "print_models", held)
    monkeypatch.setattr(export_jobs.planner, "export_zip", tiny_zip)
    app = FastAPI()
    api.add_api_routes(app)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            result = (await client.post("/api/plan", json={"case_id": "moderate"})).json()
            pid = (result["chosen"] or result["best_failed"])["plan_id"]
            assert (await client.post(f"/api/plans/{pid}/approval", json={"confirmed": True, "lock_fdi": api.STORE.lock_fdi(pid)})).status_code == 200
            download = asyncio.create_task(client.get(f"/api/plans/{pid}/stl.zip"))
            assert await asyncio.to_thread(started.wait, 60)
            other = await asyncio.wait_for(client.get(f"/api/plans/{pid}"), 30)   # generous: a parallel --slow run loads the machine
            assert other.status_code == 200
            release.set()
            output = await download
            assert output.status_code == 200 and released == [True]
            with ZipFile(BytesIO(output.content)) as archive:
                assert "print_models/README.txt" in archive.namelist()
        assert not list((tmp_path / "stl").glob("*.tmp"))   # built under a temporary name, then moved into place
    asyncio.run(scenario())
