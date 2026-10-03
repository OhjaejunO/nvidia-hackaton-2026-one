"""GET /api/case-list: samples + patient scans, each with a plan status card (#110). No mesh loads."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cualign.core import store
from cualign.core.samples import SAMPLES
from cualign.server import api

from test_patients import _scan_files


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUT_DIR", tmp_path)
    monkeypatch.setattr(store, "OUT_DIR", tmp_path)
    s = store.STORE                       # the process-global store: start every test empty
    for d in (s.cases, s.case_constraints, s.targets, s.plans):
        d.clear()
    s.active_case = None
    app = FastAPI()
    api.add_api_routes(app)
    with TestClient(app) as c:
        yield c


def _row(rows, case_id):
    return next(r for r in rows if r["case_id"] == case_id)


def test_fresh_store_lists_the_three_samples_as_plan_needed(client):
    rows = client.get("/api/case-list").json()["cases"]
    sample_ids = [s.case_id for s in SAMPLES.values() if s.available]
    assert [r["case_id"] for r in rows[:len(sample_ids)]] == sample_ids
    row = _row(rows, "poseidon-000097")
    assert row["kind"] == "sample" and row["status"] == "plan_needed" and row["status_ko"] == "계획 필요"
    assert row["plan"] is None and row["n_plans"] == 0 and row["patient"] is None and row["confirmed"] is None
    assert row["title"] == "심한 덧니, 발치 필요" and row["prescription"].startswith("제1소구치")
    assert row["reason"].startswith("총생 7.9 mm")      # the 「이유」 line under 처방 (samples only)
    assert row["badges"] == ["총생 7.9 mm", "발치"] and row["unsupported"] == []
    assert row["n_teeth"] == 14                                     # 15 STL files minus gingiva.stl


def test_cases_endpoint_is_unchanged(client):
    before = client.get("/api/cases").json()
    client.get("/api/case-list")
    assert client.get("/api/cases").json() == before


def test_patient_scan_moves_through_every_status(client):
    client.post("/api/patients", json={"alias": "3월 상담 A"})
    client.post("/api/patients/P0001/scans", files=_scan_files())
    case_id = "P0001-S1"

    row = _row(client.get("/api/case-list").json()["cases"], case_id)
    assert row["kind"] == "patient" and row["status"] == "scan_check" and row["status_ko"] == "스캔 확인 필요"
    assert row["confirmed"] is False and row["plan"] is None and row["n_plans"] == 0
    assert row["patient"] == {"patient_id": "P0001", "alias": "3월 상담 A", "scan_id": "S1"}
    assert row["subtitle"] == "3월 상담 A · S1" and row["n_teeth"] == 14 and row["unsupported"] == []

    client.post(f"/api/patients/P0001/scans/S1/confirm")
    row = _row(client.get("/api/case-list").json()["cases"], case_id)
    assert row["confirmed"] is True and row["status"] == "plan_needed" and row["plan"] is None

    failing = client.post("/api/plan", json={"case_id": case_id, "stage_cap": 1}).json()
    assert failing["chosen"] is None and failing["best_failed"]   # stage_cap=1 cannot pass
    row = _row(client.get("/api/case-list").json()["cases"], case_id)
    assert row["status"] == "violation" and row["status_ko"] == "위반 있음"
    assert row["plan"]["passed"] is False and row["plan"]["violations"] > 0
    assert row["n_plans"] == len(failing["tried"])

    passing = client.post("/api/plan", json={"case_id": case_id, "allow_extraction": False,
                                              "clear_stage_cap": True}).json()
    plan_id = passing["tried"][-1]["plan_id"]                        # representative = newest plan of the case
    row = _row(client.get("/api/case-list").json()["cases"], case_id)
    assert row["status"] == "awaiting_approval" and row["status_ko"] == "승인 대기"
    assert row["plan"]["plan_id"] == plan_id and row["plan"]["passed"] is True and row["plan"]["violations"] == 0

    assert client.post(f"/api/plans/{plan_id}/approval", json={"confirmed": True, "lock_fdi": api.STORE.lock_fdi(plan_id)}).status_code == 200
    row = _row(client.get("/api/case-list").json()["cases"], case_id)
    assert row["status"] == "approved" and row["status_ko"] == "승인됨" and row["plan"]["plan_id"] == plan_id
