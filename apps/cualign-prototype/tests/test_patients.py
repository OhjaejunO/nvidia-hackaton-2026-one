"""Patient flow: register → upload a scan → input check → plan. Local files only; no model calls."""
import numpy as np
import pytest
import trimesh
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cualign.core import store
from cualign.core.case import Case
from cualign.server import api


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


def _scan_files(drop=(), transform=None, gum=False, open_crowns=False, rename=None):
    meshes = {i: m.copy() for i, m in Case.synthetic("mild").mesh.items() if i not in drop}   # calibrated: 2 mm crowding
    if open_crowns:     # cut the gingival face off, as a segmented crown comes out of a scan
        for m in meshes.values():
            m.update_faces(m.face_normals[:, 2] > -0.5)
            m.remove_unreferenced_vertices()
    if gum:             # a plate under the crowns
        lo, hi = np.min([m.bounds[0] for m in meshes.values()], 0), np.max([m.bounds[1] for m in meshes.values()], 0)
        g = trimesh.creation.box(extents=[hi[0] - lo[0] + 6, hi[1] - lo[1] + 6, 4])
        g.apply_translation([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] - 2])
        meshes["gingiva"] = g
    out = []
    for i, m in meshes.items():
        if transform is not None:
            m.apply_transform(transform)
        name = f"{(rename or (lambda u: u))(i)}.stl" if i != "gingiva" else "gingiva.stl"
        out.append(("files", (name, m.export(file_type="stl"), "model/stl")))
    return out


def _scanner_pose(flip):
    T = trimesh.transformations.euler_matrix(0.4, -0.3 + (np.pi if flip else 0.0), 1.1)
    T[:3, 3] = [31.0, -12.0, 57.0]
    return T


def test_register_upload_check_and_plan(client):
    assert client.post("/api/patients", json={"alias": "상담 010-1234-5678"}).status_code == 400   # phone number
    assert client.post("/api/patients", json={"alias": " "}).status_code == 400
    p = client.post("/api/patients", json={"alias": "3월 상담 A", "memo": "상악 총생"}).json()
    assert p["patient_id"] == "P0001" and p["scans"] == []

    gum_only = [("files", ("gingiva.stl", b"solid g\nendsolid g\n", "model/stl"))]
    assert client.post("/api/patients/P0001/scans", files=gum_only).status_code == 400
    assert client.post("/api/patients/P9999/scans", files=_scan_files()).status_code == 404

    r = client.post("/api/patients/P0001/scans", files=_scan_files())
    assert r.status_code == 200, r.text
    scan = r.json()
    assert scan["case_id"] == "P0001-S1" and scan["check"]["ready"] and scan["check"]["n_teeth"] == 14
    assert scan["check"]["missing"] == [] and scan["check"]["crowding_mm"] > 0

    # survives a restart: a fresh store opens the scan from disk by its case id
    store.STORE.cases.pop("P0001-S1", None)
    check = client.get("/api/cases/P0001-S1/check").json()
    assert check["teeth"] == list(range(2, 16))

    # planning waits for the dentist's confirmation on the input-check screen (every planning path)
    blocked = client.post("/api/plan", json={"case_id": "P0001-S1", "allow_extraction": False})
    assert blocked.status_code == 400 and "입력 확인" in blocked.json()["detail"]
    assert client.post("/api/patients/P0001/scans/S1/confirm").json()["confirmed_at"]
    plan = client.post("/api/plan", json={"case_id": "P0001-S1", "allow_extraction": False}).json()
    assert plan["case_id"] == "P0001-S1" and plan["tried"]
    detail = client.get("/api/patients/P0001").json()
    assert [s["scan_id"] for s in detail["scans"]] == ["S1"] and detail["scans"][0]["plans"] >= 1
    assert client.get("/api/patients").json()["patients"][0]["n_scans"] == 1
    assert all(not r["case_id"].startswith("P0001") for r in client.get("/api/cases").json()["cases"])


def test_scan_with_a_missing_tooth_is_not_plannable(client):
    client.post("/api/patients", json={"alias": "결손 케이스"})
    check = client.post("/api/patients/P0001/scans", files=_scan_files(drop=(7,))).json()["check"]
    assert check["missing"] == [7] and not check["ready"] and check["unsupported"]
    assert client.post("/api/patients/P0001/scans/S1/confirm").status_code == 409   # unsupported: cannot be confirmed


def test_unknown_patient_and_case(client):
    assert client.get("/api/patients/P0404").status_code == 404
    assert client.get("/api/cases/P0404-S1/check").status_code == 404


@pytest.mark.parametrize("gum,open_crowns,basis", [(True, False, "gingiva"), (False, True, "cervical")])
def test_scanner_coordinates_are_put_into_the_core_frame(client, gum, open_crowns, basis):
    client.post("/api/patients", json={"alias": "스캐너 좌표"})
    ref = client.post("/api/patients/P0001/scans", files=_scan_files(gum=gum, open_crowns=open_crowns)).json()
    moved = client.post("/api/patients/P0001/scans", files=_scan_files(transform=_scanner_pose(flip=True), gum=gum,
                                                                         open_crowns=open_crowns)).json()
    assert moved["orientation"]["basis"] == basis and moved["orientation"]["side"] == "ok"
    assert moved["orientation"]["rotation_deg"] > 90                      # it really was upside down
    assert abs(moved["check"]["crowding_mm"] - ref["check"]["crowding_mm"]) <= 0.2
    mesh = client.get("/api/cases/P0001-S2/mesh").json()
    tops = [max(v[2] for v in t["v"]) for t in mesh["teeth"].values()]
    assert abs(float(np.median(tops))) < 0.3                              # occlusal plane at z = 0
    x = {i: np.mean([v[0] for v in mesh["teeth"][str(i)]["v"]]) for i in (2, 8, 15)}
    y8 = np.mean([v[1] for v in mesh["teeth"]["8"]["v"]])
    y2 = np.mean([v[1] for v in mesh["teeth"]["2"]["v"]])
    assert x[2] < x[15] and y8 > y2                                       # tooth 2 on the left, incisors to +y


def test_reversed_numbers_are_flagged_and_can_be_mirrored(client):
    client.post("/api/patients", json={"alias": "번호 반대"})
    scan = client.post("/api/patients/P0001/scans", files=_scan_files(gum=True, rename=lambda u: 17 - u)).json()
    assert scan["orientation"]["side"] == "reversed"
    assert client.post("/api/patients/P0001/scans/S1/confirm").status_code == 200
    fixed = client.post("/api/patients/P0001/scans/S1/mirror").json()
    assert fixed["orientation"]["side"] == "ok" and fixed["confirmed_at"] is None   # renumbering clears the confirmation
    assert fixed["teeth"] == list(range(2, 16))


def test_one_piece_scan_is_explained_and_deletes_work(client):
    client.post("/api/patients", json={"alias": "삭제 확인"})
    for name in ("upperjaw.stl", "000018.stl"):      # an all-digit scan id is not a tooth number
        r = client.post("/api/patients/P0001/scans", files=[("files", (name, b"solid a\nendsolid a\n", "model/stl"))])
        assert r.status_code == 400 and "한 덩어리" in r.json()["detail"], name
    client.post("/api/patients/P0001/scans", files=_scan_files())
    assert client.delete("/api/patients/P0001/scans/S1").json()["scans"] == []
    assert client.get("/api/cases/P0001-S1/check").status_code == 404
    assert client.delete("/api/patients/P0001").status_code == 200
    assert client.get("/api/patients/P0001").status_code == 404


# --------------------------------------------------------------------------- PR #36 review regressions
def _confirmed_plan(client, pid="P0001", sid="S1"):
    client.post(f"/api/patients/{pid}/scans/{sid}/confirm")
    res = client.post("/api/plan", json={"case_id": f"{pid}-{sid}", "extraction": [5, 12]}).json()
    return (res["chosen"] or res["best_failed"])["plan_id"]


def test_scan_ids_are_never_reused(client):
    client.post("/api/patients", json={"alias": "스캔 번호"})
    client.post("/api/patients/P0001/scans", files=_scan_files())
    client.post("/api/patients/P0001/scans", files=_scan_files())
    client.post("/api/patients/P0001/scans/S2/confirm")
    client.delete("/api/patients/P0001/scans/S1")
    new = client.post("/api/patients/P0001/scans", files=_scan_files(drop=(7,))).json()
    assert new["scan_id"] == "S3" and new["check"]["n_teeth"] == 13 and not new["check"]["confirmed"]
    ids = [s["scan_id"] for s in client.get("/api/patients/P0001").json()["scans"]]
    assert ids == ["S2", "S3"]
    r = client.post("/api/plan", json={"case_id": "P0001-S3"})       # no confirmation inherited, no plan made
    assert r.status_code == 400 or not r.json()["tried"]
    assert not [q for q in store.STORE.plans.values() if q["case_id"] == "P0001-S3"]


def test_deleting_a_patient_removes_plans_and_the_id_is_not_reused(client, tmp_path):
    client.post("/api/patients", json={"alias": "지울 환자"})
    client.post("/api/patients/P0001/scans", files=_scan_files())
    pid = _confirmed_plan(client)
    assert (tmp_path / "plans" / f"{pid}.json").exists()
    out = client.delete("/api/patients/P0001").json()
    assert out["plans"] >= 1 and not (tmp_path / "plans" / f"{pid}.json").exists()
    assert client.get(f"/api/plans/{pid}").status_code == 404
    assert client.post("/api/patients", json={"alias": "새 환자"}).json()["patient_id"] == "P0002"


def test_renumbering_makes_earlier_plans_stale(client):
    from cualign.core.service import PlanningService
    client.post("/api/patients", json={"alias": "번호 변경"})
    client.post("/api/patients/P0001/scans", files=_scan_files())
    plan_id = _confirmed_plan(client)
    target = store.STORE.targets[store.STORE.plans[plan_id]["target_id"]]
    service = PlanningService(store.STORE)
    # the plan's own strategy: the prescription refuses any other (PlanningService.target)
    tid = service.target("P0001-S1", target["info"]["strategy"], store.STORE.constraints_for("P0001-S1"))
    assert client.post(f"/api/plans/{plan_id}/approval", json={"confirmed": True, "lock_fdi": api.STORE.lock_fdi(plan_id)}).status_code == 200
    assert client.get(f"/api/plans/{plan_id}/stl.zip").status_code == 200

    check = client.post("/api/patients/P0001/scans/S1/mirror").json()
    assert check["revision"] == 2 and not check["confirmed"]
    plan = client.get(f"/api/plans/{plan_id}").json()
    assert plan["input_stale"] and plan["approval"] is None
    assert client.get(f"/api/plans/{plan_id}/stl.zip").status_code == 409
    assert client.post(f"/api/plans/{plan_id}/approval", json={"confirmed": True, "lock_fdi": api.STORE.lock_fdi(plan_id)}).status_code == 409
    with pytest.raises(ValueError):
        service.stages(tid)                         # a target made before the renumbering
    with pytest.raises(ValueError):
        service.validate(plan_id)
    # confirming what the screen showed before the change is refused; the current revision is accepted
    assert client.post("/api/patients/P0001/scans/S1/confirm", json={"revision": 1}).status_code == 409
    assert client.post("/api/patients/P0001/scans/S1/confirm", json={"revision": 2}).status_code == 200
    assert client.get(f"/api/plans/{plan_id}").json()["input_stale"]      # still the old numbering
    assert [p["input_stale"] for p in client.get("/api/plans", params={"case_id": "P0001-S1"}).json()["plans"]] == [True]
    assert target["input_revision"] == 1


def test_bad_uploads_are_refused_before_anything_is_stored(client):
    client.post("/api/patients", json={"alias": "입력 검사"})
    files = _scan_files()
    lower = files + [("files", ("36.stl", files[0][1][1], "model/stl"))]   # FDI lower left first molar (#113)
    assert "하악" in client.post("/api/patients/P0001/scans", files=lower).json()["detail"]
    dup = files + [files[0]]
    assert client.post("/api/patients/P0001/scans", files=dup).status_code == 400
    empty = [f if f[1][0] != "2.stl" else ("files", ("2.stl", b"solid x\nendsolid x\n", "model/stl")) for f in files]
    r = client.post("/api/patients/P0001/scans", files=empty)
    assert r.status_code == 400 and "2.stl" in r.json()["detail"]
    assert client.get("/api/patients/P0001").json()["scans"] == []
    assert not [d for d in (store.OUT_DIR / "patients" / "P0001" / "scans").glob("*")]   # nothing left behind


@pytest.mark.parametrize("pid", ["..", "%2e%2e", "x", "P1", "P0001%2f.."])
def test_patient_ids_are_never_used_as_paths(client, pid):
    client.post("/api/patients", json={"alias": "경로"})
    assert client.get(f"/api/patients/{pid}").status_code == 404
    assert client.delete(f"/api/patients/{pid}").status_code in (404, 405)


def test_closed_crowns_without_gum_keep_the_input_orientation_but_are_plannable(client):
    """#104 «방향 판정 불가» (v2 board 09-c): closed crowns (no cut rim to vote from) and no gingiva give no evidence
    for the occlusal side, so the scan keeps its input orientation (basis none) while still being ready to plan.
    The default `_scan_files()` is exactly that scan; the screen session uses it for the state (scripts/orientation_none_scan.py)."""
    client.post("/api/patients", json={"alias": "방향 판정 불가"})
    scan = client.post("/api/patients/P0001/scans", files=_scan_files()).json()
    assert scan["orientation"]["basis"] == "none" and scan["orientation"]["side"] == "ok"
    assert scan["check"]["ready"] and scan["check"]["n_teeth"] >= 6 and scan["check"]["unsupported"] == []
    assert client.post("/api/patients/P0001/scans/S1/confirm").status_code == 200   # the dentist may still confirm


def test_v2_rejections_and_blocks_use_the_design_wording(client, monkeypatch):
    """#112: v2 boards 08 (upload rejected, 5 kinds) and 09 (input check blocked, 3 kinds) — status codes and wording as
    designed, so the screen can show them as they are."""
    from cualign.server import api
    client.post("/api/patients", json={"alias": "설계 문구"})
    post = lambda files: client.post("/api/patients/P0001/scans", files=files)   # noqa: E731

    r = post(_scan_files() + [("files", ("38.stl", b"solid x\nendsolid x\n", "model/stl"))])   # FDI 38 = lower arch (#113)
    assert (r.status_code, r.json()["detail"]) == (400, "38.stl: 하악 번호(31~48)입니다. 지금은 상악 스캔만 받습니다.")
    r = post([("files", ("upper_arch.stl", b"solid x\nendsolid x\n", "model/stl"))])
    assert (r.status_code, r.json()["detail"]) == (400, "upper_arch.stl: 한 덩어리 악궁 스캔으로 보입니다. 지금은 치아별로 나뉜 파일"
                                                        "(11.stl … 27.stl, 선택 gingiva.stl)만 받습니다. 자동 치아 분리는 실험 단계입니다.")
    assert (api.MAX_FILE_BYTES >> 20, api.MAX_UPLOAD_BYTES >> 20) == (60, 400)
    monkeypatch.setattr(api, "MAX_FILE_BYTES", 1 << 20)                       # same wording, a 1 MB limit for the test
    r = post([("files", ("8.stl", b"\0" * ((1 << 20) + 1), "model/stl"))])
    assert (r.status_code, r.json()["detail"]) == (413, "파일이 너무 큽니다(파일당 1MB, 한 번에 400MB까지).")
    monkeypatch.setattr(api, "MAX_FILE_BYTES", 60 << 20)
    r = post(_scan_files() + [("files", ("8.stl", b"solid x\nendsolid x\n", "model/stl"))])
    assert (r.status_code, r.json()["detail"]) == (400, "8.stl: 같은 번호의 파일이 두 개 있습니다.")
    r = post(_scan_files(drop=(8,)) + [("files", ("8.stl", b"this is not an stl file at all", "model/stl"))])
    assert r.status_code == 400 and r.json()["detail"].startswith("스캔을 읽지 못했습니다: "), r.text
    assert client.get("/api/patients/P0001").json()["scans"] == []               # rejected uploads are not kept

    check = post(_scan_files(drop=(7,))).json()["check"]
    assert not check["ready"] and check["unsupported"] == ["12번 결손: 결손 공간이 있는 악궁은 아직 계획하지 않음 (연속된 치열만 지원)"]
    check = post(_scan_files(drop=tuple(range(2, 12)))).json()["check"]            # 12..15: four teeth
    assert not check["ready"] and check["unsupported"] == ["치아 4개: 악궁을 맞추기에 부족 (6개 이상 필요)"]
    scan = post(_scan_files()).json()                                              # closed crowns, no gum: no evidence
    assert scan["check"]["ready"] and scan["orientation"]["basis"] == "none"
    assert scan["orientation"]["note"] == "치아 14개 — 방향을 정할 수 없어 입력 방향 그대로 둠"
