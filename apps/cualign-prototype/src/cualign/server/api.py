"""Artifact API. Approval is checked on every export, including cached files."""
from __future__ import annotations
import asyncio
import json
import logging
import re
import uuid
import weakref

import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cualign.agent import steps
from cualign.server import export_jobs
from cualign.server.plan_events import cancel_case, setup_view
from cualign.agent.reviewer import MANUAL_RETRY
from cualign.core import Case, manual, patients, planner, samples
from cualign.server import manual_api
from cualign.core.constraints import ConstraintPatch, reason_ko
from cualign.core.service import PlanningService
from cualign.core.store import OUT_DIR, STORE

STATIC_DIR = Path(__file__).resolve().parent / "static"


class RulePlanRequest(ConstraintPatch):
    case_id: str | None = None
    parent_plan_id: str | None = None


class ReplayRequest(BaseModel):
    step: str = Field(pattern="^(setup|target|stages|plan|cap|compare)$")   # plan = stages (the name before the step flow)
    base_plan_id: str | None = None


class ApprovalRequest(BaseModel):
    confirmed: bool = False
    # the locked teeth (FDI) the dentist confirmed on the screen; approval refuses without them (Store.approve)
    lock_fdi: list[int] | None = None


class ConfirmRequest(BaseModel):
    revision: int | None = None


class FollowupRequest(BaseModel):
    messages: list[dict] = Field(default_factory=list, max_length=200)


class PatientRequest(BaseModel):
    alias: str = Field(max_length=patients.ALIAS_MAX)
    memo: str = Field(default="", max_length=patients.MEMO_MAX)


MAX_FILE_BYTES = 60 * 1024 * 1024       # one tooth or gum STL (PoC limit)
MAX_UPLOAD_BYTES = 400 * 1024 * 1024    # one scan upload


FDI_ONLY_STEMS = set(range(17, 19)) | set(range(21, 29))   # stems that exist only in FDI on the upper arch

log = logging.getLogger(__name__)


def _batch_is_fdi(names) -> bool:
    """A scan upload is FDI-numbered when any file carries a stem only FDI has (17·18, 21..28); a full FDI arch
    always does. Otherwise the stems are the app's Universal 1..16 (existing scans, tests). 11..16 alone stay
    ambiguous and are read as Universal."""
    stems = {int(Path(n or "").stem) for n in names if Path(n or "").stem.isdigit()}
    return bool(stems & FDI_ONLY_STEMS)


def _scan_file_name(name: str, fdi: bool = False) -> tuple[str | None, str]:
    """(stored name, kind): kind is "tooth", "gum", "lower" (lower arch, not supported) or "other".

    Stored names are Universal 1..16. With fdi=True the stems are the dentist's FDI numbers (#113): 18..11 → 1..8,
    21..28 → 9..16, 31..48 → lower. With fdi=False they are Universal: 1..16 → as is, 17..32 → lower.
    """
    name = Path(name or "").name
    if name.lower() == "gingiva.stl":
        return "gingiva.stl", "gum"
    # a tooth number is 1..48 without padding: "000018.stl" is a scan id, not tooth 18
    if not (name.lower().endswith(".stl") and re.fullmatch(r"[1-9]|[1-4]\d", Path(name).stem)):
        return None, "other"
    n = int(Path(name).stem)
    if fdi:
        if 11 <= n <= 18:
            return f"{19 - n}.stl", "tooth"
        if 21 <= n <= 28:
            return f"{n - 12}.stl", "tooth"
        return None, "lower" if 31 <= n <= 48 else "other"
    if n <= 16:
        return f"{n}.stl", "tooth"
    return None, "lower"   # Universal 17..32 and any FDI lower stem in a Universal batch


def _patient_json(pid: str) -> dict:
    p = patients.get_patient(pid)
    for s in p["scans"]:
        s["plans"] = sum(1 for q in STORE.plans.values() if q["case_id"] == s["case_id"])
    return p


_GUM_FILLED: dict[tuple, dict] = {}   # (case_id, gum vertex count, gum face count) -> {"gum_filled", "gum_fill"}


def gum_filled_view(case_id: str, case, gum: dict) -> dict:
    """`gum_filled` (same {v, f} shape as `gum`, the tooth trench covered, see core/gum_fill.py) and `gum_fill` (what was
    done). Computed once per case and gum (a re-uploaded scan changes the counts) and kept in memory."""
    key = (case_id, len(gum["v"]), len(gum["f"]))
    if key not in _GUM_FILLED:
        import numpy as np
        import trimesh
        from cualign.core import gum_fill
        source = "scan" if case.gum_scan is not None else "procedural"
        mesh = trimesh.Trimesh(np.asarray(gum["v"], float), np.asarray(gum["f"]), process=False)
        try:
            filled, meta = gum_fill.fill_sockets(mesh, case.anchor, [case.mesh[i] for i in case.ids])
        except Exception as e:   # noqa: BLE001 - the viewer keeps the plain gum rather than losing the case
            filled, meta = mesh, {"filled": False, "sockets": [], "error": str(e)}
        _GUM_FILLED[key] = {"gum_filled": gum if filled is mesh else gum_fill.to_json(filled),
                            "gum_fill": {"source": source, **meta}}
    return _GUM_FILLED[key]


_TEETH_CLOSED: dict[tuple, tuple] = {}   # (case_id, per-tooth vertex and face counts) -> ({tooth: closed {v, f}}, {tooth: how})


def closed_teeth_view(case_id: str, teeth: dict) -> tuple[dict, dict]:
    """`teeth` with each crown's open margin closed by a root stub, or a smoothed dome where the margin is split or short
    (core/ipr_cut.closed_json) — the view only — and {tooth: "stub" | "dome" | "closed"}. Kept in memory per case and
    scan, like gum_filled_view."""
    key = (case_id, tuple((k, len(t["v"]), len(t["f"])) for k, t in sorted(teeth.items())))
    if key not in _TEETH_CLOSED:
        from cualign.core.ipr_cut import closed_json
        done = {k: closed_json(t) for k, t in teeth.items()}
        how = {k: d[1] for k, d in done.items()}
        if dome := sorted((k for k, h in how.items() if h == "dome"), key=int):
            log.info("cuAlign: %s crowns %s closed by a dome, not a root stub (split or short margin)", case_id, dome)
        _TEETH_CLOSED[key] = ({k: d[0] for k, d in done.items()}, how)
    return _TEETH_CLOSED[key]


_MESH_JSON: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()   # Case -> the scan part of /mesh as JSON text


def mesh_base_json(cid: str, case) -> str:
    """The scan part of the mesh response (viewer_json with the crowns closed, gum_filled, gum_fill) as JSON text,
    made once per loaded case: decimating the crowns again and FastAPI's encoding of the ~2.8 MB of nested lists took
    ~0.7 s of every opening. Keyed by the Case object, so a re-uploaded scan (a new Case) gets its own."""
    if case not in _MESH_JSON:
        data = case.viewer_json()
        data["teeth"], data["teeth_cap"] = closed_teeth_view(cid, data["teeth"])   # a bared side shows no hole (the IPR-cut crowns are closed already)
        _CLOSED_CROWNS[case] = data["teeth"]
        data.update(gum_filled_view(cid, case, data["gum"]))
        _MESH_JSON[case] = json.dumps(data, separators=(",", ":"))
    return _MESH_JSON[case]


_CLOSED_CROWNS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()   # Case -> the crowns /mesh sends ({tooth: {v, f}})


def closed_cut_json(cid: str, case, cut) -> dict:
    """`teeth_cut` / `ipr_cut` like Case.cut_json, but each cut crown is the crown /mesh shows (decimated, its margin
    closed by the root stub) sliced on the cut's own planes, not the core's cut of the raw scan decimated afterwards: that
    one closes the open margin with a flat fan and, decimated with it, showed a jagged grey band along the gum line next
    to the scan's crowns. The planes and amounts are the core's (`cut.ipr_cut`); the view only."""
    from cualign.core.ipr_cut import CAP_TOL_DECIMATED, _fill_loops, _slice, cut_faces
    import trimesh
    info = getattr(cut, "ipr_cut", None) or {}
    if info and case not in _CLOSED_CROWNS:
        mesh_base_json(cid, case)
    teeth, faces = {}, {}
    for i, e in info.items():
        t = _CLOSED_CROWNS[case][str(i)]
        m = trimesh.Trimesh(np.asarray(t["v"], float), np.asarray(t["f"]), process=False)
        for pl in e["planes"]:
            m = _fill_loops(_slice(m, np.asarray(pl["n"], float), float(pl["c"])))
        teeth[str(i)] = {"v": np.round(m.vertices, 3).tolist(), "f": m.faces.tolist()}
        faces[str(i)] = {"mm": e["mm"], "faces": cut_faces(m, e["planes"], tol=CAP_TOL_DECIMATED).tolist()}
    return {"teeth_cut": teeth, "ipr_cut": faces}


def ipr_cut_view(cid: str, case, plan_id: str | None = None, target_id: str | None = None) -> dict:
    """`teeth_cut` / `ipr_cut` (see Case.cut_json) of the crowns with this case's IPR cut (#62), and `plan_id`: the plan
    whose IPR it is - `plan_id` if given, else the case's representative plan (approved, else latest). No plan, or a
    plan without IPR: `plan_id` None / the plan, and both maps empty. With `target_id` (the step flow's target turn,
    before any plan exists) the cut is the target's own; then `target_id` is in the response and `plan_id` is None."""
    if target_id:
        t = STORE.targets.get(target_id)
        if t is None or t["case_id"] != cid:
            return {"plan_id": None, "target_id": None, "teeth_cut": {}, "ipr_cut": {}}
        return {"plan_id": None, "target_id": target_id, **closed_cut_json(cid, case, planner.cut_case(case, t["info"]))}
    rec = STORE.plans.get(plan_id) if plan_id else _representative_plan(STORE.current_plan_ids_for(cid))
    if rec is None or rec["case_id"] != cid:
        return {"plan_id": None, "teeth_cut": {}, "ipr_cut": {}}
    return {"plan_id": rec["plan_id"], **closed_cut_json(cid, case, planner.cut_case(case, rec["info"]))}


PREVIOUS_CALCULATION = "이전 계산의 계획입니다(계산 코어가 바뀌었습니다). 이 계획을 기준으로 이어갈 수 없으니 다시 계산해 주세요."


def _summary(pid):
    p = STORE.plan_json(pid)
    return {"plan_id": pid, "case_id": p["case_id"], "parent_plan_id": p["parent_plan_id"],
            "strategy": p["strategy"], "n_stages": p["info"]["n_stages"], "months": p["info"]["months"],
            "passed": p["passed"], "violations": len(p["violations"]), "by_type": planner.summarize(p["violations"]),
            "constraints": p["constraints"], "review": p["review"], "approval": p["approval"],
            "input_stale": p["input_stale"],   # same rule as GET /api/plans/{id}: the scan was renumbered since
            "manual": (p.get("target") or {}).get("source") == "manual",   # staged from a hand-edited target (직접 이동)
            "previous_calculation": p["previous_calculation"],   # computed by another core version: listed as «이전 계산» only (10)
            "rule_run": p.get("rule_run")}   # None unless a rule plan made it: its group, the chosen plan, all_failed, tried


def rule_based_plan(case_id=None, allow_extraction=None, stage_cap=None, order=None, *,
                    changes=None, parent_plan_id=None, extraction=None, ipr_surfaces=None):
    cid, case = STORE.load_case(case_id)
    why = planner.unsupported_reasons(case)
    if why:
        return {"case_id": cid, "chosen": None, "unsupported": why, "tried": []}
    before = STORE.constraints_for(cid, parent_plan_id)
    if changes is None:
        changes = {"stage_cap": stage_cap}
        if extraction is not None:            # the prescribed teeth (#56); [] = non-extraction
            changes["extraction"] = list(extraction)
        elif allow_extraction is not None:    # legacy: false clears, true needs the teeth already prescribed
            changes["allow_extraction"] = allow_extraction
        if order is not None:
            changes["order"] = order
        if ipr_surfaces is not None:          # per-contact IPR (#57), Universal numbers as the CLI parsed them
            changes["ipr_surfaces"] = list(ipr_surfaces)
    c = before.patched(changes)
    c.check_case(case.ids)
    STORE.add_edit(cid, before, c)   # the form's 다시 계산 / 조건으로 계산: the dentist set these by hand (request_check)
    service = PlanningService(STORE)
    ids = service.compare(cid, c, parent_plan_id=parent_plan_id)
    STORE.case_constraints[cid] = c
    # the plans of this one request are alternatives to each other: each carries the run (group, the plan chosen, whether
    # every allowed strategy broke a rule, what was tried), so the screen shows one card and knows it after a reload
    first = _rule_plan_result(cid, ids)
    picked = first.get("chosen") or first.get("best_failed")
    run = {"group_id": "g" + uuid.uuid4().hex[:8], "chosen_plan_id": picked["plan_id"] if picked else None,
           "all_failed": first["chosen"] is None, "tried": [p["strategy"] for p in first["tried"]]}
    for pid in ids:
        STORE.plans[pid]["rule_run"] = dict(run)
        STORE.set_review(pid, {"status": "skipped", "attempts": 0,   # persists the plan, rule_run with it
                              "message": "규칙 폴백 — 검토 에이전트 미실행", "error": None})
    out = _rule_plan_result(cid, ids)
    chosen = out.get("chosen") or out.get("best_failed")
    if chosen:   # the fallback stands in for a stages turn: the flow is at stages with this plan
        STORE.set_flow(cid, "stages", constraints=c, target_id=STORE.plans[chosen["plan_id"]].get("target_id"), plan_id=chosen["plan_id"])
    return out


def _constraint_dump(record: dict) -> dict:
    constraints = record["constraints"]
    if isinstance(constraints, dict):
        return constraints
    return constraints.model_dump(mode="json")


def _rule_plan_result(cid, ids):
    tried = [_summary(pid) for pid in ids]
    chosen = next((p for p in tried if p["passed"]), None)
    result = {"case_id": cid, "chosen": chosen, "tried": tried}
    if chosen is None:
        result["best_failed"] = min(tried, key=lambda p: (p["violations"], STORE._record(p["plan_id"])["info"]["space_deficit_mm"]))
    return result


_CASE_LIST_STATUS_KO = {"scan_check": "스캔 확인 필요", "plan_needed": "계획 필요", "violation": "위반 있음",
                        "awaiting_approval": "승인 대기", "approved": "승인됨"}


def _representative_plan(plan_ids: list[str]) -> dict | None:
    """승인된 계획이 있으면 그중 가장 최근 것, 없으면 그 케이스의 가장 최근 계획(#110)."""
    approved = [STORE.plans[pid] for pid in plan_ids if STORE.plans[pid]["approval"]]
    if approved:
        return approved[-1]
    return STORE.plans[plan_ids[-1]] if plan_ids else None


def _plan_row(rec: dict | None) -> dict | None:
    if rec is None:
        return None
    return {"plan_id": rec["plan_id"], "strategy": rec["strategy"], "n_stages": rec["info"]["n_stages"],
            "passed": not rec["violations"], "violations": len(rec["violations"])}


def _case_status(confirmed: bool | None, rep: dict | None) -> str:
    if confirmed is False:      # patient scan the dentist has not checked yet
        return "scan_check"
    if rep is None:
        return "plan_needed"
    if rep["approval"]:
        return "approved"
    if rep["violations"]:
        return "violation"
    return "awaiting_approval"


def add_api_routes(app: FastAPI, review=None, followup=None):
    """`review(plan_id)` runs the bounded reviewer outside a chat request: the dentist's «검토 다시 요청», and the
    chat stream's fallback when the agent skipped the reviewer (plan_events.py reads it from app.state).
    None when no reviewer model is wired. `followup(messages)` writes the question card shown after each agent turn
    (#90); None means no cards."""
    app.state.cualign_review = review

    @app.post("/api/followup")
    async def next_followup(req: FollowupRequest):
        """{"question": {...}} or {"question": null}: the card is optional, so this never fails the screen."""
        if followup is None:
            return {"question": None}
        return {"question": await followup(req.messages)}

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/ui/")

    @app.get("/api/cases")
    async def list_cases():
        return {"cases": STORE.available_cases(), "active": STORE.active_case}

    @app.get("/api/case-list")
    async def case_list():
        """샘플 3건과 환자 스캔 전체를 계획 상태와 함께 한 화면에 준다(#110). 메시는 로드하지 않는다."""
        rows = []
        for s in samples.SAMPLES.values():
            if not s.available:
                continue
            n_teeth = sum(1 for f in s.folder.glob("*.stl") if f.name != "gingiva.stl")
            plan_ids = STORE.plan_ids_for(s.case_id)
            rep = _representative_plan(plan_ids)
            status = _case_status(None, rep)
            rows.append({"case_id": s.case_id, "kind": "sample", "title": s.title, "subtitle": s.summary,
                        "status": status, "status_ko": _CASE_LIST_STATUS_KO[status], "plan": _plan_row(rep),
                        "n_plans": len(plan_ids), "prescription": s.prescription, "reason": s.reason, "badges": list(s.badges),
                        "patient": None, "confirmed": None, "n_teeth": n_teeth or None, "crowding_mm": None,
                        "unsupported": []})
        for p in patients.list_patients():
            for sc in p["scans"]:
                confirmed = sc.get("confirmed_revision") == sc.get("revision", 1)
                plan_ids = STORE.plan_ids_for(sc["case_id"])
                rep = _representative_plan(plan_ids)
                status = _case_status(confirmed, rep)
                rows.append({"case_id": sc["case_id"], "kind": "patient", "title": p["alias"],
                            "subtitle": f"{p['alias']} · {sc['scan_id']}",
                            "status": status, "status_ko": _CASE_LIST_STATUS_KO[status], "plan": _plan_row(rep),
                            "n_plans": len(plan_ids), "prescription": None, "badges": [],
                            "patient": {"patient_id": p["patient_id"], "alias": p["alias"], "scan_id": sc["scan_id"]},
                            "confirmed": confirmed, "n_teeth": len(sc["teeth"]), "crowding_mm": None,
                            "unsupported": []})
        return {"cases": rows}

    @app.post("/api/cases/{case_id}/activate")
    async def activate_case(case_id: str):
        try:
            cid, case = STORE.load_case(case_id)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e))
        # Opening a case is the scan only: no rule preview is computed any more (step flow, .report/15-step-flow.md);
        # «에이전트 없이 계산» (POST /api/plan) stays as the fallback when an agent turn fails. The case starts from its
        # prescription: a stage cap or move order that a replay or an agent turn left at the case level is dropped here
        # (it stays on the plan it was made for, and the next turn from that plan carries it), so the screen's «조건이
        # 처방과 다릅니다» fires only when the prescription itself was changed. `active_plan` carries the newest plan's own
        # conditions for the 조건 tab; `flow` is how far the step flow went (setup conditions, target_id, plan_id) so a
        # refresh restores a case that was only set up or only targeted.
        STORE.case_constraints[cid] = STORE.constraints_for(cid).prescription()
        out = {"case_id": cid, "n_teeth": len(case.ids), "crowding_mm": planner.crowding_mm(case),
               "constraints": STORE.constraints_for(cid).model_dump(mode="json"), "active_plan": None,
               "flow": STORE.flow_json(cid), "unsupported": planner.unsupported_reasons(case)}
        ids = STORE.current_plan_ids_for(cid)   # a plan of another core version (previous_calculation) is never the case's current plan (10)
        if ids:
            out["active_plan"] = {"plan_id": ids[-1], "constraints": _constraint_dump(STORE._record(ids[-1]))}
        return out

    @app.post("/api/cases/{case_id}/restart")
    async def restart_case(case_id: str):
        """「처음부터」 on the case header: the step flow and the case's conditions start over (plans and targets stay).
        Answers like /activate, so the screen draws the case again from 초기."""
        try:
            cid, _ = STORE.load_case(case_id)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e))
        STORE.restart_case(cid)
        return await activate_case(cid)

    @app.get("/api/plans/{plan_id}/cut")
    async def plan_cut(plan_id: str):
        """The IPR-cut crowns of this plan alone: {plan_id, teeth_cut, ipr_cut} (what mesh?plan_id= adds, without the
        gum and the whole dentition again — the screen switches plans often, step flow (11))."""
        p = require_plan(plan_id)
        cid, case = STORE.load_case(p["case_id"])
        return ipr_cut_view(cid, case, plan_id)

    @app.get("/api/cases/{case_id}/targets/{target_id}/cut")
    async def target_cut(case_id: str, target_id: str):
        """The IPR-cut crowns of a target arrangement alone: {plan_id: null, target_id, teeth_cut, ipr_cut}."""
        t = STORE.targets.get(target_id)
        if t is None or t["case_id"] != case_id:
            raise HTTPException(404, "목표 배열이 없습니다")
        cid, case = STORE.load_case(case_id)
        return ipr_cut_view(cid, case, None, target_id)

    @app.get("/api/cases/{case_id}/setup/cut")
    async def setup_cut(case_id: str):
        """The crowns cut as the setup's prescription strips them: {plan_id: null, target_id: null, teeth_cut, ipr_cut}.
        셋업 comes before any target or plan (the case opens with the scan only), so its cut is the prescribed
        `ipr_surfaces` of the step flow's conditions, each contact cut on both its teeth (half each, or per face as
        `ipr_amounts` splits it). No setup: both maps empty."""
        try:
            cid, case = STORE.load_case(case_id)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e))
        c = (STORE.flow.get(cid) or {}).get("constraints")
        surfaces = [[int(a), int(b), float(mm)] for a, b, mm in (c.ipr_surfaces if c is not None else ())]
        if c is not None and c.ipr_amounts:   # 직접 이동's right-click IPR: each tooth face as prescribed, not half each
            from cualign.core.ipr_cut import cut_ipr
            return {"plan_id": None, "target_id": None, **closed_cut_json(cid, case, cut_ipr(case, surfaces, faces=c.face_amounts()))}
        info = {"ipr_surfaces": surfaces, "ipr_applied_teeth": sorted({t for a, b, _ in surfaces for t in (a, b)})}
        return {"plan_id": None, "target_id": None, **closed_cut_json(cid, case, planner.cut_case(case, info if surfaces else None))}

    def require_target(case_id: str, target_id: str) -> dict:
        t = STORE.targets.get(target_id)
        if t is None or t["case_id"] != case_id:
            raise HTTPException(404, "목표 배열이 없습니다")
        return t

    def target_view(case_id: str, target_id: str) -> dict:
        t = STORE.targets[target_id]
        target, info = t["target"], t["info"]
        stage = {str(i): np.round(np.asarray(v, float), 4).tolist() for i, v in target.items() if v is not None}
        yaw = {str(i): round(float(y), 3) for i, y in getattr(target, "yaw", {}).items()}
        c = t["constraints"]
        case = STORE.cases.get(case_id)
        return {"target_id": target_id, "case_id": case_id, "strategy": info.get("strategy"),
                "constraints": c if isinstance(c, dict) else c.model_dump(mode="json"),
                "info": info, "target": info, "violations": [], "passed": True, "summary": steps.target_summary(info),
                "stages": [stage], "rotations": [yaw],
                "pivots": {str(i): np.round(p, 4).tolist() for i, p in case.pos0.items()} if case else {},
                # the axes 직접 이동 moves along (core/manual.py): mesial, buccal, occlusal per crown
                "frames": {str(i): f for i, f in manual.frames(case).items()} if case else {}}

    @app.get("/api/cases/{case_id}/targets/{target_id}")
    async def get_target(case_id: str, target_id: str):
        """The target arrangement a target turn made, in the shape of a plan response with one stage (the final
        arrangement): stages[0] = displacement per tooth, rotations[0] = yaw per tooth, pivots, target/info,
        violations [] (nothing was validated yet), frames (the crowns' own axes for 직접 이동). The screen draws it as
        「목표」."""
        require_target(case_id, target_id)
        return target_view(case_id, target_id)

    # 직접 이동 (server/manual_api.py): the scan start, the drag check, apply, and staging a hand-edited target
    manual_api.add_manual_routes(app, store=lambda: STORE, require_target=require_target, target_view=target_view, summary=_summary)

    @app.get("/api/cases/{case_id}/mesh")
    async def case_mesh(case_id: str, plan_id: str | None = None, target_id: str | None = None):
        try:
            cid, case = STORE.load_case(case_id)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e))
        # the scan part is cached text; only the plan's (or target's) cut is made per request, and spliced in
        cut = json.dumps(ipr_cut_view(cid, case, plan_id, target_id), separators=(",", ":"))   # ?target_id= : the target turn's cut, before any plan
        return Response(mesh_base_json(cid, case)[:-1] + "," + cut[1:], media_type="application/json")

    @app.get("/api/cases/{case_id}/gum")
    async def case_gum(case_id: str):
        """The gum alone: the scan (or procedural ridge) as `gum` and the trench-covered copy as `gum_filled`."""
        try:
            cid, case = STORE.load_case(case_id)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e))
        gum = case.viewer_json()["gum"]
        return {"gum": gum, **gum_filled_view(cid, case, gum)}

    @app.post("/api/cases/upload")
    async def upload_case(files: list[UploadFile]):
        folder = OUT_DIR / "uploads" / uuid.uuid4().hex[:8]
        folder.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in files:
            name = Path(f.filename or "").name
            if not name.lower().endswith(".stl") or not (Path(name).stem.isdigit() or name.lower() == "gingiva.stl"):
                continue
            (folder / name).write_bytes(await f.read())
            n += Path(name).stem.isdigit()
        if n == 0:   # gingiva.stl alone is display-only; there is nothing to plan
            raise HTTPException(400, "upload per-tooth STL files named <tooth_id>.stl (Universal numbering, upper arch)")
        case = Case.from_dir(folder)
        cid = f"upload-{folder.name}"
        STORE.add_case(cid, case)
        return {"case_id": cid, "n_teeth": len(case.ids), "crowding_mm": planner.crowding_mm(case)}

    @app.get("/api/patients")
    async def list_patients():
        return {"patients": [{**p, "n_scans": len(p["scans"])} for p in patients.list_patients()]}

    @app.post("/api/patients")
    async def create_patient(req: PatientRequest):
        try:
            return patients.create_patient(req.alias, req.memo)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/patients/{pid}")
    async def get_patient(pid: str):
        try:
            return _patient_json(pid)
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.post("/api/patients/{pid}/scans")
    async def upload_patient_scan(pid: str, files: list[UploadFile]):
        data, other, lower, total = {}, [], [], 0
        fdi = _batch_is_fdi(f.filename for f in files)
        for f in files:
            name, kind = _scan_file_name(f.filename, fdi)
            if kind == "lower":
                lower.append(Path(f.filename).name)
                continue
            if name is None:
                if (f.filename or "").lower().endswith((".stl", ".ply", ".obj")):
                    other.append(Path(f.filename).name)
                continue
            if name in data:
                raise HTTPException(400, f"{name}: 같은 번호의 파일이 두 개 있습니다.")
            blob = await f.read(MAX_FILE_BYTES + 1)
            total += len(blob)
            if len(blob) > MAX_FILE_BYTES or total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"파일이 너무 큽니다(파일당 {MAX_FILE_BYTES >> 20}MB, 한 번에 {MAX_UPLOAD_BYTES >> 20}MB까지).")
            data[name] = blob
        if lower:
            raise HTTPException(400, f"{', '.join(lower[:3])}: 하악 번호(31~48)입니다. 지금은 상악 스캔만 받습니다.")
        if other and not any(Path(n).stem.isdigit() for n in data):
            raise HTTPException(400, f"{', '.join(other[:3])}: 한 덩어리 악궁 스캔으로 보입니다. 지금은 치아별로 나뉜 파일"
                                     "(11.stl … 27.stl, 선택 gingiva.stl)만 받습니다. 자동 치아 분리는 실험 단계입니다.")
        try:
            scan = patients.add_scan(pid, data)
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        try:
            STORE.cases.pop(scan["case_id"], None)
            _, case = STORE.load_case(scan["case_id"])
        except Exception as e:  # unreadable for the planner: do not keep the scan
            patients.remove_scan(pid, scan["scan_id"])
            STORE.forget_case(scan["case_id"])
            raise HTTPException(400, f"스캔을 읽지 못했습니다: {type(e).__name__}")
        return {**scan, "check": _check(scan["case_id"], case)}

    def _check(case_id, case):
        out = {"case_id": case_id, **planner.intake_report(case)}
        m = patients._CASE_RE.match(case_id)
        if m and patients.case_folder(case_id) is not None:
            sc = patients._scan(patients.get_patient(m.group(1)), m.group(2))
            out.update(orientation=sc.get("orientation"), revision=sc.get("revision", 1),
                       confirmed=sc.get("confirmed_revision") == sc.get("revision", 1),
                       confirmed_at=sc.get("confirmed_at") if sc.get("confirmed_revision") == sc.get("revision", 1) else None)
        return out

    def _patient_scan(pid, sid):
        try:
            patients._scan(patients.get_patient(pid), sid)
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/cases/{case_id}/check")
    async def case_check(case_id: str):
        try:
            _, case = STORE.load_case(case_id)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e))
        return _check(case_id, case)

    @app.post("/api/patients/{pid}/scans/{sid}/confirm")
    async def confirm_scan(pid: str, sid: str, req: ConfirmRequest | None = None):
        _patient_scan(pid, sid)
        _, case = STORE.load_case(f"{pid}-{sid}")
        if planner.unsupported_reasons(case):
            raise HTTPException(409, "지원하지 않는 스캔은 계획용으로 확인할 수 없습니다.")
        try:
            return patients.confirm_scan(pid, sid, req.revision if req else None)
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/patients/{pid}/scans/{sid}/mirror")
    async def mirror_scan(pid: str, sid: str):
        _patient_scan(pid, sid)
        patients.mirror_scan_numbers(pid, sid)
        case_id = f"{pid}-{sid}"
        for plan_id, plan in STORE.plans.items():      # plans of the old numbering stay as history, never approved
            if plan["case_id"] == case_id and plan["approval"]:
                plan["approval"] = None
                STORE._persist(plan_id)
        STORE.cases.pop(case_id, None)       # the numbers changed: reload from disk
        _, case = STORE.load_case(case_id)
        return _check(case_id, case)

    @app.delete("/api/patients/{pid}/scans/{sid}")
    async def delete_scan(pid: str, sid: str):
        _patient_scan(pid, sid)
        STORE.forget_case(patients.remove_scan(pid, sid))    # with its plans, targets and files
        return _patient_json(pid)

    @app.delete("/api/patients/{pid}")
    async def delete_patient(pid: str):
        try:
            case_ids = patients.delete_patient(pid)
        except KeyError as e:
            raise HTTPException(404, str(e))
        plans = sum(len(STORE.forget_case(c)) for c in case_ids)
        return {"deleted": pid, "scans": len(case_ids), "plans": plans}

    @app.get("/api/plans")
    async def list_plans(case_id: str | None = None):
        return {"plans": [_summary(pid) for pid in reversed(list(STORE.plans))
                          if case_id is None or STORE._record(pid)["case_id"] == case_id]}

    def require_plan(pid):
        if pid not in STORE.plans:
            raise HTTPException(404, f"unknown plan {pid}")
        return STORE.plans[pid]

    @app.get("/api/plans/{plan_id}")
    async def get_plan(plan_id: str):
        require_plan(plan_id)
        return STORE.plan_json(plan_id)

    @app.post("/api/plans/{plan_id}/approval")
    async def approve_plan(plan_id: str, req: ApprovalRequest):
        require_plan(plan_id)
        if not req.confirmed:
            raise HTTPException(400, "의사의 명시적 확인이 필요합니다.")
        try:
            approved = STORE.approve(plan_id, req.lock_fdi)
        except ValueError as e:
            raise HTTPException(409, str(e))
        _start_export(plan_id)   # the zip is built from now on, in the background (answer-polish (9))
        return approved

    def _start_export(plan_id: str):
        p = STORE.plans[plan_id]
        _, case = STORE.load_case(p["case_id"])
        case = planner.cut_case(case, p["info"])     # the stage files and print models carry the IPR cut (#62)
        return export_jobs.start(plan_id, case, p["stages"], p["case_id"], OUT_DIR / "stl" / f"{plan_id}.zip",
                                 (p.get("approval") or {}).get("fingerprint"))

    @app.get("/api/plans/{plan_id}/export-status")
    async def export_status(plan_id: str):
        """{building, done, total, ready}: how far the approved plan's zip is (the screen shows «N/20 단계»), with stl (the
        zip's .stl count) once ready and error when it failed. Before approval, or for a plan whose build never started,
        building false and ready false."""
        p = require_plan(plan_id)
        return export_jobs.status(plan_id, len(p["stages"]))

    @app.post("/api/plans/{plan_id}/review")
    async def request_review(plan_id: str):
        # Recovery for a plan the agent left unreviewed (skipped the call) or whose review failed.
        p = require_plan(plan_id)
        if review is None:
            raise HTTPException(503, "검토 모델이 연결되지 않았습니다.")
        if p["review"]["status"] not in MANUAL_RETRY:
            raise HTTPException(409, "미실행이거나 실패한 검토만 다시 요청할 수 있습니다.")
        await review(plan_id)
        return STORE.plan_json(plan_id)

    @app.delete("/api/plans/{plan_id}/approval")
    async def revoke_approval(plan_id: str):
        require_plan(plan_id)
        return STORE.revoke(plan_id)

    @app.get("/api/plans/{plan_id}/stl.zip")
    async def plan_stl(plan_id: str):
        p = require_plan(plan_id)
        try:
            STORE.require_approved(plan_id)
        except ValueError as e:
            raise HTTPException(409, str(e))
        # The build the approval started (or one started now), keyed by the approval's fingerprint so a stale file of
        # an earlier approval is never served: ready -> the cached zip at once; still building -> wait for it, off the
        # event loop as before (#119). Building takes 5-11 s on the samples here (mostly the print models).
        job = _start_export(plan_id)
        await asyncio.to_thread(job.wait)
        if job.error or not job.path.exists():
            raise HTTPException(500, f"내보내기 파일을 만들지 못했습니다: {job.error or 'no file'}")
        models = job.report
        note = f"{models['status']}; files={models['n_files']}" + (f"; reason={models['reason']}" if models.get("reason") else "")
        return FileResponse(str(job.path), media_type="application/zip", filename=f"cualign_{plan_id}_stages.zip",
                            headers={"X-Cualign-Print-Models": note})

    @app.get("/api/cases/{case_id}/replay/{step}")
    async def recorded_head(case_id: str, step: str):
        """What 건너뛰기 can show at once while POST .../replay recomputes the plans (up to ~9 s on 000001): the
        recording's step, date and reasoning, read from the file, nothing computed. 404 as the POST gives it."""
        from cualign.core import recorded
        rec = recorded.load(case_id, step)
        if rec is None or samples.get(case_id) is None:
            raise HTTPException(404, recorded.NO_RECORDING)
        step = recorded.ALIASES.get(step, step)
        return {"recorded": True, "step": step, "step_ko": recorded.STEP_KO[step], "recorded_at": rec["recorded_at"],
                **({"reasoning": rec["reasoning"]} if rec.get("reasoning") else {})}

    @app.post("/api/cases/{case_id}/replay")
    async def replay_recorded(case_id: str, req: ReplayRequest):
        """A sample's recorded agent answer (core/recorded.py) instead of a model turn: the plans are recomputed by the
        rule engine under the recorded constraints, the answer's stage counts are checked against them, the recorded
        review is attached, and the side effects are the agent turn's (plans stored, case constraints, context line).
        The agent turn still running on the case is cut first, not waited for (plan_events.cancel), and the
        computation runs off the event loop: on the loop it held every other request for its seconds."""
        cancel_case(case_id, "건너뛰기: the recorded answer was asked for")
        return await asyncio.to_thread(replay_now, case_id, req)

    def replay_now(case_id: str, req: ReplayRequest) -> dict:
        from cualign.core import recorded
        # the screen asks for the recording of the step it sent; when the server ran the turn as another step (a free
        # sentence's words, step-intent), the recording of that one (Store.turn_steps, set when the turn opened)
        turn = STORE.turn_steps.pop(case_id, None)
        if turn and turn["screen"] == recorded.ALIASES.get(req.step, req.step):
            req = req.model_copy(update={"step": turn["turn"]})
        rec = recorded.load(case_id, req.step)
        if rec is None or samples.get(case_id) is None:
            raise HTTPException(404, recorded.NO_RECORDING)
        try:
            cid, case = STORE.load_case(case_id)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e))
        patch = ConstraintPatch.model_validate({k: v for k, v in rec["constraints"].items() if k in ConstraintPatch.model_fields}).changes()
        try:
            constraints = STORE.constraints_for(cid, req.base_plan_id).patched(patch)
            constraints.check_case(case.ids)
        except ValueError as e:
            raise HTTPException(400, reason_ko(e))
        if req.base_plan_id and req.base_plan_id in STORE.plans and STORE.previous_calculation(req.base_plan_id):
            raise HTTPException(409, PREVIOUS_CALCULATION)
        step = recorded.ALIASES.get(req.step, req.step)
        out = {"recorded": True, "step": step, "answer_md": rec["answer_md"], "plan_selected": None, "plans": [], "recorded_at": rec["recorded_at"]}
        if rec.get("reasoning"):   # the live turn's reasoning events; the screen plays them before the answer
            out["reasoning"] = rec["reasoning"]
        STORE.replays[cid] = {"step": step, "step_ko": recorded.STEP_KO[step], "plan_id": None, "recorded_at": rec["recorded_at"]}
        if step == "setup":            # the recorded setup turn: its conditions become the case's, no plan
            STORE.case_constraints[cid] = constraints
            STORE.set_flow(cid, "setup", constraints=constraints, target_id=None, plan_id=None)
            out.update(setup_view(STORE, cid, constraints))   # constraints (ipr_surfaces filled in) + conditions_ko, as step_done setup
            return out
        if step == "target":           # the recorded target turn: the rule engine makes the target again, under the recorded strategy
            strategy = rec.get("strategy") or (planner.strategies_for(list(planner.STRATEGIES), constraints) or ["expansion"])[0]
            try:
                tid = PlanningService(STORE).target(cid, strategy, constraints)
            except ValueError as e:
                raise HTTPException(400, reason_ko(e))
            STORE.case_constraints[cid] = constraints
            STORE.set_flow(cid, "target", constraints=constraints, target_id=tid, plan_id=None)
            out["target_id"] = tid
            out["summary"] = steps.target_summary(STORE.targets[tid]["info"])
            return out
        if rec["review"] is None:      # the recorded turn made no plan (a question back, e.g. a comparison under an extraction prescription)
            return out
        try:
            ids = PlanningService(STORE).compare(cid, constraints, parent_plan_id=req.base_plan_id)
        except ValueError as e:
            raise HTTPException(400, reason_ko(e))
        STORE.case_constraints[cid] = constraints
        by_strategy = {STORE._record(pid)["strategy"]: _summary(pid) for pid in ids}
        wanted = recorded.selected_strategy(rec["answer_md"])
        selected = by_strategy.get(wanted) or next((p for p in by_strategy.values() if p["passed"]), None) \
            or min(by_strategy.values(), key=lambda p: p["violations"])
        STORE.set_review(selected["plan_id"], dict(rec["review"]))
        selected = _summary(selected["plan_id"])
        STORE.replays[cid]["plan_id"] = selected["plan_id"]
        STORE.set_flow(cid, "stages", constraints=constraints, target_id=STORE.plans[selected["plan_id"]].get("target_id"),
                       plan_id=selected["plan_id"])
        out.update({"answer_md": recorded.substitute(rec["answer_md"], by_strategy, selected),
                    "plan_selected": {"plan_id": selected["plan_id"], "parent_plan_id": selected["parent_plan_id"], "review": selected["review"]},
                    "plans": [_summary(pid) for pid in ids]})
        return out

    @app.post("/api/plan")
    async def rule_plan(req: RulePlanRequest):
        try:
            # the request is a ConstraintPatch already validated once (FDI -> Universal, #57): validating its dump again
            # converted ipr_surfaces twice and refused the form's prescription (screen report 24, step flow (13))
            changes = {k: v for k, v in req.changes().items() if k not in ("case_id", "parent_plan_id")}
            return rule_based_plan(req.case_id, changes=changes, parent_plan_id=req.parent_plan_id)
        except (KeyError, FileNotFoundError, ValueError) as e:
            raise HTTPException(400, reason_ko(e) if isinstance(e, ValueError) else str(e))

    if STATIC_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
