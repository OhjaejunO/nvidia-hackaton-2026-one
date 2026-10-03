"""직접 이동 routes (core/manual.py behind them): the scan-position target a 셋업 edit starts from, the check while
dragging, 적용 as a new target, staging that target without the agent, and the 셋업 right-click (발치 · IPR per tooth
face) that changes the case's prescription. The scan start is not stored until an edit on it is applied (SCAN). Added by api.add_api_routes, which passes the target helpers it shares
with GET …/targets/{id} and the store as a getter (tests swap api.STORE)."""
from __future__ import annotations

from typing import Callable, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cualign.core import manual
from cualign.core.constraints import Constraints, reason_ko
from cualign.core.fdi import from_fdi, label
from cualign.core.limits import IPR_PER_SURFACE
from cualign.core.service import PlanningService
from cualign.server.plan_events import setup_view


SCAN = "scan"   # the 셋업 start's target id: made afresh on each request, stored only as the parent of an applied edit


class ToothPose(BaseModel):
    d: list[float] = Field(min_length=3, max_length=3)   # total translation from the scan, mm (world)
    yaw: float = 0.0                                      # total turn about the crown's vertical axis, degrees


class ManualEditRequest(BaseModel):
    """The crowns the dentist moved by hand (Universal numbers; the screen converts from FDI). Others keep the base."""
    teeth: dict[int, ToothPose] = Field(default_factory=dict, max_length=32)


class TargetStagesRequest(BaseModel):
    parent_plan_id: str | None = None


class SetupEditRequest(BaseModel):
    """One change to the case's prescription from the 셋업 right-click. `tooth` is FDI (the number the dentist reads);
    `face` and `mm` (0.05 steps, at most IPR_PER_SURFACE off that face) are for op "ipr"."""
    op: Literal["extract", "unextract", "ipr", "unipr"]
    tooth: int
    face: Literal["mesial", "distal"] | None = None
    mm: float | None = Field(default=None, gt=0, le=IPR_PER_SURFACE)


def neighbour(t: int, face: str) -> int:
    """The tooth across a face (Universal upper arch: 1..8 the right side with 8 = FDI 11, 9..16 the left)."""
    mesial = t + 1 if t <= 8 else t - 1
    return mesial if face == "mesial" else 2 * t - mesial


def setup_edit(c: Constraints, ids, t: int, req: SetupEditRequest) -> Constraints:
    """The prescription with one right-click change. 발치 takes the tooth's IPR contacts with it; IPR sets one tooth face
    (its contact's amount is the sum of the two faces, the split kept in ipr_amounts unless it is half each)."""
    name = label(t)
    if req.op == "extract":
        if t in c.extraction:
            raise ValueError(f"{name}은 이미 발치로 처방되어 있습니다.")
        return c.patched({"extraction": [*c.extraction, t],
                          "ipr_surfaces": [list(r) for r in c.ipr_surfaces if t not in r[:2]],
                          "ipr_amounts": [list(r) for r in c.ipr_amounts if t not in r[:2]]})
    if req.op == "unextract":
        if t not in c.extraction:
            raise ValueError(f"{name}은 발치로 처방되어 있지 않습니다.")
        return c.patched({"extraction": [x for x in c.extraction if x != t]})
    faces = c.face_amounts()
    if req.op == "ipr":
        if req.face is None or req.mm is None:
            raise ValueError("IPR 은 면(근심/원심)과 양(mm)이 필요합니다.")
        if abs(req.mm / 0.05 - round(req.mm / 0.05)) > 1e-6:
            raise ValueError("IPR 양은 0.05mm 단위입니다.")
        n = neighbour(t, req.face)
        if t in c.extraction or n not in ids or n in c.extraction:
            raise ValueError(f"{name} {'근심' if req.face == 'mesial' else '원심'} 쪽에 IPR 할 이웃 치아가 없습니다.")
        faces[(t, n)] = round(req.mm, 4)
        touched = {(min(t, n), max(t, n))}
    else:   # unipr: this tooth's faces off, its neighbours' faces stay
        touched = {(min(a, b), max(a, b)) for (a, b), mm in faces.items() if a == t and mm > 0}
        if not touched:
            raise ValueError(f"{name}에 처방된 IPR 이 없습니다.")
        faces = {k: (0.0 if k[0] == t else mm) for k, mm in faces.items()}
    surfaces = {(a, b): mm for a, b, mm in c.ipr_surfaces}
    amounts = {(x, y): mm for x, y, mm in c.ipr_amounts}
    for a, b in touched:
        fa, fb = faces.get((a, b), 0.0), faces.get((b, a), 0.0)
        amounts.pop((a, b), None)
        amounts.pop((b, a), None)
        if fa + fb <= 0:
            surfaces.pop((a, b), None)
            continue
        surfaces[(a, b)] = round(fa + fb, 4)
        if abs(fa - fb) > 1e-9:   # half each needs no split
            amounts.update({k: v for k, v in (((a, b), fa), ((b, a), fb)) if v > 0})
    return c.patched({"ipr_surfaces": [[a, b, mm] for (a, b), mm in surfaces.items()],
                      "ipr_amounts": [[x, y, mm] for (x, y), mm in amounts.items()]})


def add_manual_routes(app: FastAPI, *, store: Callable, require_target: Callable, target_view: Callable, summary: Callable):
    def _scan(case_id: str):
        """The 셋업 start (manual.scan_start under the case's confirmed conditions) as a stored target reads, unstored."""
        S = store()
        try:
            cid, case = S.load_case(case_id)
            S.require_current_input(cid)
            constraints = S.constraints_for(cid)
            target, info = manual.scan_start(case, constraints)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, reason_ko(e))
        return cid, case, {"case_id": cid, "target": target, "info": info, "constraints": constraints, "input_revision": None}

    def _edited(case_id: str, target_id: str, req: ManualEditRequest):
        S = store()
        if target_id == SCAN:
            _, case, t = _scan(case_id)
        else:
            t = require_target(case_id, target_id)
            _, case = S.load_case(case_id)
        try:
            if target_id != SCAN:
                S.require_current_input(case_id, t.get("input_revision"))   # the scan may have changed since
            new, changed = manual.apply_edits(t["target"], {i: p.model_dump() for i, p in req.teeth.items()}, t["constraints"])
        except ValueError as e:
            raise HTTPException(400, reason_ko(e))
        return t, case, new, changed

    @app.post("/api/cases/{case_id}/setup/conditions")
    async def setup_conditions(case_id: str, req: SetupEditRequest):
        """직접 이동 on 셋업, right-click 발치 / IPR (and their 취소): the case's conditions with that one change, through
        Constraints.patched like any other condition change. They become the case's (the next turn's plan_context
        starts from them) and the flow's setup (a later target or plan is stale). Answers as step_done setup."""
        S = store()
        try:
            cid, case = S.load_case(case_id)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e))
        if (S.flow.get(cid) or {}).get("constraints") is None:
            raise HTTPException(409, "셋업이 끝난 뒤에 처방을 바꿀 수 있습니다.")
        try:
            t = from_fdi(req.tooth)
            if t not in case.ids:
                raise ValueError(f"이 스캔에 {req.tooth}번 치아가 없습니다.")
            old = S.constraints_for(cid)
            new = setup_edit(old, set(case.ids), t, req)
            new.check_case(case.ids)
        except ValueError as e:
            raise HTTPException(400, reason_ko(e))
        S.case_constraints[cid] = new
        S.add_edit(cid, old, new)   # the dentist's hand edit sets the field it changes (core/request_check)
        S.set_flow(cid, "setup", constraints=new, target_id=None, plan_id=None)
        return {"step": "setup", **setup_view(S, cid, new)}

    @app.post("/api/cases/{case_id}/targets/scan")
    async def scan_target(case_id: str):
        """The target a 셋업 edit starts from: a target with every crown where the scan has it (the case's confirmed conditions: the
        prescribed extraction teeth removed, only prescribed IPR cut; strategy "manual", info.source "scan"), to move by
        hand with …/scan/check and …/scan/manual. Not stored: opening the edit alone leaves no target behind (…/scan/manual
        stores it as the applied edit's parent). Answers as GET …/targets/{id} with target_id "scan"."""
        S = store()
        cid, _, t = _scan(case_id)
        S.targets[SCAN] = t   # target_view reads the store: the start is there only for this answer
        try:
            return target_view(cid, SCAN)
        finally:
            S.targets.pop(SCAN, None)

    @app.post("/api/cases/{case_id}/targets/{target_id}/check")
    async def check_manual(case_id: str, target_id: str, req: ManualEditRequest):
        """직접 이동 while dragging: the edited arrangement is checked, not stored. {max_move_mm, max_yaw_deg, min_stages,
        min_months, overlaps: [{teeth, overlap_mm3}], changed: [teeth]} (see core/manual.check)."""
        t, case, new, changed = _edited(case_id, target_id, req)
        return {**manual.check(case, new, t["info"]), "changed": changed}

    @app.post("/api/cases/{case_id}/targets/{target_id}/manual")
    async def save_manual(case_id: str, target_id: str, req: ManualEditRequest):
        """직접 이동 적용: a new target from this one with the crowns moved by hand (info.source "manual",
        parent_target_id, manual_teeth). It becomes the case's target in the flow, so the next stages turn (or
        …/stages) stages it. Answers as GET …/targets/{new id}."""
        S = store()
        t, case, new, changed = _edited(case_id, target_id, req)
        if not changed:
            raise HTTPException(400, "옮긴 치아가 없습니다.")
        if target_id == SCAN:   # the scan start is stored now, as the edit's parent
            target_id = S.put_target(case_id, t["target"], t["info"], t["constraints"])
        info = manual.manual_info(t["info"], target_id, changed, t["info"].get("manual_teeth", ()))
        tid = S.put_target(case_id, new, info, t["constraints"])
        S.set_flow(case_id, "target", constraints=t["constraints"], target_id=tid, plan_id=None)
        return target_view(case_id, tid)

    @app.post("/api/cases/{case_id}/targets/{target_id}/stages")
    async def stage_target(case_id: str, target_id: str, req: TargetStagesRequest):
        """This target staged and validated without the agent (a hand-edited target's 「에이전트 없이 단계 계산」): one
        plan, review marked not run, the flow at stages with it. Answers as a row of GET /api/plans."""
        S = store()
        require_target(case_id, target_id)
        try:
            pid = PlanningService(S).stages(target_id, req.parent_plan_id)
        except ValueError as e:
            raise HTTPException(400, reason_ko(e))
        S.set_review(pid, {"status": "skipped", "attempts": 0, "message": "에이전트 없이 계산 — 검토 에이전트 미실행", "error": None})
        t = S.targets[target_id]
        S.set_flow(case_id, "stages", constraints=t["constraints"], target_id=target_id, plan_id=pid)
        return summary(pid)
