"""In-process artifact store shared by the agent tools and the web API (same FastAPI process).

Plans are also persisted as JSON under CUALIGN_OUT (default ./out) so `nat run` results survive.
"""
from __future__ import annotations

import json
import os
import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np

from . import patients, request_check, samples
from .case import Case
from .constraints import Constraints
from .fdi import to_fdi
from .synth import PRESETS

OUT_DIR = Path(os.environ.get("CUALIGN_OUT", "out"))


ID_HEX = 8  # id length after the "p"/"t" prefix: 32 (uuid4) was long enough for the model to miscopy between tool calls (#126)


def _new_id(prefix: str, taken) -> str:
    """A fresh id not in `taken` (plans loaded from disk are in it too, so a restart cannot reuse one)."""
    while (new := prefix + secrets.token_hex(ID_HEX // 2)) in taken:
        pass
    return new


class _Stage(dict):
    """One stage read back from a plan file. Tooth ids are ints, and yaw is the saved rotation."""

    def __init__(self, moves: dict, yaw: dict | None = None):
        super().__init__({int(k): v for k, v in moves.items()})
        self.yaw = {int(k): float(v) for k, v in (yaw or {}).items()}


def _plan_from_file(data: dict) -> dict:
    """The file is the plan. Callers read case_id, review, and stages on the plan itself."""
    rotations = data.get("rotations") or []
    data["stages"] = [_Stage(st, rotations[i] if i < len(rotations) else None) for i, st in enumerate(data.get("stages") or [])]
    if isinstance(data.get("constraints"), dict):
        removed = (data.get("target") or {}).get("removed") or (data.get("info") or {}).get("removed") or ()
        legacy = "extraction" not in data["constraints"]
        data["constraints"] = Constraints.from_saved(data["constraints"], removed)   # files from before #56 too
        if legacy and data.get("approval"):
            # approved under conditions without the extraction teeth: the approval does not carry over (#98 review);
            # shown as not approved, so the screen asks for a new approval instead of offering an export it refuses
            data["approval"] = None
    data["_from_disk"] = True
    return data


def planner_version() -> str:
    """A fingerprint of the calculation core (planner, arch, ipr_cut, limits, print_model source): a plan stored under
    another fingerprint was computed by other code and is never reused as the case's current plan (answer-polish (10):
    after #138 a stored 20-stage plan was shown as the case's, so the new target arrangement seemed unchanged)."""
    from cualign.core import arch, ipr_cut, limits, planner, print_model
    h = hashlib.sha256()
    for mod in (planner, arch, ipr_cut, limits, print_model):
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()[:16]


PLANNER_VERSION = planner_version()


class Store:
    def __init__(self):
        self.cases: dict[str, Case] = {}
        self.replays: dict[str, dict] = {}   # case_id -> the last replayed recording {step, plan_id, recorded_at} (core/recorded.py)
        self.case_constraints: dict[str, Constraints] = {}
        # case_id -> how far the step flow went: {"step": setup|target|stages, "constraints": Constraints (the setup
        # conditions), "target_id", "plan_id"} (agent/steps.py); what activate returns as `flow` so a refresh restores it
        self.flow: dict[str, dict] = {}
        # case_id -> the last chat turn's recording {"screen": the one its 건너뛰기 asks for, "turn": the one the server's
        # step makes} (recorded.replay_step): a free sentence's step can differ from the step the screen sent (step-intent)
        self.turn_steps: dict[str, dict] = {}
        self.active_case: str | None = None
        self.targets: dict[str, dict] = {}
        self.plans: dict[str, dict] = {}
        # case_id -> {"seg": n, "entries": [request_check.ask/edit + "seg"]}: what the dentist asked on the case, in
        # order, every turn (setup too) and every hand edit; 처음부터 opens a new segment (restart_case). Every plan is
        # compared with the entries of its segment (check_request), at approval too. Kept on disk like the plans.
        self.requests: dict[str, dict] = {}
        self._load_persisted_plans()
        self._load_persisted_requests()

    # ------------------------------------------------------------------ cases
    def available_cases(self) -> list[dict]:
        # real scans with the dentist's prescription first (the start screen shows these); the synthetic presets stay
        # loadable by name for the agent, the tests and the CLI, but the screen does not offer them (#46)
        rows = [{"case_id": s.case_id, "kind": "sample", "title": s.title, "prescription": s.prescription,
                 "summary": s.summary, "badges": list(s.badges), "request": s.request, "note": s.note, "available": s.available,
                 "constraints": s.initial_constraints().model_dump(mode="json")} for s in samples.SAMPLES.values()]
        rows += [{"case_id": k, "kind": "synthetic", "crowding_mm": v["crowding_mm"]} for k, v in PRESETS.items()]
        env = os.environ.get("CUALIGN_CASE_DIR")
        if env and Path(env).exists():
            rows.append({"case_id": "scan", "kind": "stl-folder", "path": env})
        for k in self.cases:
            # patient scans are listed under their patient (/api/patients), never among samples or to the model
            if k not in {r["case_id"] for r in rows} and patients.case_folder(k) is None:
                rows.append({"case_id": k, "kind": "loaded"})
        return rows

    def load_case(self, case_id: str | None = None) -> tuple[str, Case]:
        if case_id is None:
            case_id = self.active_case or ("scan" if os.environ.get("CUALIGN_CASE_DIR") else "moderate")
        if case_id not in self.cases:
            if (sample := samples.get(case_id)) is not None:
                if not sample.available:
                    raise FileNotFoundError(f"sample {case_id} is not installed ({sample.folder})")
                self.cases[case_id] = Case.from_dir(sample.folder)
                # the case opens with its prescription as the planning constraints (until the dentist changes them)
                self.case_constraints.setdefault(case_id, sample.initial_constraints())
            elif case_id in PRESETS:
                self.cases[case_id] = Case.synthetic(case_id)
            elif case_id == "scan":
                self.cases[case_id] = Case.from_dir(os.environ["CUALIGN_CASE_DIR"])
            elif (folder := patients.case_folder(case_id)) is not None:
                self.cases[case_id] = Case.from_dir(folder)
            elif Path(case_id).is_dir():
                self.cases[case_id] = Case.from_dir(case_id)
            else:
                raise KeyError(f"unknown case_id {case_id!r}; available: {[r['case_id'] for r in self.available_cases()]}")
        self.active_case = case_id
        return case_id, self.cases[case_id]

    def add_case(self, case_id: str, case: Case) -> None:
        self.cases[case_id] = case
        self.active_case = case_id

    # ------------------------------------------------------------------ targets / plans
    def constraints_for(self, case_id: str, parent_plan_id: str | None = None) -> Constraints:
        if parent_plan_id is not None:
            parent = self.plans.get(parent_plan_id)
            if parent is None:
                raise ValueError("parent plan must exist in the same case")
            record = self._record(parent_plan_id)
            if record["case_id"] != case_id:
                raise ValueError("parent plan must exist in the same case")
            constraints = record["constraints"]
            if isinstance(constraints, dict):
                removed = (record.get("target") or {}).get("removed") or (record.get("info") or {}).get("removed") or ()
                return Constraints.from_saved(constraints, removed)
            return constraints
        return self.case_constraints.get(case_id, Constraints())

    def _load_persisted_plans(self) -> None:
        """A plan file belongs to one case. A new process reads them back instead of sharing another case's stages."""
        folder = OUT_DIR / "plans"
        if not folder.is_dir():
            return
        files = sorted(folder.glob("p*.json"), key=lambda path: path.stat().st_mtime)
        for path in files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            pid = data.get("plan_id")
            if not pid or not data.get("case_id") or pid in self.plans:
                continue
            if (data.get("info") or {}).get("planner_version") != PLANNER_VERSION and samples.get(data["case_id"]) is not None:
                # a sample case's plan from another version of the core: dropped, the samples are recomputed on demand
                # (a patient's plan is kept, flagged previous_calculation, see plan_json)
                path.unlink(missing_ok=True)
                (OUT_DIR / "stl" / f"{pid}.zip").unlink(missing_ok=True)
                continue
            self.plans[pid] = _plan_from_file(data)

    def _record(self, pid: str) -> dict:
        return self.plans[pid]

    def plan_ids_for(self, case_id: str) -> list[str]:
        return [pid for pid in self.plans if self._record(pid)["case_id"] == case_id]

    def previous_calculation(self, pid: str) -> bool:
        """The plan was computed by another version of the core (planner_version): shown as «이전 계산», never the
        case's current plan, never a base plan (answer-polish (10))."""
        return (self._record(pid).get("info") or {}).get("planner_version") != PLANNER_VERSION

    def current_plan_ids_for(self, case_id: str) -> list[str]:
        return [pid for pid in self.plan_ids_for(case_id) if not self.previous_calculation(pid)]

    # ------------------------------------------------------------------ input revision (patient scans)
    def require_current_input(self, case_id: str, revision: int | None = None) -> None:
        """Planning, approval and export need a patient scan the dentist confirmed, in the revision the plan was made
        from. Samples and anonymous uploads (no patient record) pass."""
        st = patients.input_state(case_id)
        if st is None:
            return
        if st.get("deleted"):
            raise ValueError("삭제된 스캔입니다.")
        if not st["confirmed"]:
            raise ValueError("입력 확인 전 스캔입니다. 입력 확인 화면에서 치아 번호와 방향을 확인한 뒤 계획하세요.")
        if revision is not None and revision != st["revision"]:
            raise ValueError("스캔 번호가 바뀐 뒤의 이전 계획입니다. 새 입력으로 다시 계획하세요.")

    def input_stale(self, case_id: str, revision: int | None) -> bool:
        st = patients.input_state(case_id)
        return st is not None and (st.get("deleted", False) or not st["confirmed"] or revision != st["revision"])

    def set_flow(self, case_id: str, step: str, **what) -> dict:
        """Record that `step` is done for the case (setup: constraints; target: target_id; stages: plan_id)."""
        flow = self.flow.setdefault(case_id, {"step": None, "constraints": None, "target_id": None, "plan_id": None})
        flow["step"] = step
        flow.update(what)
        return flow

    def flow_json(self, case_id: str) -> dict | None:
        flow = self.flow.get(case_id)
        if flow is None:
            return None
        c = flow["constraints"]
        return {**flow, "constraints": c.model_dump(mode="json") if c is not None else None}

    def restart_case(self, case_id: str) -> None:
        """「처음부터」: forget how far the step flow went and the case-level conditions (back to Constraints(), not the
        sample's prescription — the prescription chip puts it back). Targets and plans stay: plan ids, approvals and
        exports keep their records, and the screen folds the plans as 지난 계획. What was asked so far stays with those
        plans: a new segment of requests starts, and each plan keeps being compared with its own (check_request)."""
        self.flow.pop(case_id, None)
        self.case_constraints[case_id] = Constraints()
        self.requests.setdefault(case_id, {"seg": 0, "entries": []})["seg"] += 1
        self._persist_requests(case_id)

    def forget_case(self, case_id: str) -> list[str]:
        """Drop everything derived from a deleted patient scan: case, conditions, targets, plans and their files."""
        self.cases.pop(case_id, None)
        self.case_constraints.pop(case_id, None)
        self.flow.pop(case_id, None)
        self.requests.pop(case_id, None)
        self._requests_file(case_id).unlink(missing_ok=True)
        if self.active_case == case_id:
            self.active_case = None
        for tid in [t for t, v in self.targets.items() if v["case_id"] == case_id]:
            del self.targets[tid]
        gone = [pid for pid in self.plans if self._record(pid)["case_id"] == case_id]
        for pid in gone:
            del self.plans[pid]
            for f in (OUT_DIR / "plans" / f"{pid}.json", OUT_DIR / "stl" / f"{pid}.zip"):
                f.unlink(missing_ok=True)
        return gone

    def discard(self, plan_ids, target_ids=()) -> None:
        """Drop what a cancelled agent turn made (server/plan_events.cancel): it was never offered, and left in place
        its plans would show up in the case's plan list after the 건너뛰기 that cut it."""
        for tid in list(target_ids):
            self.targets.pop(tid, None)
        for pid in list(plan_ids):
            if self.plans.pop(pid, None) is not None:
                (OUT_DIR / "plans" / f"{pid}.json").unlink(missing_ok=True)

    def put_target(self, case_id: str, target: dict, info: dict, constraints: Constraints | None = None) -> str:
        tid = _new_id("t", self.targets)
        st = patients.input_state(case_id)
        self.targets[tid] = {"case_id": case_id, "target": target, "info": info,
                            "constraints": constraints or self.constraints_for(case_id),
                            "input_revision": st["revision"] if st else None}
        return tid

    # ------------------------------------------------------------------ what the dentist asked (core/request_check)
    def add_request(self, case_id: str, text: str, opened: Constraints) -> None:
        """A turn's sentence, as the turn opens (plan_events): setup turns too, whatever the turn then does."""
        self._log(case_id, request_check.ask(text, opened))

    def add_edit(self, case_id: str, before: Constraints, after: Constraints) -> None:
        """Conditions the dentist changed by hand on the screen (the form sent with a turn or 다시 계산, a setup edit):
        they set the fields they change. Nothing is kept when none of the checked fields or the stage cap changed."""
        if any(getattr(before, f) != getattr(after, f) for f in (*request_check.FIELDS_KO, "stage_cap")):
            self._log(case_id, request_check.edit(before, after))

    def _log(self, case_id: str, entry: dict) -> None:
        r = self.requests.setdefault(case_id, {"seg": 0, "entries": []})
        r["entries"].append({**entry, "seg": r["seg"]})
        self._persist_requests(case_id)

    def check_request(self, pid: str) -> dict | None:
        """The plan against everything asked on its case in its segment, as of now (request_check.check_case); None
        when nothing was asked. A request file that could not be read is never a match."""
        p = self._record(pid)
        r = self.requests.get(p["case_id"])
        if r is not None and r.get("unreadable"):
            return {"status": "needs_check", "reasons_ko": [],
                    "unread_ko": [f"요청 기록을 읽지 못했습니다({r['unreadable']})"], "unchecked": []}
        seg = p.get("request_seg", 0)
        return request_check.check_case([e for e in (r or {}).get("entries", ()) if e["seg"] == seg], p["constraints"])

    def lock_fdi(self, pid: str) -> list[int]:
        """The plan's locked teeth as the dentist reads them (FDI), sorted: what the approval confirms."""
        return sorted(to_fdi(t) for t in self._record(pid)["constraints"].lock)

    def _requests_file(self, case_id: str) -> Path:
        return OUT_DIR / "requests" / (quote(case_id, safe="") + ".json")

    def _persist_requests(self, case_id: str) -> None:
        r = self.requests[case_id]
        rows = [{k: v.model_dump(mode="json") if isinstance(v, Constraints) else v for k, v in e.items()}
                for e in r["entries"]]
        path = self._requests_file(case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"case_id": case_id, "seg": r["seg"], "entries": rows}, ensure_ascii=False),
                        encoding="utf-8")

    def _load_persisted_requests(self) -> None:
        folder = OUT_DIR / "requests"
        if not folder.is_dir():
            return
        for path in folder.glob("*.json"):
            case_id = unquote(path.stem)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entries = [{k: Constraints.model_validate(v) if k in ("opened", "before", "after") else v
                            for k, v in e.items()} for e in data["entries"]]
                self.requests[case_id] = {"seg": int(data["seg"]), "entries": entries}
            except (OSError, ValueError, KeyError, TypeError) as e:
                # fail-closed: a case whose requests cannot be read has no plan that matches them (check_request)
                self.requests[case_id] = {"seg": 0, "entries": [], "unreadable": f"{path.name}: {e}"}

    def put_plan(self, case_id: str, target_id: str | None, stages: list[dict], info: dict,
                 violations: list[dict], stage_cap: int | None, strategy: str,
                 constraints: Constraints | None = None, parent_plan_id: str | None = None) -> str:
        inherited = self.constraints_for(case_id, parent_plan_id)
        c = constraints or inherited.patched({"stage_cap": stage_cap, "order": info.get("order", inherited.order)})
        pid = _new_id("p", self.plans)
        self.plans[pid] = {"plan_id": pid, "case_id": case_id, "target_id": target_id, "stages": stages,
            "info": {**info, "planner_version": PLANNER_VERSION}, "violations": violations, "stage_cap": c.stage_cap, "strategy": strategy,
            "constraints": c, "parent_plan_id": parent_plan_id,
            "review": {"status": "not_requested", "attempts": 0, "message": "", "error": None},
            "approval": None,
            "input_revision": self.targets.get(target_id or "", {}).get("input_revision")}
        # checked here, where every plan is made, against everything asked on the case so far (check_request); the
        # approval checks again against what was asked by then
        seg = self.requests.get(case_id, {}).get("seg", 0)
        if seg:
            self.plans[pid]["request_seg"] = seg
        if (check := self.check_request(pid)) is not None:
            self.plans[pid]["request_check"] = check
        self._persist(pid)
        return pid

    def plan_json(self, pid: str) -> dict:
        p = self.plans[pid]
        if p.get("_from_disk"):
            constraints = p["constraints"]
            if not isinstance(constraints, dict):
                constraints = constraints.model_dump(mode="json")
            data = {k: v for k, v in p.items() if k != "_from_disk"}
            data["constraints"] = constraints
            data["stages"] = [{str(i): v for i, v in st.items()} for st in p["stages"]]
            data["rotations"] = [{str(i): float(y) for i, y in getattr(st, "yaw", {}).items()} for st in p["stages"]]
            data["passed"] = not p["violations"]
            data["input_stale"] = self.input_stale(p["case_id"], p.get("input_revision"))
            data["previous_calculation"] = self.previous_calculation(pid)
            return data
        tinfo = self.targets.get(p["target_id"] or "", {}).get("info", {})
        return {"plan_id": pid, "case_id": p["case_id"], "strategy": p["strategy"], "stage_cap": p["stage_cap"],
                "parent_plan_id": p["parent_plan_id"], "constraints": p["constraints"].model_dump(mode="json"),
                "review": dict(p["review"]), "approval": dict(p["approval"]) if p["approval"] else None,
                "info": p["info"], "target": tinfo, "violations": p["violations"], "passed": not p["violations"],
                "input_revision": p.get("input_revision"),
                "input_stale": self.input_stale(p["case_id"], p.get("input_revision")),
                "previous_calculation": self.previous_calculation(pid),
                # a rule plan's run (the strategies one POST /api/plan tried): only on those plans, so the approval
                # fingerprint of every other plan, and of plan files written before it, stays what it was
                **({"rule_run": dict(p["rule_run"])} if p.get("rule_run") else {}),
                # what the dentist asked on the case against the plan's conditions (core/request_check), the same way:
                # only on plans of a case where something was asked, and the segment only after a 처음부터
                **({"request_check": dict(p["request_check"])} if p.get("request_check") else {}),
                **({"request_seg": p["request_seg"]} if p.get("request_seg") else {}),
                "stages": [{str(i): np.round(v, 4).tolist() for i, v in st.items()} for st in p["stages"]],
                # degrees about each crown's vertical axis through its centroid (pivot), per stage
                "rotations": [{str(i): round(float(y), 3) for i, y in getattr(st, "yaw", {}).items()} for st in p["stages"]],
                "pivots": {str(i): np.round(c, 4).tolist() for i, c in self.cases[p["case_id"]].pos0.items()}
                if p["case_id"] in self.cases else {}}

    def fingerprint(self, pid: str) -> str:
        if self.plans[pid].get("_from_disk"):
            data = self.plan_json(pid)
            data.pop("approval", None)
            return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        data = self.plan_json(pid)
        data.pop("approval")
        # Hash full precision coordinates, not the rounded display values.
        data["stages"] = [{str(i): np.asarray(v).tolist() for i, v in st.items()} for st in self.plans[pid]["stages"]]
        return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def approve(self, pid: str, lock_fdi: list[int] | None = None) -> dict:
        """Approval opens only on a match with everything asked on the case by now, and only with the locked teeth the
        dentist confirmed on the screen (`lock_fdi`, FDI): the server records them with the approval."""
        p = self._record(pid)
        self.require_current_input(p["case_id"], p.get("input_revision"))
        if p["violations"]:
            raise ValueError("규칙 위반 계획은 승인할 수 없습니다.")
        if p["review"]["status"] not in ("passed", "skipped"):
            raise ValueError("검토가 완료되지 않았습니다. 검토 실패/미실행 계획은 승인할 수 없습니다.")
        check = self.check_request(pid)
        if check is None and p.get("request_check"):
            # fail-closed: the plan was checked once, and what it was checked against is gone
            raise ValueError("이 계획을 대조한 요청 기록을 찾지 못했습니다. 다시 계획한 뒤 승인하세요.")
        if check is not None and check != p.get("request_check"):
            p["request_check"] = check          # what the screen shows is what refused it
            self._persist(pid)
        if check is not None and check["status"] == "mismatch":
            raise ValueError("요청 문장과 계획의 조건이 다릅니다: " + "; ".join(check["reasons_ko"]) +
                             ". 조건을 고쳐 다시 계획한 뒤 승인하세요.")
        if check is not None and check["status"] != "match":
            raise ValueError("요청 문장에 읽지 못한 부분이 있어 확인이 필요합니다: " +
                             ", ".join(check["unread_ko"] + [request_check.FIELDS_KO[f] for f in check["unchecked"]]) +
                             ". 번호마다 동작을 붙여 다시 말하거나(예: 「13번 고정해줘」) 조건 탭에서 고친 뒤 다시 계획하세요.")
        locked = self.lock_fdi(pid)
        if lock_fdi is None:
            raise ValueError("고정 치아 목록(FDI)을 화면에서 확인한 뒤 승인하세요.")
        if sorted(lock_fdi) != locked:
            raise ValueError(f"확인한 고정 치아({', '.join(map(str, sorted(lock_fdi))) or '없음'})와 계획의 고정 치아"
                             f"({', '.join(map(str, locked)) or '없음'})가 다릅니다. 다시 확인하세요.")
        p["approval"] = {"status": "approved", "approved_at": datetime.now(timezone.utc).isoformat(),
                         "fingerprint": self.fingerprint(pid), "confirmed_lock_fdi": locked}
        self._persist(pid)
        return self.plan_json(pid)

    def revoke(self, pid: str) -> dict:
        self._record(pid)["approval"] = None
        self._persist(pid)
        return self.plan_json(pid)

    def require_approved(self, pid: str) -> None:
        p = self._record(pid)
        self.require_current_input(p["case_id"], p.get("input_revision"))
        if not p["approval"] or p["approval"]["fingerprint"] != self.fingerprint(pid):
            raise ValueError("의사가 현재 계획을 승인한 뒤 내보낼 수 있습니다.")

    def set_review(self, pid: str, result: dict) -> dict:
        record = self._record(pid)
        record["review"] = dict(result)
        record["approval"] = None
        self._persist(pid)
        return dict(result)

    def _persist(self, pid: str) -> None:
        d = OUT_DIR / "plans"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{pid}.json").write_text(json.dumps(self.plan_json(pid), ensure_ascii=False), encoding="utf-8")


STORE = Store()
