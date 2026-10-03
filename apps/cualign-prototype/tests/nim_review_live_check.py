"""Real NIM check of review recovery: «검토 다시 요청» and the reviewer memo's output rail.

Builds the actual app from configs/workflow.yml in this process (CuAlignWorker: NIM reviewer, Guardrails, API),
so plans can be put in the store in the states the route recovers: one the agent never reviewed (not_requested)
and one whose review failed. Then POSTs /api/plans/{id}/review for each, as the UI button does.
The server's own fallback for a skipped review only runs when the agent skips it, which a model call cannot be
made to do; that path is covered offline (tests/test_rails_middleware.py::test_skipped_review_runs_on_server...).
Never prints the API key. Writes the result to out/nim-live/review-recovery.json.

Costs real NVIDIA usage: up to four reviewer calls (two per plan) and an output rail check per memo.
Run: uv run --frozen python tests/nim_review_live_check.py
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from nat.runtime.loader import load_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "nim-live"


def main():
    load_dotenv(ROOT / ".env")
    if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
        raise SystemExit("NVIDIA_API_KEY is not set (put it in apps/cualign-prototype/.env)")
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUALIGN_OUT", str(ROOT / "out"))
    from cualign.agent import reviewer
    from cualign.core.constraints import Constraints
    from cualign.core.service import PlanningService
    from cualign.core.store import STORE
    from cualign.server.worker import CuAlignWorker

    config = ROOT / "configs" / "workflow.yml"
    os.environ["NAT_CONFIG_FILE"] = str(config)
    verdict = {}
    with TestClient(CuAlignWorker(load_config(config)).build_app()) as client:
        verdict["memo_rail_wired"] = reviewer.MEMO_CHECK is not None
        svc = PlanningService(STORE)
        STORE.load_case("moderate")
        unreviewed = svc.stages(svc.target("moderate", "expansion_ipr", Constraints(allow_extraction=False)))
        failed = svc.stages(svc.target("moderate", "expansion_ipr", Constraints(allow_extraction=False)))
        STORE.set_review(failed, {"status": "failed", "attempts": 2, "message": "검토 실패 — 계획을 승인할 수 없습니다.",
                                  "error": "timeout"})
        for name, pid in (("not_requested", unreviewed), ("failed", failed)):
            started = time.monotonic()
            res = client.post(f"/api/plans/{pid}/review")
            row = {"http": res.status_code, "elapsed_s": round(time.monotonic() - started, 1)}
            if res.status_code == 200:
                review = res.json()["review"]
                row.update(status=review["status"], attempts=review["attempts"], error=review["error"],
                           rails=review.get("rails"), memo_chars=len(review["message"]),
                           memo_head=review["message"][:120])
                if review["status"] == "passed" and res.json()["passed"]:
                    ap = client.post(f"/api/plans/{pid}/approval", json={"confirmed": True, "lock_fdi": STORE.lock_fdi(pid)})
                    row["approval_http"] = ap.status_code
            else:
                row["detail"] = res.json().get("detail")
            if row.get("status") == "passed":  # a finished review is not run again
                row["second_request_http"] = client.post(f"/api/plans/{pid}/review").status_code
            verdict[name] = row
    (OUT / "review-recovery.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
