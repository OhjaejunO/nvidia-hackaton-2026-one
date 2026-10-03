"""One real NIM conversation through the actual server, exercising the four features.

Runs `nat serve` with configs/workflow.yml (CuAlignWorker -> PlanEventsASGI + Guardrails),
then drives /chat/stream the way the UI does and reports what came back.
Never prints the API key. Writes the event trace to out/nim-live/.

Costs real NVIDIA usage: two conversations plus Guardrails input/output rails.
Run: uv run --frozen python tests/nim_live_check.py
"""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from cualign.core.fdi import from_fdi, to_fdi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "nim-live"
OUT.mkdir(parents=True, exist_ok=True)

REQUEST = "moderate 케이스로. 발치 없이 12개월 안에 끝나는 계획 짜줘. 앞니 총생부터 풀고."
REVISION = "13번 치아는 움직이지 말고 다시 짜줘."


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def wait_ready(url, proc, timeout=180):
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise SystemExit(f"server exited early with code {proc.returncode}")
            try:
                r = await client.get(url + "/api/cases", timeout=5)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(1.0)
    raise SystemExit("server did not become ready in time")


async def converse(client, url, text, ctx, log, messages=None, step=None):
    """POST /chat/stream exactly like app.js and collect the parsed SSE events. `messages` is the whole conversation
    the screen would send (ending with the user's `text`); without it the turn is a single user message. `step` is
    the turn's step (setup|target|stages), a top-level body field like the screen sends it."""
    events, answer = [], ""
    body = {"messages": messages or [{"role": "user", "content": text}], "cualign": ctx, **({"step": step} if step else {})}
    started = time.monotonic()
    async with client.stream("POST", url + "/chat/stream", json=body,
                             headers={"Accept": "text/event-stream"}, timeout=300) as r:
        if r.status_code != 200:
            raise SystemExit(f"/chat/stream -> HTTP {r.status_code} {await r.aread()}")
        buf = ""
        async for chunk in r.aiter_text():
            buf += chunk
            frames = buf.split("\n\n")
            buf = frames.pop()
            for frame in frames:
                if frame.strip().startswith("{"):  # NAT reports a workflow exception as a bare JSON line
                    try:
                        events.append(("error", json.loads(frame)))
                    except json.JSONDecodeError:
                        pass
                    continue
                # NAT uses the SSE field name itself (`intermediate_data:`), not `event:`.
                name = None
                payload = None
                for line in frame.splitlines():
                    i = line.find(": ")
                    if i < 0:
                        continue
                    field, raw = line[:i], line[i + 2:]
                    if field == "event":
                        name = raw.strip()
                    else:
                        name, payload = name or field, raw
                if payload is None or payload.strip() == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                events.append((name, obj))
                if name == "data":
                    ch = (obj.get("choices") or [{}])[0]
                    delta = ch.get("delta", {}).get("content") or ch.get("message", {}).get("content") or obj.get("value") or ""
                    if isinstance(delta, str):
                        answer += delta
    elapsed = time.monotonic() - started
    log.append({"request": text, "elapsed_s": round(elapsed, 1),
                "events": [{"event": n, "data": o} for n, o in events], "answer": answer})
    return events, answer, elapsed


async def lock_fdi(client, url, plan_id):
    """The plan's locked teeth as the screen's approval sends them (FDI): the dentist ticks them (Store.approve)."""
    d = (await client.get(url + f"/api/plans/{plan_id}", timeout=30)).json()
    return sorted(to_fdi(t) for t in d["constraints"]["lock"])


def pick(events, name):
    return next((o for n, o in events if n == name), None)


def tool_steps(events):
    names = []
    for n, o in events:
        if n == "intermediate_data":
            label = o.get("name") or ""
            if label:
                names.append(label)
    return names


async def main():
    load_dotenv(ROOT / ".env")
    if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
        raise SystemExit("NVIDIA_API_KEY is not set (put it in apps/cualign-prototype/.env)")

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PYTHONUTF8": "1", "CUALIGN_OUT": str(ROOT / "out")}
    logfile = open(OUT / "server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "nat.cli.main", "serve", "--config_file", str(ROOT / "configs" / "workflow.yml"),
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT), env=env, stdout=logfile, stderr=subprocess.STDOUT)

    log, verdict = [], {}
    try:
        await wait_ready(url, proc)
        async with httpx.AsyncClient() as client:
            await client.post(url + f"/api/cases/moderate/activate", timeout=30)

            # --- turn 1: plan from scratch -------------------------------------
            ctx = {"request_id": "live-1", "case_id": "moderate", "base_plan_id": None,
                   "constraints": {"allow_extraction": False, "stage_cap": 52, "order": "anterior_first"}}
            events, answer, elapsed = await converse(client, url, REQUEST, ctx, log)
            sel = pick(events, "plan_selected")
            verdict["turn1_elapsed_s"] = round(elapsed, 1)
            verdict["turn1_tools"] = tool_steps(events)
            verdict["turn1_answer_nonempty"] = bool(answer.strip())
            verdict["turn1_plan_selected"] = bool(sel)
            verdict["turn1_plan_error"] = pick(events, "plan_error")
            verdict["turn1_stream_error"] = pick(events, "error")
            if not sel:
                raise SystemExit(json.dumps(verdict, ensure_ascii=False, indent=2))
            plan_id = sel["plan_id"]
            verdict["turn1_plan_id"] = plan_id
            verdict["turn1_parent"] = sel["parent_plan_id"]
            verdict["turn1_review"] = sel["review"]

            detail = (await client.get(url + f"/api/plans/{plan_id}", timeout=30)).json()
            verdict["turn1_constraints"] = detail["constraints"]
            verdict["turn1_passed"] = detail["passed"]
            verdict["turn1_stages"] = detail["info"]["n_stages"]
            verdict["turn1_no_extraction_kept"] = detail["constraints"]["allow_extraction"] is False
            verdict["turn1_removed_teeth"] = detail["target"].get("removed")

            # --- approval gate --------------------------------------------------
            before = await client.get(url + f"/api/plans/{plan_id}/stl.zip", timeout=60)
            verdict["export_before_approval_status"] = before.status_code
            ap = await client.post(url + f"/api/plans/{plan_id}/approval",
                                   json={"confirmed": True, "lock_fdi": await lock_fdi(client, url, plan_id)}, timeout=30)
            verdict["approval_status"] = ap.status_code
            verdict["approval_detail"] = ap.json().get("detail") if ap.status_code != 200 else "approved"
            if ap.status_code == 200:
                after = await client.get(url + f"/api/plans/{plan_id}/stl.zip", timeout=120)
                verdict["export_after_approval_status"] = after.status_code
                verdict["export_bytes"] = len(after.content)

            # --- turn 2: revision keeps the earlier constraints -----------------
            # No lock in the context: the model must read «13번» (FDI) from the sentence. FDI 13 is app tooth 6; a
            # pre-injected Universal [13] (FDI 25) once hid that the live model locked the wrong tooth.
            ctx2 = {"request_id": "live-2", "case_id": "moderate", "base_plan_id": plan_id, "constraints": {}}
            events2, answer2, elapsed2 = await converse(client, url, REVISION, ctx2, log)
            sel2 = pick(events2, "plan_selected")
            verdict["turn2_elapsed_s"] = round(elapsed2, 1)
            verdict["turn2_plan_selected"] = bool(sel2)
            verdict["turn2_answer_nonempty"] = bool(answer2.strip())
            verdict["turn2_stream_error"] = pick(events2, "error")
            if sel2:
                verdict["turn2_plan_id"] = sel2["plan_id"]
                verdict["turn2_parent_is_turn1"] = sel2["parent_plan_id"] == plan_id
                verdict["turn2_review"] = sel2["review"]
                d2 = (await client.get(url + f"/api/plans/{sel2['plan_id']}", timeout=30)).json()
                verdict["turn2_constraints"] = d2["constraints"]
                verdict["turn2_lock_applied"] = d2["constraints"]["lock"] == [from_fdi(13)]
                verdict["turn2_request_check"] = d2.get("request_check")
                verdict["turn2_no_extraction_kept"] = d2["constraints"]["allow_extraction"] is False
                verdict["turn2_order_kept"] = d2["constraints"]["order"]
                verdict["turn2_approval_reset"] = d2["approval"] is None
                verdict["turn2_locked_tooth_still"] = all(
                    st.get(str(from_fdi(13))) == [0, 0, 0] for st in d2["stages"])
            ctxev = pick(events2, "plan_context")
            verdict["turn2_plan_context"] = ctxev.get("constraints") if ctxev else None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        logfile.close()
        (OUT / "trace.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
