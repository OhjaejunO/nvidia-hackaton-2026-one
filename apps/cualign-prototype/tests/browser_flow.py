"""Browser acceptance using synthetic plans and a fake reviewer, never NIM.
Run: uv run --frozen --with playwright python tests/browser_flow.py
Requires installed Chrome or PLAYWRIGHT_CHROMIUM_EXECUTABLE. UI CDN access is required.
"""
import asyncio
import json
import math
import os
import re
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from playwright.async_api import async_playwright

from cualign.agent import register
from cualign.agent.context import CURRENT_RUN
from cualign.agent.reviewer import review_plan
from cualign.core import store as store_module
from cualign.server import api, plan_events

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_patients import _scan_files   # noqa: E402  synthetic per-tooth scans, as the API tests use


# the newest plan of 000131 whose mesh the server cuts (#22): the plan list is newest first, so the run's own IPR plan comes first
FIND_CUT_PLAN = """(async () => {
  const plans = (await (await fetch('/api/plans?case_id=poseidon-000131')).json()).plans;
  for (const p of plans.filter((p) => /ipr/.test(p.strategy))) {
    const m = await (await fetch(`/api/cases/poseidon-000131/mesh?plan_id=${p.plan_id}`)).json();
    if (Object.keys(m.teeth_cut).length) return p.plan_id;
  }
  return null;
})()"""


def document_has_plan(body_class: str) -> bool:
    return "has-plan" in body_class.split()


def write_scan(folder: Path, **kw) -> list[str]:
    """Synthetic scan files on disk for the file chooser: drop=(4,) leaves a gap, rename flips the numbering."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for _, (name, data, _mime) in _scan_files(**kw):
        (folder / name).write_bytes(data)
        paths.append(str(folder / name))
    return paths

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "browser-acceptance"
OUT.mkdir(parents=True, exist_ok=True)
store_module.OUT_DIR = api.OUT_DIR = OUT
app = FastAPI()


@app.get("/api/cases/{case_id}/targets/{target_id}")
async def fake_target(case_id: str, target_id: str):
    """The target state in the plan's shape with one stage (the final one) — for the tests' t-fake, which the server's
    own route (#146) does not have; registered before the server's routes so it answers first."""
    pid = next((pid for pid in reversed(list(store_module.STORE.plans)) if store_module.STORE._record(pid)["case_id"] == case_id), None)
    if pid is None:
        pid = api.rule_based_plan(case_id)["chosen"]["plan_id"]
    plan = store_module.STORE.plan_json(pid)
    return {**plan, "target_id": target_id, "plan_id": None, "stages": plan["stages"][-1:], "rotations": (plan.get("rotations") or [])[-1:],
            "violations": []}


@app.get("/api/cases/{case_id}/targets/{target_id}/cut")
async def fake_target_cut(case_id: str, target_id: str):
    """The t-fake target's cut crowns (#22): the case's latest plan's cut with plan_id None and the target_id."""
    cid, case = store_module.STORE.load_case(case_id)
    pid = next((pid for pid in reversed(list(store_module.STORE.plans)) if store_module.STORE._record(pid)["case_id"] == cid), None)
    return {**api.ipr_cut_view(cid, case, pid), "plan_id": None, "target_id": target_id}


@app.get("/api/cases/{case_id}/mesh")
async def fake_mesh(case_id: str, plan_id: str | None = None, target_id: str | None = None):
    """The mesh route with `target_id` (server round 15, not on main yet): the target's cut dentition — shaped here as the
    case's latest plan's cut with plan_id None and the target_id. Without target_id it is the server's own route."""
    cid, case = store_module.STORE.load_case(case_id)
    data = case.viewer_json()
    data.update(api.gum_filled_view(cid, case, data["gum"]))
    if target_id:
        pid = next((pid for pid in reversed(list(store_module.STORE.plans)) if store_module.STORE._record(pid)["case_id"] == cid), None)
        data.update(api.ipr_cut_view(cid, case, pid))
        data.update({"plan_id": None, "target_id": target_id})
    else:
        data.update(api.ipr_cut_view(cid, case, plan_id))
    return data


async def fake_manual_review(plan_id):
    class Memo:
        async def ainvoke(self, messages):
            return "검토 메모 초안 (가짜 검토)"
    return await review_plan(plan_id, Memo(), manual=True)


api.add_api_routes(app, review=fake_manual_review)
app.add_middleware(plan_events.PlanEventsASGI)


SETUP_TURNS = [0]   # the setup turns the fake agent answered
STAGE_TOOL_S = [3.5]   # how long the slow stage turn's stage tool runs (a live NIM turn is 10–20 s with its reasoning)


def sse(name, obj):
    return f"event: {name}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n"


def nat_step(kind, tool, args, sid):
    """A tool row's NAT event as the server sends it (worker.py ToolStepsOnly): a start with the arguments, an end with
    the same id whose output is only 완료."""
    body = "**Function Input:**\n```json\n" + json.dumps(args) + "\n```" + ("\n\n**Function Output:**\n완료" if kind == "End" else "")
    return "intermediate_data: " + json.dumps({"id": sid, "name": f"Function {kind}: cualign__{tool}", "payload": body}, ensure_ascii=False) + "\n\n"


@app.post("/chat/stream")
async def fake_chat(request: Request):
    """The agent by step (#20 contract; the middleware already read the cualign block): setup → step_done with the
    constraints it read, target → step_done with a target id, stages → the real tools, so plan_selected comes from
    the middleware. The reviewer fails only on the 앞니 IPR turn."""
    body = json.loads(await request.body())
    run = CURRENT_RUN.get()
    step = body.get("step") or (run.step if run else None)   # the middleware (#146) reads the top-level step into the run
    text = next((m["content"] for m in reversed(body.get("messages", [])) if m.get("role") == "user"), "")
    async def generate():
        if "단계 도구가 오래" in text:
            # the stage turn as a live NIM run sends it (8000 log 19:05:01–02): plan_stages, then select_plan at once —
            # the stage tool held STAGE_TOOL_S here, so the screen shows its round while the agent works
            async with register.cualign(register.CuAlignToolConfig(), None) as group:
                tools = await group.get_all_functions()
                tool = lambda name: next(fn for key, fn in tools.items() if key.split("__")[-1] == name)
                yield nat_step("Start", "plan_stages", {"target_id": "t-fake"}, "g1"); await asyncio.sleep(STAGE_TOOL_S[0])
                result = await tool("compare_strategies").ainvoke(register.CompareInput())
                yield nat_step("End", "plan_stages", {"target_id": "t-fake"}, "g1")
                pick = {"plan_id": next((p for p in result["plans"] if p["passed"]), result["plans"][0])["plan_id"]}
                yield nat_step("Start", "select_plan", pick, "g2")
                await tool("select_plan").ainvoke(register.PlanIdInput(**pick))
                yield nat_step("End", "select_plan", pick, "g2")
                yield sse("step_done", {"request_id": run.request_id, "step": "stages"})
                yield 'data: {"value":"단계를 만들었습니다."}\n\n'
            return
        if step == "setup":
            # the two real setup turns of 2026-09-28, in their order, the waits shortened. The first as 000097's (8000 log
            # 17:22:41–17:23:28): nothing, then (since #189) the planner's reasoning, the set_constraints row, then the answer
            # and step_done together (the output rails hold the answer). The next as 000001's: no tool event at all — the form already carries the
            # prescription, the agent answers directly — only the answer and step_done; the screen writes 조건 설정.
            SETUP_TURNS[0] += 1
            run.constraints = run.constraints.patched({"extraction": [5, 12]})   # the conditions the answer states (a case started over has none)
            if SETUP_TURNS[0] == 1:
                await asyncio.sleep(1.0)
            # the planner's reasoning comes first, as NAT sends it (rails_middleware.ReasoningRelay, worker.ToolStepsOnly)
            for i, line in enumerate(("We need to read the prescription. It gives extraction 14 and 24.", "So we set the constraints.")):
                yield "intermediate_data: " + json.dumps({"id": f"r{i}", "type": "reasoning", "name": "reasoning", "payload": line}) + "\n\n"
                await asyncio.sleep(0.3)
            if SETUP_TURNS[0] == 1:
                args = '**Input:**\n```json\n{"extraction": [5, 12]}\n```'
                yield "intermediate_data: " + json.dumps({"id": "s1", "name": "Function Start: cualign__set_constraints", "payload": args}) + "\n\n"
                yield "intermediate_data: " + json.dumps({"id": "s1", "name": "Function End: cualign__set_constraints", "payload": args + "\n\n**Output:**\n완료"}) + "\n\n"
                await asyncio.sleep(2.5)
            yield 'data: {"value":"처방을 조건으로 옮겼습니다. 발치 14·24, IPR 면당 0.25 mm."}\n\n'
            yield sse("step_done", {"request_id": run.request_id, "step": "setup", "constraints": run.constraints.model_dump(mode="json")})
            return
        if step == "target":
            # the target tool runs a moment first (target-reveal.js draws the arch and settles the crowns meanwhile)
            yield nat_step("Start", "propose_target", {"strategy": "extraction"}, "t1"); await asyncio.sleep(2.0)
            yield nat_step("End", "propose_target", {"strategy": "extraction"}, "t1")
            yield sse("step_done", {"request_id": run.request_id, "step": "target", "target_id": "t-fake",
                                    "summary": {"crowding_mm": 7.9, "space_mm": 15.8, "strategy": "extraction", "extraction": [14, 24], "ipr": []}})
            yield 'data: {"value":"목표 배열을 만들었습니다. 초기와 비교해 보세요."}\n\n'
            return
        async with register.cualign(register.CuAlignToolConfig(), None) as group:
            tools = await group.get_all_functions()
            def tool(name):
                return next(fn for key, fn in tools.items() if key.split("__")[-1] == name)
            # Exercise actual NAT tool functions, but no external LLM. Each call streams its start and end rows as NAT
            # does, a moment apart so the screen shows them running (the tool line's seconds, the 3D chip's words, the
            # rule check's sweep).
            yield nat_step("Start", "set_constraints", {"ipr_exclude": [12, 11, 21, 22]}, "s1"); await asyncio.sleep(0.3)
            await tool("set_constraints").ainvoke({"ipr_exclude": [12, 11, 21, 22]})   # FDI, stored as Universal 7..10
            yield nat_step("End", "set_constraints", {"ipr_exclude": [12, 11, 21, 22]}, "s1")
            yield nat_step("Start", "compare_strategies", {}, "s2"); await asyncio.sleep(0.3)
            result = await tool("compare_strategies").ainvoke(register.CompareInput())
            yield nat_step("End", "compare_strategies", {}, "s2")
            selected = next((p for p in result["plans"] if p["passed"]), result["plans"][0])
            pick = {"plan_id": selected["plan_id"]}
            yield nat_step("Start", "validate", pick, "s3"); await asyncio.sleep(1.2)
            await tool("validate").ainvoke(register.PlanIdInput(**pick))
            yield nat_step("End", "validate", pick, "s3")
            yield nat_step("Start", "select_plan", pick, "s4")
            await tool("select_plan").ainvoke(register.PlanIdInput(**pick))
            yield nat_step("End", "select_plan", pick, "s4")
            if "앞니 빼고" in text:
                class Empty:
                    async def ainvoke(self, messages):
                        return ""
                yield nat_step("Start", "reviewer", pick, "s5"); await asyncio.sleep(0.3)
                await review_plan(selected["plan_id"], Empty())
                yield nat_step("End", "reviewer", pick, "s5")
            yield sse("step_done", {"request_id": run.request_id, "step": "stages"})
            # An unrelated ID in text must not control selection.
            yield 'data: {"value":"단계를 만들었습니다. 이전 후보 plan_id: p999는 선택하지 않습니다."}\n\n'
    return StreamingResponse(generate(), media_type="text/event-stream")


# a wait for "the next frame" to be drawn, in ms: generous, so it holds under load and at a lowered ?fps= (app.js)
FRAME = 250

# one browser_flow at a time on this machine, whatever the worktree: two at once (each a server and a software-WebGL
# Chrome) took the whole CPU and timed each other out. A live run touches its lock every minute, so one untouched for
# 15 minutes is a run that died: it is taken over.
LOCK = Path(tempfile.gettempdir()) / "cualign-browser-flow.lock"
LOCK_STALE = 15 * 60


def take_lock():
    t0 = time.monotonic(); said = 0
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {Path.cwd()}".encode()); os.close(fd)
            def beat():   # a live run keeps its lock fresh however long it takes
                while True:
                    try: os.utime(LOCK)
                    except FileNotFoundError: return
                    time.sleep(60)
            threading.Thread(target=beat, daemon=True).start()
            print(f"browser_flow 시작 ({int(time.monotonic() - t0)}초 기다림)", flush=True)
            return
        except FileExistsError:
            try:
                if time.time() - LOCK.stat().st_mtime > LOCK_STALE:
                    LOCK.unlink(missing_ok=True); continue
            except FileNotFoundError:
                continue
        waited = time.monotonic() - t0
        if waited > LOCK_STALE:
            raise SystemExit(f"another browser_flow still holds {LOCK} after {LOCK_STALE // 60} minutes")
        if waited >= said:
            print(f"다른 browser_flow 가 끝나기를 기다리는 중 … {int(waited)}초", flush=True); said += 10
        time.sleep(1)


async def main():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets":[sock]}, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    browser = None
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("test server did not start")
        async with async_playwright() as pw:
            executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
            # the 3D on the machine's GPU (Direct3D 11 through ANGLE), not SwiftShader: software WebGL took whole cores. Playwright's
            # own --enable-unsafe-swiftshader is dropped too, so a GPU that is not there fails the 3D steps instead of falling
            # back to the CPU unseen.
            browser = await pw.chromium.launch(executable_path=executable, headless=True, ignore_default_args=["--enable-unsafe-swiftshader"],
                args=["--headless=new", "--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
            page = await browser.new_page(viewport={"width":1500,"height":1000})
            errors = []
            page.on("pageerror", lambda error: (errors.append(str(error)), print("PAGEERROR:", error, file=sys.stderr)))   # visible when a step times out
            page.on("dialog", lambda dialog: dialog.accept())
            # ?case=<id> is the documented dev/test hook; the base sample is 000097 「심한 덧니, 발치 필요」 (#15).
            await page.goto(url + "/ui/?case=poseidon-000097")
            await page.wait_for_function("!document.body.classList.contains('start')")

            # Rule-based plan from the sidebar's 조건 tab (no model). The plan on screen is the canvas's data-plan-id;
            # the right panel's plan picker shows it in one line (#111). Helpers read the test hook, not the removed result card.
            plan_on_screen = "(window.__cualign.state.plan?.plan_id ?? '')"
            async def on_screen():
                return await page.evaluate(plan_on_screen)
            async def approval():
                return await page.evaluate("window.__cualign.state.plan?.approval ?? null")
            async def pick_plan(pid):   # the picker: ▾ opens the list, the row picked goes on screen and the list closes
                if await page.locator("#planMenu").is_hidden():
                    await page.locator("#planMore").click()
                await page.locator(f'#planMenu .plan-row[data-plan="{pid}"]').click()
            # The agent drives the steps (#20). The case opens as the scan: no plan, the strip at 초기 with the rest closed,
            # the 스캔 tab with the facts, and the agent's first word with the sample's prescription as one chip.
            assert not document_has_plan(await page.evaluate("document.body.className"))
            assert await page.locator("body").evaluate("b => b.classList.contains('step-initial')")
            assert await page.locator('#flow button[data-step="setup"]').is_disabled() and await page.locator('#flow button[data-step="stages"]').is_disabled()
            assert await page.locator("#tabScan").get_attribute("aria-selected") == "true"
            assert "치아" in await page.locator("#scanFacts").inner_text() and "총생" in await page.locator("#scanFacts").inner_text()
            # the tooth chart: 14 crowns by FDI, clean until the setup reads the prescription (no 발치 처방 mark, no 처방 row);
            # a click on it selects in the 3D too
            scan_rx = "[[...document.querySelectorAll('#toothChart g.tooth.ext')].map(g => g.dataset.fdi).sort(), [...document.querySelectorAll('#scanFacts dt')].some(d => d.textContent === '처방')]"
            assert await page.locator("#toothChart g.tooth").count() == 14
            assert await page.evaluate(scan_rx) == [[], False]
            await page.locator('#toothChart g.tooth[data-fdi="11"]').click()
            assert await page.evaluate("[...window.__cualign.state.selected]") == ["8"] and await page.locator("#toothChart g.tooth.sel").count() == 1
            await page.locator('#toothChart g.tooth[data-fdi="11"]').click()
            assert await page.evaluate("window.__cualign.state.selected.size") == 0
            assert await page.locator("#tabCond").evaluate("b => getComputedStyle(b).pointerEvents") == "none"
            assert "처방을 적어 주세요" in await page.locator(".msg.assistant").last.inner_text()
            next_chip = lambda label: page.locator(".next:not(.done) button", has_text=label)
            assert await next_chip("이 케이스의 처방 넣기").count() == 1
            # turn 1 (setup): the chip sends the prescription; step_done{setup} fills the conditions, the extracted crowns flash
            # red and go, the strip opens 셋업, the next chips ask for the target
            # which reaches the transcript first, the reasoning line or the answer (reasoning events, stream-reasoning)
            await page.evaluate("""() => { const seen = window.__firstSeen = [];
              new MutationObserver(() => {
                if (!seen.includes('thinking') && document.querySelector('.trace .thinking')) seen.push('thinking');
                if (!seen.includes('answer') && [...document.querySelectorAll('.msg.assistant')].some((m) => m.textContent.includes('처방을 조건으로 옮겼습니다'))) seen.push('answer');
              }).observe(document.getElementById('transcript'), { subtree: true, childList: true, characterData: true }); }""")
            # per frame from here: the extracted crowns' lifting copies (scan-reveal ②) and the moment step_done lands (its reasoning line)
            await page.evaluate("""() => { const w = window, s = w.__cualign.state; w.__fx = [];
              const tick = () => { const g = s.teeth['5']?.parent, teeth = new Set(Object.values(s.teeth));
                const clones = g ? g.children.filter((o) => o.isMesh && !teeth.has(o) && o.material?.depthWrite === false && o.material.opacity < 1).length : 0;
                const told = [...document.querySelectorAll('.trace .reason')].some((r) => r.textContent.startsWith('처방을 읽었습니다'));
                w.__fx.push({ t: performance.now(), clones, told });
                if (w.__fx.length < 3000) requestAnimationFrame(tick); };
              tick(); }""")
            await next_chip("이 케이스의 처방 넣기").click()
            # the 3D reads the scan only once the agent's first line is in the transcript (here a reasoning line, 1 s in)
            await page.wait_for_timeout(500)
            assert await page.locator(".trace .step, .trace .thinking").count() == 0 and await page.locator(".fx-num").count() == 0
            await page.wait_for_function("document.querySelector('.trace .step, .trace .thinking')", timeout=10000)
            await page.wait_for_function("document.querySelectorAll('.fx-num').length > 0", timeout=5000)
            await page.wait_for_function("document.body.classList.contains('step-setup') && !document.querySelector('#sendBtn').disabled", timeout=60000)
            # the reasoning line reached the transcript before the answer, and folded to a count once the answer came
            assert await page.evaluate("window.__firstSeen") == ["thinking", "answer"], await page.evaluate("window.__firstSeen")
            assert (await page.locator(".trace").last.locator(".thinking").inner_text()).startswith("모델 추론 2줄")
            assert await page.locator("#cExtract").input_value() == "14, 24"
            assert await page.locator("#tabCond").get_attribute("aria-selected") == "true"
            # no dimming over the 3D during a turn (only a chip under it); the chip's dot settles as a check and it goes 0.6 s later
            assert await page.evaluate("getComputedStyle(document.getElementById('viewCanvas')).opacity") == "1"
            assert await page.locator("#workNote").get_attribute("data-state") == "ok" and await page.locator("#workNote .pulse-dot.ok").count() == 1
            await page.wait_for_function("!document.getElementById('workNote').classList.contains('on')", timeout=2000)
            # the reasoning in sentences: the tool row and its line show without a click, the raw call only under 자세히
            trace = page.locator(".trace").last
            assert (await trace.locator(".head").inner_text()).endswith("도구 1회 · 조건 설정")
            assert await trace.locator(".step").is_visible() and "조건 설정" in await trace.locator(".step").inner_text()
            reasons = await trace.locator(".reason").all_inner_texts()
            assert reasons[0] == "처방을 읽고 계획 조건으로 옮깁니다." and reasons[1].startswith("처방을 읽었습니다 — 발치 치아 14, 24번, 고정 치아 없음"), reasons
            assert reasons[2] == "총생 7.9 mm 만큼 공간이 필요합니다.", reasons
            assert not await trace.locator(".raw pre").is_visible() and "extraction" not in await trace.inner_text()
            await trace.locator(".raw > summary", has_text="자세히").click()
            assert await trace.locator(".raw pre").is_visible() and "extraction" in await trace.locator(".raw pre").inner_text()
            # the extracted crowns lift and fade after step_done (0.6 s): a copy is on screen 0.3 s in, none 0.8 s in, then the setup lands
            fx = await page.evaluate("window.__fx")
            told = next(f["t"] for f in fx if f["told"])
            after = [(round(f["t"] - told), f["clones"]) for f in fx if told <= f["t"] <= told + 1500]
            assert any(n for dt, n in after if 200 <= dt <= 450), after
            assert not any(n for dt, n in after if dt >= 800), after
            removed_crown = "() => { const s = window.__cualign.state; return s.teeth[String(s.setup.extraction[0])].visible; }"
            await page.wait_for_function(f"!({removed_crown})()", timeout=5000)      # red for a moment, then gone (#20)
            await page.wait_for_function(f"JSON.stringify({scan_rx}) === JSON.stringify([['14', '24'], true])", timeout=10000)      # the 스캔 tab: 14·24 and 처방 with the 3D
            assert (await page.evaluate("location.hash")).endswith("&step=setup")
            assert await page.locator('#flow button[data-step="setup"]').is_enabled() and await page.locator('#flow button[data-step="target"]').is_disabled()
            assert await next_chip("목표 배열 만들기").count() == 1 and await next_chip("조건 바꾸기").count() == 1
            # 「처음부터」 on the case header: back to 초기 in one click (flow and conditions forgotten on the server), the address
            # only #case=…, the scan tab, no cut or label, and the prescription chip again; the chip runs the setup turn again
            await page.locator("#restartBtn").click()
            await page.wait_for_function("document.body.classList.contains('step-initial') && window.__cualign.state.progress === 'initial' && !window.__cualign.state.setup", timeout=30000)
            assert await page.evaluate("location.hash") == "#case=poseidon-000097" and await page.evaluate("window.__cualign.state.plan") is None
            assert await page.locator("#tabScan").get_attribute("aria-selected") == "true" and await page.locator("#flow button[data-step=\"setup\"]").is_disabled()
            assert await page.locator(".trace").count() == 0 and await page.locator(".ipr-mark, .extract-mark").count() == 0
            assert await next_chip("이 케이스의 처방 넣기").count() == 1 and await page.locator(".restart-link").count() == 0
            assert await page.locator(".msg.system", has_text="조건이 처방과 다릅니다").count() == 0      # empty conditions on purpose, no warning
            assert await page.evaluate(scan_rx) == [[], False]      # the 스캔 tab clean again: the marks came with the setup, they go with it
            await next_chip("이 케이스의 처방 넣기").click()
            await page.wait_for_function("document.body.classList.contains('step-setup') && !document.querySelector('#sendBtn').disabled", timeout=60000)
            again = (await page.locator("#cExtract").input_value(), await page.locator(".next:not(.done) button").all_inner_texts())
            assert again[0] == "14, 24" and "목표 배열 만들기" in again[1], again
            # turn 2 (target): step_done{target_id} → GET /targets → the 3D shows the target state (one stage); 초기 ↔ 목표 look back
            await next_chip("목표 배열 만들기").click()
            # 치열궁을 따라 자리 잡는다 (target-reveal.js): 1 s into the target tool the arch line is drawn and the setup stands
            # as the ghost; after the landing neither is left
            await page.locator('.step.tool.running[data-tool="propose_target"]').wait_for(timeout=30000)
            await page.wait_for_timeout(1000)
            assert await page.evaluate("[window.__targetReveal.lineShown, window.__targetReveal.ghostShown]") == [True, True]
            await page.wait_for_function("document.body.classList.contains('step-target') && window.__cualign.state.targetId === 't-fake' && !document.querySelector('#sendBtn').disabled", timeout=60000)
            await page.wait_for_function("!window.__targetReveal.lineShown && !window.__targetReveal.ghostShown", timeout=10000)
            assert await page.evaluate("window.__cualign.state.stage") == 1 and not document_has_plan(await page.evaluate("document.body.className"))
            assert await page.locator("#tabRules").get_attribute("aria-selected") == "true"      # 목표 → 규칙, the target's checks
            assert "단계 계획 전" in await page.locator("#rulesFor").inner_text()
            assert await page.locator('#flow button[data-step="stages"]').is_disabled()
            assert "확보 15.8 mm 로 총생 7.9 mm 를 해결했습니다." in await page.locator(".trace").last.locator(".reason").all_inner_texts()
            await page.locator('#flow button[data-step="initial"]').click()
            await page.wait_for_function("document.body.classList.contains('step-initial') && window.__cualign.state.stage === 0")
            assert "&step=" not in await page.evaluate("location.hash")
            await page.locator('#flow button[data-step="target"]').click()
            await page.wait_for_function("document.body.classList.contains('step-target') && window.__cualign.state.stage === 1")
            assert await next_chip("단계 만들기").count() == 1 and await next_chip("12개월 안에").count() == 1
            assert await next_chip("비발치안과 비교").count() == 0      # an extraction prescription: no comparison chip
            # turn 3 (stages): the real tools make the plan; plan_selected lands it on 단계 with the slider
            await next_chip("단계 만들기").click()
            # a tool line runs with its seconds counting (「0.4s」), then settles as 「완료 · 0.4s」 with a check
            running = page.locator(".trace").last.locator(".step.tool.running .time")
            await running.first.wait_for(timeout=60000)
            assert re.fullmatch(r"\d+\.\ds", await running.first.inner_text()), await running.first.inner_text()
            await page.wait_for_function(f"{plan_on_screen}.startsWith('p') && document.body.classList.contains('step-stages') && !document.querySelector('#sendBtn').disabled"
                f" && document.querySelector('#viewCanvas').dataset.planId === {plan_on_screen}", timeout=120000)
            parent = await on_screen()
            # the selected plan opens at its last stage, the target reached
            assert await page.evaluate("window.__cualign.state.stage === window.__cualign.state.plan.stages.length && window.__cualign.state.stage > 0")
            assert await next_chip("승인하고 내보내기").count() == 1
            tool_rows = page.locator(".trace").last.locator(".step.tool")
            assert await tool_rows.count() == 4 and await tool_rows.locator(".pulse-dot.ok").count() == 4
            times = await tool_rows.locator(".time").all_inner_texts()
            assert all(re.fullmatch(r"완료 · \d+\.\ds", t) for t in times), times
            reasons = await page.locator(".trace").last.locator(".reason").all_inner_texts()
            k = next(i for i, r in enumerate(reasons) if "단계(약 " in r)      # after the tools' own lines
            assert reasons[k + 1].startswith("규칙 검증 — 위반 "), reasons
            assert await page.locator("#tabStages").get_attribute("aria-selected") == "true"     # 단계 → 단계 표
            # the strip is open end to end now: the step in the address, a reload and the strip's buttons keep it
            await page.locator('#flow button[data-step="initial"]').click()
            await page.wait_for_function("document.body.classList.contains('step-initial') && window.__cualign.state.stage === 0")
            assert not await page.locator(".stage-bar").is_visible()
            assert await page.locator("#tabScan").get_attribute("aria-selected") == "true"
            await page.locator('#flow button[data-step="setup"]').click()
            await page.wait_for_function("document.body.classList.contains('step-setup')")
            assert (await page.evaluate("location.hash")).endswith("&step=setup")
            assert await page.locator("#tabCond").get_attribute("aria-selected") == "true"
            # a tab picked by hand holds while the step stays; the next step takes the panel again
            await page.locator("#tabScan").click()
            await page.locator('#flow button[data-step="setup"]').click()
            assert await page.locator("#tabScan").get_attribute("aria-selected") == "true"
            await page.locator('#flow button[data-step="target"]').click()
            await page.wait_for_function("document.body.classList.contains('step-target') && window.__cualign.state.stage === 1")
            assert await page.locator("#tabRules").get_attribute("aria-selected") == "true"
            await page.locator('#flow button[data-step="stages"]').click()
            await page.wait_for_function("document.body.classList.contains('step-stages') && window.__cualign.state.stage === 0")
            assert await page.locator(".stage-bar").is_visible()
            # 치료 전 (stage 0) is the scan with every crown; the extracted crowns are gone from stage 1 on, no translucent silhouette
            crown = "() => { const s = window.__cualign.state, id = String(s.plan.target.removed[0]), m = s.teeth[id]; return [m.visible, m.material.opacity]; }"
            assert await page.evaluate(crown) == [True, 1]
            await page.locator("#stageSlider").fill("1")
            await page.wait_for_function("window.__cualign.state.stage === 1")
            assert await page.evaluate(crown) == [False, 1]
            await page.locator("#firstBtn").click(); await page.wait_for_function("window.__cualign.state.stage === 0")
            # the 3D turns without a pole clamp (#18): a 720° vertical drag comes back to the start, the distance never changes,
            # and a horizontal drag afterwards keeps the arch plane level (no roll drift)
            await page.locator('.view-rail [data-view="frontal"]').click()
            cam = "() => { const c = window.__cualign.camera, t = window.__cualign.controls.target; return { p: c.position.toArray(), d: c.position.distanceTo(t), rz: new c.position.constructor(1, 0, 0).applyQuaternion(c.quaternion).z }; }"
            start = await page.evaluate(cam)
            px = 4 * math.pi / await page.evaluate("window.__cualign.controls.speed")
            for _ in range(8):
                await page.mouse.move(960, 300); await page.mouse.down(); await page.mouse.move(960, 300 + px / 8, steps=15)
                await page.wait_for_timeout(120); await page.mouse.up()      # a still pointer at release: no inertia
                await page.wait_for_timeout(FRAME)
            end = await page.evaluate(cam)
            assert math.dist(start["p"], end["p"]) < 8 and abs(start["d"] - end["d"]) < 1e-3, (start, end)
            await page.mouse.move(960, 500); await page.mouse.down(); await page.mouse.move(1260, 500, steps=10); await page.wait_for_timeout(120); await page.mouse.up()
            assert abs((await page.evaluate(cam))["rz"]) < 1e-3
            # a drag to the right moves the crown nearest the camera to the right on screen, in every preset
            near = """() => { const { camera, state } = window.__cualign; let best = null;
                for (const [id, m] of Object.entries(state.teeth)) { if (!m.visible) continue; const p = state.center[id].clone().add(m.position), d = p.distanceTo(camera.position); if (!best || d < best.d) best = { id, d, p }; }
                return best.id; }"""
            screen_x = "(id) => { const { camera, state } = window.__cualign; return state.center[id].clone().add(state.teeth[id].position).project(camera).x; }"
            for view in ["occlusal", "frontal", "left", "right"]:
                await page.locator(f'.view-rail [data-view="{view}"]').click(); await page.wait_for_timeout(FRAME)   # project() reads the last frame's camera
                tooth = await page.evaluate(near); x0 = await page.evaluate(screen_x, tooth)
                await page.mouse.move(960, 500); await page.mouse.down(); await page.mouse.move(1000, 500, steps=5); await page.wait_for_timeout(120); await page.mouse.up()
                await page.wait_for_timeout(FRAME)
                assert await page.evaluate(screen_x, tooth) > x0, view
            await page.mouse.dblclick(960, 500)
            await page.wait_for_function("document.querySelector('.view-rail [data-view=\"occlusal\"]').getAttribute('aria-pressed') === 'true'")
            await page.reload()
            await page.wait_for_function("document.body.classList.contains('step-stages') && document.body.classList.contains('has-plan')", timeout=120000)
            assert await on_screen() == parent
            assert await page.locator("#viewCanvas").get_attribute("data-plan-id") == parent
            assert await page.locator(f'#planCur .plan-row.current[data-plan="{parent}"]').is_visible()
            assert await page.locator("#stlLink").get_attribute("href") is None
            parent_detail = store_module.STORE.plan_json(parent)
            assert parent_detail["passed"] and parent_detail["review"]["status"] == "passed"      # the server reviewed what the agent did not
            assert "규칙 통과 · 검토 완료" in await page.locator("#planCur .plan-row.current .pill").inner_text()
            assert int(await page.locator("#stageSlider").get_attribute("max")) == parent_detail["info"]["n_stages"]
            await page.locator("#stageSlider").fill(str(parent_detail["info"]["n_stages"]))
            await page.locator("#stageSlider").dispatch_event("input")
            before = await page.locator("#viewCanvas").screenshot()

            # Export is gated on a passing, reviewed plan; approving happens through the export popover, which then
            # clicks the hidden download link itself.
            assert await page.locator("#exportBtn").is_enabled()
            assert await page.locator('#rail button[data-go="export"]').is_enabled()      # the rail item is the visible 내보내기
            await page.locator('#rail button[data-go="export"]').click()
            await page.wait_for_selector("#exportPop.in")
            assert await page.locator('#rail button[data-go="export"]').get_attribute("aria-expanded") == "true"
            # the card hangs off the rail item and a press outside it (here the 3D, which takes its own presses) closes it
            await page.locator("#viewCanvas").click(position={"x": 30, "y": 30})
            await page.wait_for_selector("#exportPop", state="hidden")
            assert await page.locator('#rail button[data-go="export"]').get_attribute("aria-expanded") == "false"
            assert await page.evaluate("window.__cualign.state.selected.size") == 0      # that press only closed: no tooth picked
            await page.locator('#rail button[data-go="export"]').click()
            await page.wait_for_selector("#exportPop:not([hidden])")
            # the link's place over the 3D records what it says while the zip builds (펄스 점: 만드는 중 → a check)
            await page.evaluate("""() => { window.__exportTexts = []; const s = document.querySelector('#stlLink span');
                new MutationObserver(() => window.__exportTexts.push(s.textContent)).observe(s, { childList: true, characterData: true, subtree: true }); }""")
            t0 = time.monotonic()
            async with page.expect_download(timeout=180000) as download:      # a real scan's zip takes a while (#119; 000097 with the filled gum)
                await page.locator("#exportGo").click()
            print(f"STL download started after {time.monotonic() - t0:.1f}s")
            await page.wait_for_function("!!window.__cualign.state.plan?.approval")
            await page.wait_for_selector("#exportState:not([hidden]) #stlLink.ready .pulse-dot.ok")
            assert any(t.startswith("STL 만드는 중") for t in await page.evaluate("window.__exportTexts"))
            n_stages = store_module.STORE.plan_json(parent)["info"]["n_stages"]
            assert (await page.locator("#stlLink span").inner_text()).startswith(f"STL 내려받기 ({n_stages}단계 · ")
            assert await page.locator('#planCur .plan-row.current .pill').inner_text() == "승인됨"
            assert parent in await page.locator("#stlLink").get_attribute("href")
            assert (await page.locator("#exportBtn").text_content()) == "STL 내려받기"
            assert (await page.locator('#rail button[data-go="export"] span').inner_text()) == "STL 받기"
            await (await download.value).save_as(OUT / "approved-stages.zip")

            # Revoke the approval (the 검토 fold on the 규칙 tab) and confirm export gates again.
            await page.locator("#tabRules").click()
            await page.locator("#planReview").evaluate("(el) => { el.open = true; }")
            await page.locator("#revokeBtn").click()
            await page.wait_for_function("!window.__cualign.state.plan?.approval")
            assert (await page.locator("#exportBtn").text_content()) == "내보내기"
            assert await page.locator("#stlLink").get_attribute("href") is None
            # Re-approve so the plan is on screen as approved again before the next edit invalidates it.
            await page.locator('#rail button[data-go="export"]').click()
            await page.wait_for_selector("#exportPop:not([hidden])")
            await page.locator("#exportGo").click()
            await page.wait_for_function("!!window.__cualign.state.plan?.approval")

            # Editing the conditions makes the on-screen (approved) plan stale for export until replanned.
            await page.locator("#tabCond").click()      # the flow moved the sidebar to 단계 표 (#15)
            await page.locator("#cLock").fill("13")
            assert await page.locator("#exportBtn").is_disabled()
            assert await page.locator('#rail button[data-go="export"]').is_disabled()
            assert "조건이 바뀜" in await page.locator("#condState").inner_text()
            assert await page.locator("#condApply").is_visible()      # a hand edit re-runs the setup turn (#20); here the stages turn takes the form as it is
            await page.locator("#chatInput").fill("13번은 움직이지 말고 다시 짜줘")
            await page.locator("#sendBtn").click()
            await page.wait_for_function(f"(p) => {plan_on_screen} !== p && !document.querySelector('#sendBtn').disabled"
                f" && document.querySelector('#viewCanvas').dataset.planId === {plan_on_screen}", arg=parent, timeout=120000)
            child = await on_screen()
            assert await page.evaluate("window.__cualign.state.plan.parent_plan_id") == parent
            assert await approval() is None
            assert "조건과 같음" in await page.locator("#condState").inner_text()
            assert await page.locator("#stlLink").get_attribute("href") is None
            assert await page.locator("#viewCanvas").get_attribute("data-plan-id") == child
            await page.locator("#stageSlider").evaluate("(el) => { el.value=el.max; el.dispatchEvent(new Event('input')); }")
            after = await page.locator("#viewCanvas").screenshot()
            assert before != after
            await page.screenshot(path=str(OUT / "revision.png"))

            # The decision bar for this revision lets the dentist undo it; revert, then pick the child plan again
            # from the plan picker's list (#111) so the rest of the script continues from it.
            await page.locator(".decision").last.locator('[data-act="revert"]').click()
            await page.wait_for_function(f"(p) => {plan_on_screen} === p", arg=parent)
            assert await page.locator(f'#planCur .plan-row.current[data-plan="{parent}"]').count() == 1
            # two plans: the picker shows ▾; the list's other row puts it on screen and the 단계 표 head follows
            assert await page.locator("#planMore").is_visible() and await page.locator("#planMenu").is_hidden()
            facts_before = await page.locator("#stageFacts").inner_text()
            await pick_plan(child)
            await page.wait_for_function(f"(p) => {plan_on_screen} === p", arg=child)
            assert await page.locator("#viewCanvas").get_attribute("data-plan-id") == child
            assert await page.locator(f'#planCur .plan-row.current[data-plan="{child}"]').is_visible()
            assert await page.locator(f'#planMenu .plan-row[data-plan="{parent}"]').count() == 1 and await page.locator("#planMenu").is_hidden()
            assert await page.locator("#stageFacts").inner_text() != facts_before, facts_before
            # the sidebar follows the card: its 규칙 tab names the plan on screen
            await page.locator("#tabRules").click()
            assert (await page.locator("#rulesFor").inner_text()).startswith("계획 ")
            assert await page.locator("#ruleCards .rule").count() >= 4
            await page.screenshot(path=str(OUT / "plan-cards.png"))

            # Send a real HTTP stream through PlanEventsASGI and the actual tool group. The fake reviewer fails and
            # the streamed text carries an unrelated plan id that must not steer selection.
            await page.locator("#chatInput").fill("IPR은 앞니 빼고 다시 짜줘")
            await page.locator("#sendBtn").click()
            await page.wait_for_function("window.__cualign.state.plan?.review?.status === 'failed' && !document.querySelector('#sendBtn').disabled")
            selected = await on_screen()
            assert selected != "p999" and selected != child
            assert await page.evaluate("window.__cualign.state.plan.parent_plan_id") == child
            assert "검토 실패" in await page.locator("#planCur .plan-row.current .pill").inner_text()
            assert await page.locator("#cLock").input_value() == "13"
            assert await page.locator("#cExclude").input_value() == "11, 12, 21, 22"      # Universal 7,8,9,10 in FDI (#113)
            assert await page.locator("#exportBtn").is_disabled()
            assert await page.locator("#stlLink").get_attribute("href") is None
            # a comparison turn adds one card per plan it made (an extraction prescription makes one, #56); the selected one is
            # 보는 중. Both lists count: the reload in the flow walk above folded the preview into 지난 계획
            assert await page.locator("#planMenu .plan-row").count() >= 3
            assert await page.locator(f'#planCur .plan-row.current[data-plan="{selected}"]').is_visible()
            assert await page.locator(".msg.error").last.is_visible()
            decision2 = page.locator(".decision").last
            assert await decision2.is_visible()
            assert await decision2.locator('[data-act="revert"]').count() == 1   # keeping needs no button
            assert await decision2.locator('[data-act="keep"]').count() == 0
            await page.screenshot(path=str(OUT / "review-failure.png"))

            # The dentist asks for the failed review again on the same plan: the 규칙 tab's fold is open on a failed review.
            await page.locator("#tabRules").click()
            assert await page.locator("#planReview").evaluate("(el) => el.open")
            assert await page.locator("#reviewBtn").is_visible()
            await page.locator("#reviewBtn").click()
            await page.wait_for_function("window.__cualign.state.plan?.review?.status === 'passed' && !document.querySelector('#sendBtn').disabled")
            assert await on_screen() == selected
            assert "검토 완료" in await page.locator("#planCur .plan-row.current .pill").inner_text()
            assert await page.locator("#reviewBtn").is_hidden()
            if store_module.STORE.plan_json(selected)["passed"]:
                assert await page.locator("#exportBtn").is_enabled()

            # A stale HTTP response must not overwrite a later manual selection.
            pending = asyncio.Event()
            release = asyncio.Event()
            # Obtain the payload before routing, then release the first selection last.
            async with httpx.AsyncClient() as client:
                parent_payload = (await client.get(url + "/api/plans/" + parent)).json()
            async def delayed_parent(route):
                pending.set()
                await release.wait()
                await route.fulfill(json=parent_payload)
            await page.route("**/api/plans/" + parent, delayed_parent)
            # the cards' buttons are disabled while a load runs, so the second choice goes through the test hook
            await pick_plan(parent)
            await pending.wait()
            await page.evaluate("(p) => { window.__cualign.loadPlan(p).catch(() => {}); }", child)
            await page.wait_for_function(f"(p) => {plan_on_screen} === p", arg=child)
            release.set()
            await page.wait_for_timeout(100)
            assert await on_screen() == child
            assert await page.locator("#viewCanvas").get_attribute("data-plan-id") == child

            # Reload keeps the case and the plan on screen via the #case=…&plan=… address (#90).
            assert (await page.evaluate("location.hash")) == f"#case=poseidon-000097&plan={child}&step=stages"
            case_name_before = await page.locator("#caseName").inner_text()
            await page.reload()
            await page.wait_for_function("!document.body.classList.contains('start')")
            await page.wait_for_function(f"(p) => {plan_on_screen} === p", arg=child, timeout=120000)
            assert await page.locator(f'#planCur .plan-row.current[data-plan="{child}"]').count() == 1
            assert await page.locator("#caseName").inner_text() == case_name_before
            assert await page.locator("#stlLink").get_attribute("href") is None   # approval was revoked above

            # The 조건 tab's form: a new 장수 상한 shows what changes in one line, and 다시 계산 (the rule plan, no model —
            # the same path as 에이전트 없이 계산) adds a plan card and puts that plan on screen.
            await page.locator("#tabCond").click()
            assert await page.locator("#condRecalc").is_disabled()      # the form still matches the plan on screen
            cards_before = await page.locator("#planMenu .plan-row").count()
            await page.locator("#cCap").fill("40")
            assert "장수 상한" in await page.locator("#condDiff").inner_text() and "→ 40" in await page.locator("#condDiff").inner_text()
            await page.locator("#condRecalc").click()
            await page.wait_for_function(f"(p) => {plan_on_screen} !== p && !document.body.classList.contains('streaming')", arg=child, timeout=300000)
            recalc = await on_screen()
            assert await page.evaluate("window.__cualign.state.plan.constraints.stage_cap") == 40
            assert await page.locator("#planMenu .plan-row").count() >= cards_before + 1      # one card per strategy the rule plan tried
            assert await page.locator(f'#planCur .plan-row.current[data-plan="{recalc}"]').is_visible()
            assert "조건을 바꿔 규칙으로 다시 계산했습니다 — 계획 " in await page.locator(".msg.system").last.inner_text() \
                or "허용 전략 전부 규칙 위반" in await page.locator(".msg.system").last.inner_text()

            # Going home and opening a different (real) sample card: the case opens as its scan (#20), never with the
            # previous case's plan, the strip back at 초기.
            await page.locator("#homeBtn").click()
            await page.wait_for_function("document.body.classList.contains('start')")
            await page.locator('#sampleCards .case-card[data-id="poseidon-000001"]').click()      # card → detail under it
            await page.locator("#clDetail").wait_for(state="visible", timeout=2000)   # it grows from 0 height (open animation)
            assert await page.locator("#dArch circle").count() == 14 and await page.locator("#dArch line.ipr").count() == 9   # 000001: IPR on the 9 contacts between 15…25
            assert await page.locator('#sampleCards > #clDetail.in-cards').count() == 1      # under the card row
            assert await page.locator("#intro").is_visible()   # the intro panel is back on the left (③)
            await page.locator("#dOpen").click()
            await page.wait_for_function("!document.body.classList.contains('start') && document.body.classList.contains('step-initial') && window.__cualign.state.activeCase === 'poseidon-000001'")
            assert not document_has_plan(await page.evaluate("document.body.className")) and await on_screen() == ""
            assert await page.locator('#flow button[data-step="setup"]').is_disabled()
            assert (await page.evaluate("location.hash")).startswith("#case=poseidon-000001")
            assert await page.locator("#stlLink").get_attribute("href") is None

            # My scan, the two blocking states of 입력 확인 (#104): a gap in the arch is unsupported (no planning),
            # numbers running the other way get the 좌우 번호 뒤집기 button, and after it the scan is ready.
            missing = write_scan(OUT / "scans" / "missing", drop=(4,), gum=True)
            flipped = write_scan(OUT / "scans" / "flipped", gum=True, rename=lambda u: 17 - u)
            await page.evaluate("location.hash = '#patients'")
            await page.wait_for_selector("#patientForm:visible")
            await page.fill("#pAlias", "인수 스캔")
            await page.set_input_files("#pScans", missing)
            await page.locator("#patientForm button[type=submit]").click()
            await page.wait_for_function("!document.querySelector('#checkBar').hidden && document.querySelector('#checkFacts .bad')", timeout=120000)
            facts = await page.locator("#checkFacts").inner_text()
            assert "결손" in facts and "15번" in facts, facts      # Universal 4 = FDI 15 (core writes FDI)
            assert await page.locator("#startPlan").is_hidden()      # a blocked scan cannot start a plan (#112)
            assert await page.locator("#checkBar").evaluate("e => e.classList.contains('fail')")
            assert await page.locator("#checkTitle").inner_text() == "계획할 수 없는 스캔입니다"
            assert await page.locator("#deleteScan").is_visible()
            await page.screenshot(path=str(OUT / "check-unsupported.png"))
            # 스캔 삭제 asks once in place, then the patient's scan list comes back without it
            await page.locator("#deleteScan").click()
            await page.wait_for_selector("#deleteScanPop:not([hidden])")
            await page.locator("#deleteScanGo").click()
            await page.wait_for_function("!document.querySelector('#caseGate').hidden && document.querySelector('#checkBar').hidden", timeout=60000)
            assert await page.locator("#scanList .scan-row").count() == 0
            # an upload the server refuses (#112, 보드 08): lower-arch numbers → 400 with the reason; the card shows the
            # files, the sentence as it came, and 다시 고르기
            lower = write_scan(OUT / "scans" / "lower", rename=lambda u: u + 16)
            await page.set_input_files("#scanInput", lower)
            await page.wait_for_selector("#uploadStatus .upload-fail", timeout=60000)
            why = await page.locator("#uploadStatus .upload-fail .why").inner_text()
            assert "하악" in why and "상악 스캔만" in why, why
            assert "올린 파일" in await page.locator("#uploadStatus .upload-fail .files").inner_text()
            assert await page.locator('#uploadStatus [data-act="repick"]').is_visible()
            await page.screenshot(path=str(OUT / "upload-rejected.png"))
            # a scan whose orientation cannot be decided (#104, 보드 09-c): closed synthetic crowns, no gum → basis none,
            # yet plannable; the title says so and the button asks to confirm the numbers first
            unoriented = write_scan(OUT / "scans" / "unoriented")
            await page.set_input_files("#scanInput", unoriented)
            await page.wait_for_function("!document.querySelector('#checkBar').hidden && !document.querySelector('#checkTitle').hidden", timeout=120000)
            assert await page.locator("#checkTitle").inner_text() == "방향을 정하지 못했습니다"
            assert "방향을 정할 수 없어 입력 방향 그대로 둠" in await page.locator("#checkFacts").inner_text()
            assert await page.locator("#startPlan").is_enabled() and await page.locator("#startPlan").inner_text() == "번호 확인 — 계획 시작"
            await page.screenshot(path=str(OUT / "check-unoriented.png"))
            await page.set_input_files("#scanInput", flipped)   # a second scan for the same patient
            await page.wait_for_function("document.querySelector('#checkFacts').textContent.includes('좌우 반대') && !document.querySelector('#mirrorBtn').hidden", timeout=120000)
            await page.screenshot(path=str(OUT / "check-reversed.png"))
            await page.locator("#mirrorBtn").click()
            await page.wait_for_function("document.querySelector('#mirrorBtn').hidden && !document.querySelector('#startPlan').disabled", timeout=120000)
            facts = await page.locator("#checkFacts").inner_text()
            assert "좌우 반대" not in facts and "문제 없음" in facts, facts
            pid = await page.evaluate("window.__cualign.state.patient.patient_id")
            await page.evaluate("(pid) => fetch('/api/patients/' + pid, { method: 'DELETE' }).then((r) => r.ok)", pid)   # reruns start empty

            # A case that opens with no plan because the calculation failed (#112, 보드 07). The server contract for
            # this state is not there yet, so the responses are shaped here: /activate carries plan_error, the plan
            # list is empty. The card shows the sentence, the conditions, and 이 조건으로 다시 계산 brings a plan.
            async def no_plan_activate(route):
                r = await route.fetch(); body = await r.json(); body["plan_error"] = "규칙 계산 실패 (가짜): 접촉 폭을 맞출 수 없습니다."
                await route.fulfill(json=body)
            async def empty_plans(route):
                await route.fulfill(json={"plans": []})
            await page.route("**/api/cases/poseidon-000131/activate", no_plan_activate)
            await page.route("**/api/plans?case_id=poseidon-000131", empty_plans)
            await page.locator("#homeBtn").click()
            await page.wait_for_function("document.body.classList.contains('start')")
            await page.locator('#sampleCards .case-card[data-id="poseidon-000131"]').click()
            await page.locator("#dOpen").click()
            await page.wait_for_selector("#planFail:not([hidden])", timeout=60000)
            assert "규칙 계산 실패 (가짜)" in await page.locator("#planFailMsg").inner_text()
            assert await page.locator("#planFailCond .tag").count() >= 3
            assert not document_has_plan(await page.evaluate("document.body.className"))
            await page.screenshot(path=str(OUT / "plan-failed.png"))
            await page.unroute("**/api/plans?case_id=poseidon-000131")
            # the retry sends the form as it is, the per-contact prescription in FDI (#57; the double conversion of .report/24
            # is fixed since #146): the plan that comes back is an IPR plan carrying the same contacts in Universal
            assert await page.locator("#cSurf").input_value() == "12-11 0.4, 11-21 0.4, 21-22 0.4"      # 000131's sample prescription
            sent = []
            page.on("request", lambda r: sent.append(r.post_data_json) if r.url.endswith("/api/plan") and r.method == "POST" else None)
            await page.locator("#planFailRetry").click()
            await page.wait_for_function("document.querySelector('#planFail').hidden && document.body.classList.contains('has-plan')", timeout=120000)
            assert await page.locator("#planMenu #planList .plan-row").count() >= 1
            assert sent and sent[-1]["ipr_surfaces"] == [[12, 11, 0.4], [11, 21, 0.4], [21, 22, 0.4]], sent
            plan131 = await on_screen()
            assert await page.evaluate("window.__cualign.state.plan.strategy") in ("ipr", "expansion_ipr")
            assert await page.evaluate("window.__cualign.state.plan.constraints.ipr_surfaces") == [[7, 8, 0.4], [8, 9, 0.4], [9, 10, 0.4]]
            assert await page.locator("#cSurf").input_value() == "12-11 0.4, 11-21 0.4, 21-22 0.4"      # shown back in FDI
            await page.unroute("**/api/cases/poseidon-000131/activate")
            # The 3D shows FDI numbers, never Universal (#113): the input-check screen's tooth-number labels.
            # This sample's prescription extracts Universal 5·12 = FDI 14·24.
            await page.goto(url + "/ui/#check=poseidon-000097")
            await page.wait_for_function("document.querySelectorAll('.num-label').length > 0")
            labels = await page.locator(".num-label").all_inner_texts()
            assert "14" in labels and "24" in labels, labels     # FDI for this sample's extraction (Universal 5·12)
            assert "5" not in labels, labels                     # never the bare Universal number

            # A final answer with no Korean in it: the server swaps it for a notice and sends plan_error kind=no_answer.
            # The notice shows as the server wrote it, with 다시 보내기 like an overload (server fix 2026-09-28).
            await page.goto(url + f"/ui/#case=poseidon-000131&plan={plan131}&step=stages")      # a case opens as its scan (#20); the plan and step in the address bring the stages back
            await page.wait_for_function("document.body.classList.contains('has-plan') && document.body.classList.contains('step-stages') && !document.querySelector('#sendBtn').disabled", timeout=120000)
            # IPR faces (#22): an IPR plan of this case cuts crowns (000131 prescribes IPR on the anterior; the fallback's
            # chosen plan is 확장, so pick one of the case's plans the server cuts). In the stages the cut teeth carry the cut
            # geometry, their planes in the crown's own colour (blue only in 셋업); 초기 shows the scan's crowns again
            ipr_plan = await page.evaluate(FIND_CUT_PLAN)
            assert ipr_plan, "no plan of 000131 with teeth_cut"
            await page.evaluate(f"window.__cualign.loadPlan('{ipr_plan}')")
            await page.wait_for_function(f"window.__cualign.state.plan?.plan_id === '{ipr_plan}' && !window.__cualign.state.loading", timeout=60000)
            cut_set = f"window.__cualign.state.cutSets['plan:{ipr_plan}']"
            cut_ids = await page.evaluate(f"Object.keys({cut_set} ?? {{}})")
            assert cut_ids and await page.evaluate("window.__cualign.cutKeyNow()") == f"plan:{ipr_plan}", cut_ids
            assert await page.evaluate(f"Object.entries({cut_set}).every(([id, c]) => {{ const m = window.__cualign.state.teeth[id]; return m.geometry === c.geo && m.userData.cutMesh.visible && m.userData.cutMesh.material === m.material && c.faces.index.count > 0; }})")
            # the IPR tool cursor is the setup's and a ▶ playback's only (scan-reveal.js): the slider reaching stage 1 of this
            # IPR plan, its first stage ≥ 1 on screen, brings none (it did, once per plan, on any stage change)
            await page.locator("#stageSlider").evaluate("el => { el.value = 1; el.dispatchEvent(new Event('input', { bubbles: true })); }")
            await page.wait_for_timeout(3 * FRAME)
            assert await page.evaluate("window.__cualign.state.stage") == 1 and await page.locator(".ipr-tool").count() == 0
            await page.locator('#flow button[data-step="initial"]').click()
            assert await page.evaluate("Object.values(window.__cualign.state.teeth).every((m) => m.geometry === m.userData.full && !m.userData.cutMesh.visible)")
            await page.locator('#flow button[data-step="stages"]').click()
            assert await page.evaluate(f"Object.entries({cut_set}).every(([id, c]) => window.__cualign.state.teeth[id].geometry === c.geo)")
            await page.evaluate(f"window.__cualign.loadPlan('{plan131}')")     # back to the fallback's plan for the steps below
            await page.wait_for_function(f"window.__cualign.state.plan?.plan_id === '{plan131}' && !window.__cualign.state.loading", timeout=60000)
            # each crown is that plan's cut geometry where it cuts, the scan's where it does not (확장 cuts nothing; since #143 the retry may pick IPR)
            assert await page.evaluate(f"Object.entries(window.__cualign.state.teeth).every(([id, m]) => m.geometry === (window.__cualign.state.cutSets['plan:{plan131}']?.[id]?.geo ?? m.userData.full))")
            # sequencing phases (#144, shaped here until the core ships it): info.phase_boundary splits the stage table with a
            # labelled rule (먼저 이동 / 뒤따라 이동), a tooth in info.delays waits blank before its first aligner, the slider tip names the phase
            async def phased_plan(route):
                r = await route.fetch(); body = await r.json()
                tooth = next(iter(body["stages"][0]))
                body["info"] = {**body["info"], "phase_boundary": 3, "delays": {tooth: 3}}
                await route.fulfill(json=body)
            await page.route("**/api/plans/" + plan131, phased_plan)
            await page.evaluate(f"window.__cualign.loadPlan('{plan131}')")
            await page.wait_for_function("(window.__cualign.state.plan?.info?.phase_boundary ?? 0) === 3 && !window.__cualign.state.loading", timeout=60000)
            await page.locator("#tabStages").click()
            assert await page.locator("#stageGrid .phase").count() == 1
            assert await page.evaluate("document.querySelector('#stageGrid .phase').previousElementSibling.dataset.stage === '2' && document.querySelector('#stageGrid .phase').nextElementSibling.dataset.stage === '3'")
            assert "먼저 이동 1–2" in await page.locator("#stageGrid .phase").inner_text() and "뒤따라 이동 3–" in await page.locator("#stageGrid .phase").inner_text()
            wait_cells = await page.evaluate("[...document.querySelectorAll('#stageGrid .c.wait')].map((c) => c.parentElement.dataset.stage)")
            assert wait_cells == ["1", "2"], wait_cells
            await page.evaluate("document.querySelector('#stageSlider').value = 3; document.querySelector('#stageSlider').dispatchEvent(new Event('input'))")
            assert await page.locator("#stageTip").inner_text() == "단계 3 · 뒤따라 이동 시작"
            await page.unroute("**/api/plans/" + plan131)
            await page.evaluate(f"window.__cualign.loadPlan('{plan131}')")   # the plan as the server has it, for the steps below
            await page.wait_for_function("!(window.__cualign.state.plan?.info?.phase_boundary) && !window.__cualign.state.loading", timeout=60000)
            assert await page.locator("#stageGrid .phase").count() == 0 and await page.locator("#stageTip").inner_text() == "치료 전"
            async def no_answer_stream(route):
                rid = route.request.post_data_json["cualign"]["request_id"]
                nl = chr(10)
                body = ('event: plan_error' + nl + 'data: {"request_id":"' + rid + '","case_id":"poseidon-000131","kind":"no_answer",'
                        '"message":"모델이 답을 만들지 못했습니다 (가짜 안내문)."}' + nl + nl
                        + 'data: {"value":"모델이 답을 만들지 못했습니다 (가짜 안내문)."}' + nl + nl)
                await route.fulfill(status=200, content_type="text/event-stream", body=body)
            await page.route("**/chat/stream", no_answer_stream)
            await page.locator("#chatInput").fill("발치 없이 다시 짜줘")
            await page.locator("#sendBtn").click()
            await page.wait_for_function("!document.querySelector('#retryBar').hidden && !document.querySelector('#sendBtn').disabled", timeout=60000)
            assert "가짜 안내문" in await page.locator(".msg.error").last.inner_text()
            assert await page.locator("#resendBtn").is_visible()
            # a sample case offers 건너뛰기 instead of 에이전트 없이 계산 on a failed turn (#15). Without a recording (404, faked
            # here: every sample has one since #134) the plan on screen is adopted
            assert await page.locator("#skipBtn").is_visible() and await page.locator("#retryFallback").is_hidden()
            await page.route("**/api/cases/poseidon-000131/replay", lambda route: route.fulfill(status=404, content_type="application/json", body='{"detail":"녹화된 답이 없습니다"}'))
            await page.route("**/api/cases/poseidon-000131/replay/*", lambda route: route.fulfill(status=404, content_type="application/json", body='{"detail":"녹화된 답이 없습니다"}'))   # its head (#212)
            before_skip = await on_screen()
            await page.locator("#skipBtn").click()
            await page.wait_for_selector(".msg.system:has-text('건너뛰었습니다')")      # the retry bar hides before the 404 comes back
            assert await page.locator("body").evaluate("b => b.classList.contains('step-stages')") and await page.evaluate("document.querySelector('#retryBar').hidden")
            assert "건너뜀" in await page.locator("#planCur .plan-row.current .pill").inner_text()
            assert await page.locator(".msg.system", has_text="건너뛰었습니다").count() == 1
            await page.unroute("**/api/cases/poseidon-000131/replay")
            await page.unroute("**/api/cases/poseidon-000131/replay/*")
            # the five recorded steps (#20 contract; the route is shaped here until the server knows these names): each
            # failed turn offers 건너뛰기, the recording lands like the agent's step_done / plan_selected
            seen = []
            async def replay_route(route):
                req = route.request.post_data_json; seen.append(req["step"])
                rec = {"recorded": True, "recorded_at": "2026-09-28T10:00:00", "answer_md": f"녹화된 {req['step']} 답입니다 (가짜).", "plan_selected": None, "plans": []}
                if req["step"] == "setup": rec["constraints"] = {"extraction": [], "lock": [2], "ipr_exclude": [], "ipr_limit_mm": 0.2, "stage_cap": None, "order": "simultaneous",
                                                                  "ipr_surfaces": [[7, 8, 0.4], [8, 9, 0.4], [9, 10, 0.4]]}   # 000131's prescription (#57): 12-11 · 11-21 · 21-22
                elif req["step"] == "target": rec.update(target_id="t-fake", summary={"crowding_mm": 4.2, "space_mm": 4.4, "strategy": "expansion_ipr", "extraction": [], "ipr": []})
                elif req["step"] in ("stages", "cap"): rec["plan_selected"] = {"plan_id": before_skip, "parent_plan_id": None, "review": {"status": "skipped"}}
                await route.fulfill(status=200, content_type="application/json", body=json.dumps(rec))
            await page.route("**/api/cases/poseidon-000131/replay", replay_route)
            # the recording's head (#212: drawn before the POST's recomputation comes back), for the same fake recording
            await page.route("**/api/cases/poseidon-000131/replay/*", lambda route: route.fulfill(status=200, content_type="application/json",
                body=json.dumps({"recorded": True, "step": route.request.url.rsplit("/", 1)[1], "recorded_at": "2026-09-28T10:00:00"})))
            async def skip_turn(n):
                await page.wait_for_function("!document.querySelector('#retryBar').hidden && !document.querySelector('#sendBtn').disabled", timeout=60000)
                await page.locator("#skipBtn").click()
                await page.wait_for_function(f"document.querySelectorAll('.msg.assistant.recorded').length === {n} && !window.__cualign.state.streaming", timeout=60000)
            await page.locator("#tabCond").click(); await page.locator("#condApply").click()      # setup again, by hand-edited conditions
            await skip_turn(1)
            # the recorded setup lands after the scan numbers (scan-reveal ①), with its reasoning
            await page.wait_for_function("window.__cualign.state.setup?.lock?.[0] === 2", timeout=10000)
            assert await page.evaluate("window.__cualign.state.setup.lock") == [2] and await page.locator("#cLock").input_value() == "17"      # Universal 2 = FDI 17
            assert await page.locator("body").evaluate("b => b.classList.contains('step-setup')") and await page.locator('#flow button[data-step="stages"]').is_disabled()
            # the per-contact prescription (#57): the 조건 tab shows it in FDI, the 3D marks exactly those three contacts,
            # and the patch carries it back in FDI
            assert await page.locator("#cSurf").input_value() == "12-11 0.4, 11-21 0.4, 21-22 0.4"
            await page.wait_for_function("document.querySelectorAll('.ipr-mark').length === 3", timeout=5000)   # a CSS2D mark enters the page on the next drawn frame
            # IPR marks by step (ipr-view-check): 셋업 has the dot and its mm per prescribed contact and the cut planes blue;
            # 목표 has neither (the crowns stay cut, the planes in the crown's colour)
            blue = "Object.values(window.__cualign.state.teeth).filter((m) => m.userData.cutMesh.visible && m.userData.cutMesh.material !== m.material).length"
            await page.wait_for_function(f"{blue} > 0", timeout=30000)
            assert await page.locator(".ipr-mark .ipr-mm").all_inner_texts() == ["0.40"] * 3
            assert await page.evaluate("window.__cualign.readConstraints().ipr_surfaces") == [[12, 11, 0.4], [11, 21, 0.4], [21, 22, 0.4]]
            # the recorded setup tells the same lines a live setup does, in the same order, from the recording's values
            # (the lines follow the landing, which may still be fetching after the recorded bubble shows)
            recorded_reasons = "n => [...document.querySelectorAll('.trace.recorded')].at(-1)?.querySelectorAll('.reason').length >= n"
            await page.wait_for_function(recorded_reasons, arg=2, timeout=10000)
            reasons = await page.locator(".trace.recorded").last.locator(".reason").all_inner_texts()
            assert reasons[0] == "처방을 읽고 계획 조건으로 옮깁니다." and reasons[1].startswith("처방을 읽었습니다 — "), reasons
            await next_chip("목표 배열 만들기").click()
            await skip_turn(2)
            await page.wait_for_function(recorded_reasons, arg=2, timeout=10000)
            assert await page.locator(".trace.recorded").last.locator(".reason").all_inner_texts() == [
                "확장 + IPR 전략으로 목표 배열을 만듭니다.", "확보 4.4 mm 로 총생 4.2 mm 를 해결했습니다."]
            assert await page.locator("body").evaluate("b => b.classList.contains('step-target')") and await page.evaluate("window.__cualign.state.targetId") == "t-fake"
            assert await next_chip("비발치안과 비교").count() == 1      # 000131 is a non-extraction case
            assert await page.locator(".ipr-mark, .ipr-mm").count() == 0 and await page.evaluate(blue) == 0
            await next_chip("단계 만들기").click()
            await skip_turn(3)
            await page.wait_for_function(f"document.body.classList.contains('step-stages') && {plan_on_screen} === '{before_skip}' && !window.__cualign.state.loading", timeout=60000)   # the plan (and its cut, #22) load after the recorded bubble
            await page.locator("#chatInput").fill("12개월 안에 끝나게 다시 짜줘."); await page.locator("#sendBtn").click()
            await skip_turn(4)
            await page.locator("#chatInput").fill("확장안이랑 IPR안 둘 다 만들어서 비교해줘."); await page.locator("#sendBtn").click()
            await skip_turn(5)
            assert await on_screen() == before_skip and await page.locator("#planNotice").inner_text() == ""
            assert seen == ["setup", "target", "stages", "cap", "compare"], seen
            assert "녹화된 답 · 2026-09-28" in await page.locator(".msg.assistant.recorded .recorded-tag").last.inner_text()
            await page.unroute("**/api/cases/poseidon-000131/replay")
            await page.unroute("**/api/cases/poseidon-000131/replay/*")
            await page.unroute("**/chat/stream")

            # The server keeps how far the step flow went and sends it with /activate as `flow` (#146): constraints → the strip
            # opens to 셋업 with the conditions, target_id → to 목표, plan_id → to 단계 (shaped here with the tests' t-fake target)
            async def activate_with_state(route):
                r = await route.fetch(); body = await r.json()
                body["flow"] = {"step": "target", "constraints": {"extraction": [5, 12], "lock": [], "ipr_exclude": [], "ipr_limit_mm": 0.2, "stage_cap": None, "order": "simultaneous"}, "target_id": "t-fake", "plan_id": None}
                await route.fulfill(json=body)
            await page.route("**/api/cases/poseidon-000097/activate", activate_with_state)
            await page.goto(url + "/ui/#case=poseidon-000097&step=target")
            await page.wait_for_function("document.body.classList.contains('step-target') && window.__cualign.state.targetId === 't-fake' && window.__cualign.state.activeCase === 'poseidon-000097'", timeout=120000)
            assert await page.locator("#cIpr").input_value() == "0.2" and await page.locator('#flow button[data-step="stages"]').is_disabled()
            assert await page.locator(".next:not(.done) button", has_text="단계 만들기").count() == 1      # the chips continue from the restored step
            assert "목표 배열까지" in await page.locator(".msg.assistant").last.inner_text()
            assert await page.evaluate("window.__cualign.cutKeyNow() === 'target:t-fake'")   # #22: the target's cut comes after it, under its key
            await page.wait_for_function("'target:t-fake' in window.__cualign.state.cutSets", timeout=60000)
            await page.unroute("**/api/cases/poseidon-000097/activate")

            # 단계가 자라난다 (stage-grow.js): the stage tool's start event lays the target as a ghost and the chip says
            # 단계 나누는 중; a turn that ends with no plan puts it all back
            async def stage_tool_start(route):
                ev = {"id": "sg1", "name": "cualign__plan_stages", "payload": '**Function Input:**\n```json\n{"strategy": "extraction"}\n```'}
                await route.fulfill(status=200, headers={"Content-Type": "text/event-stream"}, body="intermediate_data: " + json.dumps(ev) + "\n\n")
            await page.route("**/chat/stream", stage_tool_start)
            await page.locator("#chatInput").fill("단계를 만들어 줘"); await page.locator("#sendBtn").click()
            await page.wait_for_function("window.__stageGrow.running", timeout=10000)
            await page.wait_for_timeout(500)
            assert await page.evaluate("window.__stageGrow.ghostShown") and "단계 나누는 중" in await page.locator("#workNote").inner_text()
            await page.wait_for_function("!window.__stageGrow.running", timeout=10000)   # the turn ended with no plan: put back
            await page.unroute("**/chat/stream")
            # the round lasts while the agent works: select_plan follows the stage tool at once and does not land it
            # (it did, 2 s in, leaving the scan), and it lands at the plan's last stage, the slider at its end
            await page.locator("#chatInput").fill("단계 도구가 오래 도는 단계 만들기"); await page.locator("#sendBtn").click()
            await page.wait_for_function("window.__stageGrow.running", timeout=10000)
            await page.wait_for_timeout(2500)
            # its cursor is the agent's reticle, not the IPR saw disc (that one is the cut's only)
            assert await page.evaluate("[window.__stageGrow.ghostShown, window.__stageGrow.cursorKind]") == [True, "reticle"] and await page.locator(".ipr-tool").count() == 0
            await page.wait_for_function("!window.__stageGrow.running && window.__cualign.state.plan && !document.querySelector('#sendBtn').disabled", timeout=60000)
            last = await page.evaluate("String(window.__cualign.state.plan.stages.length)")
            assert await page.locator("#stageSlider").input_value() == last and "처음부터 재생" in await page.locator("#stageTip").inner_text()
            # the replay's landing (▶ 건너뛰기 or its end) is the last stage too — from 치료 전, in a new turn (sendChat clears
            # stageTouched, so the landing is not held where the dentist left the slider)
            await page.locator("#firstBtn").click()
            await page.evaluate("window.__cualign.state.stageTouched = false")
            assert await page.evaluate("window.__stageGrow.play()") is True
            assert not await page.evaluate("window.__stageGrow.ghostShown") and await page.locator("#stageSlider").input_value() == last

            # No WebGL (Chromium --disable-3d-apis): the 3D area shows the card and the rest works — the case opens,
            # the steps reach a plan, ▶ plays the stages, and the export popover approves it
            no3d = await pw.chromium.launch(executable_path=executable, headless=True, args=["--disable-3d-apis"])
            try:
                page = await no3d.new_page(viewport={"width":1500,"height":1000})
                page.on("pageerror", lambda error: (errors.append("no-3d: " + str(error)), print("PAGEERROR (no-3d):", error, file=sys.stderr)))
                await page.goto(url + "/ui/?case=poseidon-000097")
                await page.wait_for_function("!document.body.classList.contains('start') && window.__cualign.state.activeCase === 'poseidon-000097' && !window.__cualign.state.loading", timeout=120000)
                assert await page.locator("#no3d").is_visible() and "3D 를 그릴 수 없습니다" in await page.locator("#no3d").inner_text()
                assert await page.locator("body").evaluate("b => b.classList.contains('no-3d')")
                # the plan approved above (passed, reviewed); the case's saved flow may point at a later, unexportable one
                await page.goto(url + f"/ui/#case=poseidon-000097&plan={parent}&step=stages")
                await page.wait_for_function(f"document.body.classList.contains('step-stages') && document.body.classList.contains('has-plan') && {plan_on_screen} === '{parent}' && !window.__cualign.state.loading", timeout=120000)
                assert await page.locator("#no3d").is_visible()
                await page.locator("#playBtn").click()
                await page.wait_for_function("window.__cualign.state.stage >= 2", timeout=10000)
                await page.locator("#playBtn").click()
                assert await page.locator("#playBtn").get_attribute("aria-pressed") == "false"
                if await page.evaluate("!!window.__cualign.state.plan?.approval"):      # approved above: revoke, so this run approves it itself
                    await page.locator("#tabRules").click()
                    await page.locator("#planReview").evaluate("(el) => { el.open = true; }")
                    await page.locator("#revokeBtn").click()
                    await page.wait_for_function("!window.__cualign.state.plan?.approval")
                await page.wait_for_function("!document.querySelector('#rail button[data-go=\"export\"]').disabled", timeout=60000)
                await page.locator('#rail button[data-go="export"]').click()
                await page.wait_for_selector("#exportPop:not([hidden])")
                async with page.expect_download(timeout=180000):
                    await page.locator("#exportGo").click()
                await page.wait_for_function("!!window.__cualign.state.plan?.approval")
                await page.screenshot(path=str(OUT / "no-3d.png"))
            finally:
                await no3d.close()

            # 직접 이동 follow-ups (000001, its 9-contact IPR prescription landed as the setup): (a) the edit leaves 겹쳐 보기
            # alone, (c) on 셋업 a right-click on a crown gives 발치 / IPR…, IPR 0.2 on FDI 16's mesial face swaps in the cut
            # crown, (b) leaving the edit with a crown moved keeps it (stored as the 목표), and on 목표 a right-click gives no menu
            cid = "poseidon-000001"
            api.STORE.set_flow(cid, "setup", constraints=api.STORE.constraints_for(cid), target_id=None, plan_id=None)
            page = await browser.new_page(viewport={"width": 1500, "height": 1000})
            page.on("pageerror", lambda error: (errors.append("manual: " + str(error)), print("PAGEERROR (manual):", error, file=sys.stderr)))
            await page.goto(url + f"/ui/?nofx=1#case={cid}&step=setup")
            await page.wait_for_function(f"document.body.classList.contains('step-setup') && window.__cualign.state.activeCase === '{cid}' && !!window.__cualign.state.setup && !window.__cualign.state.loading", timeout=120000)
            overlay = await page.locator("#overlayBtn").get_attribute("aria-pressed")
            await page.locator("#moveBtn").click()
            await page.wait_for_function("document.body.classList.contains('manual-on') && document.querySelector('#moveTable .mv-row[data-id]')", timeout=60000)
            assert await page.locator("#overlayBtn").get_attribute("aria-pressed") == overlay            # (a)
            assert await page.locator("body").evaluate("b => b.classList.contains('step-setup')")        # the step stays 셋업
            crown_at = """(id) => { const s = window.__cualign.state, m = s.teeth[id], c = window.__cualign.camera;
              const p = s.center[id].clone().add(m.position); p.z = m.geometry.boundingBox.max.z; p.project(c);
              const r = document.querySelector('#viewCanvas').getBoundingClientRect();
              return [r.left + (p.x + 1) / 2 * r.width, r.top + (1 - p.y) / 2 * r.height]; }"""
            x, y = await page.evaluate(crown_at, "3")                                                     # FDI 16
            await page.mouse.click(x, y, button="right")
            await page.wait_for_selector(".rx-menu.menu")
            assert await page.locator(".rx-menu.menu .rx-title").inner_text() == "16번"
            assert await page.locator('.rx-menu button[data-act="extract"]').is_disabled()               # a molar: premolars only
            await page.locator('.rx-menu button[data-act="ipr"]').click()
            await page.wait_for_selector(".rx-menu.form")
            assert await page.locator('.rx-menu input[value="mesial"]').is_checked()
            await page.locator("#rxMm").fill("0.2")
            await page.locator('.rx-menu [data-act="ok"]').click()
            await page.wait_for_function("(window.__cualign.state.setup.ipr_amounts ?? []).some(([t, n, mm]) => t === 3 && n === 4 && mm === 0.2)", timeout=30000)
            await page.wait_for_function("(() => { const s = window.__cualign.state, c = s.cutSets.setup?.['3']; return !!c && s.teeth['3'].geometry === c.geo; })()", timeout=60000)   # (c) the cut crown on screen
            assert await page.locator("#cSurf").input_value() == "16-15 0.2, 15-14 0.4, 14-13 0.4, 13-12 0.4, 12-11 0.4, 11-21 0.4, 21-22 0.4, 22-23 0.4, 23-24 0.4, 24-25 0.4"
            # (b) a crown moved 1 mm mesially, then Esc: stored as the 목표 and still where it was put
            row = page.locator('#moveTable input[data-id="7"][data-key="mesial"]')
            await row.fill("1"); await row.dispatch_event("change")
            await page.locator("#viewCanvas").click(position={"x": 5, "y": 5})     # off the input (a click on nothing selects nothing)
            await page.keyboard.press("Escape")
            await page.wait_for_function("document.body.classList.contains('step-target') && !document.body.classList.contains('manual-on') && window.__cualign.state.target?.info?.source === 'manual'", timeout=60000)
            moved = await page.evaluate("(() => { const s = window.__cualign.state; return [Math.hypot(...s.target.stages[0]['7']), s.teeth['7'].position.length()]; })()")
            assert abs(moved[0] - 1) < 0.05 and moved[1] > 0.5, moved
            # on 목표 the edit has no right-click menu
            await page.locator("#moveBtn").click()
            await page.wait_for_function("document.body.classList.contains('manual-on')", timeout=60000)
            x, y = await page.evaluate(crown_at, "3")
            await page.mouse.click(x, y, button="right")
            await page.wait_for_timeout(300)
            assert await page.locator(".rx-menu").count() == 0
            await page.keyboard.press("Escape")                                         # nothing moved: the edit just ends
            await page.wait_for_function("!document.body.classList.contains('manual-on')", timeout=10000)
            assert await page.locator(".next:not(.done) button", has_text="처음부터 수동 배치").count() == 0
            # (d) a drag that starts on a crown turns the view, the crown picked or not: nothing is stored, the 목표 stays
            async def drag(x, y, dx, dy):
                await page.mouse.move(x, y); await page.mouse.down()
                for k in range(1, 11):
                    await page.mouse.move(x + dx * k / 10, y + dy * k / 10)
                await page.mouse.up()
            pose = "(id) => window.__cualign.state.target.stages[0][id]"
            saves = []
            page.on("request", lambda r: r.method == "POST" and r.url.endswith("/manual") and saves.append(r.url))
            tid, p0 = await page.evaluate("window.__cualign.state.targetId"), await page.evaluate(pose, "3")
            await page.locator("#moveBtn").click()
            await page.wait_for_function("document.body.classList.contains('manual-on')", timeout=60000)
            x, y = await page.evaluate(crown_at, "3")
            await drag(x, y, 120, 30)
            x, y = await page.evaluate(crown_at, "3")
            await page.mouse.click(x, y)                                                # picked: the handles show
            await page.wait_for_selector('#moveTable .mv-row.sel[data-id="3"]')
            await drag(x, y, 120, 30)
            assert await page.evaluate(pose, "3") == p0
            await page.locator("#moveBtn").click()                                     # 끄기: nothing moved, nothing stored
            await page.wait_for_function("!document.body.classList.contains('manual-on')", timeout=10000)
            assert not saves and await page.evaluate("window.__cualign.state.targetId") == tid and await page.evaluate(pose, "3") == p0
            # (e) a picked crown's handle moves it; Ctrl+Z puts it back
            await page.locator("#moveBtn").click()
            await page.wait_for_function("document.body.classList.contains('manual-on')", timeout=60000)
            x, y = await page.evaluate(crown_at, "3")
            await page.mouse.click(x, y)
            await page.wait_for_selector('#moveTable .mv-row.sel[data-id="3"]')
            hx, hy = await page.evaluate("""(id) => { const t = window.__cualign.state.target, c = window.__cualign.camera, V = c.position.constructor;
              const p = t.pivots[id], d = t.stages[0][id], f = t.frames[id].mesial;
              const v = new V(p[0] + d[0] + f[0] * 5.75, p[1] + d[1] + f[1] * 5.75, p[2] + d[2] + f[2] * 5.75).project(c);
              const r = document.querySelector('#viewCanvas').getBoundingClientRect();
              return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height]; }""", "3")   # the mesial arrow's head
            await drag(hx, hy, 40, 0)
            await page.wait_for_function("([id, p]) => window.__cualign.state.target.stages[0][id].some((x, k) => Math.abs(x - p[k]) >= 0.05)", arg=["3", p0], timeout=5000)
            await page.keyboard.press("Control+z")                                      # drawn on the next frame (redraw)
            await page.wait_for_function("([id, p]) => JSON.stringify(window.__cualign.state.target.stages[0][id]) === JSON.stringify(p)", arg=["3", p0], timeout=5000)
            await page.keyboard.press("Escape")                                         # back where it was: nothing to store
            await page.wait_for_function("!document.body.classList.contains('manual-on')", timeout=10000)
            assert not saves and await page.evaluate("window.__cualign.state.targetId") == tid
            await page.close()

            assert not errors, errors
            print("PASS: agent-driven steps (setup → target → stages, look back, five replay steps), export/approval, revision, plan cards (보기 switches the plan), reviewer failure, manual re-review, "
                  "stale response, reload keeps case and plan, case switch opens its own preview plan, "
                  "my scan: unsupported gap → 스캔 삭제, rejected upload card, unoriented scan, reversed numbering → mirror, "
                  "no-plan failure card → 이 조건으로 다시 계산, no_answer notice → 다시 보내기, "
                  "no WebGL (--disable-3d-apis): the card over the 3D, steps → ▶ → approval, "
                  "직접 이동: 겹쳐 보기 untouched, 셋업 right-click IPR → cut crown, leaving keeps the move, no menu on 목표, "
                  "a drag on a crown turns the view (nothing stored), a handle move undone by Ctrl+Z")
            await browser.close()
            browser = None
    finally:
        if browser is not None:
            await browser.close()
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


if __name__ == "__main__":
    take_lock()
    try:
        asyncio.run(main())
    finally:
        LOCK.unlink(missing_ok=True)
