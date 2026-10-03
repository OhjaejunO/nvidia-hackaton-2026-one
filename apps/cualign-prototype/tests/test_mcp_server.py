"""/mcp for NemoClaw (docs/nemoclaw.md): an MCP client over Streamable HTTP, a fake agent behind /chat/stream."""
import asyncio
import hashlib
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from cualign.agent.context import CURRENT_RUN
from cualign.core import store as store_module
from cualign.core.service import PlanningService
from cualign.server import api, mcp_server, plan_events
from cualign.server.plan_events import ChatContext, open_run

TOKEN = "test-token-not-a-secret"
TOOLS = {"cualign_list_cases", "cualign_plan", "cualign_get_plan", "cualign_approve_plan", "cualign_export_stl"}


@pytest.fixture
def served(tmp_path, monkeypatch):
    s = store_module.Store()
    monkeypatch.setattr(store_module, "OUT_DIR", tmp_path)
    for module in (plan_events, api, mcp_server):
        monkeypatch.setattr(module, "STORE", s)
    monkeypatch.setenv(mcp_server.TOKEN_ENV, TOKEN)
    monkeypatch.setenv(mcp_server.PUBLIC_URL_ENV, "https://cualign.test")
    seen = {}
    app = FastAPI()

    @app.post("/chat/stream")
    async def chat(request: Request):   # stands in for NAT's route and the ReAct agent
        body = await request.json()
        seen["messages"], seen["run"] = body["messages"], CURRENT_RUN.get()

        async def generate():
            run = CURRENT_RUN.get()
            svc = PlanningService(s)
            pid = svc.stages(svc.target(run.case_id, "expansion_ipr", run.constraints), run.base_plan_id)
            s.set_review(pid, {"status": "passed", "attempts": 1, "message": "검토 메모", "error": None})
            run.plan_ids.add(pid)
            run.selected_plan_id = pid
            yield 'intermediate_data: {"id":"1","name":"Function Start: propose_target","payload":""}\n\n'
            yield 'data: {"choices":[{"delta":{"content":"계획을 만들었습니다."}}]}\n\n'
        return StreamingResponse(generate(), media_type="text/event-stream")

    app.add_middleware(plan_events.PlanEventsASGI)
    api.add_api_routes(app)
    mcp_server.add_mcp_route(app)
    return app, s, seen


async def call(app, fn, token=TOKEN):
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost",
                                 headers=headers, timeout=60) as http:
        async with streamable_http_client("http://localhost/mcp", http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)


def result(res) -> dict:
    assert not res.isError, res.content
    return json.loads(res.content[0].text)


def test_lists_the_five_tools(served):
    app, _, _ = served
    names = asyncio.run(call(app, lambda s: s.list_tools()))
    assert {t.name for t in names.tools} == TOOLS


def test_plan_takes_the_ui_path(served):
    app, s, seen = served
    constraints = {"allow_extraction": False, "ipr_exclude": [7]}
    out = asyncio.run(call(app, lambda ses: ses.call_tool("cualign_plan", {
        "case_id": "moderate", "request": "계획 짜줘.", "constraints": constraints})))
    out = result(out)
    assert out["status"] == "planned" and out["answer"] == "계획을 만들었습니다."
    assert out["tools"] == ["propose_target"]
    plan = out["plan"]
    assert plan["reviewer_memo"] == "검토 메모" and plan["approval"] is None
    assert plan["ui_url"] == f"https://cualign.test/ui/?plan={plan['plan_id']}"
    # the same system message the UI path gets from open_run, then the dentist's words
    run = seen["run"]
    s.flow.clear()   # the context the turn started from: the flow (target_id) this very turn recorded is not in it
    _, expected = open_run(ChatContext(request_id=run.request_id, case_id="moderate", constraints=constraints), store=s)
    assert seen["messages"] == [expected, {"role": "user", "content": "계획 짜줘."}]
    assert run.constraints.ipr_exclude == (7,) and run.closed


def test_bad_case_is_an_error_not_a_crash(served):
    app, _, _ = served
    out = result(asyncio.run(call(app, lambda s: s.call_tool("cualign_plan", {"case_id": "nope", "request": "x"}))))
    assert out["status"] == "error"


def test_approve_never_approves_and_export_waits_for_the_dentist(served):
    app, s, _ = served

    async def flow(ses):
        plan = result(await ses.call_tool("cualign_plan", {"case_id": "moderate", "request": "계획 짜줘."}))["plan"]
        pid = plan["plan_id"]
        approve = result(await ses.call_tool("cualign_approve_plan", {"plan_id": pid}))
        export_before = result(await ses.call_tool("cualign_export_stl", {"plan_id": pid}))
        if plan["passed"]:
            s.approve(pid, s.lock_fdi(pid))   # the dentist, in the UI
        export_after = result(await ses.call_tool("cualign_export_stl", {"plan_id": pid}))
        return plan, approve, export_before, export_after

    plan, approve, before, after = asyncio.run(call(app, flow))
    assert approve["status"] == "refused" and approve["approval"] is None
    assert before["status"] == "refused"
    assert plan["passed"], "the moderate case should pass with expansion_ipr"
    assert after == {"status": "approved", "download_url": f"https://cualign.test/api/plans/{plan['plan_id']}/stl.zip"}


def test_token_is_required(served, monkeypatch):
    app, _, _ = served
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}

    async def status(**headers):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as c:
            res = await c.post("/mcp", json=init, headers={"Accept": "application/json, text/event-stream", **headers})
            return res.status_code
    assert asyncio.run(status()) == 401
    assert asyncio.run(status(Authorization="Bearer wrong")) == 401
    monkeypatch.delenv(mcp_server.TOKEN_ENV)
    assert asyncio.run(status(Authorization=f"Bearer {TOKEN}")) == 503
    # in an OpenShell sandbox the server holds only the hash (a provider secret would be a placeholder there)
    monkeypatch.setenv(mcp_server.TOKEN_HASH_ENV, hashlib.sha256(TOKEN.encode()).hexdigest().upper())
    assert asyncio.run(status(Authorization="Bearer wrong")) == 401
    assert asyncio.run(status(Authorization=f"Bearer {TOKEN}")) == 200


def test_host_must_be_allowed(served, monkeypatch):
    """Behind the proxy the Host is the address NemoClaw dials (colima's host-gateway), not localhost."""
    app, _, _ = served
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}

    async def status():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://192.168.5.2:8443") as c:
            res = await c.post("/mcp", json=init, headers={"Accept": "application/json, text/event-stream",
                                                          "Authorization": f"Bearer {TOKEN}"})
            return res.status_code
    assert asyncio.run(status()) == 421
    monkeypatch.setenv(mcp_server.ALLOWED_HOSTS_ENV, "192.168.5.2, 192.168.5.2:8443")
    assert asyncio.run(status()) == 200


def test_sse_parser_matches_the_ui_parser():
    text = ('intermediate_data: {"name":"도구"}\r\n\r\nevent: plan_selected\ndata: {"a":1,\ndata: "b":2}\n\n'
            'data: [DONE]\n\n{"message":"boom"}')
    assert mcp_server.sse_events(text) == [("intermediate_data", {"name": "도구"}), ("plan_selected", {"a": 1, "b": 2}),
                                           ("error", {"message": "boom"})]


def test_openclaw_skill_and_proxy_name_the_served_tools():
    """The OpenClaw skill and the deny list in docs/nemoclaw.md name the tools /mcp serves, so a rename breaks here."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    skill = (root / "workspace" / "skills" / "cualign-planner" / "SKILL.md").read_text(encoding="utf-8")
    assert set(re.findall(r"`(cualign_[a-z_]+)`", skill)) == TOOLS
    doc = (root / "docs" / "nemoclaw.md").read_text(encoding="utf-8")
    assert set(re.findall(r"--deny-tool (cualign_[a-z_]+)", doc)) == {"cualign_approve_plan", "cualign_export_stl"}
    caddy = (root / "nemoclaw" / "Caddyfile").read_text(encoding="utf-8")
    assert "path /mcp" in caddy and "{$CUALIGN_MCP_CERT}" in caddy and "{$CUALIGN_MCP_TOKEN}" in caddy
