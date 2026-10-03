"""#113 / leak: two real conversations through `nat serve` + /chat/stream, as the screen sends them.

A. Sample poseidon-000131, the card's «에이전트에게 계획 맡기기» sentence (sample.request) with the sample's constraints.
C. Sample poseidon-000097, its FDI-only card sentence (제1소구치 14·24 발치) with extraction cleared in the context: the model
   passes 14·24 as said (set_constraints takes FDI) and the stored plan must hold [5, 12] (Universal).
   Observed on main #115: no tool call, English reasoning streamed as the answer, no plan_selected.
B. Case moderate, «14번과 24번 발치로 계획 짜줘.» — set_constraints gets [14, 24] and the plan stores extraction [5, 12] (Universal).
Never prints the API key. Writes out/nim-live/fdi-leak.json (events, tool args, answers) and prints a verdict.

Costs real NVIDIA usage: two planning conversations (several model calls each) plus rails.
Run: uv run --frozen python tests/nim_fdi_live_check.py
"""
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "nim-live"
sys.path.insert(0, str(ROOT / "tests"))
from nim_live_check import converse, free_port, pick, tool_steps, wait_ready  # noqa: E402

ENGLISH_REASONING = re.compile(r"\b(We need|Let's|The user|universal numbering|I should|First,)\b", re.I)


def tool_args(events, name):
    """Arguments the model chose for the tool `name`, from the progress events (worker.py keeps name + args)."""
    out = []
    for n, o in events:
        if n == "intermediate_data" and (o.get("name") or "").endswith(name):
            out.append(o.get("payload") if "payload" in o else {k: v for k, v in o.items() if k != "name"})
    return out


async def main():
    load_dotenv(ROOT / ".env")
    if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
        raise SystemExit("NVIDIA_API_KEY is not set (put it in apps/cualign-prototype/.env)")
    OUT.mkdir(parents=True, exist_ok=True)
    from cualign.core.samples import SAMPLES
    sample = SAMPLES["poseidon-000131"]

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PYTHONUTF8": "1", "CUALIGN_OUT": str(ROOT / "out")}
    logfile = open(OUT / "fdi-leak-server.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "nat.cli.main", "serve", "--config_file", str(ROOT / "configs" / "workflow.yml"),
                             "--host", "127.0.0.1", "--port", str(port)], cwd=str(ROOT), env=env, stdout=logfile,
                            stderr=subprocess.STDOUT)
    log, verdict = [], {}
    try:
        await wait_ready(url, proc)
        async with httpx.AsyncClient() as client:
            # --- A: the sample card sentence ------------------------------------------------------------
            await client.post(url + f"/api/cases/{sample.case_id}/activate", timeout=60)
            ctx = {"request_id": "leak-A", "case_id": sample.case_id, "base_plan_id": None, "constraints": dict(sample.constraints)}
            events, answer, elapsed = await converse(client, url, sample.request, ctx, log)
            a = verdict["A_sample_000131"] = {"request": sample.request, "elapsed_s": round(elapsed, 1),
                                               "tools": tool_steps(events), "plan_selected": bool(pick(events, "plan_selected")),
                                               "plan_error": pick(events, "plan_error"), "stream_error": pick(events, "error"),
                                               "answer_head": answer[:400], "english_reasoning_in_answer": bool(ENGLISH_REASONING.search(answer)),
                                               "universal_app_numbers_in_answer": re.findall(r"앱 번호[^\n]{0,30}", answer)}
            if a["plan_selected"]:
                d = (await client.get(url + f"/api/plans/{pick(events, 'plan_selected')['plan_id']}", timeout=30)).json()
                a["constraints"] = d["constraints"]
                a["passed"] = d["passed"]
            # --- B: FDI extraction on the synthetic case -------------------------------------------------
            await client.post(url + "/api/cases/moderate/activate", timeout=60)
            ctx = {"request_id": "fdi-B", "case_id": "moderate", "base_plan_id": None, "constraints": {}}
            events, answer, elapsed = await converse(client, url, "14번과 24번 발치로 계획 짜줘.", ctx, log)
            b = verdict["B_fdi_extraction"] = {"elapsed_s": round(elapsed, 1), "tools": tool_steps(events),
                                                "set_constraints_args": tool_args(events, "set_constraints"),
                                                "plan_selected": bool(pick(events, "plan_selected")), "plan_error": pick(events, "plan_error"),
                                                "stream_error": pick(events, "error"), "answer_head": answer[:400],
                                                "universal_in_answer": re.findall(r"(?<![\d.])(?:[1-9]|1[0-6])번", answer)}
            if b["plan_selected"]:
                d = (await client.get(url + f"/api/plans/{pick(events, 'plan_selected')['plan_id']}", timeout=30)).json()
                b["constraints_extraction"] = d["constraints"]["extraction"]
                b["removed"] = d["target"].get("removed")
                b["tools_got_5_12"] = d["constraints"]["extraction"] == [5, 12]
            # --- C: the sample card's FDI-only prescription (no app-number tail) with no extraction preset -----
            s97 = SAMPLES["poseidon-000097"]
            await client.post(url + f"/api/cases/{s97.case_id}/activate", timeout=60)
            ctx = {"request_id": "fdi-C", "case_id": s97.case_id, "base_plan_id": None, "constraints": {"extraction": []}}
            events, answer, elapsed = await converse(client, url, s97.request, ctx, log)
            c = verdict["C_sample_000097_fdi_only"] = {"request": s97.request, "elapsed_s": round(elapsed, 1), "tools": tool_steps(events),
                                                        "set_constraints_args": tool_args(events, "set_constraints"),
                                                        "plan_selected": bool(pick(events, "plan_selected")), "plan_error": pick(events, "plan_error"),
                                                        "stream_error": pick(events, "error"), "answer_head": answer[:400],
                                                        "universal_in_answer": re.findall(r"(?<![\d.])(?:[1-9]|1[0-6])번", answer)}
            if c["plan_selected"]:
                d = (await client.get(url + f"/api/plans/{pick(events, 'plan_selected')['plan_id']}", timeout=30)).json()
                c["constraints_extraction"] = d["constraints"]["extraction"]
                c["removed"] = d["target"].get("removed")
                c["tools_got_5_12"] = d["constraints"]["extraction"] == [5, 12]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        logfile.close()
    (OUT / "fdi-leak.json").write_text(json.dumps({"verdict": verdict, "log": log}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
