"""E2E rehearsal of the demo script against the real NIM, tool call by tool call (v3 batch, item 4).

Script (what the presenter does on screen), one server, one case:
  1. 시작            GET /api/cases            the sample cards are there
  2. 샘플 케이스     POST /api/cases/{id}/activate   the case opens as the scan only: no plan, conditions = the prescription
  3. 에이전트 계획    three turns, one step each (the step flow): setup -> target -> stages; each stops where its step says
     (was: one turn)   /chat/stream  the card's «에이전트에게 계획 맡기기» sentence, the sample's constraints
  4. 기간 상한       /chat/stream  «12개월 안에 끝나게 다시 짜줘.» on the selected plan
  5. 비교            /chat/stream  «확장안이랑 IPR안 둘 다 만들어서 비교해줘.» on the selected plan
  6. 검토            the reviewer ran inside each turn (plan_selected.review); POST /api/plans/{id}/review if it did not
  7. 후속 질문 카드  POST /api/followup on the conversation so far (a card or null, never an error)
  8. 내보내기        stl.zip is 409 before approval; POST approval; GET stl.zip is a ZIP with n_stages*14 tooth files
Each step records what happened and a pass flag; the script never stops early unless the server is gone.

Never prints the API key. Writes out/nim-live/e2e.json (events, tool args, answers) and prints the verdict.
Costs real NVIDIA usage: three planning conversations (several model calls each) plus rails and a followup call.
Run: uv run --frozen python -X utf8 tests/nim_e2e_live_check.py [sample_case_id]   (default poseidon-000097, the demo case)
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "nim-live"
sys.path.insert(0, str(ROOT / "tests"))
from nim_live_check import converse, free_port, lock_fdi, pick, tool_steps, wait_ready  # noqa: E402

HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-z]")
UNIVERSAL_TOOTH = re.compile(r"(?<![\d.])(?:[1-9]|10)번")         # a Universal number written to the dentist (11..16 read as FDI since #133)
INTERNAL = re.compile(r"(?<![A-Za-z_])(?:allow_extraction|ipr_exclude|ipr_limit_mm|stage_cap|plan_id|expansion_ipr|anterior_first|"
                      r"space_deficit|p[0-9a-f]{6,})(?![A-Za-z_])")
DISCLAIMER = "이 계획은 초안입니다. 최종 판단은 의사가 합니다."


def tool_names(events):
    return [n.split("Function Start: ")[1] for n in tool_steps(events) if n.startswith("Function Start: ")]


def tool_args(events, name):
    out = []
    for n, o in events:
        if n == "intermediate_data" and (o.get("name") or "").endswith(name):
            out.append(o.get("payload") if "payload" in o else {k: v for k, v in o.items() if k != "name"})
    return out


def answer_checks(answer):
    hangul, latin = len(HANGUL.findall(answer)), len(LATIN.findall(answer))
    return {"korean": bool(hangul) and hangul >= 0.3 * (hangul + latin), "disclaimer": DISCLAIMER in answer,
            "universal_numbers": UNIVERSAL_TOOTH.findall(answer), "internal_terms": INTERNAL.findall(answer)}


async def turn(client, url, messages, text, ctx, log, step=None):
    """One screen turn: the conversation so far plus `text`, as app.js sends it (`step` = the turn's step)."""
    messages = messages + [{"role": "user", "content": text}]
    events, answer, elapsed = await converse(client, url, text, ctx, log, messages=messages, step=step)
    messages = messages + [{"role": "assistant", "content": answer}]
    selected = pick(events, "plan_selected")
    rec = {"request": text, "elapsed_s": round(elapsed, 1), "tools": tool_names(events),
           "plan_selected": selected["plan_id"] if selected else None, "review": (selected or {}).get("review"),
           "plan_error": pick(events, "plan_error"), "stream_error": pick(events, "error"),
           "answer": answer, **answer_checks(answer)}
    return messages, events, rec


async def main(case_id):
    load_dotenv(ROOT / ".env")
    if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
        raise SystemExit("NVIDIA_API_KEY is not set (put it in apps/cualign-prototype/.env)")
    OUT.mkdir(parents=True, exist_ok=True)
    from cualign.core.samples import SAMPLES
    sample = SAMPLES[case_id]

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PYTHONUTF8": "1", "CUALIGN_OUT": str(ROOT / "out")}
    logfile = open(OUT / "e2e-server.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "nat.cli.main", "serve", "--config_file", str(ROOT / "configs" / "workflow.yml"),
                             "--host", "127.0.0.1", "--port", str(port)], cwd=str(ROOT), env=env, stdout=logfile,
                            stderr=subprocess.STDOUT)
    log, steps = [], {}
    try:
        await wait_ready(url, proc)
        async with httpx.AsyncClient() as client:
            # 1. 시작
            cases = (await client.get(url + "/api/cases", timeout=30)).json()
            ids = [c.get("case_id") for c in (cases.get("cases") if isinstance(cases, dict) else cases)]
            steps["1_start"] = {"cases": ids, "pass": case_id in ids}
            # 2. 샘플 케이스 — the step flow: opening is the scan only (no plan, conditions = the prescription)
            r = await client.post(url + f"/api/cases/{case_id}/activate", timeout=120)
            opened = r.json()
            plans = (await client.get(url + f"/api/plans?case_id={case_id}", timeout=30)).json()["plans"]
            steps["2_open_case"] = {"status": r.status_code, "constraints": opened.get("constraints"), "flow": opened.get("flow"),
                                    "rule_plans": [(p["strategy"], p["n_stages"], p["passed"]) for p in plans],
                                    "pass": r.status_code == 200 and "plan_error" not in opened and plans == [] and opened.get("flow") is None}
            shown = None
            # 3. 에이전트 계획 — three turns, one step each: setup (conditions, asks back) -> target (the arrangement, asks
            # back) -> stages (the plan, selected and reviewed). Each turn must stop where its step says.
            messages = []
            ctx = {"request_id": "e2e-3a", "case_id": case_id, "base_plan_id": None, "constraints": {}}   # the case opened with the prescription; the model reads the sentence
            messages, ev, rec = await turn(client, url, messages, sample.request, ctx, log, step="setup")
            done = pick(ev, "step_done")
            rec["step_done"] = done
            rec["pass"] = bool(done) and done.get("step") == "setup" and not rec["plan_selected"] and rec["korean"] and "?" in rec["answer"] \
                and not rec["universal_numbers"] and not rec["internal_terms"] and rec["plan_error"] is None \
                and not any(t in rec["tools"] for t in ("cualign__propose_target", "cualign__plan_stages", "reviewer"))
            steps["3a_setup"] = rec
            ctx = {"request_id": "e2e-3b", "case_id": case_id, "base_plan_id": None, "constraints": {}}
            messages, ev, rec = await turn(client, url, messages, "이대로 목표 배열을 만들어줘.", ctx, log, step="target")
            done = pick(ev, "step_done")
            rec["step_done"] = done
            target_ok = bool(done) and done.get("step") == "target" and bool(done.get("target_id"))
            if target_ok:
                t = await client.get(url + f"/api/cases/{case_id}/targets/{done['target_id']}", timeout=30)
                target_ok = t.status_code == 200 and len(t.json()["stages"]) == 1
                rec["target_summary"] = done.get("summary")
            rec["pass"] = target_ok and not rec["plan_selected"] and rec["korean"] and "?" in rec["answer"] \
                and not rec["universal_numbers"] and not rec["internal_terms"] and rec["plan_error"] is None \
                and "cualign__plan_stages" not in rec["tools"] and "reviewer" not in rec["tools"]
            steps["3b_target"] = rec
            ctx = {"request_id": "e2e-3c", "case_id": case_id, "base_plan_id": None, "constraints": {}}
            messages, ev, rec = await turn(client, url, messages, "단계로 나눠줘.", ctx, log, step="stages")
            done = pick(ev, "step_done")
            rec["step_done"] = done
            rec["pass"] = bool(rec["plan_selected"]) and rec["korean"] and rec["disclaimer"] and not rec["universal_numbers"] \
                and not rec["internal_terms"] and "reviewer" in rec["tools"] and rec["plan_error"] is None \
                and bool(done) and done.get("step") == "stages" and done.get("plan_id") == rec["plan_selected"]
            steps["3c_stages"] = rec
            selected = rec["plan_selected"] or shown
            # 4. 기간 상한
            ctx = {"request_id": "e2e-4", "case_id": case_id, "base_plan_id": selected, "constraints": {}}
            messages, ev, rec = await turn(client, url, messages, "12개월 안에 끝나게 다시 짜줘.", ctx, log)
            rec["set_constraints_args"] = tool_args(ev, "set_constraints")
            cap_ok = False
            if rec["plan_selected"]:
                d = (await client.get(url + f"/api/plans/{rec['plan_selected']}", timeout=30)).json()
                rec["stage_cap"], rec["n_stages"], rec["passed"] = d["constraints"]["stage_cap"], d["info"]["n_stages"], d["passed"]
                cap_ok = d["constraints"]["stage_cap"] == round(8 * 30.4 / 14) and d["parent_plan_id"] == selected
            rec["pass"] = cap_ok and rec["korean"] and "17단계" in rec["answer"] and not rec["universal_numbers"] and not rec["internal_terms"]
            steps["4_stage_cap"] = rec
            selected = rec["plan_selected"] or selected
            # 5. 비교
            ctx = {"request_id": "e2e-5", "case_id": case_id, "base_plan_id": selected, "constraints": {}}
            messages, ev, rec = await turn(client, url, messages, "확장안이랑 IPR안 둘 다 만들어서 비교해줘.", ctx, log)
            rec["compare_args"] = tool_args(ev, "compare_strategies")
            lines = [ln for ln in rec["answer"].splitlines() if re.search(r"(확장|IPR)\s*(전략|안|:)", ln)]
            rec["plan_lines"] = lines
            if sample.constraints.get("extraction"):
                # an extraction prescription (000097) allows only the extraction plan (#56): the right answer to a
                # comparison of non-extraction strategies is a Korean question back, no tool, no plan
                rec["note"] = "extraction prescribed: comparison refused with a question, by design"
                # the model may call compare_strategies first (it returns the extraction plan only) and then ask: no plan selected
                rec["pass"] = not rec["plan_selected"] and rec["korean"] and "?" in rec["answer"] and rec["plan_error"] is None \
                    and not rec["universal_numbers"] and not rec["internal_terms"]
            else:
                rec["pass"] = "cualign__compare_strategies" in rec["tools"] and rec["korean"] and len(lines) >= 2 \
                    and not rec["universal_numbers"] and not rec["internal_terms"] and rec["plan_error"] is None
            steps["5_compare"] = rec
            selected = rec["plan_selected"] or selected
            # 6. 검토
            d = (await client.get(url + f"/api/plans/{selected}", timeout=30)).json()
            review = d["review"]
            if not review or review.get("status") not in ("passed", "failed"):
                r = await client.post(url + f"/api/plans/{selected}/review", timeout=120)
                review = r.json().get("review") if r.status_code == 200 else {"status": f"http {r.status_code}", "body": r.text[:200]}
            memo = (review or {}).get("memo") or (review or {}).get("message") or ""
            steps["6_review"] = {"plan": selected, "review_status": (review or {}).get("status"), "memo_head": memo[:300],
                                 "snake_case_in_memo": re.findall(r"\b\w+_(?:mm|deg|teeth|id)\b", memo),
                                 "pass": (review or {}).get("status") == "passed" and not re.findall(r"\b\w+_(?:mm|deg|teeth|id)\b", memo)}
            # 7. 후속 질문 카드
            r = await client.post(url + "/api/followup", json={"messages": messages}, timeout=60)
            card = r.json().get("question") if r.status_code == 200 else None
            steps["7_followup"] = {"status": r.status_code, "card": card, "pass": r.status_code == 200}
            # 8. 내보내기 — only a passing, reviewed plan can be approved; a failed plan is refused with the reason (409)
            before = await client.get(url + f"/api/plans/{selected}/stl.zip", timeout=60)
            appr = await client.post(url + f"/api/plans/{selected}/approval",
                                     json={"confirmed": True, "lock_fdi": await lock_fdi(client, url, selected)}, timeout=60)
            step = {"plan": selected, "plan_passed": d["passed"], "before_approval": before.status_code, "approval": appr.status_code}
            if d["passed"]:
                after = await client.get(url + f"/api/plans/{selected}/stl.zip", timeout=300)
                n_stl = None
                if after.status_code == 200:
                    with zipfile.ZipFile(BytesIO(after.content)) as z:   # tooth files only (print_models/ holds the gingiva models)
                        n_stl = len([n for n in z.namelist() if n.endswith(".stl") and not n.startswith("print_models/")])
                kept = opened["n_teeth"] - len((d.get("target") or {}).get("removed") or [])   # extracted crowns are not exported
                step.update({"after_approval": after.status_code, "stl_files": n_stl, "n_stages": d["info"]["n_stages"],
                             "n_teeth": opened["n_teeth"], "teeth_exported": kept,
                             "pass": before.status_code == 409 and appr.status_code == 200 and after.status_code == 200
                             and n_stl == d["info"]["n_stages"] * kept})
            else:
                step.update({"refusal": appr.json().get("detail") if appr.status_code == 409 else None,
                             "pass": before.status_code == 409 and appr.status_code == 409,
                             "note": "the selected plan failed the rules: approval refused as designed, no export in this rehearsal"})
            steps["8_export"] = step
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        logfile.close()
    (OUT / "e2e.json").write_text(json.dumps({"case": case_id, "steps": steps, "log": log}, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in steps.items():
        print(f"{'PASS' if v.get('pass') else 'FAIL'}  {k}  " + json.dumps({a: b for a, b in v.items() if a not in ("answer", "pass", "review")},
                                                                         ensure_ascii=False)[:300])
    print("ALL PASS" if all(v.get("pass") for v in steps.values()) else "SOME FAILED")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "poseidon-000097"))
