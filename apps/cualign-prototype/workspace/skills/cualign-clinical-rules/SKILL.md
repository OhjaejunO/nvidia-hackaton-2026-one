---
name: cualign-clinical-rules
description: Validate and stage clear-aligner (투명교정) treatment plans against published clinical limits using the cuAlign tool set. Use when an agent or developer must check per-aligner movement, IPR, arch expansion, stage count, or tooth-collision constraints; choose or switch a space-gaining strategy (expansion, IPR, extraction); or turn a natural-language constraint like "no extraction, within 12 months" into stage caps. Draft-only — never diagnose or prescribe.
---

# cuAlign Clinical Rules

## Scope and safety boundary

- This skill produces **draft staging plans** and **rule-violation reports**. The dentist makes every clinical decision.
- Never state a diagnosis, never prescribe wear time or treatment, never claim a plan is final. Output wording: "규칙 통과 (의사 검토 전 초안)".
- Do not invent clinical numbers. Use only the table below. If a constraint is not in the table, say so and ask the dentist.
- Do not send patient scan data (STL/PLY meshes) to any endpoint. Meshes stay local; the reviewer receives computed plan summaries.

## Clinical limits (sources: MDPI Applied Sciences 2024 staging review · Nature IJOS 2025 expert consensus · Align Technology 2016 press release)

| Constraint | Value | Note |
|---|---|---|
| Linear movement per aligner | 0.25 mm | hard limit for `plan_stages` |
| Angular / rotational movement per aligner | 1° / 2° | rotation (2°) enforced for incisor derotation; angulation/torque not supported |
| Wear period per aligner | 14 days | months = n_stages * 14 / 30.4 |
| IPR per surface | user limit 0..0.25 mm | PoC space calculation cap; not a patient-specific clinical recommendation |
| Unilateral arch expansion | max 2 mm | stage the expansion if over 1 mm |
| Extraction threshold | space deficit over 8 mm | consider only if the user allows extraction |

## Strategy rules

1. Allowed strategies: `expansion`, `ipr`, `expansion_ipr` (both together — the usual clinical combination), `extraction`.
2. Extraction is the dentist's prescription (#56). If the request forbids extraction ("발치 없이", "발치는 절대 안 돼"), set extraction=[] and never call `propose_target` with `extraction`. If it names the teeth ("14번과 24번 발치" — the dentist speaks FDI), set extraction to them as said ([14, 24]; set_constraints takes FDI and the server converts; only premolars FDI 14, 15, 24, 25 are supported) and plan the extraction strategy only. If extraction is wanted without teeth, ask which teeth — never pick them. Comparison does not change this constraint.
3. Default order when nothing is specified: `expansion` → `ipr` → `expansion_ipr` (extraction only when prescribed, and then alone). Each strategy gains a different amount of space; if the gain does not cover the crowding, `plan_stages` reports `space_deficit` and you move to the next strategy.
4. Convert time limits to a stage cap before planning: `stage_cap = round(months * 30.4 / 14)`. "12개월" → 26, "10개월" → 22, "8개월" → 17.
5. After `plan_stages` reports a failure, switch to the next allowed strategy and rerun `propose_target → plan_stages`. `plan_stages` already validates; `validate` only re-checks an existing plan. Stop when `passed` is true or all allowed strategies are exhausted.
6. When every allowed strategy fails, report the plan with the fewest violations and state plainly which constraint would have to be relaxed (e.g. "발치 없이는 8개월이 안 됩니다. 20개월이면 IPR 42장으로 됩니다"). Do not soften this.

## Tool sequence (NeMo Agent Toolkit function group `cualign`)

```
cualign__get_constraints    {"unused": ""} -> confirmed constraints
cualign__set_constraints    {"lock": [13], "ipr_exclude": [12,11,21,22]} -> patched constraints (FDI as the dentist said: «13번 고정, 앞니 12·11·21·22번 IPR 제외»)
cualign__clinical_limits    {"unused": ""}
cualign__load_case          {"case_id": "moderate"}
cualign__propose_target     {"strategy": "expansion_ipr"} -> target_id
cualign__plan_stages        {"target_id": "..."} -> plan_id, constraints, parent_plan_id
cualign__validate           {"plan_id": "..."} -> same constraints and rule results
cualign__compare_strategies {"allowed": ["expansion", "ipr", "expansion_ipr"]} -> plans
cualign__select_plan        {"plan_id": "..."} -> final UI selection
reviewer                   {"plan_id": "..."} -> memo or failed status (max 2 attempts / 40s)
cualign__export_stl         {"plan_id": "..."} -> approved plan ZIP only
```

The server context (the first system message of the request) already carries the confirmed constraints and may carry the case summary (`case`), the clinical limits (`limits`) and this skill text (`skill`). Do not call `get_constraints`, `load_case`, `clinical_limits` or `load_skill` for what it already provides; each such call costs one model round-trip. Call them only when the message lacks that item.

set_constraints changes ONLY fields explicitly requested in this turn, before target generation.
Preserve omitted fields. Tooth lists are full lists: merge additions with existing values.
Only explicit [] clears a tooth list; stage_cap:null clears the stage limit.
Use ipr_limit_mm for the per-surface limit and order for movement order.
Never relax a forbidden strategy or discard locks/exclusions to obtain a passing plan.
The server attaches the selected parent to newly created plans. A revision never inherits approval.
Select the final candidate once, then call reviewer once. Report review failure, never invent its memo.
Only the doctor can approve in the UI; there is no approval tool.
Rule failures and review failures block approval. Rule fallback is explicitly labelled reviewer-not-run.

Tool names carry the `cualign__` prefix exactly. Read the confirmed constraints first (from the server context; `get_constraints` only if missing). Ask ONE question if the requested change is ambiguous. Do not repeat questions already answered by the UI or a parent plan.

## Reading a violation report

| `type` | Meaning | Usual fix |
|---|---|---|
| `move_limit` | a tooth moves more than 0.25 mm between consecutive stages | more stages (only if the cap allows) or a strategy with smaller total movement |
| `collision` | two teeth's convex-hull overlap grows by more than 1 mm³ over where they started | strategy that creates space first (IPR/expansion), or reorder staging |
| `rotation_limit` | a tooth turns more than 2° between consecutive stages | more stages (only if the cap allows) |
| `stage_cap` | n_stages exceeds the allowed cap | strategy with less total movement, `simultaneous` order, or ask the dentist to relax the time limit |
| `space_deficit` | the strategy cannot gain enough space for the crowding | next strategy in the ladder; if extraction is forbidden and nothing passes, say how many mm are missing |

## Final answer format (Korean)

Language justification: the users are Korean dentists and the hackathon submission is reviewed in Korean; the UI is Korean-only. If a request arrives in another language, answer in that language instead.

Lead with one sentence (전략 · 단계 수(약 개월) · 규칙 통과 또는 위반 종류별 건수), then 조건 · 검토 결과 · 의사 확인 필요 as short bullets, in the Korean labels the workflow instructions give (발치 치아, 고정 치아, IPR 제외 치아, IPR 한도, 단계 상한, 이동 순서; 확장, IPR, 확장 + IPR, 발치). Never write plan ids, tool or field names, English enum values, formulas or counters such as attempts=. End with the sentence: "이 계획은 초안입니다. 최종 판단은 의사가 합니다."
