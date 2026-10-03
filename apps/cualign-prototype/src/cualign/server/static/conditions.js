// app.js 에서 나눈 모듈: the 조건 form — read, fill, compare with the plan's, the action buttons, approve and review.
import { $, FIELD_READ, api, fdi, fdiList, fieldErrors, monthsOfCap, normSurfaces, state, surfacesKo } from "./state.js";
import { stepIndex } from "./viewer.js";
// defined in later modules: app.js hands them in (wire) before any of them is called
let addMsg, closeExportPop, exportPopKey, planNo, pollExportStatus, renderPlanList, renderRail, renderResult;
export function wire(fns) { ({ addMsg, closeExportPop, exportPopKey, planNo, pollExportStatus, renderPlanList,
    renderRail, renderResult } = fns); }
function readConstraints() {
  const teeth = (id) => FIELD_READ[id]();
  const ipr = FIELD_READ.cIpr(), cap = FIELD_READ.cCap();
  // extraction: the prescribed teeth (#56); the app never picks them, 발치 허용 off empties the field (non-extraction)
  const out = { extraction: teeth("cExtract"), lock: teeth("cLock"), ipr_exclude: teeth("cExclude"),
    ipr_limit_mm: ipr, stage_cap: cap, clear_stage_cap: cap === null, order: $("cOrder").value };
  // IPR per contact (#57), FDI in the patch — this field only. Sent when there is one or one is being cleared; a server
  // before core #143 rejects the key (ConstraintPatch forbids extras), so an untouched empty field sends nothing
  const surf = FIELD_READ.cSurf();
  if (surf.length || state.plan?.constraints?.ipr_surfaces?.length || state.setup?.ipr_surfaces?.length) out.ipr_surfaces = surf;
  return out;
}
function fillConstraints(c) {
  $("cExtract").value = (c.extraction || []).map(fdi).sort((a,b) => a-b).join(", ");
  $("cAllowExt").checked = (c.extraction || []).length > 0; $("cExtract").disabled = !$("cAllowExt").checked;
  $("cLock").value = (c.lock || []).map(fdi).sort((a,b) => a-b).join(", ");
  $("cSurf").value = surfacesKo(c.ipr_surfaces);
  $("cExclude").value = (c.ipr_exclude || []).map(fdi).sort((a,b) => a-b).join(", ");
  $("cIpr").value = c.ipr_limit_mm;
  $("cCap").value = c.stage_cap ?? "";
  $("cMonths").value = c.stage_cap ? monthsOfCap(c.stage_cap) : "";
  $("cOrder").value = c.order;
  renderCondState();
}
// the form's conditions as short words: 비발치 · 고정 13번 · IPR 면당 0.25mm · 동시
function condWords() {
  const nums = (id) => $(id).value.split(/[,\s]+/).filter(Boolean);
  const ext = nums("cExtract"), lock = nums("cLock"), excl = nums("cExclude"), parts = [];
  parts.push(ext.length ? "발치 " + ext.join("·") + "번" : "비발치");   // the form already reads FDI (#113)
  if (lock.length) parts.push("고정 " + lock.join("·") + "번");
  const surf = $("cSurf").value.split(",").map((s) => s.trim()).filter(Boolean);   // a prescription replaces the even rule (#57), as the server's conditions_ko does
  if (surf.length) parts.push("IPR 처방 " + surf.map((s) => s + "mm").join(" · "));
  else {
    if (excl.length) parts.push("IPR 제외 " + excl.join("·") + "번");
    parts.push("면당 " + ($("cIpr").value || "0") + " mm");
  }
  parts.push($("cCap").value ? "상한 " + $("cCap").value + "장" : "단계 상한 없음");
  parts.push("이동 " + ($("cOrder").options[$("cOrder").selectedIndex]?.textContent ?? ""));
  return parts;
}
// A case with no plan whose calculation failed (#112, 보드 07): the server's sentence as it came, the conditions it
// was tried with, and one way out. Cleared by the next plan list that has rows.
function renderPlanFail() {
  const box = $("planFail"), msg = state.planError;
  box.hidden = !msg;
  if (!msg) return;
  $("planFailMsg").textContent = msg;
  $("planFailCond").replaceChildren(...condWords().map((t) => { const c = document.createElement("span"); c.className = "tag"; c.textContent = t; return c; }));
  $("planFailRetry").disabled = state.streaming || state.loading;
}
// The 조건 tab's two lines: whose conditions the form holds, and whether it still matches the plan on screen (#111)
function renderCondState() {
  const p = state.plan, n = planNo(p?.plan_id);
  $("condFor").textContent = p ? `계획 ${n}의 조건 · 고치면 셋업부터 다시` : state.setup ? "에이전트가 읽은 조건 · 고치면 셋업부터 다시" : "계획을 만들 조건";
  const dirty = constraintsDirty(), el = $("condState"), busy = state.streaming || state.loading;
  // a field that does not read: its sentence in red under it, the field outlined
  const errs = fieldErrors();
  for (const id of ["cExtract", "cLock", "cSurf", "cExclude", "cIpr", "cCap"]) {
    const box = $(id).parentElement.querySelector(".err");
    box.textContent = errs[id] ?? ""; box.hidden = !errs[id];
    $(id).setAttribute("aria-invalid", String(!!errs[id]));
  }
  el.textContent = !p ? "" : dirty ? "조건이 바뀜 · 「다시 계산」으로 새 계획" : "보고 있는 계획의 조건과 같음";
  el.classList.toggle("changed", dirty);
  const diff = p && !Object.keys(errs).length ? condDiff(p.constraints) : "";
  $("condDiff").textContent = diff;
  $("condRecalc").hidden = !p;   // 다시 계산 changes a plan on screen; before one exists the retry bar's 에이전트 없이 계산 does
  $("condRecalc").disabled = busy || !diff;
  $("condApply").hidden = stepIndex(state.progress) < 1;   // a hand-edited condition re-runs the setup turn (#20)
  $("condApply").disabled = busy;
}
// what the form changes against the plan on screen, in one line: 「발치 14·24 → 비발치, 장수 상한 없음 → 32」
function condDiff(base) {
  const c = readConstraints(), out = [];
  const teeth = (ids) => fdiList([...(ids ?? [])].sort((a, b) => fdi(a) - fdi(b)));
  const ext = (ids) => ids?.length ? "발치 " + teeth(ids) : "비발치";
  if (teeth(c.extraction) !== teeth(base.extraction)) out.push(`${ext(base.extraction)} → ${ext(c.extraction)}`);
  for (const [k, name] of [["lock", "고정"], ["ipr_exclude", "IPR 제외"]])
    if (teeth(c[k]) !== teeth(base[k])) out.push(`${name} ${teeth(base[k]) || "없음"} → ${teeth(c[k]) || "없음"}`);
  const surf = (s) => surfacesKo(normSurfaces(s)) || "없음", cs = c.ipr_surfaces && normSurfaces(c.ipr_surfaces, true);   // the form's are FDI
  if (cs && surf(cs) !== surf(base.ipr_surfaces)) out.push(`IPR 처방 ${surf(base.ipr_surfaces)} → ${surf(cs)}`);
  if (Math.abs(c.ipr_limit_mm - (base.ipr_limit_mm ?? 0.25)) > 1e-6) out.push(`면당 IPR ${base.ipr_limit_mm ?? 0.25} → ${c.ipr_limit_mm} mm`);
  if ((c.stage_cap ?? null) !== (base.stage_cap ?? null)) out.push(`장수 상한 ${base.stage_cap ?? "없음"} → ${c.stage_cap ?? "없음"}`);
  const order = (v) => [...$("cOrder").options].find((o) => o.value === v)?.textContent ?? v;
  if (c.order !== (base.order ?? "simultaneous")) out.push(`이동 순서 ${order(base.order ?? "simultaneous")} → ${order(c.order)}`);
  return out.join(", ");
}
function constraintsDirty() {
  if (!state.plan) return false;
  try {
    const c = readConstraints();
    delete c.clear_stage_cap;
    const surfChanged = JSON.stringify(normSurfaces(c.ipr_surfaces, true)) !== JSON.stringify(normSurfaces(state.plan.constraints.ipr_surfaces));
    delete c.ipr_surfaces;
    return surfChanged || Object.keys(c).some(k => JSON.stringify(c[k]) !== JSON.stringify(state.plan.constraints[k]));
  } catch { return true; }
}
function updateActions() {
  const p = state.plan, busy = state.streaming || state.loading;
  const dirty = constraintsDirty(), allowed = p && !busy && !dirty;
  for (const id of ["sendBtn", "caseBtn"]) $(id).disabled = !!busy;
  renderCondState();
  $("constraints").disabled = !!busy;
  const exportable = allowed && p.passed && !p.input_stale && ["passed", "skipped"].includes(p.review.status);
  $("exportBtn").disabled = !exportable || !!state.stlBusy;
  renderRail();
  // the zip of the plan on screen only: a build or a download of another plan does not show here
  const st = p?.approval ? state.exportStatus[p.plan_id] : null;
  const zipBusy = !!p && (state.stlBusy === p.plan_id || !!st?.building);
  const progress = zipBusy && st?.total ? `${st.done ?? 0}/${st.total}` : null;
  const building = "STL 만드는 중" + (progress ? " · " + progress : "…");
  $("exportBtn").textContent = zipBusy ? building : p?.approval ? "STL 내려받기" : "내보내기";
  // why the button is off, in the same order as the gate above; nothing while a turn or a load is running
  $("exportWhy").textContent = exportable || busy ? "" : !p ? "계획이 없습니다" : p.input_stale ? "이전 입력의 계획"
    : dirty ? "조건이 바뀜 · 새 계획 뒤 승인" : !p.passed ? `규칙 위반 ${(p.violations ?? []).length}건 · 조건을 바꿔 다시 계획`
    : "검토 실패 · 검토 다시 요청";
  // the rail item is the only visible 내보내기 (#13 polish): dim until the plan can be approved, the reason in its tooltip
  const railExport = document.querySelector('#rail button[data-go="export"]');
  railExport.querySelector("span").textContent = zipBusy ? (progress ?? "만드는 중…") : p?.approval ? "STL 받기" : "내보내기";
  railExport.title = $("exportWhy").textContent ? "내보내기 · " + $("exportWhy").textContent : p?.approval ? "단계별 STL(zip)을 내려받습니다" : "승인하고 STL 내보내기";
  $("exportSkip").hidden = !["skipped", "not_requested"].includes(p?.review?.status);
  $("revokeBtn").hidden = !p?.approval;
  $("revokeBtn").disabled = !allowed;
  if (!exportable || state.exportPopFor && state.exportPopFor !== exportPopKey()) closeExportPop();
  // Recovery when the agent skipped the reviewer or the review failed: the dentist asks for it on this plan.
  $("reviewBtn").hidden = !p || !["not_requested", "failed"].includes(p.review.status);
  $("reviewBtn").disabled = !allowed;
  const link = $("stlLink");
  if (allowed && p.approval) link.href = "/api/plans/" + encodeURIComponent(p.plan_id) + "/stl.zip";
  else link.removeAttribute("href");
  // 펄스 점 at the download link's place: building N/총 → a check and the link with what it holds; failed → X, the reason, 다시 만들기
  const failed = !zipBusy && !!st?.error, ready = !zipBusy && !!st?.ready;
  $("exportState").hidden = !p?.approval;
  const dot = link.querySelector(".pulse-dot");
  dot.hidden = !zipBusy && !ready && !failed;   // approved but no build known (a restarted server): the link builds it
  dot.classList.toggle("ok", ready); dot.classList.toggle("fail", failed);
  link.classList.toggle("ready", ready); link.classList.toggle("fail", failed);
  const n = p?.stages?.length ?? p?.info?.n_stages;
  link.querySelector("span").textContent = zipBusy ? building : failed ? "STL 만들지 못함 · " + st.error
    : ready ? `STL 내려받기 (${n}단계 · ${st.stl}개)` : "STL 내려받기";
  link.title = failed ? st.error : "";
  $("exportRetry").hidden = !failed;
  if (p?.approval && (!st || st.building)) pollExportStatus(p.plan_id);   // a reload or a revisit picks the build up
  if (p && dirty) $("planNotice").textContent = "조건 변경됨 — 새 계획을 생성한 뒤 승인하세요. 현재 3D는 이전 계획입니다.";
}
// the plan's locked teeth as the dentist reads them (FDI, sorted): what approval asks the dentist to confirm
const lockFdi = (p) => (p?.constraints?.lock ?? []).map(fdi).sort((a, b) => a - b);
// an approval counts on the export card only with the locked teeth the dentist ticked, equal to the plan's: one without
// them (approved before round 4) or with others asks for the tick again (Codex re-check 2026-10-03 (4))
const confirmedApproval = (p) => !!p?.approval && Array.isArray(p.approval.confirmed_lock_fdi)
  && [...p.approval.confirmed_lock_fdi].sort((a, b) => a - b).join() === lockFdi(p).join();
// approve: the export card's 확정 (POST, also over an approval that does not count); otherwise 승인 취소 (DELETE)
async function approveCurrent(approve = false) {
  const p = state.plan;
  if (!p || state.streaming || state.loading || constraintsDirty()) return;
  const revoke = !approve && !!p.approval;   // the 내보내기 popover asked the question; 승인 취소 needs none
  state.loading = true; updateActions();
  try {
    // the locked teeth the export card showed and the dentist ticked: the server records them, and refuses without
    const result = await api("/api/plans/" + encodeURIComponent(p.plan_id) + "/approval",
      { method: revoke ? "DELETE" : "POST", headers: { "Content-Type": "application/json" },
        ...(revoke ? {} : { body: JSON.stringify({ confirmed: true, lock_fdi: lockFdi(p) }) }) });
    delete state.exportStatus[p.plan_id];   // approval starts (or finds) the build: ask the server again
    if (state.plan?.plan_id === p.plan_id) { state.plan = result; renderResult(result); }
  } catch (e) { addMsg("error", e.message); }
  finally { state.loading = false; updateActions(); }
}

async function reviewCurrent() {
  const p = state.plan;
  if (!p || state.streaming || state.loading || constraintsDirty()) return;
  state.loading = true; updateActions();
  const row = state.planRows[p.plan_id];
  if (row) { row.review = { ...(row.review ?? {}), status: "running" }; renderPlanList(); }
  try {
    const result = await api("/api/plans/" + encodeURIComponent(p.plan_id) + "/review", { method: "POST" });
    if (state.plan?.plan_id === p.plan_id) {
      state.plan = result; renderResult(result);
      if (result.review.status === "failed") addMsg("error", result.review.message + " (" + result.review.error + ")");
    }
  } catch (e) {
    addMsg("error", "검토 요청 실패: " + e.message);
    if (row) row.review = state.plan?.plan_id === p.plan_id ? state.plan.review : row.review;   // not left 검토 중
    if (state.plan?.plan_id === p.plan_id) renderResult(state.plan);
  }
  finally { state.loading = false; updateActions(); }
}

export { approveCurrent, condWords, confirmedApproval, fillConstraints, lockFdi, readConstraints, renderCondState,
  renderPlanFail, reviewCurrent, updateActions };
