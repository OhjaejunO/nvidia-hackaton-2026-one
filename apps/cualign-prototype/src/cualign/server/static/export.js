// app.js 에서 나눈 모듈: the export popover, toast, the STL build poll and download.
import { $, capOfMonths, monthsOfCap, state } from "./state.js";
import { canvas, setView } from "./viewer.js";
import { onPointerMove } from "./pick.js";
import { approveCurrent, confirmedApproval, lockFdi, reviewCurrent, updateActions } from "./conditions.js";
import { planNo } from "./plans.js";
import { addMsg } from "./chat.js";
import { showPop } from "./panes.js";
// The export confirm card, for approving and for downloading again: beside the rail item that opened it (the anchor button
// itself is hidden), vertically centred on it with its caret. It closes on a press anywhere outside it, Esc, and when the
// plan or the step on screen changes; focus goes to its main button and back to the rail item.
const railExportBtn = () => document.querySelector('#rail button[data-go="export"]');
const exportPopKey = () => state.plan?.plan_id + "|" + state.step;
function openExportPop() {
  const p = state.plan, again = confirmedApproval(p), pop = $("exportPop");
  $("exportTitle").textContent = again ? "다시 내려받기" : "승인하고 내려받기";
  $("exportWhat").textContent = [$("caseName").textContent.split(" · 총생")[0], `계획 ${planNo(p.plan_id)}`, `${p.stages?.length ?? 0}단계`, "STL zip"].join(" · ");
  $("exportAsk").textContent = again ? "의사 승인으로 확정된 계획입니다. 같은 zip 을 다시 받습니다." : "의사 승인으로 확정합니다. 조건·3D·검토 결과를 확인하셨나요?";
  $("exportGo").textContent = again ? "다시 내려받기" : "확정하고 내려받기";
  if (again) $("exportSkip").hidden = true;
  // approval asks the dentist to tick the locked teeth (FDI) the server checks and records; what was asked on the
  // case and does not match, or was not read, is said here (the server refuses it either way)
  const lock = lockFdi(p), chk = p.request_check;
  $("exportLock").textContent = `고정 치아 ${lock.length ? lock.join(", ") + "번" : "없음"} — 확인했습니다`;
  $("exportLockRow").hidden = again;
  $("exportLockOk").checked = false;
  $("exportGo").disabled = !again;
  // an approval the server took off as it read the plan back (approval_voided), or one that does not count here
  const off = p.approval_voided ? "이전 승인을 해제했습니다 — " + p.approval_voided.reason_ko
    : p.approval && !again ? "고정 치아 확인이 맞지 않는 승인입니다. 고정 치아를 확인하고 다시 승인하세요." : "";
  const said = chk?.status === "mismatch" ? "요청과 다릅니다: " + chk.reasons_ko.join("; ")
    : chk?.status === "needs_check" ? "확인 필요 — 읽지 못한 요청: " + chk.unread_ko.join(", ") : "";
  $("exportCheck").hidden = again || !(off || said);
  $("exportCheck").textContent = [off, said].filter(Boolean).join(" · ");
  state.exportPopFor = exportPopKey();
  showPop(pop, true);
  const r = railExportBtn().getBoundingClientRect(), mid = r.top + r.height / 2;
  const top = Math.max(8, Math.min(innerHeight - pop.offsetHeight - 8, mid - pop.offsetHeight / 2));
  pop.style.top = `${Math.round(top)}px`; pop.style.left = `${Math.round(r.right + 10)}px`;
  pop.style.setProperty("--caret-y", `${Math.round(mid - top)}px`);
  railExportBtn().setAttribute("aria-expanded", "true");
  $("exportGo").focus({ preventScroll: true });
}
function closeExportPop() {
  if (!state.exportPopFor) return;
  state.exportPopFor = null;
  showPop($("exportPop"), false);
  railExportBtn().setAttribute("aria-expanded", "false");
  if (document.activeElement === document.body || $("exportPop").contains(document.activeElement)) railExportBtn().focus({ preventScroll: true });
}
$("exportBtn").addEventListener("click", () => { if (state.exportPopFor) closeExportPop(); else if (state.plan) openExportPop(); });
// capture on window: the 3D canvas and drawers.js take their presses for themselves at the document. A press on the 3D
// only closes the card (as drawers.js does): no tooth pick, no rotation
let exportSwallowUp = false;
window.addEventListener("pointerdown", (e) => {
  if (!state.exportPopFor || e.target.closest?.('#exportPop, #rail button[data-go="export"]')) return;
  closeExportPop();
  if (e.target.closest?.("#viewCanvas")) { exportSwallowUp = true; e.stopImmediatePropagation(); }
}, true);
window.addEventListener("pointerup", (e) => { if (exportSwallowUp) { exportSwallowUp = false; e.stopImmediatePropagation(); } }, true);
window.addEventListener("keydown", (e) => { if (state.exportPopFor && e.key === "Escape") closeExportPop(); }, true);
$("exportCancel").addEventListener("click", closeExportPop);
$("exportLockOk").addEventListener("change", () => { $("exportGo").disabled = !$("exportLockOk").checked; });
$("exportGo").addEventListener("click", async () => {
  closeExportPop();
  if (!confirmedApproval(state.plan)) await approveCurrent(true);
  if (confirmedApproval(state.plan) && $("stlLink").hasAttribute("href")) $("stlLink").click();
});
// The server builds the zip on every download (5–11 s on a real scan, #119), and a link click shows nothing until the
// file arrives: fetch it so the button says so meanwhile (and a refused download shows its reason instead of being
// saved as the file), then hand it to the browser. Every download path clicks the hidden link, which lands here.
// One transcript card per approval once its file arrives; 다시 내려받기 downloads again.
// One toast per approval once its file arrives (#20: no line in the agent panel, no 다시 내려받기; the rail button downloads again)
const doneCards = new Set();
function toast(text, ms = 3200) {
  const el = $("toast"); el.textContent = text; showPop(el, true);
  clearTimeout(toast.timer); toast.timer = setTimeout(() => showPop(el, false), ms);
}
// While the zip builds (from approval on, or for a download), the server reports its progress (contract: GET
// /api/plans/{id}/export-status → {building, done, total, ready, stl?, error?}), kept per plan id; the rail label and the
// link show N/총. One poll per plan, only while that plan is on screen and approved; it stops when the build is over.
const exportPolls = new Set();
async function pollExportStatus(planId) {
  if (exportPolls.has(planId)) return;
  exportPolls.add(planId);
  try {
    while (state.plan?.plan_id === planId && state.plan.approval) {
      const r = await fetch(`/api/plans/${encodeURIComponent(planId)}/export-status`);
      if (r.status === 404) return;   // the server has no progress to give
      const st = await r.json();
      state.exportStatus[planId] = st;
      if (state.plan?.plan_id === planId) updateActions();
      if (!st.building && state.stlBusy !== planId) return;
      await new Promise((res) => setTimeout(res, 500));
    }
  } catch { /* no progress to show */ }
  finally { exportPolls.delete(planId); }
}
async function downloadStl() {
  const p = state.plan, link = $("stlLink");
  if (state.stlBusy || !p?.approval || !link.hasAttribute("href")) return;
  state.stlBusy = p.plan_id; updateActions();
  pollExportStatus(p.plan_id);
  try {
    const r = await fetch(link.href);
    if (!r.ok) { const err = await r.json().catch(() => ({})); throw new Error(err.detail || ("HTTP " + r.status)); }
    const url = URL.createObjectURL(await r.blob());
    const a = document.createElement("a");
    a.href = url; a.download = `cualign_${p.plan_id}_stages.zip`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  } catch (err) {
    addMsg("error", `STL 내려받기 실패: ${err.message}`);
    return;
  } finally {
    state.stlBusy = null; updateActions();
  }
  const key = p.plan_id + "@" + p.approval.approved_at;
  toast(`${doneCards.has(key) ? "" : "승인 완료 · "}단계별 STL ${p.stages?.length ?? p.info?.n_stages ?? "?"}장을 내려받았습니다`);
  doneCards.add(key);
}
$("stlLink").addEventListener("click", (e) => { e.preventDefault(); downloadStl(); });
$("exportRetry").addEventListener("click", () => { delete state.exportStatus[state.plan?.plan_id]; downloadStl(); });   // stl.zip starts a new build
$("revokeBtn").addEventListener("click", () => approveCurrent());   // 승인 취소 (not the click event as `approve`)
$("reviewBtn").addEventListener("click", reviewCurrent);
$("cMonths").addEventListener("input", () => { const m = Number($("cMonths").value); $("cCap").value = m > 0 ? capOfMonths(m) : ""; });
$("cCap").addEventListener("input", () => { const cap = Number($("cCap").value); $("cMonths").value = cap > 0 ? monthsOfCap(cap) : ""; });
$("constraints").addEventListener("input", () => updateActions());
canvas.addEventListener("pointermove", onPointerMove);
canvas.addEventListener("dblclick", () => setView("occlusal"));   // back to the first view (#15)
canvas.addEventListener("pointerleave", () => { $("tip").hidden = true; });
for (const b of document.querySelectorAll(".view-rail button[data-view]")) b.addEventListener("click", () => setView(b.dataset.view));

export { closeExportPop, exportPopKey, pollExportStatus, toast };
