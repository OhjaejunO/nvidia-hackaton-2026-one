"""Does what the dentist asked on a case agree with the conditions a plan was made with? (#113 follow-up, 2026-10-03)

A live revision «13번 치아는 움직이지 말고 다시 짜줘.» locked FDI 25: the model sent 13 to a tool that took Universal
numbers, and the rules, the reviewer and the rails all passed the plan, because each checks the stored conditions and
none reads the dentist's sentence. This reads the tooth numbers a sentence names for a field (고정 · IPR 제외 · 발치),
added or released, and the months of a time limit, and compares them with a plan's conditions. Approval opens only on
a match (Store.approve).

Only numbers are read: upper-arch FDI 11..18 and 21..28, right before the field's words, or with «도» and no verb after
it («기존 25번 고정에 13번도»), taking the field and action of the number before it in the same sentence. A verb after
«도» is that number's own: «25번 고정하고 13번도 빼줘» releases 13.

What is not read is never a match (Codex re-check 2026-10-03 (3): «13번도 같이 해줘» dropped 13 as unread, and the plan
that left 13 out passed). A tooth number no field takes, a negated verb («고정하지 마»), a numbered tooth outside the
upper arch («35번»), a time the months pattern does not take («1년»), and a field the sentences name only in words
(«오른쪽 송곳니») that the plan changed, all make the result «확인 필요» (needs_check), which approval refuses like a
mismatch. A later sentence that reads the same tooth (or months), or a hand edit of it, settles it.

The sentences add up per case, in order (check_case): every turn's sentence, setup turns included, and every condition
the dentist changed by hand on the screen (an edit sets the fields it changed), folded onto the conditions the first of
them opened with. A plan is compared with all of them, whatever path made it (Codex re-check 2026-10-03 (3): a sentence
of a later turn, or of a setup turn that made no target, was lost).
"""
from __future__ import annotations

import re

from .fdi import from_fdi, label
from .limits import stage_cap_from_months

FIELDS_KO = {"lock": "고정 치아", "ipr_exclude": "IPR 제외 치아", "extraction": "발치 치아"}

# a tooth number the dentist reads (FDI upper arch), not part of an IPR contact (11-21), an amount or a duration
TOOTH = re.compile(r"(?<![\d.\-~])(1[1-8]|2[1-8])(?![\d.\-~])(?!\s*(?:개월|mm|단계|일|도|%|주|년|회|장))")
# what may stand between a tooth number and the words of its field: more numbers, 번, 치아, particles, separators
GAP = re.compile(r"^(?:\s|번|치아|들|기존|앞니|[은는도을를이가만의에]|[,·/]|과|와|및|랑|이랑|하고|그리고|\d{2})*$")
CUES = [   # (field, action, words); the first that follows a number with only GAP between takes it
    ("lock", "release", re.compile(r"고정\s*(?:을|은|는|도)?\s*(?:풀|해제|빼)|움직여도")),
    ("lock", "add", re.compile(r"움직이지\s*(?:말|마|않)|고정")),
    ("ipr_exclude", "add", re.compile(r"(?:IPR|ipr)\s*(?:은|는|을|를|도)?\s*(?:하지\s*(?:말|마|않)|빼|제외|금지|안\s*(?:해|함|하))")),
    ("extraction", "add", re.compile(r"(?<!비)발치(?!\s*(?:없이|는\s*(?:절대\s*)?안|금지|하지|않|안\s*(?:해|돼)))")),
]
IPR_THEN_TEETH = re.compile(r"(?:IPR|ipr)[^.?!\n]*$")          # «IPR 은 앞니 12·11·21·22번 빼고»: IPR earlier in the clause
EXCEPT = re.compile(r"빼고|빼|제외")
CLEAR = {"lock": re.compile(r"고정\s*(?:치아\s*)?(?:을|은|는)?\s*(?:모두|전부|다)\s*(?:풀|해제|빼)"),
         "extraction": re.compile(r"비발치|발치\s*(?:없이|는\s*(?:절대\s*)?안|금지|하지\s*(?:말|않)|안\s*(?:해|돼))")}
WORDS = {"lock": re.compile(r"고정|움직이지|움직여도"), "ipr_exclude": re.compile(r"IPR|ipr"),
         "extraction": re.compile(r"발치")}
MONTHS = re.compile(r"(?<!\d)(\d{1,2})\s*개월\s*(?:안|이내|내|까지|로)")
ALSO = re.compile(r"^\s*(?:번|치아)\s*(?:치아\s*)?도")                # «13번도», «13번 치아도»
SENTENCE_END = re.compile(r"[.?!\n]")
CLAUSE_END = re.compile(r"[.?!\n,·]|\d")                               # what follows a verb: up to the next number
NO_VERB = re.compile(r"\s*(?:요|해\s*(?:줘|주세요|줘요)?|부탁\S*)?\s*")   # «13번도», «13번도 해줘»: the one before
AWAY = {"lock": ("release", re.compile(r"^\s*(?:빼|풀|해제)")),         # «13번도 빼줘» after a lock releases it
        "ipr_exclude": ("add", re.compile(r"^\s*(?:빼|제외)"))}
# not read, so never a match: a negated verb in the number's clause, a numbered tooth TOOTH does not take, a time
NEGATED = re.compile(r"지\s*(?:는\s*)?(?:말|마|않)|안\s*(?:해|돼|되|하|함|풀|빼|움직)|않|못")
NUMBERED = re.compile(r"(?<![\d.\-~])\d{1,2}\s*번")
DURATION = re.compile(r"(?<![\d.])\d{1,2}\s*(?:개월|년|주)")
SNIPPET_END = re.compile(r"[.?!\n,]")


def _snippet(text: str, at: int) -> str:
    return "«" + SNIPPET_END.split(text[at:], 1)[0].strip()[:30] + "»"


def read_request(text: str) -> dict:
    """What the sentence asks of each field: {field: {"add": {Universal}, "release": {Universal}, "clear": bool}},
    "months" (the time limit, None when not stated) and "unread": [(Universal tooth | None, "time" | "tooth", what
    the dentist wrote)] for what it says that is not read. Numbers are FDI as the dentist speaks them."""
    out = {f: {"add": set(), "release": set(), "clear": bool(CLEAR[f].search(text)) if f in CLEAR else False}
           for f in FIELDS_KO}
    unread = []
    last = None                                   # (end, field, action) of the number read before this one
    teeth = list(TOOTH.finditer(text))
    for m in teeth:
        tooth, rest = from_fdi(int(m.group(1))), text[m.end():]
        best, negated = None, False
        for field, action, cue in CUES:
            c = cue.search(rest)
            if c and GAP.match(rest[:c.start()]) and (best is None or c.start() < best[0]):
                best = (c.start(), field, action, c.end())
        c = EXCEPT.search(rest)
        if (c and GAP.match(rest[:c.start()]) and IPR_THEN_TEETH.search(text[:m.start()])
                and (best is None or c.start() < best[0])):
            best = (c.start(), "ipr_exclude", "add", c.end())
        also = ALSO.match(rest)
        if best is not None:
            negated = bool(NEGATED.search(CLAUSE_END.split(rest[best[3]:], 1)[0]))   # «13번 고정하지 마»
        elif last and also and not SENTENCE_END.search(text[last[0]:m.start()]):
            tail = CLAUSE_END.split(rest[also.end():], 1)[0]
            cue = next(((f, a) for f, a, words in CUES if words.search(tail)), None)   # «13번도 꼭 고정»
            away = AWAY.get(last[1])
            negated = bool(NEGATED.search(tail))
            if cue:
                best = (None, *cue)
            elif away and away[1].match(tail):
                best = (None, last[1], away[0])
            elif NO_VERB.fullmatch(tail):
                best = (None, last[1], last[2])
            # another verb («같이 해줘», «봐줘», or 빼줘 after 발치) is no field's: the number stays unread
        if best is None or negated:
            unread.append((tooth, "tooth", _snippet(text, m.start())))
            continue
        out[best[1]][best[2]].add(tooth)
        last = (m.end(), best[1], best[2])
    starts = {m.start() for m in teeth}
    unread += [(None, "tooth", _snippet(text, m.start())) for m in NUMBERED.finditer(text) if m.start() not in starts]
    months = list(MONTHS.finditer(text))
    taken = [range(m.start(), m.end()) for m in months]
    unread += [(None, "time", _snippet(text, m.start())) for m in DURATION.finditer(text)
               if not any(m.start() in r for r in taken)]
    if len({int(m.group(1)) for m in months}) > 1:     # two different limits in one sentence: which one is not read
        unread += [(None, "time", _snippet(text, m.start())) for m in months]
        months = []
    out["months"] = int(months[0].group(1)) if months else None
    out["unread"] = unread
    return out


def ask(text: str, opened) -> dict:
    """A sentence of a turn and the conditions the turn opened with, as Store keeps it."""
    return {"kind": "ask", "text": text, "opened": opened}


def edit(before, after) -> dict:
    """Conditions the dentist changed by hand on the screen (the form, a setup edit), as Store keeps it."""
    return {"kind": "edit", "before": before, "after": after}


def check(text: str, before, after) -> dict:
    """check_case for one sentence: `text` asked on `before`, the plan made with `after` (Constraints)."""
    return check_case([ask(text, before)], after)


def check_case(entries, after) -> dict | None:
    """{"status": "match" | "mismatch" | "needs_check", "reasons_ko": [...], "unread_ko": [...], "unchecked": [fields]}
    for a case's sentences and hand edits in order (ask, edit) against a plan's conditions `after`; None when there is
    no sentence (nothing to compare: a rule-API or MCP-only case). Reasons are Korean, tooth numbers FDI."""
    if not any(e["kind"] == "ask" for e in entries):
        return None
    base = entries[0]["opened"] if entries[0]["kind"] == "ask" else entries[0]["before"]
    want = {f: set(getattr(base, f)) for f in FIELDS_KO}
    named, worded, months, unread = set(), set(), None, []
    for e in entries:
        if e["kind"] == "edit":
            for f in FIELDS_KO:
                b, a = set(getattr(e["before"], f)), set(getattr(e["after"], f))
                if a != b:
                    want[f] = a
                    named.add(f)
                    unread = [u for u in unread if u[0] not in a ^ b]          # the dentist set those teeth by hand
            if e["before"].stage_cap != e["after"].stage_cap:
                months, unread = None, [u for u in unread if u[1] != "time"]
            continue
        asked = read_request(e["text"])
        for f in FIELDS_KO:
            w = asked[f]
            if w["add"] or w["release"] or w["clear"]:
                named.add(f)
                if (f == "extraction" and w["add"]) or w["clear"]:
                    want[f] = set(w["add"])            # a prescription names all the teeth to extract
                else:
                    want[f] = (want[f] - w["release"]) | w["add"]   # lock and ipr_exclude are full lists
                unread = [u for u in unread if u[0] not in w["add"] | w["release"]]   # said again, and read
            elif WORDS[f].search(e["text"]):
                worded.add(f)
        if asked["months"] is not None:
            months, unread = asked["months"], [u for u in unread if u[1] != "time"]
        unread += asked["unread"]
    reasons, unchecked = [], []
    for f, ko in FIELDS_KO.items():
        a, b = set(getattr(after, f)), set(getattr(base, f))
        if f not in named:
            if a != b:
                if f in worded:
                    unchecked.append(f)   # the sentences talk about the field without numbers: not read
                else:
                    reasons.append(f"{ko}가 요청 문장에 없이 바뀌었습니다({label(sorted(b)) if b else '없음'} → "
                                   f"{label(sorted(a)) if a else '없음'})")
            continue
        if missing := sorted(want[f] - a):
            reasons.append(f"{ko}에 요청한 {label(missing)}이 없습니다")
        if extra := sorted(a - want[f]):
            reasons.append(f"{ko}에 요청에 없던 {label(extra)}이 들어갔습니다")
    if months is not None:
        cap = stage_cap_from_months(months)
        if after.stage_cap != cap:
            reasons.append(f"요청한 {months}개월(단계 상한 {cap}단계)과 저장된 단계 상한"
                           f"({f'{after.stage_cap}단계' if after.stage_cap else '없음'})이 다릅니다")
    unread_ko = list(dict.fromkeys(u[2] for u in unread))
    status = "mismatch" if reasons else "needs_check" if unread_ko or unchecked else "match"
    return {"status": status, "reasons_ko": reasons, "unread_ko": unread_ko, "unchecked": unchecked}
