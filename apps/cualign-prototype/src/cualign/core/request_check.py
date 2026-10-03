"""Does the turn's sentence agree with the conditions its plan was made with? (#113 follow-up, 2026-10-03)

A live revision «13번 치아는 움직이지 말고 다시 짜줘.» locked FDI 25: the model sent 13 to a tool that took Universal
numbers, and the rules, the reviewer and the rails all passed the plan, because each checks the stored conditions and
none reads the dentist's sentence. This reads the tooth numbers the sentence names for a field (고정 · IPR 제외 · 발치),
added or released, and the months of a time limit, and compares them with the change the turn made to the conditions
(the conditions the turn opened with -> the plan's). A disagreement blocks approval (Store.approve).

Only numbers are read: upper-arch FDI 11..18 and 21..28, right before the field's words, or with «도» and no field words
after it («기존 25번 고정에 13번도»), taking the field and action of the number before it in the same sentence (Codex
review 2026-10-03: read only forwards, the plan's own sentence blocked the right plan and passed the one that left 13
out). A tooth named by its name
(«오른쪽 송곳니») is not read: a field the sentence names without numbers that the turn changed is listed as unchecked,
never as a match.
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


def read_request(text: str) -> dict:
    """What the sentence asks of each field: {field: {"add": {Universal}, "release": {Universal}, "clear": bool}},
    plus "months" (the time limit, None when not stated). Numbers are FDI as the dentist speaks them."""
    out = {f: {"add": set(), "release": set(), "clear": bool(CLEAR[f].search(text)) if f in CLEAR else False}
           for f in FIELDS_KO}
    last = None                                   # (end, field, action) of the number read before this one
    for m in TOOTH.finditer(text):
        tooth, rest = from_fdi(int(m.group(1))), text[m.end():]
        best = None
        for field, action, cue in CUES:
            c = cue.search(rest)
            if c and GAP.match(rest[:c.start()]) and (best is None or c.start() < best[0]):
                best = (c.start(), field, action)
        c = EXCEPT.search(rest)
        if (c and GAP.match(rest[:c.start()]) and IPR_THEN_TEETH.search(text[:m.start()])
                and (best is None or c.start() < best[0])):
            best = (c.start(), "ipr_exclude", "add")
        if best is None and last and ALSO.match(rest) and not SENTENCE_END.search(text[last[0]:m.start()]):
            best = (None, last[1], last[2])
        if best is not None:
            out[best[1]][best[2]].add(tooth)
            last = (m.end(), best[1], best[2])
    months = MONTHS.search(text)
    out["months"] = int(months.group(1)) if months else None
    return out


def check(text: str, before, after) -> dict:
    """{"status": "match" | "mismatch", "reasons_ko": [...], "unchecked": [fields]} for the sentence `text` and the
    turn's change of conditions `before` -> `after` (Constraints). Reasons are Korean, tooth numbers FDI."""
    asked, reasons, unchecked = read_request(text), [], []
    for field, ko in FIELDS_KO.items():
        want, b, a = asked[field], set(getattr(before, field)), set(getattr(after, field))
        named = want["add"] | want["release"]
        if not named and not want["clear"]:
            if a != b:
                if WORDS[field].search(text):
                    unchecked.append(field)   # the sentence talks about the field without numbers: not read
                else:
                    reasons.append(f"{ko}가 요청 문장에 없이 바뀌었습니다({label(sorted(b)) if b else '없음'} → "
                                   f"{label(sorted(a)) if a else '없음'})")
            continue
        if field == "extraction" and want["add"]:
            expected = want["add"]                      # a prescription names all the teeth to extract
        elif want["clear"]:
            expected = set(want["add"])
        else:                                           # lock and ipr_exclude are full lists: the rest stays
            expected = (b - want["release"]) | want["add"]
        if missing := sorted(expected - a):
            reasons.append(f"{ko}에 요청한 {label(missing)}이 없습니다")
        if extra := sorted(a - expected):
            reasons.append(f"{ko}에 요청에 없던 {label(extra)}이 들어갔습니다")
    if asked["months"] is not None:
        cap = stage_cap_from_months(asked["months"])
        if after.stage_cap != cap:
            reasons.append(f"요청한 {asked['months']}개월(단계 상한 {cap}단계)과 저장된 단계 상한"
                           f"({f'{after.stage_cap}단계' if after.stage_cap else '없음'})이 다릅니다")
    return {"status": "mismatch" if reasons else "match", "reasons_ko": reasons, "unchecked": unchecked}
