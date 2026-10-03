"""Confirmed planning constraints. Values are snapshots, never implicit resets.

Extraction is the dentist's prescription of which teeth to extract (#56): `extraction` lists them (Universal
numbers), () is non-extraction. The app never picks the teeth. IPR is prescribed per contact (#57): `ipr_surfaces`
lists (tooth, neighbour, mm) - the amount taken off that contact in total, half from each tooth - and the plan strips
those contacts only. With the list empty the older uniform rule applies (every contact of the teeth not in
`ipr_exclude`, `ipr_limit_mm` per contact). The cap is IPR_PER_SURFACE per tooth surface, so 2x that per contact. `allow_extraction` survives only as a derived value
in the output (screens, logs, evaluations read it); as an input, false clears the list and true without teeth is
refused with ExtractionTeethNeeded, so the caller asks the dentist which teeth.
`ipr_amounts` (optional, 직접 이동's right-click IPR) splits a prescribed contact per tooth face: (tooth, neighbour, mm
off that tooth's face); a contact listed there takes its faces' amounts (a face not listed takes none) instead of half
each, and its `ipr_surfaces` amount is their sum. Empty (the default) changes nothing and is left out of the dump.
"""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_serializer, model_validator
from .limits import IPR_PER_SURFACE, PREMOLARS

Tooth = Annotated[int, Field(strict=True, ge=2, le=15)]
Surface = tuple[Tooth, Tooth, float]         # (tooth, its neighbour, mm at that contact) - Universal numbers
Order = Literal["simultaneous", "anterior_first", "sequential"]
CONTACT_MAX_MM = 2 * IPR_PER_SURFACE         # a contact is two tooth surfaces


def _fdi_pair(a: int, b: int) -> str:
    from .fdi import to_fdi
    return f"{to_fdi(a)}-{to_fdi(b)}"


def surfaces_ko(surfaces) -> str:
    """IPR prescription as the dentist reads it: '11-21 0.4mm, 12-11 0.4mm' (FDI), '' when there is none."""
    return ", ".join(f"{_fdi_pair(a, b)} {mm:g}mm" for a, b, mm in surfaces)
EXTRACTABLE = frozenset(PREMOLARS)   # the layout (anchored on the first molars) supports premolar extraction only


class ExtractionTeethNeeded(ValueError):
    """Extraction was allowed without saying which teeth: the app does not choose them."""

    def __init__(self):
        super().__init__("발치할 치아 번호가 필요합니다(예: 14번과 24번). 앱은 발치 치아를 고르지 않습니다.")


def reason_ko(e: Exception) -> str:
    """A constraint error in the words shown to the dentist: the validator's own sentence, not pydantic's dump."""
    errors = getattr(e, "errors", None)
    if callable(errors):
        msgs = [str(x.get("msg", "")).removeprefix("Value error, ") for x in errors()]
        return "; ".join(m for m in msgs if m) or str(e)
    return str(e)


def _legacy_extraction(data: dict, current: tuple = ()) -> dict:
    """Map the old allow_extraction input onto extraction. false clears; true keeps a prescription already there and
    otherwise needs the teeth; a value contradicting the given list is an input error."""
    if "allow_extraction" not in data:
        return data
    data = dict(data)
    allow = data.pop("allow_extraction")
    if allow is None:
        return data
    teeth = data.get("extraction")
    if teeth is not None:
        if bool(allow) != bool(teeth):
            raise ValueError("allow_extraction과 extraction이 서로 다릅니다.")
        return data
    if not allow:
        data["extraction"] = ()
    elif not current:
        raise ExtractionTeethNeeded()
    return data


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    extraction: tuple[Tooth, ...] = ()
    lock: tuple[Tooth, ...] = ()
    ipr_exclude: tuple[Tooth, ...] = ()
    ipr_limit_mm: float = Field(default=IPR_PER_SURFACE, ge=0, le=IPR_PER_SURFACE, allow_inf_nan=False)
    ipr_surfaces: tuple[Surface, ...] = ()
    ipr_amounts: tuple[Surface, ...] = ()    # (tooth, neighbour, mm off this tooth's face) - see the module docstring
    stage_cap: int | None = Field(default=None, gt=0)
    order: Order = "simultaneous"

    @model_serializer(mode="wrap")
    def _dump_without_empty_amounts(self, handler):
        data = handler(self)
        if not self.ipr_amounts:   # without per-face amounts the dump is what it always was (saved plans, golden sets)
            data.pop("ipr_amounts", None)
        return data

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy(cls, data):
        return _legacy_extraction(data) if isinstance(data, dict) else data

    @field_validator("ipr_surfaces")
    @classmethod
    def contacts_within_the_cap(cls, value):
        """Each surface names two neighbouring teeth once, with a positive amount within the cap."""
        out: dict[tuple[int, int], float] = {}
        for a, b, mm in value:
            a, b = sorted((a, b))
            if b != a + 1:
                raise ValueError(f"IPR 접촉면은 이웃한 두 치아여야 합니다: {_fdi_pair(a, b)}")
            if (a, b) in out:
                raise ValueError(f"IPR 접촉면이 두 번 처방되었습니다: {_fdi_pair(a, b)}")
            if not mm > 0:
                raise ValueError(f"IPR 양은 0보다 커야 합니다: {_fdi_pair(a, b)} {mm:g}mm")
            if mm > CONTACT_MAX_MM + 1e-9:
                raise ValueError(f"IPR 처방 {_fdi_pair(a, b)} {mm:g}mm: 접촉면당 최대 {CONTACT_MAX_MM:g}mm"
                                 f"(치아 면당 {IPR_PER_SURFACE:g}mm)를 넘습니다")
            out[(a, b)] = float(mm)
        return tuple((a, b, mm) for (a, b), mm in sorted(out.items()))

    @field_validator("ipr_amounts")
    @classmethod
    def faces_within_the_cap(cls, value):
        """Each face names a tooth and its neighbour once, with an amount from 0 to the per-surface cap."""
        out: dict[tuple[int, int], float] = {}
        for t, n, mm in value:
            if abs(t - n) != 1:
                raise ValueError(f"IPR 면은 이웃한 두 치아여야 합니다: {_fdi_pair(t, n)}")
            if (t, n) in out:
                raise ValueError(f"IPR 면이 두 번 처방되었습니다: {_fdi_pair(t, n)}")
            if not 0 <= mm <= IPR_PER_SURFACE + 1e-9:
                raise ValueError(f"IPR 면당 양은 0~{IPR_PER_SURFACE:g}mm 입니다: {_fdi_pair(t, n)} {mm:g}mm")
            out[(t, n)] = float(mm)
        return tuple((t, n, mm) for (t, n), mm in sorted(out.items()))

    @model_validator(mode="after")
    def _faces_add_up_to_their_contact(self):
        contacts = {(a, b): mm for a, b, mm in self.ipr_surfaces}
        faces: dict[tuple[int, int], float] = {}
        for t, n, mm in self.ipr_amounts:
            faces[(min(t, n), max(t, n))] = faces.get((min(t, n), max(t, n)), 0.0) + mm
        for (a, b), total in faces.items():
            if (a, b) not in contacts:
                raise ValueError(f"처방되지 않은 접촉면의 IPR 면입니다: {_fdi_pair(a, b)}")
            if abs(total - contacts[(a, b)]) > 1e-6:
                raise ValueError(f"IPR 면의 합이 접촉면 처방과 다릅니다: {_fdi_pair(a, b)} {total:g}mm ≠ {contacts[(a, b)]:g}mm")
        return self

    def face_amounts(self) -> dict[tuple[int, int], float]:
        """{(tooth, neighbour): mm off that face} for every prescribed contact: its listed faces, else half each."""
        split = {(min(t, n), max(t, n)) for t, n, _ in self.ipr_amounts}
        out = {(t, n): mm for t, n, mm in self.ipr_amounts}
        for a, b, mm in self.ipr_surfaces:
            if (a, b) in split:
                out.setdefault((a, b), 0.0); out.setdefault((b, a), 0.0)
            else:
                out[(a, b)] = out[(b, a)] = mm / 2
        return out

    @model_validator(mode="after")
    def _prescribed_contacts_not_excluded(self):
        hit = sorted({t for a, b, _ in self.ipr_surfaces for t in (a, b) if t in self.ipr_exclude})
        if hit:
            from .fdi import label
            raise ValueError(f"IPR 제외 치아에 IPR 이 처방되었습니다: {label(hit)}")
        return self

    @field_validator("extraction", "lock", "ipr_exclude")
    @classmethod
    def unique_teeth(cls, value):
        return tuple(sorted(set(value)))

    def describe_ko(self) -> str:
        """The 조건 line the dentist reads, with FDI tooth numbers (#113). The server writes it so the model copies it
        instead of converting numbers itself (a live answer turned ipr_exclude 2..15 into 25..43)."""
        from .fdi import to_fdi
        from .limits import DAYS_PER_MONTH, WEAR_DAYS
        teeth = lambda ts: ", ".join(str(to_fdi(t)) for t in ts) + "번" if ts else "없음"   # noqa: E731
        cap = f"{self.stage_cap}단계(약 {round(self.stage_cap * WEAR_DAYS / DAYS_PER_MONTH, 1)}개월)" if self.stage_cap else "없음"
        order = {"simultaneous": "동시", "anterior_first": "앞니 먼저", "sequential": "순차"}[self.order]
        ipr = (f"IPR 처방 {surfaces_ko(self.ipr_surfaces)} · " if self.ipr_surfaces else
               f"IPR 제외 치아 {teeth(self.ipr_exclude)} · IPR 한도 면당 {self.ipr_limit_mm:g}mm · ")
        return f"발치 치아 {teeth(self.extraction)} · 고정 치아 {teeth(self.lock)} · {ipr}단계 상한 {cap} · 이동 순서 {order}"

    @field_validator("extraction")
    @classmethod
    def supported_extraction(cls, value):
        if set(value) - EXTRACTABLE:
            raise ValueError(f"소구치(4, 5, 12, 13) 발치만 계획할 수 있습니다: {sorted(set(value) - EXTRACTABLE)}")
        return value

    @model_validator(mode="after")
    def _not_both_locked_and_extracted(self):
        if set(self.extraction) & set(self.lock):
            raise ValueError(f"발치할 치아를 고정할 수 없습니다: {sorted(set(self.extraction) & set(self.lock))}")
        return self

    @computed_field
    @property
    def allow_extraction(self) -> bool:
        return bool(self.extraction)

    @classmethod
    def from_saved(cls, data: dict, removed=()) -> "Constraints":
        """A constraints snapshot read back from a plan file. Files written before #56 have allow_extraction and no
        teeth: an extraction plan's prescription is the teeth it removed; a plan that removed none only had
        extraction allowed, which is non-extraction now. Never picks teeth."""
        if "extraction" not in data and "allow_extraction" in data:
            allowed = bool(data["allow_extraction"])
            data = {k: v for k, v in data.items() if k != "allow_extraction"}
            # only an extraction the plan was allowed to make becomes its prescription; a forbidden one stays
            # forbidden, so the validator still reports it (#98 review)
            data["extraction"] = sorted(set(removed or ())) if allowed else []
        return cls.model_validate(data)

    def prescription(self) -> "Constraints":
        """The dentist's prescription part: extraction, lock, IPR exclusions and limit. The stage cap and the move order
        are plan conditions (they belong to the plan they were made for and reach the next turn through that plan), so
        a case keeps none of them between openings (answer-polish (8), 2026-09-28)."""
        return self.model_copy(update={"stage_cap": None, "order": "simultaneous"})

    def patched(self, changes: dict) -> "Constraints":
        changes = _legacy_extraction(changes, self.extraction)
        if "ipr_surfaces" in changes and "ipr_amounts" not in changes and self.ipr_amounts:
            # a new contact list (the form, a turn) keeps the face split only of the contacts it leaves as they were
            kept = {(min(int(a), int(b)), max(int(a), int(b))): float(mm) for a, b, mm in changes["ipr_surfaces"]}
            old = {(a, b): mm for a, b, mm in self.ipr_surfaces}
            changes = {**changes, "ipr_amounts": [f for f in self.ipr_amounts
                                                  if kept.get((min(f[0], f[1]), max(f[0], f[1]))) == old.get((min(f[0], f[1]), max(f[0], f[1])))]}
        return Constraints.model_validate({**self.model_dump(exclude={"allow_extraction"}), **changes})

    def check_case(self, ids):
        contacts = {t for a, b, _ in self.ipr_surfaces for t in (a, b)}
        if (set(self.lock) | set(self.ipr_exclude) | set(self.extraction) | contacts) - set(ids):
            raise ValueError("constraint refers to a tooth absent from this case")

    def teeth_fdi(self) -> dict:
        """The tooth fields as the set_constraints tool takes them (FDI, ToolConstraintPatch): the model keeps the
        existing teeth of a full list by copying these, never by converting the Universal fields itself."""
        from .fdi import to_fdi
        return {"extraction": [to_fdi(t) for t in self.extraction], "lock": [to_fdi(t) for t in self.lock],
                "ipr_exclude": [to_fdi(t) for t in self.ipr_exclude],
                "ipr_surfaces": [[to_fdi(a), to_fdi(b), mm] for a, b, mm in self.ipr_surfaces]}


class ConstraintPatch(BaseModel):
    """null means keep. A model filling every schema field with null must not change anything.

    Clearing is always explicit: [] empties a tooth list, clear_stage_cap drops the stage cap.
    Do not rely on exclude_unset here — a tool call arrives with every field present.
    """
    model_config = ConfigDict(extra="forbid")
    extraction: list[Tooth] | None = Field(
        default=None, description="Teeth the dentist prescribed to extract (Universal numbers). [] = non-extraction. "
                                  "Never choose teeth yourself: if extraction is allowed but no teeth were named, ask.")
    allow_extraction: bool | None = Field(
        default=None, description="Deprecated: prefer extraction. false = non-extraction; true alone is refused "
                                  "(the teeth are needed).")
    lock: list[Tooth] | None = None
    ipr_exclude: list[Tooth] | None = None
    ipr_limit_mm: float | None = Field(default=None, ge=0, le=IPR_PER_SURFACE, allow_inf_nan=False)
    ipr_surfaces: list[list[float]] | None = Field(
        default=None, description="IPR the dentist prescribed per contact, FDI numbers: [[11, 21, 0.4], [12, 11, 0.4]] = "
                                  "0.4 mm at the 11-21 contact and at the 12-11 contact (half off each tooth). Only these "
                                  "contacts are stripped. [] = no per-contact prescription (the uniform rule with "
                                  "ipr_exclude and ipr_limit_mm applies). Never invent contacts or amounts.")
    stage_cap: int | None = Field(default=None, gt=0)
    clear_stage_cap: bool = False
    order: Order | None = None

    @field_validator("ipr_surfaces")
    @classmethod
    def fdi_to_universal(cls, value):
        """The dentist's FDI pairs, stored as the app's Universal numbers (#113)."""
        if value is None:
            return None
        from .fdi import from_fdi
        out = []
        for row in value:
            if len(row) != 3:
                raise ValueError("IPR 처방은 [치아, 이웃 치아, mm] 세 값이어야 합니다")
            a, b, mm = row
            if a != int(a) or b != int(b):
                raise ValueError("IPR 처방의 치아 번호는 정수(FDI)여야 합니다")
            out.append([from_fdi(int(a)), from_fdi(int(b)), float(mm)])
        return out

    def changes(self):
        result = {k: v for k, v in self.model_dump().items()
                  if v is not None and k not in ("clear_stage_cap", "stage_cap")}
        if self.clear_stage_cap:
            if self.stage_cap is not None:
                raise ValueError("clear_stage_cap and stage_cap cannot be set together")
            result["stage_cap"] = None
        elif self.stage_cap is not None:
            result["stage_cap"] = self.stage_cap
        return result


FdiTooth = Annotated[int, Field(strict=True, ge=11, le=28)]   # upper-arch FDI; 19 and 20 are refused by from_fdi


class ToolConstraintPatch(ConstraintPatch):
    """The set_constraints tool's input: the tooth lists in the dentist's FDI numbers as spoken, like ipr_surfaces.
    The model only copies numbers; the conversion to Universal happens here, in the server (#113 · a live revision
    sent lock [13] for «13번 치아» and FDI 25 was locked). The UI form, the recorded answers and the MCP tool keep
    ConstraintPatch (Universal), so Tooth and the stored constraints do not change.

    The fields keep the numbers as sent and changes() converts them, once: the agent's tool call is validated twice
    (LangChain's args_schema, then NAT's input conversion), so a converting validator converted twice. That refused
    every ipr_surfaces prescription on the agent path (FDI 11 -> 8 -> «not an upper-arch FDI number: 8»)."""
    extraction: list[FdiTooth] | None = Field(
        default=None, description="Teeth the dentist prescribed to extract, FDI numbers as spoken: «14번과 24번 발치» = "
                                  "[14, 24]. [] = non-extraction. Never choose teeth yourself: if extraction is allowed "
                                  "but no teeth were named, ask.")
    lock: list[FdiTooth] | None = Field(
        default=None, description="Teeth that must not move, FDI numbers as spoken: «13번 치아는 움직이지 말고» = [13]. "
                                  "A full list: keep the teeth already in teeth_fdi.lock. [] clears it.")
    ipr_exclude: list[FdiTooth] | None = Field(
        default=None, description="Teeth with no IPR, FDI numbers as spoken. A full list: keep the teeth already in "
                                  "teeth_fdi.ipr_exclude. [] clears it.")

    @field_validator("extraction", "lock", "ipr_exclude")
    @classmethod
    def fdi_teeth(cls, value):
        """Upper-arch FDI numbers only (19 and 20 are not teeth), kept as sent."""
        from .fdi import from_fdi
        for t in value or ():
            from_fdi(t)
        return value

    @field_validator("ipr_surfaces")
    @classmethod
    def fdi_to_universal(cls, value):   # the same name replaces ConstraintPatch's converting validator
        """Checked as ConstraintPatch checks it, kept as sent (FDI)."""
        if value is not None:
            ConstraintPatch(ipr_surfaces=value)
        return value

    def changes(self):
        """ConstraintPatch.changes() in the app's Universal numbers: the one conversion of the tool's FDI input."""
        from .fdi import from_fdi
        out = super().changes()
        for field in ("extraction", "lock", "ipr_exclude"):
            if field in out:
                out[field] = [from_fdi(t) for t in out[field]]
        if "ipr_surfaces" in out:
            out["ipr_surfaces"] = ConstraintPatch(ipr_surfaces=out["ipr_surfaces"]).ipr_surfaces
        return out
