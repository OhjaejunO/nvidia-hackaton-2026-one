"""workspace/ is the one source of the agent definitions (OpenClaw workspace convention, DLI NemoClaw 03b).

The planner's instructions moved from configs/workflow.yml into workspace/AGENTS.md. The golden set (#73) is tuned to
that wording, so the string NAT hands the model must stay byte-identical to the one inlined before the move.
"""
import hashlib
import re
from pathlib import Path

from nat.utils.io.yaml_tools import yaml_load

from cualign.core import skills as S

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "workspace"
# sha256 of workspace/AGENTS.md after the #120 batch (7231 chars: the c98bef0 snapshot plus the #113 FDI table, the
# conditions_ko line, the IPR-cap rule and the Korean-question rule) and the v3 batch (7476 chars: IPR contacts are not
# a numbering puzzle, after the 2026-09-28 leak on the FDI-only sample text; then 8041 chars after the E2E rehearsal:
# select_plan after a comparison, no tooth numbers from tool results; then the #57 per-contact IPR prescription, and
# 9273 chars with the step flow's «Steps» paragraph, .report/15: setup and target turns stop and ask back, stages takes
# the context's target_id; then a per-contact amount up to 0.5 is inside the cap, after the 000001 recording refused 0.4; then the cap-already-met sentence, screen report 28; then 9794 chars at 14 days an aligner, wear-14d:
# the months-to-cap formula and the examples' stage counts and months; then 10073 chars when set_constraints took FDI
# tooth numbers (the server converts them, teeth_fdi keeps a full list) and the request_check sentence, after a live
# «13번 치아» revision locked FDI 25).
INSTRUCTIONS_SHA256 = "5fe7cc8b6da032182381604a3adb238f4b776e3b2582928c4a3f75a459fa50dc"
OPENCLAW = ROOT / "nemoclaw" / "openclaw-2026.7.1"
HANGUL = re.compile("[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")


def _instructions() -> str:
    return yaml_load(ROOT / "configs" / "workflow.yml")["workflow"]["additional_instructions"]


def test_nat_reads_the_instructions_from_workspace_agents_md():
    raw = (ROOT / "configs" / "workflow.yml").read_text(encoding="utf-8")
    assert "additional_instructions: file://../workspace/AGENTS.md" in raw
    assert "You are cuAlign, a staging-plan DRAFT assistant" not in raw   # no inline copy of the planner's instructions
    assert _instructions() == (WS / "AGENTS.md").read_text(encoding="utf-8")


def test_instructions_are_byte_identical_to_the_pre_move_snapshot():
    text = _instructions()
    assert len(text) == 10073
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == INSTRUCTIONS_SHA256
    assert hashlib.sha256((WS / "AGENTS.md").read_bytes()).hexdigest() == INSTRUCTIONS_SHA256


def test_agents_md_carries_the_core_rules():
    text = _instructions()
    assert "Never diagnose or prescribe" in text
    assert "FIRST set_constraints" in text and "Never call propose_target before" in text   # condition order
    assert "Call compare_strategies ONLY when the user asks to compare strategies" in text
    assert "Never silently relax these constraints" in text and "never loosen conditions" in text
    assert "No tool can approve a plan" in text and "the dentist must approve in the UI" in text


def test_convention_files_and_no_memory_folder():
    for name in ("SOUL.md", "AGENTS.md", "IDENTITY.md", "USER.md", "TOOLS.md", "HEARTBEAT.md", "MEMORY.md", "README.md",
                 "README.ko.md", "desk/AGENTS.md"):
        assert (WS / name).is_file(), name
    assert not (WS / "memory").exists()   # memory is off (MEMORY.md says why)


def _sentences(text: str) -> list[str]:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    parts = []
    for line in text.splitlines():
        cells = line.strip().strip("|").split("|") if line.lstrip().startswith("|") else [line]
        for cell in cells:
            parts += [s for s in re.split(r"(?<=[.!?])\s+", cell) if re.search(r"\w", s)]
    return parts


def test_english_workspace_files_follow_ste():
    # DESIGN.md §11 of the stable-agent-documentation-guidebook: English originals have no Hangul, and Vale counts
    # words as \S*\w\S* tokens. Context files keep 20 words or fewer, the README 25. AGENTS.md and the skill bodies
    # are out of scope: their bytes are tied to the golden set (#73, #96).
    for name, limit in (("SOUL.md", 20), ("IDENTITY.md", 20), ("USER.md", 20), ("TOOLS.md", 20), ("MEMORY.md", 20),
                        ("desk/AGENTS.md", 20), ("README.md", 25)):
        text = (WS / name).read_text(encoding="utf-8")
        assert not HANGUL.search(text), name
        assert "\u2014" not in text, name
        for s in _sentences(text):
            assert len(re.findall(r"\S*\w\S*", s)) <= limit, (name, s)


def test_rules_have_a_reason_and_do_not_copy_the_planner():
    for name in ("SOUL.md", "USER.md", "MEMORY.md", "desk/AGENTS.md"):
        text = (WS / name).read_text(encoding="utf-8")
        rules = [line for line in text.splitlines() if re.match(r"\d+\. \*\*", line)]
        assert rules, name
        assert all("(Reason: " in line for line in rules), name
        # The boundaries live in AGENTS.md and the skills; the desk files point there instead of copying them.
        for copied in ("Never diagnose", "Never silently relax", "No tool can approve", "이 계획은 초안입니다"):
            assert copied not in text, (name, copied)
    assert "skills/cualign-planner/SKILL.md" in (WS / "SOUL.md").read_text(encoding="utf-8")


DESK_FILES = ("SOUL.md", "IDENTITY.md", "USER.md", "TOOLS.md", "MEMORY.md", "desk/AGENTS.md")


def test_desk_files_point_only_to_files_in_the_desk_workspace():
    # R-004: the reader follows the links. The desk has its workspace and skills/, not guardrails/, openshell/, src/ or docs/.
    for name in DESK_FILES:
        text = re.sub(r"<!--.*?-->", "", (WS / name).read_text(encoding="utf-8"), flags=re.S)
        for ref in re.findall(r"`([^`\s]+)`", text):
            if "/" in ref or ref.endswith(".md"):
                assert ref == "memory/" or (WS / ref).exists(), (name, ref)
    tools = (WS / "TOOLS.md").read_text(encoding="utf-8")
    assert "skills/cualign-planner/SKILL.md" in tools and "null" not in tools   # the skill holds when and how


def test_readme_lists_the_sources_outside_the_desk_for_the_maintainer():
    readme = (WS / "README.md").read_text(encoding="utf-8")
    for path in ("src/cualign/server/mcp_server.py", "src/cualign/agent/register.py", "configs/workflow.yml",
                 "guardrails/prompts.yml", "guardrails/policy/cualign_clinical_scope_v1.0.0.md",
                 "openshell/policy.yaml", "openshell/server-policy.yaml", "docs/nemoclaw.md"):
        assert path in readme and (ROOT / path).is_file(), path
    assert "--deny-tool" in readme
    # Install step 4 points to the SOUL.md lock status; the record must be there.
    assert "records if the sandbox can lock the file" in readme
    assert "| `SOUL.md` 읽기 전용 | 강제되지 않음 |" in (ROOT / "docs" / "nemoclaw.md").read_text(encoding="utf-8")
    for name in ("SOUL.md", "TOOLS.md"):   # the runtime and tool sources are listed once, in the README
        assert "--deny-tool" not in (WS / name).read_text(encoding="utf-8"), name


def test_desk_agents_md_holds_only_the_desk_operating_rules():
    # Issue #96: no identity sentence. The desk follows SOUL.md, IDENTITY.md and the planner skill. The install copies
    # the file to AGENTS.md in the desk workspace, in place of the NemoClaw default that writes memory/ notes.
    text = re.sub(r"<!--.*?-->", "", (WS / "desk" / "AGENTS.md").read_text(encoding="utf-8"), flags=re.S)
    assert "You are" not in text
    for ref in ("`SOUL.md`", "`IDENTITY.md`", "`skills/cualign-planner/SKILL.md`", "`MEMORY.md`", "`HEARTBEAT.md`"):
        assert ref in text, ref
    assert "memory/" not in text   # no daily notes (MEMORY.md says why)
    readme = (WS / "README.md").read_text(encoding="utf-8")
    assert "Copy `desk/AGENTS.md` to `/sandbox/.openclaw/workspace/AGENTS.md`." in readme


def test_vendored_openclaw_template_matches_its_record():
    # The OpenClaw version and the sha256 of the copy are recorded once, in nemoclaw/openclaw-2026.7.1/README.md.
    rows = re.findall(r"^\| `(templates/[^`]+)` \| `[^`]+` \| `([0-9a-f]{64})` \|", (OPENCLAW / "README.md").read_text(encoding="utf-8"),
                      flags=re.M)
    assert {path for path, _ in rows} == {"templates/HEARTBEAT.md"}
    assert [p.name for p in (OPENCLAW / "templates").iterdir()] == ["HEARTBEAT.md"]   # no copy that nothing uses
    for path, digest in rows:
        assert hashlib.sha256((OPENCLAW / path).read_bytes()).hexdigest() == digest, path


def test_heartbeat_md_is_the_openclaw_default():
    # Comments only, so OpenClaw skips the heartbeat model call.
    assert (WS / "HEARTBEAT.md").read_bytes() == (OPENCLAW / "templates" / "HEARTBEAT.md").read_bytes()


def test_readme_translation_records_the_hash_of_its_original():
    ko = (WS / "README.ko.md").read_text(encoding="utf-8")
    first = ko.splitlines()[0]
    want = hashlib.sha256((WS / "README.md").read_bytes()).hexdigest()
    assert first == f"<!-- source: README.md sha256: {want} -->"
    assert "\u2014" not in ko   # fluent-korean: no em dash
    readme = (WS / "README.md").read_text(encoding="utf-8")
    assert "It is not a template for a repository context file." in readme


def test_skills_live_only_in_workspace():
    assert S.SKILLS_DIR == WS / "skills"
    assert {"cualign-clinical-rules", "cualign-planner"} <= set(S.installed())
    assert S.read_skill("cualign-planner")["name"] == "cualign-planner"
    # No copy or redirect left at the old places (an empty folder git no longer tracks does not count).
    for old in (ROOT / "skills", ROOT / "nemoclaw" / "cualign-planner"):
        assert not [p for p in old.rglob("*") if p.is_file()], old
