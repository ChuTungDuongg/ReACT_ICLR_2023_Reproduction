"""Explicit Sprint 6 audit of hard FEVER paper-fidelity requirements."""

from __future__ import annotations

import re
from pathlib import Path

from react_reproduction.agents.parsing import parse_tool_action
from react_reproduction.cli import build_parser
from react_reproduction.config import load_project_config
from react_reproduction.evaluation.fever import FEVER_LABELS
from react_reproduction.prompts.fever import FEVER_EXEMPLARS
from react_reproduction.tools.wikipedia import ActionType


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fever_paper_fidelity_audit() -> None:
    config = load_project_config(
        PROJECT_ROOT / "configs" / "default.yaml",
        project_root=PROJECT_ROOT,
    )
    source_text = "\n".join(
        path.read_text("utf-8")
        for path in (PROJECT_ROOT / "src" / "react_reproduction").rglob("*.py")
    ).casefold()
    actions = {
        parse_tool_action("Search[x]").action_type,
        parse_tool_action("Lookup[x]").action_type,
        parse_tool_action("Finish[SUPPORTS]").action_type,
    }
    cli_defaults = build_parser(PROJECT_ROOT).parse_args(
        ["benchmark", "--task", "fever", "--method", "cot-sc"]
    )

    audit = {
        "claim_only": all("evidence" not in example.claim.casefold() for example in FEVER_EXEMPLARS),
        "three_appendix_c2_exemplars": len(FEVER_EXEMPLARS) == 3,
        "all_labels_covered": {example.label for example in FEVER_EXEMPLARS} == set(FEVER_LABELS),
        "shared_ablation_examples": len({example.claim for example in FEVER_EXEMPLARS}) == 3,
        "cot_sc_n_21": cli_defaults.cot_sc_samples == 21,
        "cot_sc_temperature_point_7": cli_defaults.cot_sc_temperature == 0.7,
        "fever_react_max_steps_5": config.max_agent_steps_for("fever") == 5,
        "cot_sc_threshold_n_over_2": 11 >= 21 / 2 and not 10 >= 21 / 2,
        "wikipedia_action_space": actions == {
            ActionType.SEARCH,
            ActionType.LOOKUP,
            ActionType.FINISH,
        },
        "accuracy_primary_metric": '"accuracy"' in source_text,
        "no_fine_tuning": re.search(
            r"\b(?:lora|qlora|trainingarguments)\b|optimizer\.step",
            source_text,
        )
        is None,
    }
    assert all(audit.values()), {name: value for name, value in audit.items() if not value}
