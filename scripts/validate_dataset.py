#!/usr/bin/env python
"""Validate that the data on disk matches what the pipeline expects.

Run this BEFORE preprocess_dataset.py / generate_graphs.py / train.py to surface
missing files clearly. The validator never crashes on a missing path - it
classifies every check as [OK], [WARN], or [ERR] and prints a summary at the end.

Usage:
    python scripts/validate_dataset.py --config configs/local.yaml
    python scripts/validate_dataset.py --config configs/local.yaml --sample 10

Exit code: 0 if no [ERR] findings, 1 otherwise (suitable for CI).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

# Allow running as a script from the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from stroke_gat.config import load_config  # noqa: E402
from stroke_gat.data.service import DataService  # noqa: E402

logger = logging.getLogger(__name__)


# ANSI color codes (graceful fallback when the terminal doesn't support them)
class _C:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


class Validator:
    """Collects findings and prints a final summary."""

    def __init__(self, use_color: bool = True):
        self.findings: list[tuple[str, str]] = []  # (level, message)
        self.use_color = use_color

    def ok(self, msg: str) -> None:
        self._add("OK", msg, _C.GREEN)

    def warn(self, msg: str) -> None:
        self._add("WARN", msg, _C.YELLOW)

    def err(self, msg: str) -> None:
        self._add("ERR", msg, _C.RED)

    def info(self, msg: str) -> None:
        # Plain informational line, not counted in summary
        prefix = f"{_C.DIM}[..]{_C.END}" if self.use_color else "[..]"
        print(f"{prefix} {msg}")

    def _add(self, level: str, msg: str, color: str) -> None:
        self.findings.append((level, msg))
        prefix = f"{color}[{level:>4}]{_C.END}" if self.use_color else f"[{level:>4}]"
        print(f"{prefix} {msg}")

    def summary(self) -> int:
        """Print summary, return exit code (0 if no errors, 1 otherwise)."""
        n_ok = sum(1 for level, _ in self.findings if level == "OK")
        n_warn = sum(1 for level, _ in self.findings if level == "WARN")
        n_err = sum(1 for level, _ in self.findings if level == "ERR")
        print()
        print("=" * 60)
        print(f"Summary: {n_ok} OK, {n_warn} WARN, {n_err} ERR")
        print("=" * 60)
        if n_err > 0:
            print(f"{_C.RED}{_C.BOLD}Validation FAILED.{_C.END} Fix the [ERR] items before running the pipeline.")
            return 1
        if n_warn > 0:
            print(f"{_C.YELLOW}Validation passed with warnings.{_C.END} The pipeline will run but some subjects may be skipped.")
            return 0
        print(f"{_C.GREEN}{_C.BOLD}Validation passed.{_C.END} You're good to run the pipeline.")
        return 0


def _check_path(v: Validator, label: str, path: str | None, kind: str = "dir") -> bool:
    """Return True iff the path exists with the right kind."""
    if not path:
        v.err(f"{label}: path not configured (empty string in YAML)")
        return False
    p = Path(path)
    if not p.exists():
        v.err(f"{label}: path does not exist: {p}")
        return False
    if kind == "dir" and not p.is_dir():
        v.err(f"{label}: expected directory, found file: {p}")
        return False
    if kind == "file" and not p.is_file():
        v.err(f"{label}: expected file, found directory: {p}")
        return False
    v.ok(f"{label}: exists at {p}")
    return True


def validate_atlas(v: Validator, service: DataService) -> None:
    cfg = service._config

    if _check_path(v, "atlas_file", cfg.atlas_file, kind="file"):
        try:
            atlas_data, _ = service.load_atlas()
            import numpy as np

            n_unique = int(len(np.unique(atlas_data)))
            v.ok(f"atlas_file readable, {n_unique} unique label values")
        except Exception as e:
            v.err(f"atlas_file failed to load: {e}")

    if _check_path(v, "atlas_labels_file", cfg.atlas_labels_file, kind="file"):
        try:
            labels = service.load_atlas_labels()
            v.ok(f"atlas_labels_file: {len(labels)} region names parsed")
        except Exception as e:
            v.err(f"atlas_labels_file failed to parse: {e}")


def validate_raw_bids(v: Validator, service: DataService, sample_size: int) -> None:
    cfg = service._config

    if not cfg.raw_bids:
        v.warn("raw_bids: not configured. 3-class node classification requires the OpenNeuro ds004889 download. See README Step 1.")
        return

    if not _check_path(v, "raw_bids", cfg.raw_bids, kind="dir"):
        return

    # Subject discovery
    subjects = service.discover_subjects_bids()
    if not subjects:
        v.err(f"raw_bids: no sub-XXXX directories found at {cfg.raw_bids}")
        return
    v.ok(f"raw_bids: discovered {len(subjects)} subjects")

    # BIDS metadata files
    raw_path = Path(cfg.raw_bids)
    for fname in ("dataset_description.json", "participants.tsv"):
        fpath = raw_path / fname
        if fpath.exists():
            v.ok(f"raw_bids/{fname} present")
        else:
            v.warn(f"raw_bids/{fname} missing (not strictly required, but standard BIDS)")

    # Derivatives
    deriv = raw_path / "derivatives" / "lesion_masks"
    if deriv.exists() and deriv.is_dir():
        v.ok(f"derivatives/lesion_masks/ present")
    else:
        v.err(
            f"derivatives/lesion_masks/ NOT found at {deriv}. "
            "3-class labels (acute=1, chronic=2) cannot be derived without this."
        )

    # Sample N subjects, check modalities + masks
    sample = subjects[: min(sample_size, len(subjects))]
    v.info(f"Sampling {len(sample)} subjects for modality/mask spot-check...")

    expected_modalities = ["T1", "FLAIR", "ADC", "TRACE"]
    missing_per_modality = {m: 0 for m in expected_modalities}
    missing_per_mask = {m: 0 for m in ["General", "Acute", "Chronic"]}

    for subject_id in sample:
        try:
            modalities = service.load_subject_modalities(subject_id)
            for m in expected_modalities:
                if m not in modalities:
                    missing_per_modality[m] += 1
        except Exception as e:
            v.warn(f"{subject_id}: load_subject_modalities raised {type(e).__name__}: {e}")
            continue

        try:
            masks = service.load_subject_masks(subject_id)
            for m in ["General", "Acute", "Chronic"]:
                # zero-size array == missing
                if masks.get(m) is None or masks[m].size == 0:
                    missing_per_mask[m] += 1
        except Exception as e:
            v.warn(f"{subject_id}: load_subject_masks raised {type(e).__name__}: {e}")

    # Report sample-level findings
    n = len(sample)
    for m, missing in missing_per_modality.items():
        if missing == 0:
            v.ok(f"sample modality {m}: present in all {n} subjects")
        elif missing < n:
            v.warn(f"sample modality {m}: missing in {missing}/{n} subjects (those subjects will be skipped)")
        else:
            severity = v.err if m == "T1" else v.warn
            severity(f"sample modality {m}: missing in ALL {n} sampled subjects")

    for m, missing in missing_per_mask.items():
        if missing == 0:
            v.ok(f"sample mask {m}: present in all {n} subjects")
        elif missing < n:
            v.warn(f"sample mask {m}: missing in {missing}/{n} subjects")
        else:
            severity = v.err if m in ("Acute", "Chronic") else v.warn
            severity(f"sample mask {m}: missing in ALL {n} sampled subjects")


def validate_soop_normalized(v: Validator, service: DataService) -> None:
    cfg = service._config

    if not cfg.soop_normalized:
        v.warn("soop_normalized: not configured (only needed for notebook 01 quickstart)")
        return

    if not _check_path(v, "soop_normalized", cfg.soop_normalized, kind="dir"):
        return

    soop_path = Path(cfg.soop_normalized)
    n_flair = len(list(soop_path.glob("wsub-*_FLAIR.nii.gz")))
    n_lesion = len(list(soop_path.glob("bwsrsub-*_lesion.nii.gz")))

    if n_flair == 0:
        v.warn(f"soop_normalized: no wsub-*_FLAIR.nii.gz files found at {soop_path}")
    else:
        v.ok(f"soop_normalized: {n_flair} FLAIR files, {n_lesion} lesion masks")

    soop_subjects = service.discover_subjects_soop()
    if soop_subjects:
        v.ok(f"soop_normalized: {len(soop_subjects)} subject IDs parsed from lesion filenames")


def validate_outputs(v: Validator, service: DataService) -> None:
    """Check that output dirs are writable (or at least their parents)."""
    cfg = service._config
    for label, path in [
        ("output_preprocessed", cfg.output_preprocessed),
        ("output_graphs", cfg.output_graphs),
        ("output_models", cfg.output_models),
        ("output_visualizations", cfg.output_visualizations),
    ]:
        if not path:
            v.warn(f"{label}: path not configured")
            continue
        p = Path(path)
        if p.exists():
            v.ok(f"{label}: {p}")
        else:
            parent = p.parent
            if parent.exists() or str(parent) in (".", ""):
                v.ok(f"{label}: will be created at {p}")
            else:
                v.warn(f"{label}: neither {p} nor parent {parent} exists yet")


def main():
    parser = argparse.ArgumentParser(
        description="Validate that the data on disk matches the pipeline's expectations.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/local.yaml)")
    parser.add_argument("--sample", type=int, default=5, help="Number of subjects to spot-check (default: 5)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERR] config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    service = DataService(config.paths)

    v = Validator(use_color=not args.no_color)

    print(f"Validating against config: {config_path}\n")

    print("--- ATLAS ---")
    validate_atlas(v, service)

    print("\n--- RAW BIDS (OpenNeuro ds004889) ---")
    validate_raw_bids(v, service, sample_size=args.sample)

    print("\n--- SOOP NORMALIZED (optional) ---")
    validate_soop_normalized(v, service)

    print("\n--- OUTPUT DIRECTORIES ---")
    validate_outputs(v, service)

    sys.exit(v.summary())


if __name__ == "__main__":
    main()
