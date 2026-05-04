#!/usr/bin/env python
"""Sanity-check the dataset against the pipeline's expectations.

Usage:
    python scripts/validate_dataset.py --config configs/local.yaml
    python scripts/validate_dataset.py --config configs/local.yaml --sample 10

Run this before preprocess_dataset.py / generate_graphs.py / train.py so
missing files are reported up front. Prints one [OK] / [WARN] / [ERR] line
per check; exits non-zero if any [ERR].
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from stroke_gat.config import load_config  # noqa: E402
from stroke_gat.data.service import DataService  # noqa: E402


class Report:
    """Accumulates findings and prints a final summary."""

    def __init__(self) -> None:
        self.findings: list[tuple[str, str]] = []

    def _add(self, level: str, msg: str) -> None:
        self.findings.append((level, msg))
        print(f"[{level:>4}] {msg}")

    def ok(self, msg: str) -> None:
        self._add("OK", msg)

    def warn(self, msg: str) -> None:
        self._add("WARN", msg)

    def err(self, msg: str) -> None:
        self._add("ERR", msg)

    def summary(self) -> int:
        n_ok = sum(1 for level, _ in self.findings if level == "OK")
        n_warn = sum(1 for level, _ in self.findings if level == "WARN")
        n_err = sum(1 for level, _ in self.findings if level == "ERR")
        print()
        print(f"Summary: {n_ok} OK, {n_warn} WARN, {n_err} ERR")
        if n_err:
            print("Validation failed; fix the [ERR] items before running the pipeline.")
            return 1
        if n_warn:
            print("Validation passed with warnings; some subjects may be skipped.")
        else:
            print("Validation passed.")
        return 0


def _check_path(r: Report, label: str, path: str | None, kind: str = "dir") -> bool:
    if not path:
        r.err(f"{label}: not configured (empty in YAML)")
        return False
    p = Path(path)
    if not p.exists():
        r.err(f"{label}: does not exist: {p}")
        return False
    if kind == "dir" and not p.is_dir():
        r.err(f"{label}: expected a directory, found a file: {p}")
        return False
    if kind == "file" and not p.is_file():
        r.err(f"{label}: expected a file, found a directory: {p}")
        return False
    r.ok(f"{label}: {p}")
    return True


def check_atlas(r: Report, service: DataService) -> None:
    cfg = service._config
    if _check_path(r, "atlas_file", cfg.atlas_file, kind="file"):
        try:
            import numpy as np
            atlas, _ = service.load_atlas()
            r.ok(f"atlas readable, {len(np.unique(atlas))} unique label values")
        except Exception as e:
            r.err(f"atlas failed to load: {e}")
    if _check_path(r, "atlas_labels_file", cfg.atlas_labels_file, kind="file"):
        try:
            labels = service.load_atlas_labels()
            r.ok(f"atlas labels parsed: {len(labels)} regions")
        except Exception as e:
            r.err(f"atlas labels failed to parse: {e}")


def check_raw_bids(r: Report, service: DataService, sample_size: int) -> None:
    cfg = service._config
    if not cfg.raw_bids:
        r.warn("raw_bids: not configured. 3-class training requires the OpenNeuro ds004889 download (see README).")
        return
    if not _check_path(r, "raw_bids", cfg.raw_bids, kind="dir"):
        return

    subjects = service.discover_subjects_bids()
    if not subjects:
        r.err(f"raw_bids: no sub-XXXX directories under {cfg.raw_bids}")
        return
    r.ok(f"raw_bids: {len(subjects)} subjects discovered")

    raw = Path(cfg.raw_bids)
    for fname in ("dataset_description.json", "participants.tsv"):
        if (raw / fname).exists():
            r.ok(f"raw_bids/{fname} present")
        else:
            r.warn(f"raw_bids/{fname} missing (BIDS-recommended, not strictly required)")

    deriv = raw / "derivatives" / "lesion_masks"
    if deriv.is_dir():
        r.ok("derivatives/lesion_masks/ present")
    else:
        r.err(f"derivatives/lesion_masks/ not found at {deriv}; 3-class labels cannot be derived without it.")

    sample = subjects[: min(sample_size, len(subjects))]
    print(f"[ ..] Spot-checking {len(sample)} subjects for modalities and masks")

    expected_mods = ["T1", "FLAIR", "ADC", "TRACE"]
    miss_mod = {m: 0 for m in expected_mods}
    miss_mask = {m: 0 for m in ["General", "Acute", "Chronic"]}

    for sid in sample:
        try:
            mods = service.load_subject_modalities(sid)
        except Exception as e:
            r.warn(f"{sid}: load_subject_modalities raised {type(e).__name__}: {e}")
            mods = {}
        for m in expected_mods:
            if m not in mods:
                miss_mod[m] += 1
        try:
            masks = service.load_subject_masks(sid)
        except Exception as e:
            r.warn(f"{sid}: load_subject_masks raised {type(e).__name__}: {e}")
            masks = {}
        for m in ["General", "Acute", "Chronic"]:
            arr = masks.get(m)
            if arr is None or arr.size == 0:
                miss_mask[m] += 1

    n = len(sample)
    for m, missing in miss_mod.items():
        if missing == 0:
            r.ok(f"sample modality {m}: present in all {n}")
        elif missing < n:
            r.warn(f"sample modality {m}: missing in {missing}/{n} (those subjects will be skipped)")
        else:
            (r.err if m == "T1" else r.warn)(f"sample modality {m}: missing in all {n} sampled subjects")
    for m, missing in miss_mask.items():
        if missing == 0:
            r.ok(f"sample mask {m}: present in all {n}")
        elif missing < n:
            r.warn(f"sample mask {m}: missing in {missing}/{n}")
        else:
            (r.err if m in ("Acute", "Chronic") else r.warn)(f"sample mask {m}: missing in all {n}")


def check_soop_normalized(r: Report, service: DataService) -> None:
    cfg = service._config
    if not cfg.soop_normalized:
        r.warn("soop_normalized: not configured (only needed for notebook 01 quickstart)")
        return
    if not _check_path(r, "soop_normalized", cfg.soop_normalized, kind="dir"):
        return
    soop = Path(cfg.soop_normalized)
    n_flair = len(list(soop.glob("wsub-*_FLAIR.nii.gz")))
    n_lesion = len(list(soop.glob("bwsrsub-*_lesion.nii.gz")))
    if n_flair == 0:
        r.warn(f"soop_normalized: no wsub-*_FLAIR.nii.gz under {soop}")
    else:
        r.ok(f"soop_normalized: {n_flair} FLAIR files, {n_lesion} lesion masks")
    soop_subjects = service.discover_subjects_soop()
    if soop_subjects:
        r.ok(f"soop_normalized: {len(soop_subjects)} subject IDs parsed from lesion filenames")


def check_outputs(r: Report, service: DataService) -> None:
    cfg = service._config
    for label, path in (
        ("output_preprocessed", cfg.output_preprocessed),
        ("output_graphs", cfg.output_graphs),
        ("output_models", cfg.output_models),
        ("output_visualizations", cfg.output_visualizations),
    ):
        if not path:
            r.warn(f"{label}: not configured")
            continue
        p = Path(path)
        if p.exists():
            r.ok(f"{label}: {p}")
        else:
            parent = p.parent
            if parent.exists() or str(parent) in (".", ""):
                r.ok(f"{label}: will be created at {p}")
            else:
                r.warn(f"{label}: neither {p} nor parent {parent} exists yet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="Path to a YAML config (e.g. configs/local.yaml).")
    parser.add_argument("--sample", type=int, default=5, help="Number of subjects to spot-check (default 5).")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ ERR] config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    service = DataService(config.paths)
    r = Report()

    print(f"Validating against {config_path}\n")
    print("-- atlas --")
    check_atlas(r, service)
    print("\n-- raw BIDS (OpenNeuro ds004889) --")
    check_raw_bids(r, service, sample_size=args.sample)
    print("\n-- SOOP normalized (optional) --")
    check_soop_normalized(r, service)
    print("\n-- output directories --")
    check_outputs(r, service)

    sys.exit(r.summary())


if __name__ == "__main__":
    main()
