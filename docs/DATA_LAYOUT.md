# Data Layout Reference

This document describes the **exact file layout** the pipeline expects, plus the
**per-voxel label derivation rule**. If something here is unclear, run
`python scripts/validate_dataset.py --config configs/local.yaml` — the validator
checks every assumption documented below.

---

## 1. Raw BIDS dataset (OpenNeuro `ds004889`)

The full multi-modal stroke dataset, required for paper-faithful 3-class node classification.

```
ds004889/
├── README                                  (optional, dataset notes)
├── CHANGES                                 (optional, version history)
├── dataset_description.json                (BIDS-required, "BIDSVersion", "Name")
├── participants.tsv                        (subject-level demographics)
├── sub-1000/
│   ├── anat/
│   │   ├── sub-1000_T1w.nii.gz             T1-weighted anatomical
│   │   └── sub-1000_FLAIR.nii.gz           FLAIR
│   └── dwi/
│       ├── sub-1000_rec-ADC_dwi.nii.gz     Apparent Diffusion Coefficient map
│       └── sub-1000_rec-TRACE_dwi.nii.gz   Trace-weighted DWI
├── sub-1001/...
├── ... (~1715 subjects)
└── derivatives/
    └── lesion_masks/
        ├── sub-1000/
        │   └── dwi/
        │       ├── sub-1000_space-TRACE_desc-lesion_mask.nii.gz
        │       │       (combined lesion mask, both timings union)
        │       ├── sub-1000_space-TRACE_desc-lesionAcute_mask.nii.gz
        │       │       (acute lesion only — class 1)
        │       └── sub-1000_space-TRACE_desc-lesionChronic_mask.nii.gz
        │               (chronic lesion only — class 2)
        ├── sub-1001/...
        └── ...
```

### What's required vs optional

| File | Required for training? | Notes |
|---|---|---|
| `sub-XXXX_T1w.nii.gz` | **Yes** | Reference image for inter-modal registration |
| `sub-XXXX_FLAIR.nii.gz` | Yes | Used as a modality + for atlas overlay viz |
| `sub-XXXX_rec-ADC_dwi.nii.gz` | Recommended | Modality contributes to SLIC + graph features |
| `sub-XXXX_rec-TRACE_dwi.nii.gz` | Recommended | Modality contributes to SLIC + graph features |
| `..._desc-lesionAcute_mask.nii.gz` | **Yes for 3-class** | Without it, only class 0 / "no lesion" labels are derivable |
| `..._desc-lesionChronic_mask.nii.gz` | **Yes for 3-class** | Same |
| `..._desc-lesion_mask.nii.gz` | Optional | Used as a sanity check / fallback combined mask |

### Subject ID format

Subject IDs are `sub-` followed by alphanumerics. The pipeline scans the BIDS
root with the regex `^sub-[A-Za-z0-9]+$`, so both `sub-1000` and `sub-09A` work.
The OpenNeuro publication uses numeric IDs (`sub-1000`, `sub-1001`, ...).

---

## 2. SOOP normalized release

A **flat** directory of FLAIR + single binary lesion masks in standard MNI space.
This release does **not** contain T1, ADC, TRACE, or the acute/chronic split. It
is used by this project for two purposes:

1. **The ArterialAtlas136 atlas + label file** (required by the graph
   construction step regardless of which dataset you train on).
2. **A FLAIR-only quickstart for `notebooks/01_data_exploration.ipynb`** when the
   raw BIDS hasn't been downloaded yet.

```
SOOP_NIfTI_normalized/NIfTI/
├── ArterialAtlas136.nii.gz                 atlas volume (32 arterial regions)
├── ArterialAtlas136.txt                    region-id → name map (see Section 3)
├── FLAIR_mean_1714.nii.gz                  population mean FLAIR (auxiliary)
├── T1_mean_1714.nii.gz                     population mean T1 (auxiliary)
├── lesion_mean_1449.nii.gz                 population mean lesion (auxiliary)
├── wsub-10_FLAIR.nii.gz                    warped FLAIR for subject 10
├── wsub-100_FLAIR.nii.gz
├── wsub-1000_FLAIR.nii.gz
├── ... (1714 total)
├── bwsrsub-10_lesion.nii.gz                binary lesion mask for subject 10
├── bwsrsub-100_lesion.nii.gz
├── bwsrsub-1000_lesion.nii.gz
└── ... (1449 total — only subjects with lesions)
```

### Filename prefixes

| Prefix | Meaning |
|---|---|
| `w`     | warped to standard space |
| `bwsr`  | bias-corrected, warped, smoothed, resampled |

### Subject ID format

Numeric only, no zero-padding, no `sub-` prefix:
`wsub-10_FLAIR.nii.gz`, `wsub-100_FLAIR.nii.gz`, `wsub-1000_FLAIR.nii.gz`.
The pipeline parses these via the regex `bwsrsub-(\d+)_lesion\.nii\.gz`.

---

## 3. ArterialAtlas136 label file

`ArterialAtlas136.txt` is pipe-delimited, one region per line:

```
1|ACAL|anterior cerebral artery left|1
2|ACAR|anterior cerebral artery right|2
3|MLSL|medial lenticulostriate left|1
...
32|VBR|vertebrobasilar right|2
```

Columns: `index | abbreviation | full_name | group_id`.
The pipeline only uses `index` and `full_name`.

---

## 4. Per-voxel label derivation rule

From the user's clinical clarification: **labels are derived per voxel by
overlapping the acute and chronic mask volumes**.

```
For each voxel v:
    if v is inside the acute mask:    label(v) = 1   (acute lesion)
    elif v is inside the chronic mask: label(v) = 2   (chronic lesion)
    else:                              label(v) = 0   (no lesion / control tissue)
```

Edge cases:

- **Both masks cover a voxel** (rare overlap): acute wins (more clinically urgent).
  Implemented in `src/stroke_gat/graph/features.py` via a nested
  `torch.where(acute > 0, 1, where(chronic > 0, 2, 0))`.
- **Subject has no acute mask AND no chronic mask** (control / non-stroke):
  every voxel is class 0. The pipeline handles this transparently — missing
  masks are zero-filled in `DataService.load_subject_masks()`.
- **Voxel-to-supervoxel aggregation**: a supervoxel inherits the label of any
  voxel inside it (priority same as above: acute > chronic > none). See
  `NodeFeatureComputer.compute()`.

---

## 5. Atlas registration

`ArterialAtlas136.nii.gz` is in standard MNI space. The pipeline registers it to
each subject's T1 via nilearn `resample_to_img(..., interpolation="nearest")`,
implemented in `DataService.preprocess_subject()`. All modalities are
co-registered to T1 before graph construction.

---

## 6. Output layout

After running the pipeline, this is what you get:

```
data/
├── preprocessed/                            # written by preprocess_dataset.py
│   ├── sub-1000/
│   │   ├── T1.nii.gz                        registered + normalized
│   │   ├── FLAIR.nii.gz
│   │   ├── ADC.nii.gz
│   │   ├── TRACE.nii.gz
│   │   ├── General_mask.nii.gz              registered to T1 space
│   │   ├── Acute_mask.nii.gz
│   │   └── Chronic_mask.nii.gz
│   └── ...
├── graphs/                                  # written by generate_graphs.py
│   ├── sub-1000_supervoxel_graph.pt         torch_geometric.data.Data
│   ├── sub-1001_supervoxel_graph.pt
│   └── ...
├── subjects_metadata.csv                    # written by generate_graphs.py
│                                            # columns: subject, stroke_status,
│                                            # lesion_level, has_acute, has_chronic
└── models/                                  # output of training (configurable)

outputs/                                     # default training output_dir
├── checkpoints/
│   ├── best_model.pt                        best by composite score
│   └── latest_checkpoint.pt
├── training_results.json
└── test_results.json
```
