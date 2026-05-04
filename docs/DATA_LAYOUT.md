# Data layout

Notes on what the pipeline expects on disk and how voxel labels are derived.
If anything below seems off in your tree, run

```bash
python scripts/validate_dataset.py --config configs/local.yaml
```

which checks every path mentioned here.

## 1. Raw BIDS dataset (OpenNeuro `ds004889`)

The full multi-modal data is required for the 3-class classification used in
the paper. Expected tree:

```
ds004889/
  dataset_description.json
  participants.tsv
  sub-1000/
    anat/
      sub-1000_T1w.nii.gz
      sub-1000_FLAIR.nii.gz
    dwi/
      sub-1000_rec-ADC_dwi.nii.gz
      sub-1000_rec-TRACE_dwi.nii.gz
  sub-1001/
  ...
  derivatives/
    lesion_masks/
      sub-1000/
        dwi/
          sub-1000_space-TRACE_desc-lesion_mask.nii.gz
          sub-1000_space-TRACE_desc-lesionAcute_mask.nii.gz
          sub-1000_space-TRACE_desc-lesionChronic_mask.nii.gz
      sub-1001/
      ...
```

Per-subject required files:

| File                                    | Required? | Notes                                  |
|-----------------------------------------|-----------|----------------------------------------|
| `sub-XXXX_T1w.nii.gz`                   | yes       | reference image for inter-modal alignment |
| `sub-XXXX_FLAIR.nii.gz`                 | yes       | also used for atlas overlays           |
| `sub-XXXX_rec-ADC_dwi.nii.gz`           | recommended | fourth SLIC channel                  |
| `sub-XXXX_rec-TRACE_dwi.nii.gz`         | recommended | fourth SLIC channel                  |
| `..._desc-lesionAcute_mask.nii.gz`      | yes for 3-class labels | provides class 1            |
| `..._desc-lesionChronic_mask.nii.gz`    | yes for 3-class labels | provides class 2            |
| `..._desc-lesion_mask.nii.gz`           | optional  | combined mask, used as a sanity check  |

Subject IDs are matched with the regex `^sub-[A-Za-z0-9]+$`, so `sub-1000` and
`sub-09A` are both fine. The OpenNeuro release uses numeric IDs.

## 2. SOOP normalized release

A flat directory of warped FLAIR volumes plus a single binary lesion mask per
subject. We use it for two things:

1. The `ArterialAtlas136` atlas and its label file (needed regardless of which
   data source you train on).
2. A FLAIR-only quickstart for `notebooks/01_data_exploration.ipynb` — if you
   haven't pulled the full BIDS dataset yet, the notebook will run against this
   release.

```
SOOP_NIfTI_normalized/NIfTI/
  ArterialAtlas136.nii.gz                 (32 arterial regions)
  ArterialAtlas136.txt                    (region-id -> name)
  FLAIR_mean_1714.nii.gz                  (population mean, auxiliary)
  T1_mean_1714.nii.gz
  lesion_mean_1449.nii.gz
  wsub-{ID}_FLAIR.nii.gz                  (1714 files)
  bwsrsub-{ID}_lesion.nii.gz              (1449 files; only stroke subjects)
```

Filename prefixes follow the SOOP convention: `w` = warped to standard space;
`bwsr` = bias-corrected, warped, smoothed, resampled. Subject IDs in this
release are numeric only with no zero-padding (`wsub-10_FLAIR.nii.gz`,
`wsub-1000_FLAIR.nii.gz`); the loader matches them with
`bwsrsub-(\d+)_lesion\.nii\.gz`.

## 3. ArterialAtlas136 label file

`ArterialAtlas136.txt` is pipe-delimited, one region per line:

```
1|ACAL|anterior cerebral artery left|1
2|ACAR|anterior cerebral artery right|2
3|MLSL|medial lenticulostriate left|1
...
```

Columns: `index | abbreviation | full_name | group_id`. Only `index` and
`full_name` are used downstream.

## 4. How node labels are derived

For each voxel `v`:

```
if v in acute mask         -> 1   (acute)
elif v in chronic mask     -> 2   (chronic)
else                       -> 0   (no lesion)
```

If both masks happen to cover the same voxel, acute wins. Subjects whose
acute and chronic masks are both empty are controls and contribute only
class-0 nodes; missing mask files are zero-filled in
`DataService.load_subject_masks()`.

A supervoxel inherits the label of any voxel in it, with the same priority
(acute > chronic > none). The aggregation lives in `NodeFeatureComputer.compute`
in `src/stroke_gat/graph/features.py`.

## 5. Atlas registration

The atlas is published in standard MNI space. During preprocessing it is
resampled into each subject's T1 space with nilearn's
`resample_to_img(..., interpolation="nearest")`; FLAIR/ADC/TRACE are also
co-registered to T1 with linear interpolation. See
`DataService.preprocess_subject()`.

## 6. Outputs

After running `preprocess_dataset.py` and `generate_graphs.py`:

```
data/
  preprocessed/
    sub-1000/
      T1.nii.gz                          (registered, normalized)
      FLAIR.nii.gz
      ADC.nii.gz
      TRACE.nii.gz
      General_mask.nii.gz
      Acute_mask.nii.gz
      Chronic_mask.nii.gz
  graphs/
    sub-1000_supervoxel_graph.pt          (torch_geometric.data.Data)
    sub-1001_supervoxel_graph.pt
    ...
  subjects_metadata.csv                   (subject, stroke_status, lesion_level,
                                           has_acute, has_chronic)

outputs/                                  (default --output-dir for train.py)
  checkpoints/
    best_model.pt
    latest_checkpoint.pt
  training_results.json
  test_results.json
```
