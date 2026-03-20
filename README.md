# rodent_MRI_test

Utility script:

- `analyze_implant_distance.py` estimates implant location and DG landmarks from Bruker ParaVision T2 RARE scans using a brain-first pipeline: segment whole-brain mask per slice, estimate bilateral DG inside brain-anchored ROIs, and detect implant/fiber candidates only inside the brain mask.
- By default it analyzes blocks `5` and `13`.
- Optional `--dg-world-mm X Y Z` lets you compare implant distance to a manual DG world coordinate.
- Optional `--viz-dir <dir>` exports SVG overlays for:
  - implant slice with candidate fiber spots (dark connected components constrained to segmented brain interior)
  - DG slice with bilateral DG candidates, ipsilateral DG, and dark DG ROI spots

Example:

```bash
python3 analyze_implant_distance.py --blocks 5 13 --viz-dir viz
```

## Closed-loop AFNI/SPM T2 pipeline

Use `run_neuro_analysis.py` when you want one command that takes folders and runs full processing:

- discovers T2 NIfTI files under a root folder (`*t2*.nii*` / `*rare*.nii*`, then fallback to any NIfTI),
- auto-converts Bruker ParaVision folders to NIfTI when no NIfTI files are present (uses `brkraw`; configurable),
- runs AFNI (`3dUnifize`, `3dAutomask`, `3dSkullStrip`) and/or SPM segmentation,
- writes one `summary.json` report,
- optionally computes a **DG proxy** from SPM gray-matter segmentation (centroids of posterior left/right gray-matter clusters).

Rodent-focused defaults:

- `--species mouse` is the default and enables AFNI rodent skull-strip flags.
- SPM is **skipped** for mouse/rat unless you provide `--spm-tpm /path/to/rodent_TPM.nii`.
- only 3D NIfTI volumes are processed (4D or hidden/env files are skipped).

Example (full loop):

```bash
python3 run_neuro_analysis.py \
  --input-root /data/nii \
  --backend both \
  --spm-dir /opt/spm12 \
  --spm-tpm /data/atlas/mouse_TPM.nii \
  --out-dir analysis_out
```

AFNI-only:

```bash
python3 run_neuro_analysis.py --input-root /data/nii --backend afni --out-dir afni_out
```

SPM-only:

```bash
python3 run_neuro_analysis.py --input-root /data/nii --backend spm --spm-dir /opt/spm12 --spm-tpm /data/atlas/mouse_TPM.nii --out-dir spm_out
```

### Where to get NIfTI files

This repo accepts `.nii`/`.nii.gz` directly and can auto-convert Bruker ParaVision raw folders (`2dseq`, `visu_pars`) when needed.

Automatic conversion defaults to:

```bash
python3 run_neuro_analysis.py --input-root /path/to/bruker_study --out-dir analysis_out
```

It uses `brkraw` by default (`tonii {input} -o {output}`) and writes converted data to `<out-dir>/converted_nifti`.
When `--input-root` points to a single scan folder (for example `.../5`) or nested `pdata` path, the script retries conversion against likely scan/study ancestors and uses isolated conversion attempt directories for robust retries.
You can override this with `--bruker-converter-cmd`, `--bruker-converter-args`, `--converted-dir`, or disable it with `--no-convert-bruker`.

Manual conversion can be done with:

- `brkraw` (`https://github.com/BrkRaw/brkraw`)

Notes:
- AFNI backend requires AFNI binaries in `PATH`.
- SPM backend requires MATLAB (or SPM standalone launcher) and a valid SPM installation.
- For rodents, provide rodent TPM via `--spm-tpm` and prefer `--spm-affreg none`.
- DG output is a segmentation-derived proxy, not an atlas-validated dentate gyrus segmentation.

## Manual-assisted hippocampus script (experiment/run layout)

`mouse_hippo_manual_assisted.py` (and `mouse_hippo_manual_assisted (1).py`) now supports selecting data by `experiment` + `run` inside a project root.

You can still pass direct NIfTI inputs:

```bash
python3 mouse_hippo_manual_assisted.py \
  --input-nii /path/to/subj.nii.gz \
  --atlas-template /path/to/atlas_template.nii.gz \
  --atlas-labels /path/to/atlas_labels.nii.gz
```

Or point at a project hierarchy like `main_project_folder/<experiment>/<run>`:

```bash
python3 mouse_hippo_manual_assisted.py \
  --project-root /path/to/main_project_folder \
  --experiment expA \
  --run 11 \
  --atlas-template /path/to/atlas_template.nii.gz \
  --atlas-labels /path/to/atlas_labels.nii.gz
```

When using `--project-root/--experiment/--run` and leaving `--out-dir` at default, output is automatically named:
`analysis_out_manual_coronal_<experiment>_<run>`.

If a run folder has only Bruker raw data (no `.nii/.nii.gz` yet), the script now auto-converts first and then continues the full manual-assisted workflow end-to-end:

```bash
python3 mouse_hippo_manual_assisted.py \
  --project-root /path/to/main_project_folder \
  --experiment expA \
  --run 11 \
  --atlas-template /path/to/atlas_template.nii.gz \
  --atlas-labels /path/to/atlas_labels.nii.gz
```

Converted files are written under `<out-dir>/converted_nifti` by default (so they are experiment/run-specific when default output naming is used). You can customize with:

- `--converted-dir /path/to/converted`
- `--bruker-converter-cmd brkraw`
- `--bruker-converter-args auto`
- `--no-convert-bruker` (to require pre-existing NIfTI inputs)
