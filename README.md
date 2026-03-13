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
- auto-converts Bruker ParaVision folders to NIfTI when no NIfTI files are present (uses `bruker2nifti` by default; configurable),
- runs AFNI (`3dUnifize`, `3dAutomask`, `3dSkullStrip`) and/or SPM segmentation,
- writes one `summary.json` report,
- optionally computes a **DG proxy** from SPM gray-matter segmentation (centroids of posterior left/right gray-matter clusters).

Example (full loop):

```bash
python3 run_neuro_analysis.py \
  --input-root /data/nii \
  --backend both \
  --spm-dir /opt/spm12 \
  --out-dir analysis_out
```

AFNI-only:

```bash
python3 run_neuro_analysis.py --input-root /data/nii --backend afni --out-dir afni_out
```

SPM-only:

```bash
python3 run_neuro_analysis.py --input-root /data/nii --backend spm --spm-dir /opt/spm12 --out-dir spm_out
```

### Where to get NIfTI files

This repo accepts `.nii`/`.nii.gz` directly and can auto-convert Bruker ParaVision raw folders (`2dseq`, `visu_pars`) when needed.

Automatic conversion defaults to:

```bash
python3 run_neuro_analysis.py --input-root /path/to/bruker_study --out-dir analysis_out
```

It invokes `bruker2nifti -i {input} -o {output}` by default and writes converted data to `<out-dir>/converted_nifti`.
You can override this with `--bruker-converter-cmd`, `--bruker-converter-args`, `--converted-dir`, or disable it with `--no-convert-bruker`.

Manual conversion is still supported with tools such as:

- `Bru2`
- `bruker2nifti`
- `dcm2niix` (if exported through DICOM)

Notes:
- AFNI backend requires AFNI binaries in `PATH`.
- SPM backend requires MATLAB (or SPM standalone launcher) and a valid SPM installation.
- DG output is a segmentation-derived proxy, not an atlas-validated dentate gyrus segmentation.
