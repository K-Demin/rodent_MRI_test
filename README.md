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
- auto-converts Bruker ParaVision folders to NIfTI when no NIfTI files are present (prefers Python `bruker2nifti` by default; configurable),
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

It auto-detects a converter from `bruker2nii`, `bruker2nifti`, `Bru2Nii`, `bru2nii`, or `Bru2` and applies `-i {input} -o {output}` by default, writing converted data to `<out-dir>/converted_nifti`.
When `--input-root` points to a single scan folder (for example `.../5`) or nested `pdata` path, the script retries conversion against likely scan/study ancestors and uses isolated conversion attempt directories to avoid `FileExistsError` collisions from `bruker2nifti`.
You can override this with `--bruker-converter-cmd`, `--bruker-converter-args`, `--converted-dir`, or disable it with `--no-convert-bruker`.

Manual conversion is still supported with tools such as:

- `bruker2nifti` (Python package; install via `pip install bruker2nifti`)
- `bruker2nii` (CLI commonly installed with `bruker2nifti`)
- `Bru2Nii`
- `Bru2`
- `dcm2niix` (if exported through DICOM)

Notes:
- AFNI backend requires AFNI binaries in `PATH`.
- SPM backend requires MATLAB (or SPM standalone launcher) and a valid SPM installation.
- For rodents, provide rodent TPM via `--spm-tpm` and prefer `--spm-affreg none`.
- DG output is a segmentation-derived proxy, not an atlas-validated dentate gyrus segmentation.
