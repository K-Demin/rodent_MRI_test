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


## Running proper AFNI or SPM analysis

If you want a standard neuroimaging workflow (instead of the lightweight heuristic script), use:

- `run_neuro_analysis.py --backend afni` for AFNI preprocessing (`3dUnifize`, `3dAutomask`, `3dSkullStrip`)
- `run_neuro_analysis.py --backend spm` for SPM12 tissue segmentation via MATLAB

Example:

```bash
python3 run_neuro_analysis.py --backend afni --input-nii my_scan.nii.gz --out-dir afni_out
python3 run_neuro_analysis.py --backend spm --input-nii my_scan.nii.gz --spm-dir /opt/spm12 --out-dir spm_out
```

Notes:
- AFNI backend requires AFNI binaries in `PATH`.
- SPM backend requires MATLAB (or SPM standalone launcher) and a valid SPM installation.
- If your source data is raw Bruker ParaVision, convert to NIfTI first (e.g., `Bru2`/`bruker2nifti`).
