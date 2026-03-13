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
