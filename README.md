# rodent_MRI_test

Utility script:

- `analyze_implant_distance.py` estimates implant location and DG landmarks from Bruker ParaVision T2 RARE scans.
- By default it analyzes blocks `5` and `13`.
- Optional `--dg-world-mm X Y Z` lets you compare implant distance to a manual DG world coordinate.
- Optional `--viz-dir <dir>` exports SVG overlays for:
  - implant slice with candidate fiber spots (dark tail inside segmented brain)
  - DG slice with bilateral DG candidates, ipsilateral DG, and dark DG ROI spots

Example:

```bash
python3 analyze_implant_distance.py --blocks 5 13 --viz-dir viz
```
