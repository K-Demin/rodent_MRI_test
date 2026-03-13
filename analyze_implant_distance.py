#!/usr/bin/env python3
"""Estimate implant and DG locations in Bruker ParaVision T2 scans.

This script is intentionally dependency-free and tailored for this dataset where
block 5 and block 13 are the T2 RARE scans.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

Point3D = Tuple[float, float, float]


def _get_param(text: str, name: str) -> str | None:
    m = re.search(rf"##\${name}=(.*)", text)
    return m.group(1).strip() if m else None


def _get_array(text: str, name: str) -> List[float]:
    m = re.search(rf"##\${name}=\(.*?\)\n([^#]+)", text, re.S)
    if not m:
        return []
    s = " ".join(m.group(1).split())
    out: List[float] = []
    for tok in s.split():
        mm = re.match(r"@(\d+)\*\(([^)]+)\)", tok)
        if mm:
            out.extend([float(mm.group(2))] * int(mm.group(1)))
        else:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


def _percentile(values: Sequence[float], p: float) -> float:
    s = sorted(values)
    idx = int((len(s) - 1) * p)
    return s[max(0, min(len(s) - 1, idx))]


def _largest_component(mask: Sequence[bool], w: int, h: int) -> set[int]:
    visited = [False] * (w * h)
    best: List[int] = []
    for i, is_on in enumerate(mask):
        if not is_on or visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: List[int] = []
        while stack:
            a = stack.pop()
            comp.append(a)
            x, y = a % w, a // w
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < w and 0 <= ny < h:
                    j = ny * w + nx
                    if mask[j] and not visited[j]:
                        visited[j] = True
                        stack.append(j)
        if len(comp) > len(best):
            best = comp
    return set(best)


def _component_centroid(indices: Iterable[int], w: int) -> Tuple[float, float]:
    lst = list(indices)
    cx = sum(i % w for i in lst) / len(lst)
    cy = sum(i // w for i in lst) / len(lst)
    return cx, cy


def load_block(block_dir: Path) -> dict:
    visu_text = (block_dir / "pdata/1/visu_pars").read_text(errors="ignore")

    size = [int(v) for v in _get_array(visu_text, "VisuCoreSize")[:2]]
    w, h = size
    extent = _get_array(visu_text, "VisuCoreExtent")
    dx, dy = extent[0] / w, extent[1] / h

    frames = int(float(_get_param(visu_text, "VisuCoreFrameCount") or "0"))
    slopes = _get_array(visu_text, "VisuCoreDataSlope")
    offs = _get_array(visu_text, "VisuCoreDataOffs")
    positions = _get_array(visu_text, "VisuCorePosition")
    orientations = _get_array(visu_text, "VisuCoreOrientation")

    raw = (block_dir / "pdata/1/2dseq").read_bytes()
    values = struct.unpack("<" + "h" * (len(raw) // 2), raw)

    images: List[List[float]] = []
    for z in range(frames):
        sl = values[z * w * h : (z + 1) * w * h]
        s = slopes[z] if z < len(slopes) else 1.0
        o = offs[z] if z < len(offs) else 0.0
        images.append([v * s + o for v in sl])

    return {
        "w": w,
        "h": h,
        "dx": dx,
        "dy": dy,
        "frames": frames,
        "images": images,
        "positions": positions,
        "orientations": orientations,
    }


def pixel_to_world(block: dict, z: int, cx: float, cy: float) -> Point3D:
    w = block["w"]
    h = block["h"]
    pos = block["positions"][3 * z : 3 * z + 3]
    ori = block["orientations"][9 * z : 9 * z + 9]
    ex = (ori[0], ori[3], ori[6])
    ey = (ori[1], ori[4], ori[7])
    xoff = (cx - (w - 1) / 2) * block["dx"]
    yoff = (cy - (h - 1) / 2) * block["dy"]
    return tuple(pos[i] + xoff * ex[i] + yoff * ey[i] for i in range(3))


def _segment_brain(im: List[float], w: int, h: int) -> set[int]:
    thr = _percentile(im, 0.55)
    return _largest_component([v > thr for v in im], w, h)


def estimate_implant(block: dict) -> Tuple[int, float, float, Point3D, int, set[int]]:
    """Return (slice_idx, x_px, y_px, world_mm, n_dark_vox, brain_mask)."""
    w, h = block["w"], block["h"]
    candidates = []
    for z, im in enumerate(block["images"]):
        brain = _segment_brain(im, w, h)
        if len(brain) < 1500:
            continue
        vals = [im[i] for i in brain]
        dark_thr = _percentile(vals, 0.05)
        dark = [i for i in brain if im[i] <= dark_thr]
        if len(dark) < 20:
            continue
        cx, cy = _component_centroid(dark, w)
        score = len(dark) - 0.2 * abs(cx - w / 2) - 0.2 * abs(cy - h / 2)
        candidates.append((score, z, cx, cy, len(dark), brain))

    if not candidates:
        raise RuntimeError("No implant candidate found")

    _, z, cx, cy, n_dark, brain = max(candidates, key=lambda x: x[0])
    return z, cx, cy, pixel_to_world(block, z, cx, cy), n_dark, brain


def _dg_candidate_in_roi(
    im: List[float],
    brain: set[int],
    w: int,
    h: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Tuple[float, float] | None:
    roi = [
        i
        for i in brain
        if x_min <= (i % w) / w <= x_max and y_min <= (i // w) / h <= y_max
    ]
    if len(roi) < 80:
        return None

    roi_vals = [im[i] for i in roi]
    # DG granule layer on T2 is often relatively hypointense; take deep tail in ROI.
    thr = _percentile(roi_vals, 0.12)
    dark = [i for i in roi if im[i] <= thr]
    if len(dark) < 20:
        return None
    return _component_centroid(dark, w)


def estimate_dg(block: dict, implant_slice: int, implant_x: float) -> Tuple[int, float, float, Point3D, Point3D]:
    """Estimate ipsilateral and bilateral-midpoint DG world coordinates.

    Returns:
      (slice_idx, dg_x_px, dg_y_px, dg_ipsi_world, dg_mid_world)
    """
    w, h = block["w"], block["h"]

    # search around implant slice where hippocampus is usually visible in coronal T2.
    z0 = max(0, implant_slice - 2)
    z1 = min(block["frames"] - 1, implant_slice + 2)

    best = None
    for z in range(z0, z1 + 1):
        im = block["images"][z]
        brain = _segment_brain(im, w, h)
        if len(brain) < 1500:
            continue

        # bilateral hippocampal ROIs (dataset heuristic): mid-lateral + ventral-mid dorsal band.
        left = _dg_candidate_in_roi(im, brain, w, h, 0.15, 0.45, 0.45, 0.88)
        right = _dg_candidate_in_roi(im, brain, w, h, 0.55, 0.85, 0.45, 0.88)
        if not left or not right:
            continue

        lx, ly = left
        rx, ry = right
        # expect roughly symmetric x and similar y for bilateral DG.
        symmetry = abs((w - 1) - (lx + rx))
        y_match = abs(ly - ry)
        score = -(symmetry + 0.7 * y_match)
        if best is None or score > best[0]:
            best = (score, z, left, right)

    if best is None:
        raise RuntimeError("No DG candidates found; try manual DG coordinate input.")

    _, z, left, right = best
    # choose DG candidate nearest to the implant in image space (more robust than x-side assumptions).
    ipsi = right if abs(right[0] - implant_x) < abs(left[0] - implant_x) else left
    dgx, dgy = ipsi
    midx, midy = (left[0] + right[0]) / 2, (left[1] + right[1]) / 2

    return z, dgx, dgy, pixel_to_world(block, z, dgx, dgy), pixel_to_world(block, z, midx, midy)


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", nargs="+", default=["5", "13"], help="T2 block directories")
    p.add_argument(
        "--dg-world-mm",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional manual DG reference coordinate in scanner/world mm.",
    )
    args = p.parse_args()

    manual_dg = tuple(args.dg_world_mm) if args.dg_world_mm else None

    for b in args.blocks:
        block = load_block(Path(b))
        iz, ix, iy, implant_world, n_dark, _brain = estimate_implant(block)
        print(
            f"block {b}: implant slice={iz+1}/{block['frames']}, pixel=({ix:.2f},{iy:.2f}), "
            f"implant_world_mm=({implant_world[0]:.3f},{implant_world[1]:.3f},{implant_world[2]:.3f}), "
            f"dark_voxels={n_dark}"
        )

        if manual_dg is not None:
            d = distance(implant_world, manual_dg)
            print(f"  DG(manual)_world_mm=({manual_dg[0]:.3f},{manual_dg[1]:.3f},{manual_dg[2]:.3f})")
            print(f"  implant_to_DG_mm={d:.3f}")
            continue

        dz, dgx, dgy, dg_world_ipsi, dg_world_mid = estimate_dg(block, iz, ix)
        print(
            f"  DG(segmented) slice={dz+1}/{block['frames']}, ipsi_pixel=({dgx:.2f},{dgy:.2f}), "
            f"DG_ipsi_world_mm=({dg_world_ipsi[0]:.3f},{dg_world_ipsi[1]:.3f},{dg_world_ipsi[2]:.3f})"
        )
        print(
            f"  DG_midline_world_mm=({dg_world_mid[0]:.3f},{dg_world_mid[1]:.3f},{dg_world_mid[2]:.3f})"
        )
        print(f"  implant_to_DG_ipsi_mm={distance(implant_world, dg_world_ipsi):.3f}")
        print(f"  implant_to_DG_midline_mm={distance(implant_world, dg_world_mid):.3f}")


if __name__ == "__main__":
    main()
