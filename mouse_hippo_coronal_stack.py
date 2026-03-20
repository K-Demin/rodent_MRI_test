#!/usr/bin/env python3
"""Locate hippocampus / DG on anisotropic 2D coronal mouse T2 stacks.

Core idea
---------
These scans are not a near-isotropic 3D volume. They are a stack of thick 2D
coronal slices (e.g. 0.1 x 0.1 x 0.5 mm, 18 slices). Treating them as a fully
3D volume and searching arbitrary 3D orientation hypotheses is usually the
wrong model.

This script instead:
1) infers the subject slice axis from the thickest voxel dimension,
2) reorients the subject into a coronal slice stack (H, W, N),
3) tries a very small set of atlas hypotheses (candidate coronal axis,
   A->P vs P->A ordering, optional in-plane mirror),
4) resamples the atlas to the subject spacing,
5) scores subject slice i against atlas slice j using 2D in-plane alignment,
6) finds the best monotonic subject->atlas slice path with dynamic programming,
7) warps atlas labels slice-wise onto the subject stack,
8) writes the labels back on the ORIGINAL scan grid/affine.

This is designed for partial slabs where the anatomy progresses monotonically
through slice index and where hippocampus is easier to localize than in a full
3D whole-brain registration.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import AffineTransform, warp


# -----------------------------
# basic io / utilities
# -----------------------------


def _rm_if_exists(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _save_nifti(data: np.ndarray, affine: np.ndarray, out_path: Path, like: nib.Nifti1Image | None = None) -> None:
    hdr = like.header.copy() if like is not None else nib.Nifti1Header()
    out = nib.Nifti1Image(data, affine, header=hdr)
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)
    nib.save(out, str(out_path))


# -----------------------------
# image preprocessing
# -----------------------------


def _fix_suspicious_zooms(zooms: Iterable[float]) -> tuple[float, float, float]:
    vals = tuple(float(abs(z)) for z in list(zooms)[:3])
    if max(vals) > 5.0:
        vals = tuple(v * 0.001 for v in vals)
    return vals


def _percentile_clip(arr: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    a = float(np.percentile(finite, lo))
    b = float(np.percentile(finite, hi))
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return arr
    return np.clip(arr, a, b)


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    a = float(finite.min())
    b = float(finite.max())
    if b <= a:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - a) / (b - a)


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    lab = label(mask.astype(np.uint8), connectivity=1)
    if lab.max() == 0:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == np.argmax(counts)


def _brain_mask_2d(slice2d: np.ndarray) -> np.ndarray:
    s = np.asarray(slice2d, dtype=np.float32)
    s = _percentile_clip(s)
    nonzero = s[s > 0]
    if nonzero.size < 50:
        return np.zeros_like(s, dtype=bool)
    try:
        thr = threshold_otsu(nonzero)
    except Exception:
        thr = float(np.percentile(nonzero, 60))
    thr = max(thr * 0.7, float(np.percentile(nonzero, 25)))
    mask = s > thr
    mask = ndi.binary_closing(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    mask = _largest_cc(mask)
    mask = ndi.binary_opening(mask, iterations=1)
    return mask.astype(bool)


def _slice_has_signal(slice2d: np.ndarray) -> bool:
    return np.count_nonzero(slice2d > 0) > 100


def _slice_props(mask: np.ndarray) -> dict[str, float] | None:
    if np.count_nonzero(mask) < 50:
        return None
    rp = regionprops(mask.astype(np.uint8))
    if not rp:
        return None
    r = max(rp, key=lambda x: x.area)
    minr, minc, maxr, maxc = r.bbox
    return {
        "cy": float(r.centroid[0]),
        "cx": float(r.centroid[1]),
        "height": float(maxr - minr),
        "width": float(maxc - minc),
        "orientation": float(r.orientation),
        "area": float(r.area),
    }


def _nmi(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, bins: int = 48) -> float:
    if mask is not None:
        a = a[mask]
        b = b[mask]
    else:
        a = a.ravel()
        b = b.ravel()
    good = np.isfinite(a) & np.isfinite(b)
    a = a[good]
    b = b[good]
    if a.size < 100:
        return float("-inf")
    a = _normalize01(_percentile_clip(a))
    b = _normalize01(_percentile_clip(b))
    ah, aedges = np.histogram(a, bins=bins, range=(0, 1))
    bh, bedges = np.histogram(b, bins=bins, range=(0, 1))
    jh, _, _ = np.histogram2d(a, b, bins=(aedges, bedges))
    pxy = jh / max(float(jh.sum()), 1.0)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nzx = px > 0
    nzy = py > 0
    nzxy = pxy > 0
    hx = -np.sum(px[nzx] * np.log(px[nzx]))
    hy = -np.sum(py[nzy] * np.log(py[nzy]))
    hxy = -np.sum(pxy[nzxy] * np.log(pxy[nzxy]))
    if hxy <= 0:
        return float("-inf")
    return float((hx + hy) / hxy)


# -----------------------------
# orientation / stack helpers
# -----------------------------


@dataclass(frozen=True)
class StackSpec:
    atlas_slice_axis: int
    atlas_reverse_ap: bool
    subject_mirror_lr: bool

    @property
    def name(self) -> str:
        return f"atlasAxis{self.atlas_slice_axis}_rev{int(self.atlas_reverse_ap)}_mirror{int(self.subject_mirror_lr)}"


def _reorient_stack(vol: np.ndarray, zooms: tuple[float, float, float], slice_axis: int) -> tuple[np.ndarray, tuple[float, float, float], tuple[int, int, int]]:
    perm = tuple(ax for ax in range(3) if ax != slice_axis) + (slice_axis,)
    vol2 = np.transpose(vol, axes=perm)
    zooms2 = tuple(float(zooms[p]) for p in perm)
    return np.ascontiguousarray(vol2), zooms2, perm


def _invert_reorient_stack(vol: np.ndarray, perm: tuple[int, int, int]) -> np.ndarray:
    inv = np.argsort(np.array(perm))
    return np.transpose(vol, axes=inv)


def _resample_to_spacing(vol: np.ndarray, in_zooms: tuple[float, float, float], out_zooms: tuple[float, float, float], order: int) -> np.ndarray:
    factors = [float(in_zooms[d]) / float(out_zooms[d]) for d in range(3)]
    return ndi.zoom(vol, zoom=factors, order=order)


def _crop_atlas_to_labels(template: np.ndarray, labels: np.ndarray, crop_label_ids: list[int] | None, margin_vox=(12, 12, 6)) -> tuple[np.ndarray, np.ndarray]:
    if crop_label_ids:
        mask = np.isin(labels, np.array(crop_label_ids))
    else:
        mask = labels != 0
    if not np.any(mask):
        raise SystemExit("Atlas crop mask is empty. Check --crop-label-ids / --label-ids.")
    ijk = np.where(mask)
    mins = np.array([int(ijk[d].min()) for d in range(3)])
    maxs = np.array([int(ijk[d].max()) for d in range(3)])
    start = np.maximum(mins - np.array(margin_vox, dtype=int), 0)
    stop = np.minimum(maxs + np.array(margin_vox, dtype=int) + 1, np.array(labels.shape, dtype=int))
    tpl = template[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
    lab = labels[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
    return tpl, lab


# -----------------------------
# 2D alignment per slice pair
# -----------------------------


def _similarity_from_masks(src_mask: np.ndarray, dst_mask: np.ndarray) -> tuple[AffineTransform, dict[str, float]] | None:
    psrc = _slice_props(src_mask)
    pdst = _slice_props(dst_mask)
    if psrc is None or pdst is None:
        return None

    # regionprops orientation is measured against rows; for our purposes only a coarse guess is needed
    src_angle = -float(psrc["orientation"])
    dst_angle = -float(pdst["orientation"])
    rot = dst_angle - src_angle

    src_scale = max((pdst["width"] / max(psrc["width"], 1e-3) + pdst["height"] / max(psrc["height"], 1e-3)) / 2.0, 1e-3)

    # Build transform around source centroid -> destination centroid.
    t1 = AffineTransform(translation=(-psrc["cx"], -psrc["cy"]))
    t2 = AffineTransform(scale=(src_scale, src_scale), rotation=rot)
    t3 = AffineTransform(translation=(pdst["cx"], pdst["cy"]))
    matrix = t3.params @ t2.params @ t1.params
    return AffineTransform(matrix=matrix), {
        "rot_rad": float(rot),
        "scale": float(src_scale),
        "src_cx": float(psrc["cx"]),
        "src_cy": float(psrc["cy"]),
        "dst_cx": float(pdst["cx"]),
        "dst_cy": float(pdst["cy"]),
    }


def _refine_transform_by_angle_search(src_img: np.ndarray, src_mask: np.ndarray, dst_img: np.ndarray, dst_mask: np.ndarray, base_tform: AffineTransform) -> tuple[AffineTransform, float]:
    best_score = float("-inf")
    best_tform = base_tform

    # small extra angular refinement around the moment-based guess
    for extra_deg in (-16, -12, -8, -4, 0, 4, 8, 12, 16):
        extra = math.radians(extra_deg)
        cx = dst_img.shape[1] / 2.0
        cy = dst_img.shape[0] / 2.0
        c1 = AffineTransform(translation=(-cx, -cy))
        r = AffineTransform(rotation=extra)
        c2 = AffineTransform(translation=(cx, cy))
        M = c2.params @ r.params @ c1.params @ base_tform.params
        tform = AffineTransform(matrix=M)

        moved = warp(src_img, inverse_map=tform.inverse, output_shape=dst_img.shape, order=1, preserve_range=True)
        moved_mask = warp(src_mask.astype(np.float32), inverse_map=tform.inverse, output_shape=dst_img.shape, order=0, preserve_range=True) > 0.5
        overlap = moved_mask & dst_mask
        if overlap.sum() < 100:
            continue
        score = _nmi(moved, dst_img, mask=overlap)
        if score > best_score:
            best_score = score
            best_tform = tform

    return best_tform, float(best_score)


def _score_slice_pair(src_img: np.ndarray, src_lab: np.ndarray, dst_img: np.ndarray) -> tuple[float, AffineTransform | None]:
    src_mask = src_lab != 0
    dst_mask = _brain_mask_2d(dst_img)
    if np.count_nonzero(src_mask) < 50 or np.count_nonzero(dst_mask) < 50:
        return float("-inf"), None

    guess = _similarity_from_masks(src_mask, dst_mask)
    if guess is None:
        return float("-inf"), None
    base_tform, _ = guess
    best_tform, best_score = _refine_transform_by_angle_search(src_img, src_mask, dst_img, dst_mask, base_tform)
    return float(best_score), best_tform


# -----------------------------
# dynamic programming over slice path
# -----------------------------


def _compute_score_matrix(atlas_stack: np.ndarray, atlas_labels: np.ndarray, subject_stack: np.ndarray) -> tuple[np.ndarray, list[list[AffineTransform | None]]]:
    n_subj = subject_stack.shape[2]
    n_atlas = atlas_stack.shape[2]
    scores = np.full((n_subj, n_atlas), -np.inf, dtype=np.float32)
    tforms: list[list[AffineTransform | None]] = [[None for _ in range(n_atlas)] for _ in range(n_subj)]

    for i in range(n_subj):
        dst = subject_stack[:, :, i]
        if not _slice_has_signal(dst):
            continue
        for j in range(n_atlas):
            src = atlas_stack[:, :, j]
            lab = atlas_labels[:, :, j]
            if np.count_nonzero(lab) < 50:
                continue
            s, tfm = _score_slice_pair(src, lab, dst)
            scores[i, j] = s
            tforms[i][j] = tfm
    return scores, tforms


def _best_monotonic_path(scores: np.ndarray, expected_step: float = 1.0, step_penalty: float = 0.25) -> tuple[list[int], float]:
    n_subj, n_atlas = scores.shape
    dp = np.full((n_subj, n_atlas), -np.inf, dtype=np.float32)
    ptr = np.full((n_subj, n_atlas), -1, dtype=np.int32)

    dp[0, :] = scores[0, :]

    for i in range(1, n_subj):
        prev_valid = np.where(np.isfinite(dp[i - 1]))[0]
        for j in range(n_atlas):
            if not np.isfinite(scores[i, j]):
                continue
            best = -np.inf
            best_k = -1
            for k in prev_valid:
                if j <= k:
                    continue
                trans_cost = step_penalty * abs((j - k) - expected_step)
                cand = float(dp[i - 1, k]) + float(scores[i, j]) - trans_cost
                if cand > best:
                    best = cand
                    best_k = int(k)
            dp[i, j] = best
            ptr[i, j] = best_k

    end_j = int(np.nanargmax(dp[-1]))
    total = float(dp[-1, end_j])
    path = [end_j]
    cur = end_j
    for i in range(n_subj - 1, 0, -1):
        cur = int(ptr[i, cur])
        if cur < 0:
            # fallback if some early slices had no good matches
            cur = max(0, path[-1] - 1)
        path.append(cur)
    path.reverse()
    return path, total


# -----------------------------
# high-level subject pipeline
# -----------------------------


def _load_subject_stack(subject_nii: Path, subject_slice_axis: int | None) -> dict[str, object]:
    img = nib.load(str(subject_nii))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise SystemExit(f"Expected 3D NIfTI. Got {data.shape} for {subject_nii}")
    zooms = _fix_suspicious_zooms(img.header.get_zooms()[:3])

    slice_axis = int(np.argmax(np.array(zooms))) if subject_slice_axis is None else int(subject_slice_axis)
    stack, stack_zooms, perm = _reorient_stack(data, zooms, slice_axis)
    stack = _percentile_clip(stack)

    return {
        "img": img,
        "raw_data": data,
        "raw_zooms": zooms,
        "slice_axis": slice_axis,
        "stack": stack,
        "stack_zooms": stack_zooms,
        "perm": perm,
    }


def _load_and_prepare_atlas(atlas_template: Path, atlas_labels: Path, crop_label_ids: list[int] | None) -> dict[str, object]:
    timg = nib.load(str(atlas_template))
    limg = nib.load(str(atlas_labels))
    tdat = np.asarray(timg.get_fdata(), dtype=np.float32)
    ldat = np.asarray(limg.get_fdata())
    if tdat.shape != ldat.shape:
        raise SystemExit(f"Atlas template/labels shape mismatch: {tdat.shape} vs {ldat.shape}")
    zooms = _fix_suspicious_zooms(timg.header.get_zooms()[:3])

    tdat = _percentile_clip(tdat)
    tdat, ldat = _crop_atlas_to_labels(tdat, ldat, crop_label_ids)

    return {
        "template": tdat,
        "labels": ldat,
        "zooms": zooms,
        "template_img": timg,
        "labels_img": limg,
    }


def _apply_stack_spec(subject_stack: np.ndarray, spec: StackSpec) -> np.ndarray:
    out = subject_stack
    if spec.subject_mirror_lr:
        out = np.flip(out, axis=1)
    return np.ascontiguousarray(out)


def _invert_subject_stack_mirror(label_stack: np.ndarray, spec: StackSpec) -> np.ndarray:
    out = label_stack
    if spec.subject_mirror_lr:
        out = np.flip(out, axis=1)
    return np.ascontiguousarray(out)


def _prepare_atlas_under_spec(atlas: dict[str, object], subject_stack_zooms: tuple[float, float, float], spec: StackSpec) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    tpl = atlas["template"]
    lab = atlas["labels"]
    zooms = atlas["zooms"]

    tpl_o, zooms_o, _ = _reorient_stack(tpl, zooms, spec.atlas_slice_axis)
    lab_o, _, _ = _reorient_stack(lab, zooms, spec.atlas_slice_axis)

    if spec.atlas_reverse_ap:
        tpl_o = tpl_o[:, :, ::-1]
        lab_o = lab_o[:, :, ::-1]

    tpl_r = _resample_to_spacing(tpl_o, zooms_o, subject_stack_zooms, order=1)
    lab_r = _resample_to_spacing(lab_o, zooms_o, subject_stack_zooms, order=0)

    tpl_r = _percentile_clip(tpl_r)
    lab_r = np.asarray(np.rint(lab_r), dtype=np.int16)
    return tpl_r, lab_r, subject_stack_zooms


def _run_subject(
    subject_nii: Path,
    out_dir: Path,
    atlas_template: Path,
    atlas_labels: Path,
    label_ids: list[int] | None,
    crop_label_ids: list[int] | None,
    subject_slice_axis: int | None,
) -> dict[str, object]:
    case_dir = out_dir / subject_nii.stem.replace('.nii', '')
    _rm_if_exists(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    subject = _load_subject_stack(subject_nii, subject_slice_axis)
    atlas = _load_and_prepare_atlas(atlas_template, atlas_labels, crop_label_ids)

    subject_stack = subject["stack"]
    subject_stack_zooms = subject["stack_zooms"]

    specs = [
        StackSpec(atlas_slice_axis=a, atlas_reverse_ap=rev, subject_mirror_lr=mir)
        for a in (0, 1, 2)
        for rev in (False, True)
        for mir in (False, True)
    ]

    best = None
    summary_rows = []

    for spec in specs:
        subj_spec_stack = _apply_stack_spec(subject_stack, spec)
        atlas_spec_stack, atlas_spec_labels, _ = _prepare_atlas_under_spec(atlas, subject_stack_zooms, spec)

        # Need subject coverage to fit within atlas after resampling.
        if atlas_spec_stack.shape[2] < subj_spec_stack.shape[2] + 2:
            summary_rows.append({
                "spec": spec.name,
                "status": "skipped",
                "reason": "atlas stack too short after resampling",
                "atlas_shape": list(map(int, atlas_spec_stack.shape)),
            })
            continue

        scores, tforms = _compute_score_matrix(atlas_spec_stack, atlas_spec_labels, subj_spec_stack)
        finite_frac = float(np.isfinite(scores).mean())
        if finite_frac < 0.05:
            summary_rows.append({
                "spec": spec.name,
                "status": "skipped",
                "reason": "too few valid slice-pair scores",
                "finite_fraction": finite_frac,
                "atlas_shape": list(map(int, atlas_spec_stack.shape)),
            })
            continue

        path, total_score = _best_monotonic_path(scores)
        mean_score = float(total_score / max(subj_spec_stack.shape[2], 1))

        row = {
            "spec": spec.name,
            "atlas_slice_axis": int(spec.atlas_slice_axis),
            "atlas_reverse_ap": bool(spec.atlas_reverse_ap),
            "subject_mirror_lr": bool(spec.subject_mirror_lr),
            "total_score": float(total_score),
            "mean_score": mean_score,
            "atlas_shape": list(map(int, atlas_spec_stack.shape)),
            "path_first_last": [int(path[0]), int(path[-1])],
            "path": [int(x) for x in path],
            "finite_fraction": finite_frac,
        }
        summary_rows.append(row)

        if best is None or mean_score > best["mean_score"]:
            best = {
                "spec": spec,
                "scores": scores,
                "tforms": tforms,
                "path": path,
                "total_score": float(total_score),
                "mean_score": mean_score,
                "atlas_stack": atlas_spec_stack,
                "atlas_labels": atlas_spec_labels,
            }

    if best is None:
        raise SystemExit("No valid atlas/subject slice-matching hypothesis succeeded.")

    spec = best["spec"]
    path = best["path"]
    tforms = best["tforms"]
    atlas_spec_labels = best["atlas_labels"]
    subj_spec_stack = _apply_stack_spec(subject_stack, spec)

    pred_labels_stack = np.zeros_like(subj_spec_stack, dtype=np.int16)
    for i, j in enumerate(path):
        tfm = tforms[i][j]
        if tfm is None:
            continue
        src_lab = atlas_spec_labels[:, :, j].astype(np.float32)
        moved_lab = warp(src_lab, inverse_map=tfm.inverse, output_shape=subj_spec_stack[:, :, i].shape, order=0, preserve_range=True)
        pred_labels_stack[:, :, i] = np.rint(moved_lab).astype(np.int16)

    pred_labels_stack = _invert_subject_stack_mirror(pred_labels_stack, spec)
    pred_labels_raw = _invert_reorient_stack(pred_labels_stack, subject["perm"])

    out_labels = case_dir / "best_labels_on_original_scan_grid.nii.gz"
    _save_nifti(pred_labels_raw.astype(np.int16), subject["img"].affine, out_labels, like=subject["img"])

    out_mask = None
    centroid = None
    if label_ids:
        mask_raw = np.isin(pred_labels_raw, np.array(label_ids)).astype(np.uint8)
        out_mask = case_dir / "best_roi_mask_on_original_scan_grid.nii.gz"
        _save_nifti(mask_raw, subject["img"].affine, out_mask, like=subject["img"])
        vox = np.column_stack(np.where(mask_raw > 0))
        if vox.size > 0:
            ctr = vox.mean(axis=0)
            ctr_h = np.array([ctr[0], ctr[1], ctr[2], 1.0], dtype=float)
            centroid = [float(x) for x in (subject["img"].affine @ ctr_h)[:3]]

    summary_rows = sorted(summary_rows, key=lambda x: float(x.get("mean_score", float("-inf"))), reverse=True)

    summary = {
        "input_subject": str(subject_nii),
        "subject_shape": list(map(int, subject["raw_data"].shape)),
        "subject_zooms": [float(x) for x in subject["raw_zooms"]],
        "subject_slice_axis": int(subject["slice_axis"]),
        "subject_stack_shape": list(map(int, subject["stack"].shape)),
        "subject_stack_zooms": [float(x) for x in subject["stack_zooms"]],
        "best_hypothesis": {
            "spec": spec.name,
            "atlas_slice_axis": int(spec.atlas_slice_axis),
            "atlas_reverse_ap": bool(spec.atlas_reverse_ap),
            "subject_mirror_lr": bool(spec.subject_mirror_lr),
            "total_score": float(best["total_score"]),
            "mean_score": float(best["mean_score"]),
            "path": [int(x) for x in path],
            "final_labels_on_original_scan_grid": str(out_labels),
            "final_roi_mask_on_original_scan_grid": str(out_mask) if out_mask else None,
            "roi_centroid_mm_original_header": centroid,
        },
        "top_hypotheses": summary_rows[:10],
    }
    return summary


# -----------------------------
# CLI
# -----------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Slice-wise hippocampus/DG localization for anisotropic 2D coronal mouse T2 scans.")
    p.add_argument("--input-nii", type=Path, action="append", required=True,
                   help="One or more subject T2 NIfTI files.")
    p.add_argument("--atlas-template", type=Path, required=True)
    p.add_argument("--atlas-labels", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("analysis_out_coronal"))
    p.add_argument("--label-ids", type=int, nargs="*", default=None,
                   help="Final ROI label IDs to extract as a binary mask.")
    p.add_argument("--crop-label-ids", type=int, nargs="*", default=None,
                   help="Atlas labels used for atlas crop before matching. Default: same as --label-ids; else all nonzero labels.")
    p.add_argument("--subject-slice-axis", type=int, choices=[0, 1, 2], default=None,
                   help="Override inferred subject slice axis. Default: thickest-voxel axis.")
    args = p.parse_args()

    if not args.atlas_template.exists():
        raise SystemExit(f"Atlas template not found: {args.atlas_template}")
    if not args.atlas_labels.exists():
        raise SystemExit(f"Atlas labels not found: {args.atlas_labels}")
    for nii in args.input_nii:
        if not nii.exists():
            raise SystemExit(f"Input NIfTI not found: {nii}")

    crop_label_ids = args.crop_label_ids if args.crop_label_ids else args.label_ids

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "atlas_template": str(args.atlas_template),
        "atlas_labels": str(args.atlas_labels),
        "label_ids": [int(x) for x in (args.label_ids or [])],
        "crop_label_ids": [int(x) for x in (crop_label_ids or [])],
        "subjects": [],
    }

    for nii in args.input_nii:
        print(f"=== Processing {nii}")
        try:
            res = _run_subject(
                subject_nii=nii,
                out_dir=args.out_dir,
                atlas_template=args.atlas_template,
                atlas_labels=args.atlas_labels,
                label_ids=args.label_ids,
                crop_label_ids=crop_label_ids,
                subject_slice_axis=args.subject_slice_axis,
            )
        except Exception as exc:
            res = {
                "input_subject": str(nii),
                "status": "failed",
                "error": str(exc),
            }
        summary["subjects"].append(res)

    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Done. Summary written to: {out_json}")


if __name__ == "__main__":
    main()
