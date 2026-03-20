
#!/usr/bin/env python3
"""Manual-assisted hippocampus labeling for coronal 2D mouse T2 slabs.

This script is intentionally simple and robust:
1) infer the slice axis from the thickest voxel dimension in the subject scan
2) generate subject and atlas montages with slice indices
3) user chooses atlas coronal axis/direction and AP anchors
4) script maps subject slices -> atlas slices linearly across the chosen range
5) it auto-tests 4 in-plane flip hypotheses and chooses the best one
6) it performs 2D slice registration (ECC affine by default) and warps atlas labels
7) it writes the full multi-label atlas result back on the original subject scan grid
   and also saves a binary >0 mask plus QC overlays

This is meant for partial 2D TurboRARE coronal stacks where blind 3D automation
can be brittle and waste time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2
from scipy import ndimage


def _fix_zooms(zooms: Iterable[float]) -> tuple[float, float, float]:
    z = tuple(float(abs(v)) for v in list(zooms)[:3])
    if max(z) > 5.0:
        z = tuple(v * 0.001 for v in z)
    return z


def _move_axis_last(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(arr, axis, -1)


def _infer_subject_slice_axis(zooms: tuple[float, float, float]) -> int:
    return int(np.argmax(np.asarray(zooms)))


def _normalize_for_display(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(y)
    if not np.any(finite):
        return np.zeros_like(y, dtype=np.float32)
    vals = y[finite]
    lo = np.percentile(vals, 1.0)
    hi = np.percentile(vals, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max() + 1e-6)
    y = np.clip((y - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    y[~finite] = 0.0
    return y.astype(np.float32)


def _montage(arr3d: np.ndarray, out_png: Path, title: str, overlay3d: np.ndarray | None = None, max_slices: int | None = None) -> None:
    n = arr3d.shape[-1]
    if max_slices is not None:
        use_idx = np.linspace(0, n - 1, max_slices).round().astype(int)
        use_idx = np.unique(use_idx)
    else:
        use_idx = np.arange(n)

    cols = 6
    rows = math.ceil(len(use_idx) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.6))
    axs = np.array(axs).reshape(rows, cols)

    for ax in axs.flat:
        ax.axis("off")

    for k, i in enumerate(use_idx):
        ax = axs.flat[k]
        sl = arr3d[..., i].T
        ax.imshow(_normalize_for_display(sl), cmap="gray", origin="lower")
        if overlay3d is not None:
            ov = overlay3d[..., i].T > 0
            if np.any(ov):
                ax.contour(ov.astype(float), levels=[0.5], colors="r", linewidths=0.6)
        ax.set_title(str(int(i)), fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _crop_to_brain(slice2d: np.ndarray, pad: int = 10) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    img = _normalize_for_display(slice2d)
    thr = max(0.05, np.percentile(img, 60) * 0.25)
    mask = img > thr
    if np.count_nonzero(mask) < 50:
        h, w = img.shape
        return img, (0, h, 0, w)
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, img.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1], (y0, y1, x0, x1)


def _resample_slice_to_subject(atlas_slice: np.ndarray, atlas_label_slice: np.ndarray, atlas_zooms_xy: tuple[float, float], subj_shape_xy: tuple[int, int], subj_zooms_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    # resample atlas slice to subject in-plane pixel size, then center pad/crop to subject shape
    scale_y = atlas_zooms_xy[0] / subj_zooms_xy[0]
    scale_x = atlas_zooms_xy[1] / subj_zooms_xy[1]
    atlas_img = _normalize_for_display(atlas_slice)
    rs_img = ndimage.zoom(atlas_img, zoom=(scale_y, scale_x), order=1)
    rs_lab = ndimage.zoom(atlas_label_slice.astype(np.float32), zoom=(scale_y, scale_x), order=0)

    target_h, target_w = subj_shape_xy
    out_img = np.zeros((target_h, target_w), dtype=np.float32)
    out_lab = np.zeros((target_h, target_w), dtype=np.float32)

    sh, sw = rs_img.shape
    y0_out = max((target_h - sh) // 2, 0)
    x0_out = max((target_w - sw) // 2, 0)
    y0_in = max((sh - target_h) // 2, 0)
    x0_in = max((sw - target_w) // 2, 0)

    hh = min(target_h, sh)
    ww = min(target_w, sw)

    out_img[y0_out:y0_out + hh, x0_out:x0_out + ww] = rs_img[y0_in:y0_in + hh, x0_in:x0_in + ww]
    out_lab[y0_out:y0_out + hh, x0_out:x0_out + ww] = rs_lab[y0_in:y0_in + hh, x0_in:x0_in + ww]
    return out_img, out_lab


def _apply_flip2d(arr: np.ndarray, flip_lr: bool, flip_ud: bool) -> np.ndarray:
    out = arr
    if flip_lr:
        out = np.fliplr(out)
    if flip_ud:
        out = np.flipud(out)
    return np.ascontiguousarray(out)


def _ecc_register(moving: np.ndarray, fixed: np.ndarray, motion: str = "affine", n_iter: int = 500) -> tuple[np.ndarray, float]:
    # moving -> fixed
    mov = (_normalize_for_display(moving) * 255.0).astype(np.uint8)
    fix = (_normalize_for_display(fixed) * 255.0).astype(np.uint8)

    warp_mode = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }[motion]

    if warp_mode == cv2.MOTION_AFFINE:
        warp = np.eye(2, 3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, 1e-5)
    try:
        cc, warp = cv2.findTransformECC(
            templateImage=fix,
            inputImage=mov,
            warpMatrix=warp,
            motionType=warp_mode,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
        return warp, float(cc)
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), float("-inf")


def _warp2d(arr: np.ndarray, warp: np.ndarray, out_shape: tuple[int, int], order: int) -> np.ndarray:
    interp = cv2.INTER_NEAREST if order == 0 else cv2.INTER_LINEAR
    return cv2.warpAffine(
        arr.astype(np.float32),
        warp,
        dsize=(out_shape[1], out_shape[0]),
        flags=interp | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _prepare_atlas_orientation(tpl: np.ndarray, lab: np.ndarray, zooms: tuple[float, float, float], axis: int, reverse: bool) -> tuple[np.ndarray, np.ndarray, tuple[float, float], np.ndarray]:
    tpl_s = _move_axis_last(tpl, axis)
    lab_s = _move_axis_last(lab, axis)
    if reverse:
        tpl_s = tpl_s[..., ::-1]
        lab_s = lab_s[..., ::-1]
    inplane_axes = [a for a in range(3) if a != axis]
    inplane_zooms = (zooms[inplane_axes[0]], zooms[inplane_axes[1]])
    nonzero = np.where(np.any(lab_s > 0, axis=(0, 1)))[0]
    return tpl_s, lab_s, inplane_zooms, nonzero


def _linear_map_indices(n_subject: int, subj_start: int, subj_end: int, atlas_start: int, atlas_end: int) -> list[int | None]:
    out: list[int | None] = [None] * n_subject
    if subj_end <= subj_start:
        return out
    subj_positions = np.arange(subj_start, subj_end + 1)
    atlas_positions = np.linspace(atlas_start, atlas_end, len(subj_positions))
    atlas_positions = np.round(atlas_positions).astype(int)
    for s, a in zip(subj_positions, atlas_positions):
        out[int(s)] = int(a)
    return out


def _score_flip_combo(subject_stack: np.ndarray, atlas_stack: np.ndarray, atlas_label_stack: np.ndarray, subj_zooms_xy: tuple[float, float], atlas_zooms_xy: tuple[float, float], mapping: list[int | None], flip_lr: bool, flip_ud: bool, rep_slices: list[int], motion: str) -> float:
    scores = []
    for s in rep_slices:
        a = mapping[s]
        if a is None:
            continue
        subj = subject_stack[..., s]
        atl = atlas_stack[..., a]
        lab = atlas_label_stack[..., a]
        atl_rs, _ = _resample_slice_to_subject(atl, lab, atlas_zooms_xy, subj.shape, subj_zooms_xy)
        atl_rs = _apply_flip2d(atl_rs, flip_lr, flip_ud)
        warp, cc = _ecc_register(atl_rs, subj, motion=motion)
        if np.isfinite(cc):
            scores.append(cc)
    if not scores:
        return float("-inf")
    return float(np.median(scores))


def _overlay_qc(subject_stack: np.ndarray, label_stack: np.ndarray, out_png: Path, title: str) -> None:
    _montage(subject_stack, out_png, title=title, overlay3d=label_stack)


def main() -> None:
    p = argparse.ArgumentParser(description="Manual-assisted hippocampus labeling for coronal 2D mouse T2 slabs.")
    p.add_argument("--input-nii", type=Path, action="append", required=True)
    p.add_argument("--atlas-template", type=Path, required=True)
    p.add_argument("--atlas-labels", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("analysis_out_manual_coronal"))
    p.add_argument("--atlas-axis", type=int, choices=[0, 1, 2], default=None, help="Atlas coronal/AP slice axis. If omitted, montages for all 3 axes are created and you will be prompted.")
    p.add_argument("--atlas-reverse", action="store_true", help="Reverse atlas slice order along the chosen axis.")
    p.add_argument("--subject-start", type=int, default=None, help="First subject slice that should receive hippocampus labels.")
    p.add_argument("--subject-end", type=int, default=None, help="Last subject slice that should receive hippocampus labels.")
    p.add_argument("--atlas-start", type=int, default=None, help="First atlas slice matching subject-start.")
    p.add_argument("--atlas-end", type=int, default=None, help="Last atlas slice matching subject-end.")
    p.add_argument("--motion", choices=["translation", "euclidean", "affine"], default="affine")
    args = p.parse_args()

    if not args.atlas_template.exists():
        raise SystemExit(f"Atlas template not found: {args.atlas_template}")
    if not args.atlas_labels.exists():
        raise SystemExit(f"Atlas labels not found: {args.atlas_labels}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    atlas_tpl_img = nib.load(str(args.atlas_template))
    atlas_lab_img = nib.load(str(args.atlas_labels))
    atlas_tpl = np.asarray(atlas_tpl_img.get_fdata(), dtype=np.float32)
    atlas_lab = np.asarray(atlas_lab_img.get_fdata())
    atlas_zooms = _fix_zooms(atlas_tpl_img.header.get_zooms()[:3])

    # Prepare atlas montages for all axes so user can choose visually if needed.
    atlas_choice_json = {}
    for axis in (0, 1, 2):
        for reverse in (False, True):
            tpl_s, lab_s, _, nonzero = _prepare_atlas_orientation(atlas_tpl, atlas_lab, atlas_zooms, axis, reverse)
            title = f"atlas axis={axis} reverse={int(reverse)} | red=label | nonzero slices {int(nonzero.min()) if len(nonzero) else 'NA'}..{int(nonzero.max()) if len(nonzero) else 'NA'}"
            out_png = args.out_dir / f"atlas_axis{axis}_rev{int(reverse)}_montage.png"
            _montage(tpl_s, out_png, title=title, overlay3d=(lab_s > 0).astype(np.uint8), max_slices=min(tpl_s.shape[-1], 36))
            atlas_choice_json[f"axis{axis}_rev{int(reverse)}"] = {
                "montage": str(out_png),
                "nonzero_slice_min": int(nonzero.min()) if len(nonzero) else None,
                "nonzero_slice_max": int(nonzero.max()) if len(nonzero) else None,
                "n_slices": int(tpl_s.shape[-1]),
            }

    atlas_choice_path = args.out_dir / "atlas_montages.json"
    atlas_choice_path.write_text(json.dumps(atlas_choice_json, indent=2) + "\n")

    summary = {
        "atlas_template": str(args.atlas_template),
        "atlas_labels": str(args.atlas_labels),
        "atlas_montages_json": str(atlas_choice_path),
        "subjects": [],
    }

    for subj_path in args.input_nii:
        if not subj_path.exists():
            summary["subjects"].append({"input_subject": str(subj_path), "status": "failed", "error": "file not found"})
            continue

        case_dir = args.out_dir / subj_path.stem.replace(".nii", "")
        case_dir.mkdir(parents=True, exist_ok=True)

        subj_img = nib.load(str(subj_path))
        subj_data = np.asarray(subj_img.get_fdata(), dtype=np.float32)
        subj_zooms = _fix_zooms(subj_img.header.get_zooms()[:3])

        if subj_data.ndim != 3:
            summary["subjects"].append({"input_subject": str(subj_path), "status": "failed", "error": f"expected 3D, got {subj_data.shape}"})
            continue

        subj_axis = _infer_subject_slice_axis(subj_zooms)
        subj_stack = _move_axis_last(subj_data, subj_axis)
        inplane_axes = [a for a in range(3) if a != subj_axis]
        subj_zooms_xy = (subj_zooms[inplane_axes[0]], subj_zooms[inplane_axes[1]])
        subj_montage = case_dir / "subject_montage.png"
        _montage(subj_stack, subj_montage, title=f"subject slices | inferred slice axis={subj_axis}")

        atlas_axis = args.atlas_axis
        atlas_reverse = bool(args.atlas_reverse)
        if atlas_axis is None:
            print(f"\nSubject montage written to: {subj_montage}")
            print(f"Atlas montage summary written to: {atlas_choice_path}")
            atlas_axis = int(input("Choose atlas axis (0/1/2): ").strip())
            atlas_reverse = bool(int(input("Reverse atlas slice order? (0/1): ").strip()))

        atlas_stack, atlas_label_stack, atlas_zooms_xy, nonzero = _prepare_atlas_orientation(
            atlas_tpl, atlas_lab, atlas_zooms, atlas_axis, atlas_reverse
        )
        if len(nonzero) == 0:
            summary["subjects"].append({"input_subject": str(subj_path), "status": "failed", "error": "chosen atlas orientation has no nonzero labels"})
            continue

        # Ask for AP anchors if not provided.
        s0 = args.subject_start
        s1 = args.subject_end
        a0 = args.atlas_start
        a1 = args.atlas_end
        if None in (s0, s1, a0, a1):
            print(f"\nSubject montage written to: {subj_montage}")
            print(f"Chosen atlas montage: {args.out_dir / f'atlas_axis{atlas_axis}_rev{int(atlas_reverse)}_montage.png'}")
            print(f"Suggested atlas nonzero label slice range: {int(nonzero.min())}..{int(nonzero.max())}")
            if s0 is None:
                s0 = int(input("subject-start slice index: ").strip())
            if s1 is None:
                s1 = int(input("subject-end slice index: ").strip())
            if a0 is None:
                a0 = int(input("atlas-start slice index: ").strip())
            if a1 is None:
                a1 = int(input("atlas-end slice index: ").strip())

        s0, s1, a0, a1 = int(s0), int(s1), int(a0), int(a1)
        mapping = _linear_map_indices(subj_stack.shape[-1], s0, s1, a0, a1)

        rep = list(np.linspace(s0, s1, min(6, max(2, s1 - s0 + 1))).round().astype(int))
        flip_scores = []
        for flr in (False, True):
            for fud in (False, True):
                sc = _score_flip_combo(
                    subject_stack=subj_stack,
                    atlas_stack=atlas_stack,
                    atlas_label_stack=atlas_label_stack,
                    subj_zooms_xy=subj_zooms_xy,
                    atlas_zooms_xy=atlas_zooms_xy,
                    mapping=mapping,
                    flip_lr=flr,
                    flip_ud=fud,
                    rep_slices=rep,
                    motion=args.motion,
                )
                flip_scores.append((sc, flr, fud))
        flip_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_flr, best_fud = flip_scores[0]

        label_dtype = np.int16 if np.issubdtype(atlas_label_stack.dtype, np.integer) else np.int16
        label_stack_subject = np.zeros(subj_stack.shape, dtype=label_dtype)
        per_slice_scores = {}

        for s in range(subj_stack.shape[-1]):
            a = mapping[s]
            if a is None:
                continue
            subj_sl = subj_stack[..., s]
            atl_sl = atlas_stack[..., a]
            atl_lab_sl = atlas_label_stack[..., a]
            atl_rs, lab_rs = _resample_slice_to_subject(atl_sl, atl_lab_sl, atlas_zooms_xy, subj_sl.shape, subj_zooms_xy)
            atl_rs = _apply_flip2d(atl_rs, best_flr, best_fud)
            lab_rs = _apply_flip2d(lab_rs, best_flr, best_fud)
            warp, cc = _ecc_register(atl_rs, subj_sl, motion=args.motion)
            warped_lab = _warp2d(lab_rs.astype(np.float32), warp, subj_sl.shape, order=0)
            label_stack_subject[..., s] = np.rint(warped_lab).astype(label_dtype)
            per_slice_scores[int(s)] = float(cc)

        # Move stack back to original voxel order and save on original header/affine.
        label_raw = np.moveaxis(label_stack_subject, -1, subj_axis)
        out_labels = case_dir / "atlas_labels_on_original_scan_grid.nii.gz"
        nib.save(nib.Nifti1Image(label_raw.astype(np.int16), subj_img.affine, subj_img.header), str(out_labels))

        out_mask = case_dir / "hippocampus_mask_on_original_scan_grid.nii.gz"
        nib.save(nib.Nifti1Image((label_raw > 0).astype(np.uint8), subj_img.affine, subj_img.header), str(out_mask))

        qc_png = case_dir / "qc_overlay_montage.png"
        _overlay_qc(subj_stack, label_stack_subject, qc_png, title=f"QC overlay | atlas axis={atlas_axis} rev={int(atlas_reverse)} flipLR={int(best_flr)} flipUD={int(best_fud)}")

        info = {
            "input_subject": str(subj_path),
            "status": "ok",
            "subject_montage": str(subj_montage),
            "atlas_axis": int(atlas_axis),
            "atlas_reverse": bool(atlas_reverse),
            "subject_slice_range": [int(s0), int(s1)],
            "atlas_slice_range": [int(a0), int(a1)],
            "best_inplane_flip_lr": bool(best_flr),
            "best_inplane_flip_ud": bool(best_fud),
            "best_flip_score_median_ecc": float(best_score),
            "per_slice_scores": per_slice_scores,
            "output_labels": str(out_labels),
            "output_mask": str(out_mask),
            "qc_overlay_montage": str(qc_png),
        }
        summary["subjects"].append(info)

    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Done. Summary written to: {out_json}")


if __name__ == "__main__":
    main()
