#!/usr/bin/env python3
"""Robust hippocampus / DG localization on partial mouse T2 slabs.

Why this rewrite exists
-----------------------
The original pipeline assumes the NIfTI headers are already anatomically truthful
and then calls canonicalization / resampling very early. That is exactly the
wrong thing to do when Bruker->NIfTI conversion has produced an incorrect A/P
(or broader orientation) header: once a wrong header is canonicalized, the wrong
geometry becomes internally consistent and registration keeps failing in a way
that looks like a persistent A/P swap.

This script takes a different approach:

1) Ignore the original orientation header semantics as much as possible.
   We keep only the voxel sizes (absolute zooms) and the raw voxel arrays.

2) Put atlas and subject into a synthetic, centered voxel-space affine.
   This makes registration independent of whatever qform/sform madness came
   from conversion.

3) Crop the atlas around the structure of interest (hippocampus / DG) using the
   atlas labels directly. This is much better for partial slabs than trying to
   register to a near-whole-brain template and manually guessing an A/P crop.

4) Search orientation hypotheses automatically. By default this tries all 48
   axis-permutation / flip combinations of the subject voxel array, runs a rigid
   registration for each candidate, scores the result, and keeps the best one.

5) Warp labels back to the chosen subject orientation, then inverse-reorient the
   labels into the ORIGINAL raw subject voxel order and save them with the
   ORIGINAL subject affine/header. So the final overlay can be opened directly on
   your original scan.

Important note
--------------
If there is no external left/right marker and the anatomy is highly symmetric,
left-right can remain intrinsically ambiguous. For bilateral hippocampus/DG
localization this is usually acceptable, but for unilateral implant localization
it is not. In that case you need an acquisition-side orientation truth source.

This version also restores the Bruker->NIfTI discovery/conversion path that was
present in your earlier script. That means:

- existing NIfTIs are discovered first
- if none are found, the script can attempt Bruker conversion automatically
- scan IDs are resolved from filenames OR parent folder names more robustly,
  rather than only matching "scan-<id>_..."
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


EXCLUDED_DIR_NAMES = {
    "analysis_out",
    "mouse_atlas",
    "atlas",
    "templates",
    ".git",
    ".venv",
    "__pycache__",
}


# -----------------------------
# small utilities
# -----------------------------


def _run(cmd: list[str]) -> str:
    print("+", " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def _require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required command not found in PATH: {name}")


def _rm_if_exists(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _is_hidden_or_env_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _is_scan_like_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name.isdigit()
        and (path / "acqp").exists()
        and (path / "method").exists()
    )


def _looks_like_bruker_root(root: Path) -> bool:
    if (root / "study.MR").exists() and (root / "subject").exists():
        return True
    if (root / "pdata" / "1" / "2dseq").exists():
        return True
    if any(root.glob("pdata/*/2dseq")):
        return True
    return any(root.glob("*/pdata/*/2dseq"))


def _looks_like_t2_bruker_scan(scan_dir: Path) -> bool:
    method_path = scan_dir / "method"
    if not method_path.exists():
        return False
    try:
        method_text = method_path.read_text(errors="ignore").lower()
    except OSError:
        return False
    t2_markers = (
        "rare",
        "turborare",
        "t2",
        "rapid acquisition with relaxation enhancement",
    )
    return any(marker in method_text for marker in t2_markers)


def _is_3d_nifti(path: Path) -> bool:
    try:
        return len(nib.load(str(path)).shape) == 3
    except Exception:
        return False


def _find_t2_niftis(root: Path, exclude_paths: list[Path] | None = None) -> list[Path]:
    exclude_paths = [p.resolve() for p in (exclude_paths or [])]
    all_nifti = sorted([*root.rglob("*.nii"), *root.rglob("*.nii.gz")])

    kept: list[Path] = []
    for p in all_nifti:
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        if _is_hidden_or_env_path(rel):
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
            continue
        rp = p.resolve()
        if any(rp == ex or ex in rp.parents for ex in exclude_paths):
            continue
        name = p.name.lower()
        if any(tok in name for tok in ("annotation", "template", "atlas", "label", "roi")):
            continue
        kept.append(p)

    t2 = [p for p in kept if any(tok in p.name.lower() for tok in ("t2", "rare", "turborare"))]
    return t2 or kept


def _scan_dirs_in_tree(root: Path) -> list[Path]:
    out: list[Path] = []
    if _is_scan_like_dir(root):
        out.append(root)
    for child in sorted(root.iterdir()):
        if _is_scan_like_dir(child):
            out.append(child)
    return out


def _find_all_niftis_under(root: Path) -> list[Path]:
    files = sorted([*root.rglob("*.nii"), *root.rglob("*.nii.gz")])
    return [p for p in files if _is_3d_nifti(p)]


def _infer_scan_id_from_path(path: Path) -> int | None:
    """Best-effort scan ID inference from filename and parent folders.

    Handles common patterns such as:
      scan-5_xxx.nii.gz
      scan_5_xxx.nii.gz
      .../5/.../file.nii.gz
      .../scan-5/.../file.nii.gz
    """
    candidates = [path.stem, path.name]
    candidates.extend(parent.name for parent in path.parents[:4])

    patterns = [
        re.compile(r"(?:^|[^0-9])scan[-_]?([0-9]+)(?:[^0-9]|$)", re.IGNORECASE),
        re.compile(r"^(\d+)$"),
    ]

    for text in candidates:
        for pat in patterns:
            m = pat.search(text)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
    return None


def _find_scan_file_by_id(scans: list[Path], scan_id: int) -> Path | None:
    token = f"scan-{scan_id}_"
    for p in scans:
        if token in p.name:
            return p

    exact_matches = [p for p in scans if _infer_scan_id_from_path(p) == scan_id]
    if exact_matches:
        exact_matches.sort(key=lambda p: len(p.parts))
        return exact_matches[0]
    return None


def _bruker_input_candidates(root: Path) -> list[Path]:
    candidates = [root]
    ancestors = [root, *root.parents]
    for anc in ancestors:
        if anc == anc.parent:
            continue
        if _is_scan_like_dir(anc):
            candidates.append(anc)
            candidates.append(anc.parent)
        elif (anc / "study.MR").exists():
            candidates.append(anc)

    if "pdata" in root.parts:
        pdata_idx = root.parts.index("pdata")
        if pdata_idx > 0:
            scan_guess = Path(*root.parts[:pdata_idx])
            candidates.extend([scan_guess, scan_guess.parent])

    uniq: list[Path] = []
    for c in candidates:
        if c not in uniq:
            uniq.append(c)
    return uniq


def _resolve_bruker_converter(converter_cmd: str | None) -> str:
    resolved = converter_cmd or "brkraw"
    converter_exe = shlex.split(resolved)[0]
    _require_cmd(converter_exe)
    return resolved


def _converter_commands(
    converter_cmd: str,
    converter_args_template: str,
    input_dir: Path,
    output_dir: Path,
) -> list[list[str]]:
    scan_id = input_dir.name

    def build(template: str) -> list[str]:
        rendered = template.format(
            input=input_dir.as_posix(),
            output=output_dir.as_posix(),
            scan_id=scan_id,
        )
        return shlex.split(converter_cmd) + shlex.split(rendered)

    if converter_args_template.strip().lower() != "auto":
        return [build(converter_args_template)]

    candidates = [
        "convert {input} -o {output}",
        "convert -i {input} -o {output}",
        "convert-batch {input} -o {output}",
        "convert-batch -i {input} -o {output}",
    ]
    return [build(tpl) for tpl in candidates]


def _convert_bruker_to_nifti(
    input_root: Path,
    converted_dir: Path,
    converter_cmd: str | None,
    converter_args_template: str,
) -> list[Path]:
    converted_dir.mkdir(parents=True, exist_ok=True)
    resolved_converter = _resolve_bruker_converter(converter_cmd)
    print(f"Using Bruker converter command: {resolved_converter}")

    last_error: subprocess.CalledProcessError | None = None
    for idx, candidate in enumerate(_bruker_input_candidates(input_root), start=1):
        attempt_out = converted_dir / f"attempt_{idx}"
        if attempt_out.exists():
            shutil.rmtree(attempt_out)
        attempt_out.mkdir(parents=True, exist_ok=True)

        if candidate != input_root:
            print(f"Retrying Bruker conversion with likely study root: {candidate}")

        for cmd in _converter_commands(
            converter_cmd=resolved_converter,
            converter_args_template=converter_args_template,
            input_dir=candidate,
            output_dir=attempt_out,
        ):
            try:
                _run(cmd)
            except subprocess.CalledProcessError as exc:
                last_error = exc
                continue

            converted = _find_t2_niftis(attempt_out)
            if converted:
                return converted

    if last_error is not None:
        raise SystemExit(
            "Bruker conversion failed for all candidate input paths. "
            "Check input root and converter arguments."
        ) from last_error

    return _find_t2_niftis(converted_dir)


def _fix_suspicious_zooms(zooms: Iterable[float]) -> tuple[float, float, float]:
    vals = tuple(float(abs(z)) for z in list(zooms)[:3])
    if max(vals) > 5.0:
        vals = tuple(v * 0.001 for v in vals)
    return vals


def _centered_affine(shape: tuple[int, int, int], zooms: tuple[float, float, float]) -> np.ndarray:
    aff = np.eye(4, dtype=float)
    aff[0, 0], aff[1, 1], aff[2, 2] = zooms
    center = 0.5 * (np.array(shape, dtype=float) - 1.0)
    aff[:3, 3] = -center * np.array(zooms, dtype=float)
    return aff


def _save_nifti(data: np.ndarray, affine: np.ndarray, out_path: Path, like: nib.Nifti1Image | None = None) -> None:
    hdr = like.header.copy() if like is not None else nib.Nifti1Header()
    if np.issubdtype(data.dtype, np.integer):
        out = nib.Nifti1Image(data, affine, header=hdr)
    else:
        out = nib.Nifti1Image(data.astype(np.float32), affine, header=hdr)
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)
    nib.save(out, str(out_path))


def _bbox_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(mask):
        raise ValueError("Mask is empty; cannot compute bounding box.")
    ijk = np.where(mask)
    mins = np.array([int(ijk[d].min()) for d in range(3)], dtype=int)
    maxs = np.array([int(ijk[d].max()) for d in range(3)], dtype=int)
    return mins, maxs


def _expand_bbox(mins: np.ndarray, maxs: np.ndarray, shape: tuple[int, int, int], margin_vox: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    start = np.maximum(mins - np.array(margin_vox, dtype=int), 0)
    stop = np.minimum(maxs + np.array(margin_vox, dtype=int) + 1, np.array(shape, dtype=int))
    return start, stop


def _crop_array(arr: np.ndarray, start: np.ndarray, stop: np.ndarray) -> np.ndarray:
    return arr[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]


def _crop_affine(affine: np.ndarray, start: np.ndarray) -> np.ndarray:
    new_aff = affine.copy()
    offset = np.array([start[0], start[1], start[2], 1.0], dtype=float)
    new_aff[:3, 3] = (affine @ offset)[:3]
    return new_aff


def _percentile_clip(arr: np.ndarray, lo=0.5, hi=99.5) -> np.ndarray:
    nz = arr[np.isfinite(arr)]
    if nz.size == 0:
        return arr.astype(np.float32)
    a = np.percentile(nz, lo)
    b = np.percentile(nz, hi)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return arr.astype(np.float32)
    return np.clip(arr, a, b).astype(np.float32)


def _nmi(a: np.ndarray, b: np.ndarray, bins: int = 64, mask: np.ndarray | None = None) -> float:
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
    a = _percentile_clip(a)
    b = _percentile_clip(b)
    ah, aedges = np.histogram(a, bins=bins)
    bh, bedges = np.histogram(b, bins=bins)
    jh, _, _ = np.histogram2d(a, b, bins=(aedges, bedges))
    pxy = jh / np.maximum(jh.sum(), 1.0)
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
# orientation search machinery
# -----------------------------


@dataclass(frozen=True)
class OrientationSpec:
    perm: tuple[int, int, int]
    flips: tuple[bool, bool, bool]

    @property
    def name(self) -> str:
        p = "".join(str(x) for x in self.perm)
        f = "".join("1" if x else "0" for x in self.flips)
        return f"perm{p}_flip{f}"



def _all_orientation_specs() -> list[OrientationSpec]:
    specs: list[OrientationSpec] = []
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product([False, True], repeat=3):
            specs.append(OrientationSpec(tuple(perm), tuple(bool(x) for x in flips)))
    return specs



def _apply_spec(arr: np.ndarray, spec: OrientationSpec) -> np.ndarray:
    out = np.transpose(arr, axes=spec.perm)
    for ax, do_flip in enumerate(spec.flips):
        if do_flip:
            out = np.flip(out, axis=ax)
    return np.ascontiguousarray(out)



def _invert_spec(arr: np.ndarray, spec: OrientationSpec) -> np.ndarray:
    out = np.asarray(arr)
    for ax, do_flip in enumerate(spec.flips):
        if do_flip:
            out = np.flip(out, axis=ax)
    inv_perm = np.argsort(np.array(spec.perm))
    out = np.transpose(out, axes=inv_perm)
    return np.ascontiguousarray(out)



def _permuted_zooms(zooms: tuple[float, float, float], spec: OrientationSpec) -> tuple[float, float, float]:
    return tuple(float(zooms[i]) for i in spec.perm)


# -----------------------------
# preprocessing
# -----------------------------


def _load_standardized_raw(nii_path: Path, out_copy: Path) -> dict[str, object]:
    img = nib.load(str(nii_path))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise SystemExit(f"Expected 3D image: {nii_path}, got shape={data.shape}")
    zooms = _fix_suspicious_zooms(img.header.get_zooms()[:3])
    aff = _centered_affine(tuple(int(x) for x in data.shape), zooms)
    _save_nifti(data, aff, out_copy, like=img)
    return {
        "orig_path": str(nii_path),
        "shape": [int(x) for x in data.shape],
        "orig_header_zooms": [float(z) for z in img.header.get_zooms()[:3]],
        "used_zooms": [float(z) for z in zooms],
        "standardized_path": str(out_copy),
        "original_affine": [[float(x) for x in row] for row in img.affine.tolist()],
    }



def _maybe_n4(input_nii: Path, output_nii: Path, enabled: bool) -> Path:
    if not enabled:
        shutil.copy2(input_nii, output_nii)
        return output_nii
    _require_cmd("N4BiasFieldCorrection")
    _run(["N4BiasFieldCorrection", "-d", "3", "-i", str(input_nii), "-o", str(output_nii)])
    return output_nii



def _prepare_atlas(
    atlas_template: Path,
    atlas_labels: Path,
    out_dir: Path,
    crop_label_ids: list[int] | None,
    crop_margin_mm: float,
    do_n4: bool,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    tpl_std = out_dir / "atlas_template_std.nii.gz"
    lab_std = out_dir / "atlas_labels_std.nii.gz"
    tpl_info = _load_standardized_raw(atlas_template, tpl_std)
    lab_info = _load_standardized_raw(atlas_labels, lab_std)

    tpl_work = out_dir / "atlas_template_std_n4.nii.gz"
    _maybe_n4(tpl_std, tpl_work, enabled=do_n4)

    timg = nib.load(str(tpl_work))
    limg = nib.load(str(lab_std))
    tdat = np.asarray(timg.get_fdata(), dtype=np.float32)
    ldat = np.asarray(limg.get_fdata())
    zooms = tuple(float(abs(z)) for z in limg.header.get_zooms()[:3])

    if crop_label_ids:
        crop_mask = np.isin(ldat, np.array(crop_label_ids, dtype=ldat.dtype))
    else:
        crop_mask = ldat != 0

    if not np.any(crop_mask):
        raise SystemExit(
            "Atlas crop mask is empty. Check your --crop-label-ids / --label-ids against the atlas labels."
        )

    mins, maxs = _bbox_from_mask(crop_mask)
    margin_vox = tuple(int(math.ceil(crop_margin_mm / max(z, 1e-6))) for z in zooms)
    start, stop = _expand_bbox(mins, maxs, ldat.shape, margin_vox)

    t_crop = _crop_array(tdat, start, stop)
    l_crop = _crop_array(ldat, start, stop)
    mask_crop = (l_crop != 0).astype(np.uint8)

    crop_aff = _crop_affine(limg.affine, start)

    tpl_crop = out_dir / "atlas_template_crop.nii.gz"
    lab_crop = out_dir / "atlas_labels_crop.nii.gz"
    mask_crop_nii = out_dir / "atlas_mask_crop.nii.gz"
    _save_nifti(t_crop.astype(np.float32), crop_aff, tpl_crop, like=timg)
    _save_nifti(l_crop.astype(np.int16), crop_aff, lab_crop, like=limg)
    _save_nifti(mask_crop, crop_aff, mask_crop_nii, like=limg)

    return {
        "template_info": tpl_info,
        "labels_info": lab_info,
        "template_n4": str(tpl_work),
        "crop_label_ids": [int(x) for x in (crop_label_ids or [])],
        "crop_margin_mm": float(crop_margin_mm),
        "bbox_start": [int(x) for x in start],
        "bbox_stop": [int(x) for x in stop],
        "template_crop": str(tpl_crop),
        "labels_crop": str(lab_crop),
        "mask_crop": str(mask_crop_nii),
    }


# -----------------------------
# registration and scoring
# -----------------------------


def _register_candidate(
    fixed_template: Path,
    fixed_labels: Path,
    fixed_mask: Path,
    moving_subject: Path,
    out_dir: Path,
    use_affine_stage: bool,
) -> dict[str, object]:
    _require_cmd("antsRegistration")
    _require_cmd("antsApplyTransforms")

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "reg_"
    warped = out_dir / "reg_Warped.nii.gz"
    inverse_warped = out_dir / "reg_InverseWarped.nii.gz"

    cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "1",
        "--output", f"[{prefix},{warped},{inverse_warped}]",
        "--interpolation", "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "0",
        "--initial-moving-transform", f"[{fixed_template},{moving_subject},1]",
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{fixed_template},{moving_subject},1,32,Regular,0.25]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "-x", str(fixed_mask),
    ]
    if use_affine_stage:
        cmd += [
            "--transform", "Affine[0.05]",
            "--metric", f"MI[{fixed_template},{moving_subject},1,32,Regular,0.25]",
            "--convergence", "[1000x500x250x100,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
            "-x", str(fixed_mask),
        ]
    log_text = _run(cmd)

    affine = out_dir / "reg_0GenericAffine.mat"
    labels_in_subject = out_dir / "labels_in_subject_candidate_space.nii.gz"
    _run([
        "antsApplyTransforms",
        "-d", "3",
        "-i", str(fixed_labels),
        "-r", str(moving_subject),
        "-o", str(labels_in_subject),
        "-n", "NearestNeighbor",
        "-t", f"[{affine},1]",
    ])

    fixed_img = nib.load(str(fixed_template))
    warped_img = nib.load(str(warped))
    mask_img = nib.load(str(fixed_mask))

    fixed_arr = np.asarray(fixed_img.get_fdata(), dtype=np.float32)
    warped_arr = np.asarray(warped_img.get_fdata(), dtype=np.float32)
    mask_arr = np.asarray(mask_img.get_fdata()) > 0
    score_nmi = _nmi(fixed_arr, warped_arr, bins=64, mask=mask_arr)

    return {
        "warped_moving_to_fixed": str(warped),
        "inverse_warped": str(inverse_warped),
        "affine": str(affine),
        "labels_in_subject_candidate_space": str(labels_in_subject),
        "score_nmi": float(score_nmi),
        "registration_log": log_text,
    }


# -----------------------------
# subject pipeline
# -----------------------------


def _mask_centroid_mm(mask_data: np.ndarray, affine: np.ndarray) -> list[float] | None:
    if not np.any(mask_data > 0):
        return None
    vox = np.column_stack(np.where(mask_data > 0))
    ctr = vox.mean(axis=0)
    ctr_h = np.array([ctr[0], ctr[1], ctr[2], 1.0], dtype=float)
    mm = (affine @ ctr_h)[:3]
    return [float(x) for x in mm]



def _prepare_subject_candidate(
    subject_raw_std: Path,
    spec: OrientationSpec,
    out_dir: Path,
    do_n4: bool,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(subject_raw_std))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    zooms = tuple(float(abs(z)) for z in img.header.get_zooms()[:3])

    data_or = _apply_spec(data, spec)
    zooms_or = _permuted_zooms(zooms, spec)
    aff_or = _centered_affine(tuple(int(x) for x in data_or.shape), zooms_or)

    raw_or = out_dir / "subject_candidate_raw.nii.gz"
    _save_nifti(data_or, aff_or, raw_or, like=img)

    final_or = out_dir / "subject_candidate_final.nii.gz"
    _maybe_n4(raw_or, final_or, enabled=do_n4)

    return {
        "spec": asdict(spec),
        "candidate_raw": str(raw_or),
        "candidate_final": str(final_or),
        "shape": [int(x) for x in data_or.shape],
        "zooms": [float(x) for x in zooms_or],
    }



def _run_one_subject(
    subject_nii: Path,
    out_dir: Path,
    atlas_info: dict[str, object],
    do_n4_subject: bool,
    label_ids: list[int] | None,
    crop_label_ids: list[int] | None,
    use_affine_stage: bool,
    max_candidates: int | None,
) -> dict[str, object]:
    case_dir = out_dir / subject_nii.stem.replace(".nii", "")
    _rm_if_exists(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    orig_img = nib.load(str(subject_nii))

    subj_std = case_dir / "subject_raw_std.nii.gz"
    subj_std_info = _load_standardized_raw(subject_nii, subj_std)

    specs = _all_orientation_specs()
    if max_candidates is not None:
        specs = specs[:max_candidates]

    candidates_summary: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_spec: OrientationSpec | None = None
    best_label_img: nib.Nifti1Image | None = None

    fixed_template = Path(atlas_info["template_crop"])
    fixed_labels = Path(atlas_info["labels_crop"])
    fixed_mask = Path(atlas_info["mask_crop"])

    for idx, spec in enumerate(specs, start=1):
        cand_dir = case_dir / "candidates" / f"{idx:02d}_{spec.name}"
        prep = _prepare_subject_candidate(subj_std, spec, cand_dir / "prep", do_n4=do_n4_subject)

        try:
            reg = _register_candidate(
                fixed_template=fixed_template,
                fixed_labels=fixed_labels,
                fixed_mask=fixed_mask,
                moving_subject=Path(prep["candidate_final"]),
                out_dir=cand_dir / "reg",
                use_affine_stage=use_affine_stage,
            )
            score = float(reg["score_nmi"])
            row = {
                "candidate_index": idx,
                "name": spec.name,
                "perm": list(spec.perm),
                "flips": [bool(x) for x in spec.flips],
                "score_nmi": score,
                "candidate_dir": str(cand_dir),
            }
            candidates_summary.append(row)
            if best is None or score > float(best["score_nmi"]):
                best = {**row, "prep": prep, "reg": reg}
                best_spec = spec
                best_label_img = nib.load(reg["labels_in_subject_candidate_space"])
        except subprocess.CalledProcessError as exc:
            candidates_summary.append({
                "candidate_index": idx,
                "name": spec.name,
                "perm": list(spec.perm),
                "flips": [bool(x) for x in spec.flips],
                "status": "failed",
                "error": str(exc),
                "candidate_dir": str(cand_dir),
            })

    if best is None or best_spec is None or best_label_img is None:
        raise SystemExit(f"All candidate registrations failed for subject: {subject_nii}")

    best_label_data_candidate = np.asarray(best_label_img.get_fdata())
    best_label_data_raw_order = _invert_spec(best_label_data_candidate, best_spec)
    final_labels_native = case_dir / "best_labels_on_original_scan_grid.nii.gz"
    _save_nifti(best_label_data_raw_order.astype(np.int16), orig_img.affine, final_labels_native, like=orig_img)

    final_roi_mask_native = None
    centroid_native = None
    if label_ids:
        roi_mask_native_data = np.isin(best_label_data_raw_order, np.array(label_ids)).astype(np.uint8)
        final_roi_mask_native = case_dir / "best_roi_mask_on_original_scan_grid.nii.gz"
        _save_nifti(roi_mask_native_data, orig_img.affine, final_roi_mask_native, like=orig_img)
        centroid_native = _mask_centroid_mm(roi_mask_native_data, orig_img.affine)

    candidates_summary = sorted(
        candidates_summary,
        key=lambda x: float(x.get("score_nmi", float("-inf"))),
        reverse=True,
    )

    top_candidates = candidates_summary[: min(10, len(candidates_summary))]

    return {
        "input_subject": str(subject_nii),
        "subject_raw_standardized": subj_std_info,
        "crop_label_ids_used": [int(x) for x in (crop_label_ids or [])],
        "target_label_ids": [int(x) for x in (label_ids or [])],
        "best_candidate": {
            "name": best["name"],
            "perm": best["perm"],
            "flips": best["flips"],
            "score_nmi": float(best["score_nmi"]),
            "candidate_final": best["prep"]["candidate_final"],
            "labels_in_candidate_space": best["reg"]["labels_in_subject_candidate_space"],
            "final_labels_on_original_scan_grid": str(final_labels_native),
            "final_roi_mask_on_original_scan_grid": str(final_roi_mask_native) if final_roi_mask_native else None,
            "roi_centroid_mm_original_header": centroid_native,
        },
        "top_candidates": top_candidates,
    }


# -----------------------------
# CLI
# -----------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Robust mouse hippocampus/DG localization on partial T2 slabs by atlas ROI crop + automatic orientation search."
    )
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("analysis_out"))
    p.add_argument("--scan-ids", type=int, nargs="*", default=[5, 13])
    p.add_argument("--input-nii", type=Path, action="append", default=None)

    p.add_argument("--no-convert-bruker", action="store_true",
                   help="Disable automatic Bruker->NIfTI conversion when no NIfTIs are found.")
    p.add_argument("--converted-dir", type=Path, default=None,
                   help="Where auto-converted Bruker NIfTIs will be written. Default: <out-dir>/converted_nifti")
    p.add_argument("--bruker-converter-cmd", default=None,
                   help="Converter executable or wrapper command, e.g. 'brkraw'.")
    p.add_argument("--bruker-converter-args", default="auto",
                   help="Converter argument template. Use 'auto' to try common brkraw forms.")

    p.add_argument("--atlas-template", type=Path, required=True)
    p.add_argument("--atlas-labels", type=Path, required=True)
    p.add_argument("--label-ids", type=int, nargs="*", default=None,
                   help="Final label IDs to extract as ROI mask in subject space (e.g. DG / hippocampus labels).")
    p.add_argument("--crop-label-ids", type=int, nargs="*", default=None,
                   help="Atlas labels used to crop around the target region before registration. Default: same as --label-ids.")
    p.add_argument("--crop-margin-mm", type=float, default=2.5)
    p.add_argument("--no-n4-atlas", action="store_true")
    p.add_argument("--no-n4-subject", action="store_true")
    p.add_argument("--use-affine-stage", action="store_true",
                   help="Also run affine after rigid. Default is rigid only.")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Limit number of orientation candidates. Default: all 48.")
    args = p.parse_args()

    if not args.input_root.exists():
        raise SystemExit(f"Input root not found: {args.input_root}")
    if not args.atlas_template.exists():
        raise SystemExit(f"Atlas template not found: {args.atlas_template}")
    if not args.atlas_labels.exists():
        raise SystemExit(f"Atlas labels not found: {args.atlas_labels}")

    crop_label_ids = args.crop_label_ids if args.crop_label_ids else args.label_ids

    exclude_paths: list[Path] = [
        args.out_dir.resolve(),
        args.atlas_template.resolve(),
        args.atlas_template.parent.resolve(),
        args.atlas_labels.resolve(),
        args.atlas_labels.parent.resolve(),
    ]

    scans = args.input_nii or _find_t2_niftis(args.input_root, exclude_paths=exclude_paths)
    scans = [p for p in scans if _is_3d_nifti(p)]

    if args.input_nii is None and scans:
        scan_roots = _scan_dirs_in_tree(args.input_root)
        if scan_roots:
            t2_scan_dirs = [scan for scan in scan_roots if _looks_like_t2_bruker_scan(scan)]
            if t2_scan_dirs:
                t2_scan_tokens = {scan.name for scan in t2_scan_dirs}
                filtered = [
                    nii for nii in scans
                    if any(part in t2_scan_tokens for part in nii.relative_to(args.input_root).parts)
                ]
                if filtered:
                    scans = filtered

    converted_dir = args.converted_dir or (args.out_dir / "converted_nifti")

    if not scans:
        bruker_candidates = _bruker_input_candidates(args.input_root)
        can_attempt = any(_looks_like_bruker_root(candidate) for candidate in bruker_candidates)

        if args.no_convert_bruker or not can_attempt:
            raise SystemExit("No NIfTI scans found under input root. Convert Bruker to NIfTI first or allow auto-conversion.")

        print(f"No NIfTIs found. Attempting Bruker conversion into: {converted_dir}")
        scans = _convert_bruker_to_nifti(
            input_root=args.input_root,
            converted_dir=converted_dir,
            converter_cmd=args.bruker_converter_cmd,
            converter_args_template=args.bruker_converter_args,
        )
        scans = [p for p in scans if _is_3d_nifti(p)]

        if not scans:
            raise SystemExit("Conversion succeeded but no 3D NIfTI files were discovered.")

    broad_pool = scans[:]
    if converted_dir.exists():
        for pth in _find_all_niftis_under(converted_dir):
            if pth not in broad_pool:
                broad_pool.append(pth)

    slab_paths: dict[int, Path] = {}
    for sid in args.scan_ids:
        pth = _find_scan_file_by_id(broad_pool, sid)
        if pth is not None:
            slab_paths[sid] = pth

    missing = [sid for sid in args.scan_ids if sid not in slab_paths]
    if args.input_nii is None and missing:
        raise SystemExit(f"Could not find requested scan IDs in discovered NIfTIs: {missing}")

    subject_list = args.input_nii if args.input_nii else [slab_paths[sid] for sid in args.scan_ids]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    atlas_info = _prepare_atlas(
        atlas_template=args.atlas_template,
        atlas_labels=args.atlas_labels,
        out_dir=args.out_dir / "atlas_prepared",
        crop_label_ids=crop_label_ids,
        crop_margin_mm=args.crop_margin_mm,
        do_n4=not args.no_n4_atlas,
    )

    summary: dict[str, object] = {
        "input_root": str(args.input_root),
        "atlas_template": str(args.atlas_template),
        "atlas_labels": str(args.atlas_labels),
        "label_ids": [int(x) for x in (args.label_ids or [])],
        "crop_label_ids": [int(x) for x in (crop_label_ids or [])],
        "crop_margin_mm": float(args.crop_margin_mm),
        "use_affine_stage": bool(args.use_affine_stage),
        "max_candidates": args.max_candidates,
        "converted_dir": str(converted_dir),
        "atlas_prepared": atlas_info,
        "subjects": [],
    }

    for subj in subject_list:
        print(f"=== Processing {subj}")
        try:
            result = _run_one_subject(
                subject_nii=subj,
                out_dir=args.out_dir,
                atlas_info=atlas_info,
                do_n4_subject=not args.no_n4_subject,
                label_ids=args.label_ids,
                crop_label_ids=crop_label_ids,
                use_affine_stage=bool(args.use_affine_stage),
                max_candidates=args.max_candidates,
            )
        except Exception as exc:
            result = {
                "input_subject": str(subj),
                "status": "failed",
                "error": str(exc),
            }
        summary["subjects"].append(result)

    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Done. Summary written to: {out_json}")


if __name__ == "__main__":
    main()
