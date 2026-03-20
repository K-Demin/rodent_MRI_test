#!/usr/bin/env python3
"""Find hippocampus / DG on rodent Bruker T2 scans using direct ANTs registration
to a manually AP-cropped atlas.

Key fixes in this version
-------------------------
1) Do NOT infer atlas AP direction from affine A/P labels for cropping.
   Instead, crop the template using:
      - a MANUAL AP voxel axis
      - a MANUAL anterior side ("low" or "high")
   because the template affine/orientation may be wrong.

2) Keep subject preparation simple:
      - canonicalize with nibabel
      - isotropic resample
      - optional unifize
   No manual x/y/z flip logic.

3) Registration defaults to RIGID ONLY, because affine can introduce mirror-like
   solutions on ambiguous slab data.

4) Save extra debug information, including affine determinant inspection.

Workflow
--------
1) discover or convert Bruker scans to NIfTI
2) keep target scan IDs (default: 5 and 13)
3) prepare subject: canonicalize orientation, isotropic resample, optional unifize
4) fix suspicious atlas/header voxel units if needed
5) canonicalize atlas template + labels
6) crop atlas/template using MANUAL AP axis and MANUAL anterior side
7) register subject slab -> cropped atlas with ANTs rigid (default) or rigid+affine
8) warp cropped atlas labels into subject space with nearest-neighbor interpolation
9) optionally extract selected label IDs into a binary mask and centroid
10) write summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path


EXCLUDED_DIR_NAMES = {
    "analysis_out",
    "mouse_atlas",
    "atlas",
    "templates",
    ".git",
    ".venv",
    "__pycache__",
}


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"Required command '{name}' not found in PATH. "
            "Install it or pass a different wrapper command."
        )


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


def _scan_dirs_in_tree(root: Path) -> list[Path]:
    out: list[Path] = []
    if _is_scan_like_dir(root):
        out.append(root)
    for child in sorted(root.iterdir()):
        if _is_scan_like_dir(child):
            out.append(child)
    return out


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
        skip = False
        for ex in exclude_paths:
            if rp == ex or ex in rp.parents:
                skip = True
                break
        if skip:
            continue

        name = p.name.lower()
        if any(tok in name for tok in ("annotation", "template", "atlas", "label", "roi")):
            continue

        kept.append(p)

    if not kept:
        return []

    t2 = [p for p in kept if any(tok in p.name.lower() for tok in ("t2", "rare", "turborare"))]
    return t2 or kept


def _is_3d_nifti(path: Path) -> bool:
    try:
        import nibabel as nib
        shape = nib.load(str(path)).shape
        return len(shape) == 3
    except Exception:
        return False


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


def _get_orientation_info(nii: Path) -> dict[str, object]:
    try:
        import nibabel as nib
        img = nib.load(str(nii))
        return {
            "path": str(nii),
            "shape": [int(v) for v in img.shape[:3]],
            "zooms": [float(v) for v in img.header.get_zooms()[:3]],
            "axcodes": list(nib.aff2axcodes(img.affine)),
            "affine": [[float(x) for x in row] for row in img.affine.tolist()],
        }
    except Exception as exc:
        return {
            "path": str(nii),
            "error": str(exc),
        }


def _canonicalize_nifti(input_nii: Path, output_nii: Path) -> dict[str, object]:
    try:
        import nibabel as nib
        import numpy as np
    except Exception as exc:
        raise SystemExit("Canonical orientation requires nibabel and numpy.") from exc

    img = nib.load(str(input_nii))
    before = nib.aff2axcodes(img.affine)

    can = nib.as_closest_canonical(img)
    after = nib.aff2axcodes(can.affine)

    hdr = can.header.copy()
    aff = can.affine.copy()
    out = nib.Nifti1Image(np.asarray(can.get_fdata(), dtype="float32"), aff, hdr)
    out.set_qform(aff, code=1)
    out.set_sform(aff, code=1)
    nib.save(out, str(output_nii))

    return {
        "input": str(input_nii),
        "output": str(output_nii),
        "before_axcodes": list(before),
        "after_axcodes": list(after),
        "shape": [int(v) for v in out.shape[:3]],
        "zooms": [float(v) for v in out.header.get_zooms()[:3]],
    }


def _fix_nifti_units_if_suspicious(input_nii: Path, out_nii: Path) -> dict[str, object]:
    try:
        import nibabel as nib
    except Exception as exc:
        raise SystemExit("This script requires nibabel for NIfTI header handling.") from exc

    img = nib.load(str(input_nii))
    hdr = img.header.copy()
    aff = img.affine.copy()
    zooms = hdr.get_zooms()[:3]
    max_abs_zoom = max(abs(float(z)) for z in zooms)

    applied = False
    scale = 1.0
    if max_abs_zoom > 5.0:
        scale = 0.001
        aff[:3, :4] *= scale
        try:
            hdr.set_zooms(tuple(float(z) * scale for z in hdr.get_zooms()))
        except Exception:
            pass
        applied = True

    out_img = nib.Nifti1Image(img.get_fdata(dtype="float32"), aff, hdr)
    out_img.set_qform(aff, code=1)
    out_img.set_sform(aff, code=1)
    nib.save(out_img, str(out_nii))

    try:
        import nibabel as nib
        out_axcodes = list(nib.aff2axcodes(aff))
    except Exception:
        out_axcodes = None

    return {
        "input": str(input_nii),
        "output": str(out_nii),
        "input_zooms": [float(z) for z in zooms],
        "output_axcodes": out_axcodes,
        "applied_scale_factor": float(scale),
        "units_fix_applied": bool(applied),
    }


def _crop_template_remove_anterior_only_manual(
    template_nii: Path,
    labels_nii: Path,
    out_template_nii: Path,
    out_labels_nii: Path,
    ap_axis: int,
    anterior_side: str,
    anterior_margin_mm: float = 1.0,
) -> dict[str, object]:
    """
    Remove only the anterior part of the atlas using MANUAL voxel-side definition.

    Parameters
    ----------
    ap_axis : int
        Voxel axis corresponding to AP direction: 0, 1, or 2.
    anterior_side : str
        Which end of that voxel axis is anterior: "low" or "high".

    Notes
    -----
    This function does NOT trust affine A/P labels.
    """
    try:
        import nibabel as nib
        import numpy as np
    except Exception as exc:
        raise SystemExit("Manual anterior-only atlas cropping requires nibabel and numpy.") from exc

    if ap_axis not in (0, 1, 2):
        raise SystemExit(f"ap_axis must be 0, 1, or 2; got {ap_axis}")
    if anterior_side not in ("low", "high"):
        raise SystemExit(f"anterior_side must be 'low' or 'high'; got {anterior_side}")

    timg = nib.load(str(template_nii))
    limg = nib.load(str(labels_nii))
    tdat = np.asarray(timg.get_fdata())
    ldat = np.asarray(limg.get_fdata())

    if tdat.shape != ldat.shape:
        raise SystemExit(f"Template and labels shapes differ: {tdat.shape} vs {ldat.shape}")

    mask = ldat != 0
    if not np.any(mask):
        raise SystemExit("Atlas label volume has no nonzero voxels, so it cannot define a crop.")

    ijk = np.where(mask)
    mins = np.array([ijk[0].min(), ijk[1].min(), ijk[2].min()], dtype=int)
    maxs = np.array([ijk[0].max(), ijk[1].max(), ijk[2].max()], dtype=int)

    zooms = np.array([abs(float(z)) for z in limg.header.get_zooms()[:3]], dtype=float)
    margin_vox = int(np.ceil(anterior_margin_mm / max(zooms[ap_axis], 1e-6)))

    start = np.array([0, 0, 0], dtype=int)
    stop = np.array(ldat.shape, dtype=int)

    if anterior_side == "high":
        stop[ap_axis] = min(maxs[ap_axis] + margin_vox + 1, ldat.shape[ap_axis])
    else:
        start[ap_axis] = max(mins[ap_axis] - margin_vox, 0)

    xs = slice(start[0], stop[0])
    ys = slice(start[1], stop[1])
    zs = slice(start[2], stop[2])

    t_crop = tdat[xs, ys, zs]
    l_crop = ldat[xs, ys, zs]

    new_aff = limg.affine.copy()
    offset = np.array([start[0], start[1], start[2], 1.0], dtype=float)
    new_aff[:3, 3] = (limg.affine @ offset)[:3]

    t_out = nib.Nifti1Image(t_crop.astype("float32"), new_aff, timg.header.copy())
    l_out = nib.Nifti1Image(l_crop.astype(ldat.dtype), new_aff, limg.header.copy())

    t_out.set_qform(new_aff, code=1)
    t_out.set_sform(new_aff, code=1)
    l_out.set_qform(new_aff, code=1)
    l_out.set_sform(new_aff, code=1)

    nib.save(t_out, str(out_template_nii))
    nib.save(l_out, str(out_labels_nii))

    return {
        "template_cropped": str(out_template_nii),
        "labels_cropped": str(out_labels_nii),
        "manual_ap_axis": int(ap_axis),
        "manual_anterior_side": str(anterior_side),
        "mins_ijk": [int(v) for v in mins],
        "maxs_ijk": [int(v) for v in maxs],
        "start_ijk": [int(v) for v in start],
        "stop_ijk": [int(v) for v in stop],
        "anterior_margin_mm": float(anterior_margin_mm),
        "cropped_shape": [int(v) for v in t_crop.shape],
    }


def _extract_label_mask(labels_nii: Path, out_mask_nii: Path, label_ids: list[int]) -> dict[str, object] | None:
    try:
        import nibabel as nib
        import numpy as np
    except Exception:
        return None

    img = nib.load(str(labels_nii))
    data = np.asarray(img.get_fdata())
    mask = np.isin(data, np.array(label_ids, dtype=data.dtype))

    out = nib.Nifti1Image(mask.astype("uint8"), img.affine, img.header.copy())
    out.set_qform(img.affine, code=1)
    out.set_sform(img.affine, code=1)
    nib.save(out, str(out_mask_nii))

    return {
        "mask": str(out_mask_nii),
        "label_ids": [int(x) for x in label_ids],
        "nonzero_voxels": int(mask.sum()),
    }


def _mask_centroid_mm(mask_nii: Path) -> dict[str, list[float]] | None:
    try:
        import nibabel as nib
        import numpy as np
    except Exception:
        return None

    img = nib.load(str(mask_nii))
    data = np.asarray(img.get_fdata()) > 0
    if not data.any():
        return None

    vox = np.column_stack(np.where(data))
    xyz = vox.mean(axis=0)
    xyz_h = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=float)
    mm = (img.affine @ xyz_h)[:3].tolist()
    return {"centroid_mm": [float(v) for v in mm]}


def _split_cmd(cmd: str) -> list[str]:
    return shlex.split(cmd)


def _read_ants_affine_det(mat_path: Path) -> dict[str, object]:
    import numpy as np

    if not mat_path.exists():
        return {"path": str(mat_path), "error": "Affine file does not exist."}

    text = mat_path.read_text()
    m = re.search(r"Parameters:\s*([^\n]+)", text)
    if m is None:
        return {"path": str(mat_path), "error": "Could not find Parameters line."}

    vals = [float(x) for x in m.group(1).strip().split()]
    if len(vals) < 12:
        return {"path": str(mat_path), "error": f"Expected >=12 affine params, got {len(vals)}."}

    A = np.array(vals[:9], dtype=float).reshape(3, 3)
    det = float(np.linalg.det(A))
    return {
        "path": str(mat_path),
        "linear_matrix": [[float(x) for x in row] for row in A.tolist()],
        "determinant": det,
        "is_reflection": bool(det < 0),
    }


def _prepare_subject(
    input_nii: Path,
    out_dir: Path,
    iso_dxyz: tuple[float, float, float],
    do_unifize: bool,
) -> dict[str, object]:
    _require_cmd("3dresample")
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical = out_dir / "input_canonical.nii.gz"
    _rm_if_exists(canonical)
    canon_info = _canonicalize_nifti(input_nii, canonical)

    isotropic = out_dir / "input_iso.nii.gz"
    _rm_if_exists(isotropic)
    _run(
        [
            "3dresample",
            "-dxyz",
            str(iso_dxyz[0]),
            str(iso_dxyz[1]),
            str(iso_dxyz[2]),
            "-prefix",
            str(isotropic),
            "-input",
            str(canonical),
        ]
    )

    isotropic_canonical = out_dir / "input_iso_canonical.nii.gz"
    _rm_if_exists(isotropic_canonical)
    iso_canon_info = _canonicalize_nifti(isotropic, isotropic_canonical)

    final_input = isotropic_canonical
    unifized = None
    unifize_canon_info = None
    if do_unifize:
        _require_cmd("3dUnifize")
        unifized = out_dir / "input_final_unifized.nii.gz"
        _rm_if_exists(unifized)
        _run(["3dUnifize", "-input", str(final_input), "-prefix", str(unifized)])

        final_unifized_canonical = out_dir / "input_final_unifized_canonical.nii.gz"
        _rm_if_exists(final_unifized_canonical)
        unifize_canon_info = _canonicalize_nifti(unifized, final_unifized_canonical)
        final_input = final_unifized_canonical

    result: dict[str, object] = {
        "input": str(input_nii),
        "canonicalized": str(canonical),
        "canonicalize_info": canon_info,
        "isotropic": str(isotropic),
        "isotropic_canonical": str(isotropic_canonical),
        "isotropic_canonicalize_info": iso_canon_info,
        "final_input": str(final_input),
        "iso_dxyz": [float(v) for v in iso_dxyz],
        "orientation_checks": {
            "raw": _get_orientation_info(input_nii),
            "canonical": _get_orientation_info(canonical),
            "isotropic": _get_orientation_info(isotropic),
            "final": _get_orientation_info(final_input),
        },
    }
    if unifized is not None:
        result["unifized"] = str(unifized)
        result["unifized_canonicalize_info"] = unifize_canon_info

    return result


def run_direct_ants_to_cropped_atlas(
    input_nii: Path,
    out_dir: Path,
    template_nii: Path,
    labels_nii: Path,
    label_ids: list[int] | None,
    ants_cmd: str,
    ants_apply_cmd: str,
    anterior_crop_margin_mm: float,
    template_ap_axis: int,
    template_anterior_side: str,
    use_affine_stage: bool,
) -> dict[str, object]:
    ants_exe = _split_cmd(ants_cmd)[0]
    ants_apply_exe = _split_cmd(ants_apply_cmd)[0]
    _require_cmd(ants_exe)
    _require_cmd(ants_apply_exe)

    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_template = out_dir / "template_units_fixed.nii.gz"
    fixed_labels = out_dir / "template_labels_units_fixed.nii.gz"
    template_fix = _fix_nifti_units_if_suspicious(template_nii, fixed_template)
    labels_fix = _fix_nifti_units_if_suspicious(labels_nii, fixed_labels)

    template_canonical = out_dir / "template_units_fixed_canonical.nii.gz"
    labels_canonical = out_dir / "template_labels_units_fixed_canonical.nii.gz"
    template_canon_info = _canonicalize_nifti(fixed_template, template_canonical)
    labels_canon_info = _canonicalize_nifti(fixed_labels, labels_canonical)

    cropped_template = out_dir / "template_cropped.nii.gz"
    cropped_labels = out_dir / "template_labels_cropped.nii.gz"
    crop_info = _crop_template_remove_anterior_only_manual(
        template_nii=template_canonical,
        labels_nii=labels_canonical,
        out_template_nii=cropped_template,
        out_labels_nii=cropped_labels,
        ap_axis=template_ap_axis,
        anterior_side=template_anterior_side,
        anterior_margin_mm=anterior_crop_margin_mm,
    )

    prefix = out_dir / "reg_"
    warped = out_dir / "reg_Warped.nii.gz"
    inverse_warped = out_dir / "reg_InverseWarped.nii.gz"

    reg_cmd = _split_cmd(ants_cmd) + [
        "--dimensionality", "3",
        "--output", f"[{prefix},{warped},{inverse_warped}]",
        "--interpolation", "Linear",
        "--use-histogram-matching", "0",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--initial-moving-transform", f"[{cropped_template},{input_nii},0]",
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{cropped_template},{input_nii},1,32,Regular,0.25]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
    ]

    if use_affine_stage:
        reg_cmd += [
            "--transform", "Affine[0.1]",
            "--metric", f"MI[{cropped_template},{input_nii},1,32,Regular,0.25]",
            "--convergence", "[1000x500x250x100,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
        ]

    _run(reg_cmd)

    affine = out_dir / "reg_0GenericAffine.mat"
    affine_info = _read_ants_affine_det(affine)

    labels_in_native = out_dir / "atlas_labels_in_input_space.nii.gz"
    apply_cmd = _split_cmd(ants_apply_cmd) + [
        "-d", "3",
        "-i", str(cropped_labels),
        "-r", str(input_nii),
        "-o", str(labels_in_native),
        "-n", "NearestNeighbor",
        "-t", f"[{affine},1]",
    ]
    _run(apply_cmd)

    result: dict[str, object] = {
        "template_units_fix": template_fix,
        "labels_units_fix": labels_fix,
        "template_canonicalize_info": template_canon_info,
        "labels_canonicalize_info": labels_canon_info,
        "crop_info": crop_info,
        "registered_subject_to_cropped_atlas": str(warped),
        "inverse_warped": str(inverse_warped),
        "affine_mat": str(affine),
        "affine_info": affine_info,
        "template_labels_in_input_space": str(labels_in_native),
        "ants_cmd": ants_cmd,
        "ants_apply_cmd": ants_apply_cmd,
        "used_affine_stage": bool(use_affine_stage),
        "orientation_checks": {
            "subject_input": _get_orientation_info(input_nii),
            "template_raw": _get_orientation_info(template_nii),
            "labels_raw": _get_orientation_info(labels_nii),
            "template_fixed": _get_orientation_info(fixed_template),
            "labels_fixed": _get_orientation_info(fixed_labels),
            "template_canonical": _get_orientation_info(template_canonical),
            "labels_canonical": _get_orientation_info(labels_canonical),
            "template_cropped": _get_orientation_info(cropped_template),
            "labels_cropped": _get_orientation_info(cropped_labels),
            "labels_in_native": _get_orientation_info(labels_in_native),
        },
    }

    if label_ids:
        roi_mask = out_dir / "roi_mask_in_input_space.nii.gz"
        roi_info = _extract_label_mask(labels_in_native, roi_mask, label_ids)
        result["roi_extraction"] = roi_info or {}
        centroid = _mask_centroid_mm(roi_mask)
        if centroid:
            result["roi_centroid"] = centroid

    return result


def _run_single(
    nii: Path,
    out_dir: Path,
    species: str,
    iso_dxyz: tuple[float, float, float],
    do_unifize: bool,
    template: Path,
    labels: Path,
    label_ids: list[int] | None,
    ants_cmd: str,
    ants_apply_cmd: str,
    clean_case_dirs: bool,
    anterior_crop_margin_mm: float,
    template_ap_axis: int,
    template_anterior_side: str,
    use_affine_stage: bool,
) -> dict[str, object]:
    case_out = out_dir / nii.stem.replace(".nii", "")
    if clean_case_dirs and case_out.exists():
        shutil.rmtree(case_out)
    case_out.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "input_nii": str(nii),
        "out_dir": str(case_out),
        "species": species,
    }

    result["prepared"] = _prepare_subject(
        nii,
        case_out / "prepped",
        iso_dxyz,
        do_unifize,
    )

    result["ants_registration"] = run_direct_ants_to_cropped_atlas(
        input_nii=Path(result["prepared"]["final_input"]),
        out_dir=case_out / "ants",
        template_nii=template,
        labels_nii=labels,
        label_ids=label_ids,
        ants_cmd=ants_cmd,
        ants_apply_cmd=ants_apply_cmd,
        anterior_crop_margin_mm=anterior_crop_margin_mm,
        template_ap_axis=template_ap_axis,
        template_anterior_side=template_anterior_side,
        use_affine_stage=use_affine_stage,
    )

    return result


def _find_scan_file_by_id(scans: list[Path], scan_id: int) -> Path | None:
    token = f"scan-{scan_id}_"
    for p in scans:
        if token in p.name:
            return p
    return None


def _find_all_niftis_under(root: Path) -> list[Path]:
    files = sorted([*root.rglob("*.nii"), *root.rglob("*.nii.gz")])
    return [p for p in files if _is_3d_nifti(p)]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Find mouse hippocampus / DG on T2 scans using direct ANTs registration to a manually AP-cropped hippocampus atlas."
    )
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("analysis_out"))
    p.add_argument("--species", choices=["mouse", "rat", "human"], default="mouse")

    p.add_argument("--scan-ids", type=int, nargs="*", default=[5, 13],
                   help="Target T2 slab scan IDs.")
    p.add_argument("--input-nii", type=Path, action="append", default=None)

    p.add_argument("--no-convert-bruker", action="store_true")
    p.add_argument("--converted-dir", type=Path, default=None)
    p.add_argument("--bruker-converter-cmd", default=None)
    p.add_argument("--bruker-converter-args", default="auto")

    p.add_argument("--prep-iso-dxyz", type=float, nargs=3, default=(0.1, 0.1, 0.1),
                   metavar=("DX", "DY", "DZ"))
    p.add_argument("--no-unifize", action="store_true",
                   help="Disable 3dUnifize before ANTs.")

    p.add_argument("--atlas-template", type=Path, required=True,
                   help="MRI atlas template NIfTI.")
    p.add_argument("--atlas-labels", type=Path, required=True,
                   help="Int-valued atlas label/segmentation NIfTI.")
    p.add_argument("--label-ids", type=int, nargs="*", default=None,
                   help="Atlas label IDs to extract in subject space.")

    p.add_argument(
        "--anterior-crop-margin-mm",
        type=float,
        default=1.0,
        help="Extra margin in mm kept posterior to the removed anterior atlas cutoff.",
    )
    p.add_argument(
        "--template-ap-axis",
        type=int,
        required=True,
        choices=[0, 1, 2],
        help="MANUAL voxel axis of the template corresponding to AP direction: 0, 1, or 2.",
    )
    p.add_argument(
        "--template-anterior-side",
        required=True,
        choices=["low", "high"],
        help="Which end of template AP voxel axis is anatomically anterior.",
    )

    p.add_argument(
        "--ants-cmd",
        default="antsRegistration",
        help="ANTs registration command.",
    )
    p.add_argument(
        "--ants-apply-cmd",
        default="antsApplyTransforms",
        help="ANTs apply-transforms command.",
    )
    p.add_argument(
        "--use-affine-stage",
        action="store_true",
        help="Also run ANTs affine stage after rigid. Default is rigid only.",
    )
    p.add_argument(
        "--clean-case-dirs",
        action="store_true",
        help="Delete existing per-case output folders before reprocessing.",
    )
    args = p.parse_args()

    if not args.input_root.exists():
        raise SystemExit(f"Input root not found: {args.input_root}")
    if not args.atlas_template.exists():
        raise SystemExit(f"Atlas template not found: {args.atlas_template}")
    if not args.atlas_labels.exists():
        raise SystemExit(f"Atlas labels not found: {args.atlas_labels}")

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

    if not scans:
        bruker_candidates = _bruker_input_candidates(args.input_root)
        can_attempt = any(_looks_like_bruker_root(candidate) for candidate in bruker_candidates)

        if args.no_convert_bruker or not can_attempt:
            raise SystemExit(
                "No candidate NIfTI files found. Convert Bruker data first or allow auto-conversion."
            )

        converted_dir = args.converted_dir or (args.out_dir / "converted_nifti")
        print(f"No NIfTI files found. Attempting Bruker conversion into: {converted_dir}")
        scans = _convert_bruker_to_nifti(
            input_root=args.input_root,
            converted_dir=converted_dir,
            converter_cmd=args.bruker_converter_cmd,
            converter_args_template=args.bruker_converter_args,
        )
        scans = [p for p in scans if _is_3d_nifti(p)]

        if not scans:
            raise SystemExit("Conversion succeeded but no NIfTI files were discovered.")

    broad_pool = scans[:]
    converted_search_root = args.converted_dir or (args.out_dir / "converted_nifti")
    if converted_search_root.exists():
        for pth in _find_all_niftis_under(converted_search_root):
            if pth not in broad_pool:
                broad_pool.append(pth)

    slab_paths: dict[int, Path] = {}
    for sid in args.scan_ids:
        pth = _find_scan_file_by_id(broad_pool, sid)
        if pth is not None:
            slab_paths[sid] = pth

    missing_slabs = [sid for sid in args.scan_ids if sid not in slab_paths]
    if missing_slabs:
        raise SystemExit(f"Could not find slab scan files for IDs: {missing_slabs}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "input_root": str(args.input_root),
        "atlas_template": str(args.atlas_template),
        "atlas_labels": str(args.atlas_labels),
        "scan_ids": [int(x) for x in args.scan_ids],
        "template_ap_axis": int(args.template_ap_axis),
        "template_anterior_side": str(args.template_anterior_side),
        "used_affine_stage": bool(args.use_affine_stage),
        "cases": [],
    }

    for slab_id in args.scan_ids:
        slab_nii = slab_paths[slab_id]
        print(f"=== Processing slab scan {slab_id}")
        try:
            case_result = _run_single(
                nii=slab_nii,
                out_dir=args.out_dir,
                species=args.species,
                iso_dxyz=tuple(args.prep_iso_dxyz),
                do_unifize=not args.no_unifize,
                template=args.atlas_template,
                labels=args.atlas_labels,
                label_ids=args.label_ids,
                ants_cmd=args.ants_cmd,
                ants_apply_cmd=args.ants_apply_cmd,
                clean_case_dirs=args.clean_case_dirs,
                anterior_crop_margin_mm=args.anterior_crop_margin_mm,
                template_ap_axis=int(args.template_ap_axis),
                template_anterior_side=str(args.template_anterior_side),
                use_affine_stage=bool(args.use_affine_stage),
            )
        except subprocess.CalledProcessError as exc:
            case_result = {
                "input_nii": str(slab_nii),
                "status": "failed",
                "error": str(exc),
            }
        except Exception as exc:
            case_result = {
                "input_nii": str(slab_nii),
                "status": "failed",
                "error": str(exc),
            }

        summary["cases"].append(case_result)

    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Done. Summary written to: {out_json}")


if __name__ == "__main__":
    main()