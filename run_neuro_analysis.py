#!/usr/bin/env python3
"""Closed-loop AFNI/SPM pipeline for rodent T2 MRI folders.

Given an input folder, this script can:
1) discover candidate T2 NIfTI files,
2) run AFNI and/or SPM preprocessing,
3) derive a DG proxy coordinate from segmentation outputs,
4) emit a single machine-readable summary file.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"Required command '{name}' not found in PATH. "
            "Install the required neuroimaging package and retry."
        )


def _find_t2_niftis(root: Path) -> list[Path]:
    all_nifti = sorted([*root.rglob("*.nii"), *root.rglob("*.nii.gz")])
    if not all_nifti:
        return []

    t2 = [p for p in all_nifti if "t2" in p.name.lower() or "rare" in p.name.lower()]
    return t2 or all_nifti


def _looks_like_bruker_root(root: Path) -> bool:
    if (root / "study.MR").exists() and (root / "subject").exists():
        return True
    return any(root.glob("*/pdata/1/2dseq"))


def _convert_bruker_to_nifti(
    input_root: Path,
    converted_dir: Path,
    converter_cmd: str,
    converter_args_template: str,
) -> list[Path]:
    converter_exe = shlex.split(converter_cmd)[0]
    _require_cmd(converter_exe)

    converted_dir.mkdir(parents=True, exist_ok=True)
    converter_args = converter_args_template.format(
        input=input_root.as_posix(),
        output=converted_dir.as_posix(),
    )
    cmd = shlex.split(converter_cmd) + shlex.split(converter_args)
    _run(cmd)
    return _find_t2_niftis(converted_dir)


def run_afni(input_nii: Path, out_dir: Path) -> dict[str, str]:
    required = ["3dcopy", "3dcalc", "3dAutomask", "3dUnifize", "3dSkullStrip"]
    for cmd in required:
        _require_cmd(cmd)

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = out_dir / "input.nii.gz"
    _run(["3dcopy", str(input_nii), str(copied)])

    unifized = out_dir / "input_unifized.nii.gz"
    _run(["3dUnifize", "-input", str(copied), "-prefix", str(unifized)])

    automask = out_dir / "brain_mask.nii.gz"
    _run(["3dAutomask", "-prefix", str(automask), str(unifized)])

    skullstrip = out_dir / "brain_ss.nii.gz"
    _run(["3dSkullStrip", "-input", str(unifized), "-prefix", str(skullstrip)])

    masked = out_dir / "brain_masked.nii.gz"
    _run(
        [
            "3dcalc",
            "-a",
            str(unifized),
            "-b",
            str(automask),
            "-expr",
            "a*step(b)",
            "-prefix",
            str(masked),
        ]
    )

    return {
        "input": str(copied),
        "unifized": str(unifized),
        "mask": str(automask),
        "skullstrip": str(skullstrip),
        "masked": str(masked),
    }


def _write_spm_batch(input_nii: Path, out_dir: Path, spm_dir: Path) -> Path:
    batch = textwrap.dedent(
        f"""
        addpath('{spm_dir.as_posix()}');
        spm('defaults', 'fmri');
        spm_jobman('initcfg');

        matlabbatch{{1}}.spm.spatial.preproc.channel.vols = {{'{input_nii.as_posix()},1'}};
        matlabbatch{{1}}.spm.spatial.preproc.channel.biasreg = 0.001;
        matlabbatch{{1}}.spm.spatial.preproc.channel.biasfwhm = 60;
        matlabbatch{{1}}.spm.spatial.preproc.channel.write = [1 1];

        tpm = fullfile('{spm_dir.as_posix()}', 'tpm', 'TPM.nii');
        for k = 1:6
            matlabbatch{{1}}.spm.spatial.preproc.tissue(k).tpm = {{sprintf('%s,%d', tpm, k)}};
            matlabbatch{{1}}.spm.spatial.preproc.tissue(k).ngaus = 1;
            matlabbatch{{1}}.spm.spatial.preproc.tissue(k).native = [1 0];
            matlabbatch{{1}}.spm.spatial.preproc.tissue(k).warped = [0 0];
        end

        matlabbatch{{1}}.spm.spatial.preproc.warp.mrf = 1;
        matlabbatch{{1}}.spm.spatial.preproc.warp.cleanup = 1;
        matlabbatch{{1}}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
        matlabbatch{{1}}.spm.spatial.preproc.warp.affreg = 'mni';
        matlabbatch{{1}}.spm.spatial.preproc.warp.fwhm = 0;
        matlabbatch{{1}}.spm.spatial.preproc.warp.samp = 3;
        matlabbatch{{1}}.spm.spatial.preproc.warp.write = [1 1];

        cd('{out_dir.as_posix()}');
        spm_jobman('run', matlabbatch);
        exit;
        """
    ).strip()

    batch_path = out_dir / "spm_segment_job.m"
    batch_path.write_text(batch + "\n")
    return batch_path


def run_spm(input_nii: Path, out_dir: Path, spm_dir: Path, matlab_cmd: str) -> dict[str, str]:
    if not spm_dir.exists():
        raise SystemExit(f"SPM directory does not exist: {spm_dir}")

    matlab = matlab_cmd.split()[0]
    _require_cmd(matlab)

    out_dir.mkdir(parents=True, exist_ok=True)
    batch_path = _write_spm_batch(input_nii=input_nii, out_dir=out_dir, spm_dir=spm_dir)

    if matlab == "matlab":
        _run(["matlab", "-batch", f"run('{batch_path.as_posix()}')"])
    else:
        _run(matlab_cmd.split() + [str(batch_path)])

    stem = input_nii.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]

    outputs = {
        "gm": str(out_dir / f"c1{stem}.nii"),
        "wm": str(out_dir / f"c2{stem}.nii"),
        "csf": str(out_dir / f"c3{stem}.nii"),
        "deformation": str(out_dir / f"y_{stem}.nii"),
    }
    return outputs


def _estimate_dg_from_spm(gm_path: Path) -> dict[str, list[float]] | None:
    try:
        import nibabel as nib
        import numpy as np
    except Exception:
        return None

    if not gm_path.exists():
        return None

    img = nib.load(str(gm_path))
    gm = np.asarray(img.get_fdata(), dtype=float)
    if gm.ndim != 3 or gm.size == 0:
        return None

    thr = np.percentile(gm[gm > 0], 70) if np.any(gm > 0) else 0.0
    mask = gm >= thr
    if not np.any(mask):
        return None

    xs, ys, zs = np.where(mask)
    x_mid = float(np.median(xs))
    z_min, z_max = np.percentile(zs, 30), np.percentile(zs, 75)
    post = mask.copy()
    post[:, :, : int(z_min)] = False
    post[:, :, int(z_max) + 1 :] = False

    left = np.where(post & (np.indices(post.shape)[0] < x_mid))
    right = np.where(post & (np.indices(post.shape)[0] >= x_mid))
    if len(left[0]) == 0 or len(right[0]) == 0:
        return None

    l_vox = np.array([left[0].mean(), left[1].mean(), left[2].mean(), 1.0])
    r_vox = np.array([right[0].mean(), right[1].mean(), right[2].mean(), 1.0])
    aff = img.affine
    l_mm = (aff @ l_vox)[:3].tolist()
    r_mm = (aff @ r_vox)[:3].tolist()
    return {"left_mm": [float(v) for v in l_mm], "right_mm": [float(v) for v in r_mm]}


def _run_single(
    nii: Path,
    out_dir: Path,
    backend: str,
    spm_dir: Path,
    matlab_cmd: str,
) -> dict:
    case_out = out_dir / nii.stem.replace(".nii", "")
    case_out.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {"input_nii": str(nii), "out_dir": str(case_out), "backend": backend}

    if backend in {"afni", "both"}:
        result["afni"] = run_afni(nii, case_out / "afni")

    if backend in {"spm", "both"}:
        spm_out = run_spm(nii, case_out / "spm", spm_dir=spm_dir, matlab_cmd=matlab_cmd)
        result["spm"] = spm_out
        dg = _estimate_dg_from_spm(Path(spm_out["gm"]))
        if dg:
            result["dg_proxy"] = dg

    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="Closed-loop T2 pipeline: discover NIfTI files in a folder and run AFNI/SPM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-root", type=Path, required=True, help="Folder containing NIfTI scans")
    p.add_argument(
        "--backend",
        choices=["afni", "spm", "both"],
        default="both",
        help="Pipeline backend",
    )
    p.add_argument("--out-dir", type=Path, default=Path("analysis_out"))
    p.add_argument("--spm-dir", type=Path, default=Path("/opt/spm12"), help="SPM installation directory")
    p.add_argument(
        "--matlab-cmd",
        default="matlab",
        help="MATLAB/SPM launcher command. For SPM standalone, provide wrapper command prefix.",
    )
    p.add_argument(
        "--input-nii",
        type=Path,
        action="append",
        default=None,
        help="Optional explicit NIfTI path(s). If omitted, files are discovered in --input-root.",
    )
    p.add_argument(
        "--no-convert-bruker",
        action="store_true",
        help="Disable automatic Bruker-to-NIfTI conversion when no NIfTI files are found.",
    )
    p.add_argument(
        "--converted-dir",
        type=Path,
        default=None,
        help="Where converted NIfTI files are written (default: <out-dir>/converted_nifti).",
    )
    p.add_argument(
        "--bruker-converter-cmd",
        default="bruker2nifti",
        help="Bruker converter command.",
    )
    p.add_argument(
        "--bruker-converter-args",
        default="-i {input} -o {output}",
        help="Arguments template for Bruker converter. Available placeholders: {input}, {output}.",
    )
    args = p.parse_args()

    if not args.input_root.exists():
        raise SystemExit(f"Input root not found: {args.input_root}")

    scans = args.input_nii or _find_t2_niftis(args.input_root)
    if not scans:
        if args.no_convert_bruker or not _looks_like_bruker_root(args.input_root):
            raise SystemExit(
                "No NIfTI files found. Convert Bruker data first (e.g. Bru2, bruker2nifti, or dcm2niix), "
                "or run without --no-convert-bruker to auto-convert."
            )

        converted_dir = args.converted_dir or (args.out_dir / "converted_nifti")
        print(f"No NIfTI files found. Attempting Bruker conversion into: {converted_dir}")
        scans = _convert_bruker_to_nifti(
            input_root=args.input_root,
            converted_dir=converted_dir,
            converter_cmd=args.bruker_converter_cmd,
            converter_args_template=args.bruker_converter_args,
        )
        if not scans:
            raise SystemExit(
                "Bruker conversion completed but no NIfTI files were discovered in the converted output. "
                "Check --bruker-converter-cmd/--bruker-converter-args and source data."
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_root": str(args.input_root),
        "backend": args.backend,
        "auto_bruker_conversion": not args.no_convert_bruker,
        "cases": [],
    }
    for nii in scans:
        if not nii.exists():
            print(f"Skipping missing file: {nii}")
            continue
        print(f"=== Processing: {nii}")
        summary["cases"].append(
            _run_single(
                nii=nii,
                out_dir=args.out_dir,
                backend=args.backend,
                spm_dir=args.spm_dir,
                matlab_cmd=args.matlab_cmd,
            )
        )

    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Done. Summary written to: {out_json}")


if __name__ == "__main__":
    main()
