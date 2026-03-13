#!/usr/bin/env python3
"""Run a real AFNI or SPM pipeline on MRI data.

This script is intentionally explicit about external requirements:
- AFNI backend requires AFNI binaries in PATH (e.g. 3dUnifize, 3dSkullStrip).
- SPM backend requires MATLAB (or MCR wrapper) plus an installed SPM toolbox.

It does not try to re-implement those suites; it orchestrates them reproducibly.
"""

from __future__ import annotations

import argparse
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


def run_afni(input_nii: Path, out_dir: Path) -> None:
    """Run a basic but proper AFNI preprocessing chain."""
    required = ["3dcalc", "3dAutomask", "3dUnifize", "3dSkullStrip"]
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

    if shutil.which("@chauffeur_afni"):
        _run(
            [
                "@chauffeur_afni",
                "-ulay",
                str(unifized),
                "-olay",
                str(automask),
                "-prefix",
                str(out_dir / "qc_mask"),
                "-montx",
                "4",
                "-monty",
                "1",
                "-set_xhairs",
                "OFF",
                "-label_mode",
                "0",
            ]
        )


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

        % Tissue classes (TPM.nii)
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


def run_spm(input_nii: Path, out_dir: Path, spm_dir: Path, matlab_cmd: str) -> None:
    """Run proper SPM tissue segmentation through MATLAB."""
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run AFNI or SPM analysis on a NIfTI volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-nii", type=Path, required=True, help="Input NIfTI (.nii/.nii.gz)")
    p.add_argument("--backend", choices=["afni", "spm"], required=True)
    p.add_argument("--out-dir", type=Path, default=Path("analysis_out"))
    p.add_argument("--spm-dir", type=Path, default=Path("/opt/spm12"), help="SPM installation directory")
    p.add_argument(
        "--matlab-cmd",
        default="matlab",
        help="MATLAB/SPM launcher command. For SPM standalone, provide wrapper command prefix.",
    )
    args = p.parse_args()

    if not args.input_nii.exists():
        raise SystemExit(f"Input NIfTI not found: {args.input_nii}")

    if args.backend == "afni":
        run_afni(args.input_nii, args.out_dir)
    else:
        run_spm(args.input_nii, args.out_dir, args.spm_dir, args.matlab_cmd)

    print(f"Done. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
