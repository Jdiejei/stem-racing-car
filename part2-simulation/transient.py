"""Does the wake really switch? A time-accurate continuation of a steady car.

Takes a finished steady case (run_car --keep-runs), copies its mesh and its
converged field into a new case as the start, and runs pimpleFoam for
`duration` seconds of flow time with the same mesh, models and force function
objects. The drag history then shows whether the flow stays on the steady
solution's branch, settles on another, or keeps switching, and what the
honest time-averaged drag is.

    python transient.py RUN_DIR OUT_DIR --duration 0.15 --np 4
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

import openfoam_case as oc

SCHEMES_DDT = "ddtSchemes      { default backward; }"
PIMPLE = """
PIMPLE
{
    nOuterCorrectors 1;
    nCorrectors      2;
    nNonOrthogonalCorrectors 1;
    consistent       yes;
}

relaxationFactors { equations { ".*" 1; } }
"""


def latest_time_dir(run: Path) -> Path:
    times = []
    for d in run.iterdir():
        try:
            times.append((float(d.name), d))
        except ValueError:
            pass
    times = [t for t in times if t[0] > 0]
    if not times:
        raise FileNotFoundError(f"no solved time directory in {run}")
    return max(times)[1]


def transient_control(steady_control: str, duration_s: float, dt_s: float, max_co: float) -> str:
    c = re.sub(r"\napplication\s+\S+;", "\napplication     pimpleFoam;", steady_control)
    c = re.sub(r"\nendTime\s+\S+;", f"\nendTime         {duration_s};", c)
    c = re.sub(r"\ndeltaT\s+\S+;", f"\ndeltaT          {dt_s};\nadjustTimeStep  yes;\nmaxCo           {max_co};", c)
    c = re.sub(r"\nwriteControl\s+\S+;", "\nwriteControl    adjustableRunTime;", c)
    c = re.sub(r"\nwriteInterval\s+\S+;", f"\nwriteInterval   {duration_s};", c)
    c = re.sub(r"\ntimePrecision\s+\S+;", "\ntimePrecision   10;", c)
    return c


def transient_solution(steady_solution: str) -> str:
    s = steady_solution.replace('"(U|k|omega|ReThetat|gammaInt)"', '"(U|k|omega|ReThetat|gammaInt)(|Final)"')
    s = s.replace("solvers\n{\n    p\n", "solvers\n{\n    \"(p|pFinal)\"\n", 1)
    s = re.sub(r"\nSIMPLE\s*\{.*?\n\}\n", "\n", s, flags=re.S)
    s = re.sub(r"\nrelaxationFactors\s*\{.*\n\}\s*$", "\n", s, flags=re.S)
    return s + PIMPLE


def make_case(steady_run: Path, out: Path, duration_s: float, dt_s: float, max_co: float, n: int) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(steady_run / "constant", out / "constant")
    shutil.copytree(steady_run / "system", out / "system")
    shutil.copytree(latest_time_dir(steady_run), out / "0")
    shutil.rmtree(out / "0" / "uniform", ignore_errors=True)
    for f in ("phi",):
        (out / "0" / f).unlink(missing_ok=True)
    sys_ = out / "system"
    (sys_ / "controlDict").write_text(transient_control((sys_ / "controlDict").read_text(), duration_s, dt_s, max_co))
    (sys_ / "fvSchemes").write_text(re.sub(r"ddtSchemes\s*\{[^}]*\}", SCHEMES_DDT, (sys_ / "fvSchemes").read_text()))
    (sys_ / "fvSolution").write_text(transient_solution((sys_ / "fvSolution").read_text()))
    (sys_ / "decomposeParDict").write_text(oc._decompose_dict(n))
    return out


def _fx_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        v = re.sub(r"[()]", " ", line).split()
        rows.append((float(v[0]), float(v[1])))
    a = np.array(rows)
    return a[:, 0], a[:, 1]


def drag_history(case: Path) -> tuple[np.ndarray, np.ndarray]:
    """(t, half-car x-force) from the TOTAL forces object (`forces`, all walls);
    the per-group objects `forces_<group>` are read by group_histories."""
    files = sorted(case.glob("postProcessing/forces/*/force.dat"))
    if not files:
        raise FileNotFoundError("no postProcessing/forces/*/force.dat written")
    return _fx_series(files[-1])


def group_histories(case: Path) -> dict:
    """{group: (t, half x-force)} for each forces_<group> object (body, wheels...)."""
    out = {}
    for f in sorted(case.glob("postProcessing/forces_*/*/force.dat")):
        out[f.parent.parent.name[len("forces_"):]] = _fx_series(f)
    return out


def summarise(t, d_half, settle_frac=0.25, n_blocks=8) -> dict:
    """Drag statistics over the time after the first `settle_frac` of the run:
    mean, SD, range, and the means of n_blocks equal blocks (do they wander?)."""
    keep = t >= t[0] + settle_frac * (t[-1] - t[0])
    tt, d = t[keep], 2.0 * d_half[keep]                       # full car
    # time-weighted (the step adapts)
    w = np.gradient(tt)
    mean = float(np.sum(w * d) / np.sum(w))
    blocks = [float(np.mean(b)) for b in np.array_split(d, n_blocks)]
    return {"t_start_s": float(tt[0]), "t_end_s": float(tt[-1]), "steps": int(len(tt)),
            "D_mean_N": mean, "D_sd_N": float(np.sqrt(np.sum(w * (d - mean) ** 2) / np.sum(w))),
            "D_min_N": float(d.min()), "D_max_N": float(d.max()), "block_means_N": blocks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steady_run")
    ap.add_argument("out")
    ap.add_argument("--duration", type=float, default=0.15)
    ap.add_argument("--dt", type=float, default=1e-5)
    ap.add_argument("--max-co", type=float, default=1.5)
    ap.add_argument("--np", type=int, default=4)
    a = ap.parse_args()
    bashrc = oc.find_openfoam_bashrc(None)
    case = make_case(Path(a.steady_run), Path(a.out), a.duration, a.dt, a.max_co, a.np)
    timeout = 20000
    import subprocess
    try:
        oc._run("decomposePar -force", case, bashrc, "decomposePar.log", timeout)
        oc._run(f"mpirun -np {a.np} pimpleFoam -parallel", case, bashrc, "pimpleFoam.log", timeout)
    except subprocess.CalledProcessError as e:      # show why, in the job log
        print((e.output or "")[-4000:])
        raise
    t, d = drag_history(case)
    np.savetxt(case / "drag_history.txt", np.c_[t, 2.0 * d], header="t_s D_full_N")
    rec = summarise(t, d)
    rec["groups"] = {g: summarise(tg, dg) for g, (tg, dg) in group_histories(case).items()}
    (case / "transient.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
