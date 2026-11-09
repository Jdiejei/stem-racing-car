"""transient.py without OpenFOAM: the case rewrite and the drag statistics."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openfoam_case as oc
import transient as tr


def test_control_dict_becomes_time_accurate():
    c = tr.transient_control(oc.build_control_dict(oc.OpenFOAMRunConfig(max_iterations=1000), 1e-3),
                             0.15, 1e-5, 1.5)
    for want in ("application     pimpleFoam;", "endTime         0.15;", "deltaT          1e-05;",
                 "adjustTimeStep  yes;", "maxCo           1.5;", "writeControl    adjustableRunTime;",
                 "writeInterval   0.15;", "timePrecision   10;"):
        assert want in c
    assert "simpleFoam" not in c and "forces" in c          # force objects kept


def test_solution_gets_final_solvers_and_pimple():
    s = tr.transient_solution(oc._FV_SOLUTION)
    assert '"(p|pFinal)"' in s and '"(U|k|omega|ReThetat|gammaInt)(|Final)"' in s
    assert "PIMPLE" in s and "\nSIMPLE" not in s
    assert s.count("relaxationFactors") == 1 and '".*" 1;' in s


def test_case_built_from_the_latest_steady_field(tmp_path):
    run = tmp_path / "steady"
    oc._write(run / "system" / "controlDict", oc.build_control_dict(oc.OpenFOAMRunConfig(), 1e-3))
    oc._write(run / "system" / "fvSchemes", oc._FV_SCHEMES)
    oc._write(run / "system" / "fvSolution", oc._FV_SOLUTION)
    oc._write(run / "constant" / "polyMesh" / "points", "pts")
    oc._write(run / "0" / "U", "cold")
    oc._write(run / "1000" / "U", "converged")
    oc._write(run / "1000" / "phi", "flux")
    case = tr.make_case(run, tmp_path / "t", 0.1, 1e-5, 1.5, 4)
    assert (case / "0" / "U").read_text() == "converged"
    assert not (case / "0" / "phi").exists()
    assert "default backward;" in (case / "system" / "fvSchemes").read_text()
    assert "numberOfSubdomains 4;" in (case / "system" / "decomposeParDict").read_text()


def test_statistics_see_a_switching_wake():
    t = np.linspace(0, 0.2, 4001)
    d_half = np.where((t // 0.025) % 2 == 0, 0.20, 0.23)     # 0.40 N / 0.46 N full car, switching
    r = tr.summarise(t, d_half)
    assert r["D_mean_N"] == pytest.approx(0.43, abs=0.005)
    assert r["D_min_N"] == pytest.approx(0.40) and r["D_max_N"] == pytest.approx(0.46)
    assert r["D_sd_N"] == pytest.approx(0.03, abs=0.003)


def test_total_force_comes_from_the_total_object_only(tmp_path):
    for obj, fx in (("forces", 0.2), ("forces_wheelR", 0.05)):
        oc._write(tmp_path / "postProcessing" / obj / "0" / "force.dat",
                  "# Time total\n0.001 (%g 0 0) (0 0 0) (0 0 0)\n0.002 (%g 0 0) (0 0 0) (0 0 0)\n" % (fx, fx))
    t, d = tr.drag_history(tmp_path)
    assert list(d) == [0.2, 0.2]
    assert set(tr.group_histories(tmp_path)) == {"wheelR"}


def test_stage_commands_use_an_absolute_case_path(tmp_path, monkeypatch):
    """A relative case path broke every stage: `cd rel && source .. && cd rel` resolves
    the second cd against the first (transient runs, 2026-11-09)."""
    import subprocess
    seen = {}

    class P:
        returncode, stdout = 0, ""

    def fake(args, cwd, **kw):
        seen["cmd"], seen["cwd"] = args[-1], cwd
        return P()

    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "case").mkdir()
    oc._run("decomposePar", Path("case"), "bashrc", "x.log", 10)
    absolute = str((tmp_path / "case").resolve())
    assert seen["cmd"].count(f"cd '{absolute}'") == 2 and seen["cwd"] == absolute
