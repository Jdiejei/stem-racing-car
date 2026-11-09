"""Mesh morphing (noise fix 3) without OpenFOAM: the moveMesh call is faked."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openfoam_case as oc


def _case(tmp_path, cfg):
    run = tmp_path / "run"
    oc._write(run / "system" / "controlDict", oc.build_control_dict(cfg, 1e-3))
    oc._write(run / "system" / "fvSolution", oc._FV_SOLUTION)
    oc._write(run / "constant" / "polyMesh" / "points", "blockMesh points")
    ref = tmp_path / "ref"
    for f in ("points", "faces", "owner", "neighbour", "boundary"):
        oc._write(ref / "constant" / "polyMesh" / f, f"ref {f}")
    return run, ref


def test_morph_moves_the_reference_mesh_and_restores_the_case(tmp_path, monkeypatch):
    cfg = oc.OpenFOAMRunConfig(max_iterations=1000, morph_from=str(tmp_path / "ref"))
    run, _ = _case(tmp_path, cfg)
    seen = {}

    def fake_run(cmd, cwd, bashrc, log, timeout):
        seen["cmd"] = cmd
        seen["control"] = (cwd / "system" / "controlDict").read_text()
        seen["solution"] = (cwd / "system" / "fvSolution").read_text()
        seen["pd"] = (cwd / "0" / "pointDisplacement").read_text()
        seen["dyn"] = (cwd / "constant" / "dynamicMeshDict").read_text()
        seen["mesh_points"] = (cwd / "constant" / "polyMesh" / "points").read_text()
        oc._write(cwd / str(oc.MORPH_STEPS) / "polyMesh" / "points", "moved points")
        return ""

    monkeypatch.setattr(oc, "_run", fake_run)
    control0 = (run / "system" / "controlDict").read_text()
    oc._morph(run, cfg, "bashrc", 60)

    assert seen["cmd"] == "moveMesh -noFunctionObjects"
    assert seen["mesh_points"] == "ref points"                 # started from the reference
    assert f"endTime         {oc.MORPH_STEPS};" in seen["control"]
    assert f"writeInterval   {oc.MORPH_STEPS};" in seen["control"]
    assert '"cellDisplacement.*"' in seen["solution"]
    assert "surfaceDisplacement" in seen["pd"] and "car.stl" in seen["pd"]
    assert "displacementLaplacian" in seen["dyn"]
    # result: moved points in place, reference topology, case as it was
    assert (run / "constant" / "polyMesh" / "points").read_text() == "moved points"
    assert (run / "constant" / "polyMesh" / "faces").read_text() == "ref faces"
    assert (run / "system" / "controlDict").read_text() == control0
    assert '"cellDisplacement.*"' not in (run / "system" / "fvSolution").read_text()
    assert not (run / str(oc.MORPH_STEPS)).exists()
    assert not (run / "constant" / "dynamicMeshDict").exists()
    assert not (run / "0" / "pointDisplacement").exists()


def test_morph_without_a_reference_mesh_fails_loudly(tmp_path):
    cfg = oc.OpenFOAMRunConfig(morph_from=str(tmp_path / "nowhere"))
    run, _ = _case(tmp_path, cfg)
    with pytest.raises(FileNotFoundError):
        oc._morph(run, cfg, "bashrc", 60)


def test_snapping_is_unchanged_without_morph(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(oc, "_run", lambda cmd, *a: calls.append(cmd) or "")
    oc._mesh(tmp_path, oc.OpenFOAMRunConfig(), "bashrc", 60)
    assert calls == ["snappyHexMesh -overwrite", "checkMesh"]


def test_warm_start_maps_the_reference_field_after_meshing(tmp_path, monkeypatch):
    ref = tmp_path / "ref"
    oc._write(ref / "system" / "controlDict", "x")
    calls = []
    monkeypatch.setattr(oc, "_run", lambda cmd, *a: calls.append(cmd) or "")
    oc._mesh(tmp_path / "run", oc.OpenFOAMRunConfig(warm_from=str(ref)), "bashrc", 60)
    assert calls == ["snappyHexMesh -overwrite",
                     f"mapFields '{ref}' -sourceTime latestTime -consistent", "checkMesh"]


def test_warm_start_without_a_case_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(oc, "_run", lambda cmd, *a: "")
    with pytest.raises(FileNotFoundError):
        oc._mesh(tmp_path, oc.OpenFOAMRunConfig(warm_from=str(tmp_path / "nope")), "bashrc", 60)


def test_morph_moves_in_small_steps():
    pd = oc.morph_point_displacement()
    assert f"velocity        ({oc.MORPH_STEP_M:g} {oc.MORPH_STEP_M:g} {oc.MORPH_STEP_M:g});" in pd
    assert oc.MORPH_STEPS * oc.MORPH_STEP_M >= 0.004          # reaches a 4 mm local cut


def _mesh_with(tmp_path, monkeypatch, check_text):
    calls = []
    monkeypatch.setattr(oc, "_morph", lambda *a: calls.append("morph"))
    def fake_run(cmd, cwd, bashrc, log, timeout):
        calls.append(cmd)
        return check_text if log == "checkMesh_morph.log" else "Mesh OK."
    monkeypatch.setattr(oc, "_run", fake_run)
    oc._mesh(tmp_path, oc.OpenFOAMRunConfig(morph_from=str(tmp_path / "ref")), "bashrc", 60)
    return calls


def test_a_good_reshape_is_kept(tmp_path, monkeypatch):
    calls = _mesh_with(tmp_path, monkeypatch, "Mesh OK.")
    assert calls == ["morph", "checkMesh", "checkMesh"] and not (tmp_path / "MORPH_FALLBACK").exists()


def test_a_bad_reshape_falls_back_to_a_fresh_mesh_and_says_so(tmp_path, monkeypatch):
    calls = _mesh_with(tmp_path, monkeypatch,
                       " ***Error in face pyramids: 1 faces are incorrectly oriented.\nFailed 1 mesh checks.\n")
    assert calls == ["morph", "checkMesh", "blockMesh", "snappyHexMesh -overwrite", "checkMesh"]
    assert "incorrectly oriented" in (tmp_path / "MORPH_FALLBACK").read_text()
