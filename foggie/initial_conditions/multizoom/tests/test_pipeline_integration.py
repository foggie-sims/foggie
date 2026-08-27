"""Tests for the ics_refactor pipeline integration.

These exercise the parts that need no simulation data: group discovery from
the registry, the guard rails, and the contract that a rendered group config
parses back through multizoom's own config reader.
"""
import os
import sys

import numpy as np
import pytest
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from foggie.initial_conditions.multizoom import config as mzconfig
from foggie.initial_conditions.multizoom import pipeline_integration as pi
from foggie.initial_conditions.multizoom import lagrangian_regions


def _registry(rows, with_column=True):
    cols = dict(halo_id=[r[0] for r in rows],
                box=[r[1] for r in rows],
                enabled=[r[2] for r in rows],
                rvir_min=[r[3] for r in rows])
    if with_column:
        cols[pi.GROUP_COLUMN] = [r[4] for r in rows]
    return Table(cols)


def test_no_column_means_no_groups():
    """The production registry has no multizoom_group column; that must be
    silently 'no groups', never an error."""
    t = _registry([(1, "25Mpc_DM_512", True, 0.0, "")], with_column=False)
    assert pi.registry_groups(t) == {}


def test_groups_collected_and_disabled_skipped():
    t = _registry([
        (11, "25Mpc_DM_512", True,  0.0, "dwarfs"),
        (22, "25Mpc_DM_512", True,  0.0, "dwarfs"),
        (33, "25Mpc_DM_512", False, 0.0, "dwarfs"),   # disabled
        (44, "25Mpc_DM_512", True,  0.0, ""),         # ungrouped
        (55, "25Mpc_DM_512", True,  0.0, "pair"),
    ])
    groups = pi.registry_groups(t)
    assert groups == {"dwarfs": [11, 22], "pair": [55]}


def test_group_spanning_two_boxes_is_refused():
    """A group shares one parent box and one noise realization."""
    t = _registry([
        (11, "25Mpc_DM_512", True, 0.0, "mixed"),
        (22, "25Mpc_DM_256", True, 0.0, "mixed"),
    ])
    with pytest.raises(RuntimeError, match="spans several boxes"):
        pi.group_box("mixed", t)


def test_unknown_group_is_refused():
    t = _registry([(11, "25Mpc_DM_512", True, 0.0, "dwarfs")])
    with pytest.raises(RuntimeError, match="no enabled members"):
        pi.group_box("nope", t)


def test_rendered_config_parses_back(tmp_path, monkeypatch):
    """The rendered group config must satisfy multizoom's own parser."""
    class FakeBox:
        sim_name = "25Mpc_DM_512"
        template_config = "25Mpc_DM_512_planck18.conf"
        refine_centers = False
        def music_exe_dir_path(self):
            return "/path/to/music"

    monkeypatch.setattr(pi.pconfig, "foggie_ics_dir", lambda: str(tmp_path))
    monkeypatch.setattr(pi.pbuild, "center_for_level",
                        lambda box, h, lv, d, rv: np.array([0.5, 0.4, 0.3]))
    monkeypatch.setattr(pi.pbuild, "halo_center_and_radius",
                        lambda box, h, rv: (None, 205.0))
    t = _registry([(11, "25Mpc_DM_512", True, 0.0, "dwarfs"),
                   (22, "25Mpc_DM_512", True, 400.0, "dwarfs")])

    text = pi.render_group_config(FakeBox(), "dwarfs", [11, 22], 2, t)
    path = tmp_path / "group.conf"
    path.write_text(text)

    params = mzconfig.parse_multizoom_config(str(path))
    assert list(params["halos"]) == ["11", "22"]
    assert params["tag"] == "11+22"
    # multi-halo must use the exact particle mask, never a spanning hull
    assert params["shape_type"] == "exact"
    assert params["new_ics_directory"].endswith("multizoom_dwarfs")
    np.testing.assert_allclose(params["halos"]["11"]["center"][0],
                               [0.5, 0.4, 0.3])
    assert params["halos"]["22"]["radius"] == (205.0, "kpc")


def test_trim_lagrangian_outliers(tmp_path):
    """Strays beyond 5x the 99th-percentile radius are dropped; a genuinely
    extended cloud is left alone."""
    rng = np.random.default_rng(3)
    core = rng.normal(loc=0.5, scale=0.002, size=(500, 3))
    strays = np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]])
    fn = tmp_path / "pts.dat"
    np.savetxt(fn, np.vstack([core, strays]))
    dropped = lagrangian_regions.trim_lagrangian_outliers(str(fn))
    assert dropped == 2
    assert len(np.loadtxt(fn)) == 500

    # a diffuse cloud: too many "outliers" to be strays, so nothing is trimmed
    diffuse = rng.normal(loc=0.5, scale=0.05, size=(500, 3))
    fn2 = tmp_path / "diffuse.dat"
    np.savetxt(fn2, diffuse)
    assert lagrangian_regions.trim_lagrangian_outliers(str(fn2)) == 0
    assert len(np.loadtxt(fn2)) == 500
