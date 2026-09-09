import numpy as np
import pytest

from multizoom import config as mzconfig

MULTI_CONF = """
[setup]
simulation_name = testsim
mode = merge

[region]
final_redshift = 0.0
radius_factor = 1.5
shape_type = exact
halo_radius_units = kpc

[halo:5016]
halo_center = 0.493, 0.508, 0.461
halo_radius = 205.

[halo:5033]
halo_center = 0.201, 0.774, 0.312
halo_mass = 8.3e11
halo_mass_units = Msun/h
radius_factor = 2.0
"""

LEGACY_CONF = """
[setup]
simulation_name = legacy

[region]
final_redshift = 0.0
halo_center = 0.5, 0.5, 0.5
halo_radius = 200.
shape_type = convex_hull
"""


def _write(tmp_path, text, name="test.conf"):
    fn = tmp_path / name
    fn.write_text(text)
    return str(fn)


def test_multi_halo_parse(tmp_path):
    params = mzconfig.parse_multizoom_config(_write(tmp_path, MULTI_CONF))
    assert list(params["halos"]) == ["5016", "5033"]
    assert params["tag"] == "5016+5033"
    assert params["mode"] == "merge"
    h1 = params["halos"]["5016"]
    assert h1["radius"] == (205.0, "kpc")
    assert h1["radius_factor"] == 1.5
    np.testing.assert_allclose(h1["center"][0], [0.493, 0.508, 0.461])
    h2 = params["halos"]["5033"]
    assert h2["mass"] == (8.3e11, "Msun/h")
    assert "radius" not in h2
    assert h2["radius_factor"] == 2.0


def test_legacy_single_halo(tmp_path):
    params = mzconfig.parse_multizoom_config(_write(tmp_path, LEGACY_CONF))
    assert list(params["halos"]) == ["0"]
    assert params["halos"]["0"]["radius"] == (200.0, "kpc")
    assert params["mode"] == "union"


def test_multi_halo_requires_exact(tmp_path):
    text = MULTI_CONF.replace("shape_type = exact",
                              "shape_type = convex_hull")
    with pytest.raises(RuntimeError, match="exact"):
        mzconfig.parse_multizoom_config(_write(tmp_path, text))


def test_missing_center_fails(tmp_path):
    text = MULTI_CONF.replace("halo_center = 0.493, 0.508, 0.461\n", "")
    with pytest.raises(RuntimeError, match="5016"):
        mzconfig.parse_multizoom_config(_write(tmp_path, text))


def test_mass_and_radius_warns_and_prefers_mass(tmp_path):
    text = MULTI_CONF.replace("halo_mass = 8.3e11",
                              "halo_mass = 8.3e11\nhalo_radius = 300.")
    with pytest.warns(UserWarning, match="Defaulting to mass"):
        params = mzconfig.parse_multizoom_config(_write(tmp_path, text))
    halo = params["halos"]["5033"]
    assert "mass" in halo and "radius" not in halo


def test_music_env_parse():
    env = mzconfig.parse_music_env(
        "LD_LIBRARY_PATH=/opt/hdf5/lib;OMP_PROC_BIND=true")
    assert env == {"LD_LIBRARY_PATH": "/opt/hdf5/lib",
                   "OMP_PROC_BIND": "true"}
    assert mzconfig.parse_music_env(None) == {}
    with pytest.raises(RuntimeError):
        mzconfig.parse_music_env("NOEQUALSSIGN")


def test_rockstar_catalog(tmp_path):
    catalog_fn = tmp_path / "out_0.list"
    catalog_fn.write_text(
        "#ID DescID Mvir Vmax Vrms Rvir Rs Np X Y Z\n"
        "#a = 1.000000\n"
        "12 -1 1e12 180 190 250 30 1000 12.5 20.0 5.0\n"
        "34 -1 5e11 150 160 180 25 500 2.5 7.5 22.5\n")
    catalog = mzconfig.read_rockstar_catalog(str(catalog_fn))
    center, rvir = mzconfig.halo_from_catalog(catalog, "34", 25.0)
    np.testing.assert_allclose(center, [0.1, 0.3, 0.9])
    assert rvir == 180.0
    with pytest.raises(RuntimeError, match="not found"):
        mzconfig.halo_from_catalog(catalog, "99", 25.0)
