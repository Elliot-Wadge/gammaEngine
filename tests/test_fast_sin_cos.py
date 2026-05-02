import numpy as np
from gammaEngine.core import fast_cos, fast_sin


def test_fast_sin():
    rng = np.random.default_rng(0)
    test_angles = rng.random(300)*2*np.pi
    N = 4096
    angles = np.linspace(0,2*np.pi+4*np.pi/N,N)
    sin_table = np.sin(angles)
    cos_table = np.cos(angles)
    dangle = (2*np.pi + 2*np.pi/N)/N
    inv_dangle = 1/dangle

    for angle in test_angles:

        
        if not np.isclose(np.sin(angle), fast_sin(angle, sin_table, dangle, inv_dangle), atol=5e-3):
            print(np.sin(angle), fast_sin(angle, sin_table, dangle, inv_dangle))
            print()

        assert np.isclose(np.sin(angle), fast_sin(angle, sin_table, dangle, inv_dangle), atol=5e-3)


def test_fast_cos():
    rng = np.random.default_rng(0)
    test_angles = rng.random(300)*2*np.pi
    N = 4096
    angles = np.linspace(0,2*np.pi+4*np.pi/N,N)
    sin_table = np.sin(angles)
    cos_table = np.cos(angles)
    dangle = (2*np.pi + 2*np.pi/N)/N
    inv_dangle = 1/dangle

    for angle in test_angles:

        
        
        if not np.isclose(np.cos(angle), fast_cos(angle, cos_table, dangle, inv_dangle), atol=5e-3):
            print(np.cos(angle), fast_cos(angle, cos_table, dangle, inv_dangle))
            print()

        assert np.isclose(np.cos(angle), fast_cos(angle, cos_table, dangle, inv_dangle), atol=5e-3)