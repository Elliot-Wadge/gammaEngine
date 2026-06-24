import numpy as np
from gammaEngine.shell import gamma, match_gamma_function, prescreen
import pymedphys

def make_grid_3D(nx=10, ny=10, nz=10, spacing=1.0):
    x = np.linspace(0, (nx - 1) * spacing, nx)
    y = np.linspace(0, (ny - 1) * spacing, ny)
    z = np.linspace(0, (nz - 1) * spacing, nz)

    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    return z, y, x, zz, yy, xx


def gaussian_3D(xx, yy, zz, sigma, x0, y0, z0):
    return np.exp(-((xx - x0)**2 + (yy - y0)**2 + (zz - z0)**2) / (2 * sigma**2))


def make_grid(nx=101, ny=101, spacing=1.0):
    x = np.linspace(0, (nx-1)*spacing, nx)
    y = np.linspace(0, (ny-1)*spacing, ny)
    xv, yv = np.meshgrid(x, y)
    return x, y, xv, yv


def gaussian_2D(xx,yy,sigma,x0,y0):
    return np.exp(-((xx-x0)**2 + (yy-y0)**2)/2/sigma**2)


def gaussian(x, b, c):
    return np.exp(-(x-b)**2/(2*c**2))


def test_match():
    f = match_gamma_function(([1],[1],[1]), False)
    assert f.__name__ == '_gamma_3D'
    f = match_gamma_function(([1],[1]), False)
    assert f.__name__ == '_gamma_2D'
    f = match_gamma_function(([1],), False)
    assert f.__name__ == '_gamma_1D'
    f = match_gamma_function(([1],[1],[1]), True)
    assert f.__name__ == '_gamma_3D_pr'
    f = match_gamma_function(([1],[1]), True)
    assert f.__name__ == '_gamma_2D_pr'
    f = match_gamma_function(([1],), True)
    assert f.__name__ == '_gamma_1D_pr'


def test_gamma_against_pymedphys():

    x, y, xx, yy = make_grid()
    z1 = gaussian_2D(xx, yy, 9.5, 0, 0)*200
    z2 = gaussian_2D(xx, yy, 10, 1, 0)*200 + 2
    coords1 = np.stack((yy.ravel(), xx.ravel()), axis=1)
    gamma_pmp = pymedphys.gamma((y,x), z1, (y,x), z2, 2, 2, 0, 100, local_gamma=False, interp_algo='scipy', global_normalisation=np.max(z2))
    gamma_me = gamma(coords1, (y,x), z1.ravel(), z2, 2.0, 2.0, 0.02, 100/np.max(z2))
    gamma_me = gamma_me.reshape(z1.shape)

    assert np.allclose(gamma_pmp, gamma_me, atol=3.1e-3)
    assert abs(np.mean(gamma_pmp - gamma_me)) < 1e-4

    x_ref = np.arange(-10,10,0.1, dtype=np.float64)
    x_eval = x_ref + 0.1
    y_ref = gaussian(x_ref, 0.7, 2)*250
    y_eval = gaussian(x_ref, 0, 1.98)*250

    gamma_me = gamma(x_eval,
                     (x_ref,),
                     y_eval,
                     y_ref,
                     2,
                     2,
                     0.001,
                     100/np.max(y_ref))


    gamma_pmp = pymedphys.gamma(x_eval, y_eval, x_ref, y_ref, 2, 2, interp_fraction=2000, local_gamma=False, lower_percent_dose_cutoff=0, global_normalisation=np.max(y_ref))
    assert abs(np.mean(gamma_pmp - gamma_me)) < 1e-4
    assert np.allclose(gamma_pmp, gamma_me, atol=3.1e-3)


    z, y, x, zz, yy, xx = make_grid_3D(nx=5, ny=5, nz=5)
    dose_ref = gaussian_3D(xx, yy, zz, 4, 0, 0, 0) * 50
    dose_eval = gaussian_3D(xx, yy, zz, 5, 0.5, 0, 0) * 50 + 1
    coords_eval = np.stack(
        (zz.ravel(), yy.ravel(), xx.ravel()),
        axis=1
    )
    gamma_me = gamma(
        coords_eval,
        (z,y,x),
        dose_eval.ravel(),
        dose_ref,
        2.0,
        2.0,
        0.05,
        100/np.max(dose_ref)
    )
    gamma_pmp = pymedphys.gamma(
        (z, y, x),
        dose_eval,
        (z, y, x),
        dose_ref,
        2,
        2,
        0,
        40,
        local_gamma=False,
        interp_algo='scipy',
        global_normalisation=np.max(dose_ref)
    )
    gamma_me = gamma_me.reshape(dose_ref.shape)
    assert abs(np.mean(gamma_pmp - gamma_me)) < 1e-2