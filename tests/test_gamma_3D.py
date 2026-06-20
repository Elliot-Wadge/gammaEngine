import numpy as np
from gammaEngine.core import _gamma_3D, _gamma_3D_pr
import pymedphys



def make_grid_3D(nx=10, ny=10, nz=10, spacing=1.0):
    x = np.linspace(0, (nx - 1) * spacing, nx)
    y = np.linspace(0, (ny - 1) * spacing, ny)
    z = np.linspace(0, (nz - 1) * spacing, nz)

    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    return z, y, x, zz, yy, xx


def gaussian_3D(xx, yy, zz, sigma, x0, y0, z0):
    return np.exp(-((xx - x0)**2 + (yy - y0)**2 + (zz - z0)**2) / (2 * sigma**2))


def test_compare_to_pymedphys_3D():

    # Create grid
    z, y, x, zz, yy, xx = make_grid_3D(nx=10, ny=10, nz=10)

    # Reference and evaluation dose
    dose_ref = gaussian_3D(xx, yy, zz, 5, 0, 0, 0) * 100
    dose_eval = gaussian_3D(xx, yy, zz, 5, 1, 0, 0) * 100 + 1

    # Coordinates for evaluation points (z, y, x)
    coords_eval = np.stack(
        (zz.ravel(), yy.ravel(), xx.ravel()),
        axis=1
    )

    # Your implementation
    gamma_me = _gamma_3D(
        coords_eval,
        z,
        y,
        x,
        dose_eval.ravel(),
        dose_ref,
        2.0,
        2.0,
        0.02,
        1
    )

    # PyMedPhys gamma (3D)
    gamma_pmp = pymedphys.gamma(
        (z, y, x),
        dose_eval,
        (z, y, x),
        dose_ref,
        2,
        2,
        0,
        100,
        local_gamma=False,
        interp_algo='scipy'
    )

    gamma_me = gamma_me.reshape(dose_ref.shape)

    # Assertions
    assert abs(np.mean(gamma_pmp - gamma_me)) < 1e-2

    
def test_pymedphys_speed_3D(benchmark):


    # Create grid
    z, y, x, zz, yy, xx = make_grid_3D()

    # Reference and evaluation dose
    dose_ref = gaussian_3D(xx, yy, zz, 10, 0, 0, 0) * 100
    dose_eval = gaussian_3D(xx, yy, zz, 10, 1, 0, 0) * 100 + 1

    # Coordinates for evaluation points (z, y, x)
    coords_eval = np.stack(
        (zz.ravel(), yy.ravel(), xx.ravel()),
        axis=1
    )

    gamma_pmp = pymedphys.gamma(
        (z, y, x),
        dose_ref,
        (z, y, x),
        dose_eval,
        2,
        2,
        0,
        10,
        local_gamma=False,
        interp_algo='scipy'
    )

    def run_metric():
        reg = gamma_pmp = pymedphys.gamma(
                (z, y, x),
                dose_ref,
                (z, y, x),
                dose_eval,
                2,
                2,
                0,
                10,
                local_gamma=False,
                interp_algo='scipy'
            )
        return reg

    reg = benchmark(run_metric)



def test_gamma_speed_3D(benchmark):


    # Create grid
    z, y, x, zz, yy, xx = make_grid_3D()

    # Reference and evaluation dose
    dose_ref = gaussian_3D(xx, yy, zz, 10, 0, 0, 0) * 100
    dose_eval = gaussian_3D(xx, yy, zz, 10, 1, 0, 0) * 100 + 1

    # Coordinates for evaluation points (z, y, x)
    coords_eval = np.stack(
        (zz.ravel(), yy.ravel(), xx.ravel()),
        axis=1
    )

    gamma_me = _gamma_3D(
        coords_eval,
        z,
        y,
        x,
        dose_ref.ravel(),
        dose_eval,
        2.0,
        2.0,
        0.2,
        1
    )

    def run_metric():
        reg = gamma_me = _gamma_3D(
                            coords_eval,
                            z,
                            y,
                            x,
                            dose_ref.ravel(),
                            dose_eval,
                            2.0,
                            2.0,
                            0.2,
                            1
                        )
        return reg

    reg = benchmark(run_metric)


def test_gamma_speed_3D_pr(benchmark):


    # Create grid
    z, y, x, zz, yy, xx = make_grid_3D()

    # Reference and evaluation dose
    dose_ref = gaussian_3D(xx, yy, zz, 10, 0, 0, 0) * 100
    dose_eval = gaussian_3D(xx, yy, zz, 10, 1, 0, 0) * 100 + 1

    # Coordinates for evaluation points (z, y, x)
    coords_eval = np.stack(
        (zz.ravel(), yy.ravel(), xx.ravel()),
        axis=1
    )

    gamma_me = _gamma_3D_pr(
        coords_eval,
        z,
        y,
        x,
        dose_ref.ravel(),
        dose_eval,
        2.0,
        2.0,
        0.2,
        1
    )

    def run_metric():
        reg = gamma_me = _gamma_3D_pr(
                            coords_eval,
                            z,
                            y,
                            x,
                            dose_ref.ravel(),
                            dose_eval,
                            2.0,
                            2.0,
                            0.2,
                            1
                        )
        return reg

    reg = benchmark(run_metric)


def debug():
    z, y, x, zz, yy, xx = make_grid_3D()
    dose_ref = np.ones(zz.shape, dtype=np.float64)*100
    

    gamma_me = _gamma_3D(
        np.array([[0,0.4,0]], dtype=np.float64),
        z,
        y,
        x,
        np.array([98], dtype=np.float64),
        dose_ref,
        2.0,
        2.0,
        0.02,
        1
    )
    print(gamma_me)


def test_planes_3D():
    # Define axes
    z_ax = np.arange(-25, 26, 1, dtype=np.float64)
    y_ax = np.arange(-25, 26, 1, dtype=np.float64)
    x_ax = np.arange(-25, 26, 1, dtype=np.float64)

    rng = np.random.default_rng(0)
    params = rng.random((15, 3)) * 2

    d_thresh = 1.0
    dist_thresh = 1.0

    zz, yy, xx = np.meshgrid(z_ax, y_ax, x_ax, indexing='ij')

    for param in params:
        slope, intercept, threshold = param
        dose_ref = slope*(xx-intercept)
        coord_eval = np.array([[0.0, 0.0, 0.0]])
        dose_eval = np.array([0.0])

        
        min_gamma = abs(slope*intercept/np.sqrt(1+slope**2))/threshold
        
        

        # Execute 3D Gamma
        res = _gamma_3D(
            coord_eval=coord_eval,
            ref_z_axes=z_ax,
            ref_y_axes=y_ax,
            ref_x_axes=x_ax,
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            dose_threshold=threshold,
            distance_threshold=threshold,
            interp_res=0.01, # Increased for speed; lower for higher precision
            norm=1.0
        )[0]

        assert np.isclose(res, min_gamma, atol=1e-3, rtol=1e-3)


def test_planes_3D_pr():
    # Define axes
    z_ax = np.arange(-25, 26, 1, dtype=np.float64)
    y_ax = np.arange(-25, 26, 1, dtype=np.float64)
    x_ax = np.arange(-25, 26, 1, dtype=np.float64)

    rng = np.random.default_rng(0)
    params = rng.random((100, 3)) * 2

    d_thresh = 1.0
    dist_thresh = 1.0

    zz, yy, xx = np.meshgrid(z_ax, y_ax, x_ax, indexing='ij')

    for param in params:
        slope, intercept, threshold = param
        dose_ref = slope*(xx-intercept)
        coord_eval = np.array([[0.0, 0.0, 0.0]])
        dose_eval = np.array([0.0])

        
        min_gamma = abs(slope*intercept/np.sqrt(1+slope**2))/threshold
        
        

        # Execute 3D Gamma
        res = _gamma_3D_pr(
            coord_eval=coord_eval,
            ref_z_axes=z_ax,
            ref_y_axes=y_ax,
            ref_x_axes=x_ax,
            dose_eval=dose_eval,
            dose_ref=dose_ref,
            dose_threshold=threshold,
            distance_threshold=threshold,
            interp_res=0.01, # Increased for speed; lower for higher precision
            norm=1.0
        )[0]

        if min_gamma <= 1:
            print(min_gamma)
            assert bool(res)
        else:
            assert not bool(res)



if __name__ == '__main__':
    test_planes_3D()