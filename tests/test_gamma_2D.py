import numpy as np
from gammaEngine.core import _gamma_2D
import pymedphys

def make_grid(nx=101, ny=101, spacing=1.0):
    x = np.linspace(0, (nx-1)*spacing, nx)
    y = np.linspace(0, (ny-1)*spacing, ny)
    xv, yv = np.meshgrid(x, y)
    return x, y, xv, yv

def gaussian_2D(xx,yy,sigma,x0,y0):
    return np.exp(-((xx-x0)**2 + (yy-y0)**2)/2/sigma**2)


def plane(xx, yy, a, b, c, d):
    return a*(xx - c) + b*(yy - d) 


def test_compare_to_pymedphys():


    x, y, xx, yy = make_grid()
    z1 = gaussian_2D(xx, yy, 10, 0, 0)*100
    z2 = gaussian_2D(xx, yy, 10, 1, 0)*100 + 1
    coords1 = np.stack((yy.ravel(), xx.ravel()), axis=1)
    gamma_pmp = pymedphys.gamma((y,x), z1, (y,x), z2, 2, 2, 0, 100, local_gamma=False, interp_algo='scipy')
    gamma_me = _gamma_2D(coords1, y, x, z1.ravel(), z2, 2.0, 2.0, 0.02, 1)
    gamma_me = gamma_me.reshape(z1.shape)

    assert np.allclose(gamma_pmp, gamma_me, atol=3.1e-3)
    assert abs(np.mean(gamma_pmp - gamma_me)) < 1e-4


def test_planes_x():
    x = np.arange(-25,26,1,dtype=np.float64)
    y = np.arange(-25,26,1,dtype=np.float64)

    rng = np.random.default_rng(0)
    params = rng.random((20,3))*5
    
    xx, yy = np.meshgrid(x,y)

    for param in params:
        slope, intercept, threshold = param
        z = plane(xx, yy, slope, 0, intercept, 0)
        # derived by calculating the min distance between the origin and a line w given slope and intercept and then rescaling by the threshold
        # this assumes that dose threshold and distance threshold are equal
        min_gamma = abs(slope*intercept/np.sqrt(1+slope**2))/threshold
        res = _gamma_2D(np.array([[0,0]]), x, y, np.array([0]), z, threshold, threshold,  interp_res=0.01, norm=1)[0]

        assert np.isclose(res, min_gamma, atol=1e-6, rtol=5e-4)


def test_planes_y():
    x = np.arange(-25,26,1,dtype=np.float64)
    y = np.arange(-25,26,1,dtype=np.float64)

    rng = np.random.default_rng(0)
    params = rng.random((30,3))*5
    
    xx, yy = np.meshgrid(x,y)

    for param in params:
        slope, intercept, threshold = param
        z = plane(xx, yy, 0, slope, 0, intercept)
        # derived by calculating the min distance between the origin and a line w given slope and intercept and then rescaling by the threshold
        # this assumes that dose threshold and distance threshold are equal
        min_gamma = abs(slope*intercept/np.sqrt(1+slope**2))/threshold
        res = _gamma_2D(np.array([[0,0]]), x, y, np.array([0]), z, threshold, threshold,  interp_res=0.01, norm=1)[0]

        assert np.isclose(res, min_gamma, atol=1e-6, rtol=5e-4)


def test_gamma_speed_2D(benchmark):


    x, y, xx, yy = make_grid(nx=501, ny=501)
    z1 = gaussian_2D(xx, yy, 10, 0, 0)*100
    z2 = gaussian_2D(xx, yy, 10, 1, 0)*100 + 1
    coords1 = np.stack((yy.ravel(), xx.ravel()), axis=1)
    
    # warmup 
    gamma_me = _gamma_2D(coords1, y, x, z1.ravel(), z2, 2.0, 2.0, 0.1, 1)
    gamma_me = gamma_me.reshape(z1.shape)

    def run_metric():
        reg = _gamma_2D(coords1, y, x, z1.ravel(), z2, 2.0, 2.0, 0.1, 1)
        return reg

    reg = benchmark(run_metric)



def test_pymedphys_speed_2D(benchmark):


    x, y, xx, yy = make_grid(nx=501, ny=501)
    z1 = gaussian_2D(xx, yy, 10, 0, 0)*100
    z2 = gaussian_2D(xx, yy, 10, 1, 0)*100 + 1
    

    # warmup
    gamma_pmp = pymedphys.gamma((y,x), z1, (y,x), z2, 2, 2, 0, 20, local_gamma=False, interp_algo='scipy')

    def run_metric():
        reg = pymedphys.gamma((y,x), z1, (y,x), z2, 2, 2, 0, 20, local_gamma=False, interp_algo='scipy')
        return reg

    reg = benchmark(run_metric) 


    