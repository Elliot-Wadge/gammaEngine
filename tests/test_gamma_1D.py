import numpy as np
from gammaEngine.core import _gamma_1D
import pymedphys
from matplotlib import pyplot as plt

def line(x, a, b):
    return a*(x - b)


def gaussian(x, b, c):
    return np.exp(-(x-b)**2/(2*c**2))


def test_lines_x():
    x = np.arange(-25,26,1,dtype=np.float64)
    

    rng = np.random.default_rng(0)
    params = rng.random((100,3))*5
    
    
    for param in params:
        slope, intercept, threshold = param
        dose_ref = line(x, slope, intercept)
        # derived by calculating the min distance between the origin and a line w given slope and intercept and then rescaling by the threshold
        # this assumes that dose threshold and distance threshold are equal
        min_gamma = abs(slope*intercept/np.sqrt(1+slope**2))/threshold
        res = _gamma_1D(np.array([0], dtype=float), 
                        x, 
                        np.array([0], dtype=float), 
                        dose_ref, 
                        threshold, 
                        threshold, 
                        0.0001, 
                        1)[0]

        assert np.isclose(res, min_gamma, atol=1e-6, rtol=5e-4)


def test_compare_to_pymedphys():
    x_ref = np.arange(-10,10,0.1, dtype=np.float64)
    x_eval = x_ref + 0.1
    y_ref = gaussian(x_ref, 0.5, 2)*100
    y_eval = gaussian(x_ref, 0, 1.98)*100

    gamma_me = _gamma_1D(x_eval,
                         x_ref,
                         y_eval,
                         y_ref,
                         2,
                         2,
                         0.001,
                         1)


    gamma_pmp = pymedphys.gamma(x_eval, y_eval, x_ref, y_ref, 2, 2, interp_fraction=2000, local_gamma=False, lower_percent_dose_cutoff=0)

    # plt.figure()
    # plt.plot(x_eval, gamma_me, label='me')
    # plt.plot(x_eval, gamma_pmp, label='pymedphys')
    # plt.legend()
    # plt.savefig('1d_test_fig.png')
    assert abs(np.mean(gamma_pmp - gamma_me)) < 1e-4
    assert np.allclose(gamma_pmp, gamma_me, atol=3.1e-3)
    

def test_gamma_speed_1D(benchmark):


    x_ref = np.arange(-10,10,0.1, dtype=np.float64)
    x_eval = x_ref + 0.1
    y_ref = gaussian(x_ref, 0.5, 2)*100
    y_eval = gaussian(x_ref, 0, 1.98)*100
    
    # warmup 
    gamma_me = _gamma_1D(x_eval,
                         x_ref,
                         y_eval,
                         y_ref,
                         2,
                         2,
                         0.001,
                         1)

    def run_metric():
        reg = _gamma_1D(x_eval,
                         x_ref,
                         y_eval,
                         y_ref,
                         2,
                         2,
                         0.001,
                         1)
        return reg

    reg = benchmark(run_metric)


def test_pymedphys_speed_1D(benchmark):


    x_ref = np.arange(-10,10,0.1, dtype=np.float64)
    x_eval = x_ref + 0.1
    y_ref = gaussian(x_ref, 0.5, 2)*100
    y_eval = gaussian(x_ref, 0, 1.98)*100
    
    # warmup 
    gamma_pmp = pymedphys.gamma(x_eval, y_eval, x_ref, y_ref, 2, 2, interp_fraction=2000, local_gamma=False, lower_percent_dose_cutoff=0)

    def run_metric():
        reg = pymedphys.gamma(x_eval, y_eval, x_ref, y_ref, 2, 2, interp_fraction=2000, local_gamma=False, lower_percent_dose_cutoff=0)
        return reg

    reg = benchmark(run_metric)


if __name__ == '__main__':
    test_lines_x()