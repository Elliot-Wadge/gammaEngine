from gammaEngine.interpolation import f_linear_interp, f_bilinear_interp
import numpy as np

def test_linear_interp():
    res = 0.1
    x = np.arange(-10,10,res)
    dx = 1/res
    y = 2*x
    rng = np.random.default_rng(0)
    samples = rng.random(100)*20 - 10

    for sample in samples:
        interp_val = f_linear_interp(sample, x , y, dx)
        result = 2*sample
        assert np.isclose(result, interp_val, atol=1e-12, rtol=1e-12)


def test_bilinear_interp():
    dx_val = 0.1
    dy_val = 0.1

    x = np.arange(-10, 10, dx_val)
    y = np.arange(-10, 10, dy_val)

    dx = 1 / dx_val
    dy = 1 / dy_val

    rng = np.random.default_rng(0)
    xs = rng.random(200) * 20 - 10
    ys = rng.random(200) * 20 - 10

    # exact bilinear function (polynomial -> exact for bilinear interp)
    def f(x, y):
        return 3*x + 5*y + 2*x*y + 7

    # build grid
    values = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            values[i, j] = f(x[j], y[i])

    for xv, yv in zip(xs, ys):
        interp_val = f_bilinear_interp(
            yv, xv, y, x, values, dy, dx
        )
        true_val = f(xv, yv)

        assert np.isclose(true_val, interp_val, atol=1e-10, rtol=1e-10)

if __name__ == '__main__':
    test_linear_interp()