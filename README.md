# gammaEngine

High-performance gamma analysis for radiotherapy quality assurance.

gammaEngine is a fast NumPy/Numba-based implementation of gamma index evaluation designed for clinical and research QA workflows. It focuses on computational efficiency, numerical correctness, and flexible handling of both regular and irregular dose grids for comparison between treatment planning system dose planes and measurement data.

---

## Key Features

- High-performance gamma computation using Numba-acceleration
- Parallel execution support for large-scale dose evaluations
- Efficient interpolation and indexing strategies for structured grids
- Support for both regular and irregular coordinate systems
- Designed for integration into QA and research pipelines

---

## Design Goals

gammaEngine is built around three primary goals:

### Performance
Core routines are compiled with Numba and optimized for:
- Tight loop execution
- Parallel execution where appropriate

### Numerical clarity
- Explicit handling of grid geometry
- Deterministic interpolation behaviour
- Clear separation of coordinate mapping and gamma evaluation

### Flexibility
- Works with regular dose grids directly
- Supports irregular measurement grids via explicit coordinate mapping
- Consistent evaluation framework across different data sources

---

## Performance

gammaEngine is designed for high-throughput gamma evaluation.

Typical behaviour:
- 1D comparisons: large speedups compared to general-purpose implementations
- 2D comparisons: significant acceleration, especially when parallelized
- 3D comparisons: increasingly memory-bound but still substantially faster in many practical cases

Performance improvements come primarily from:
- Eliminating general-purpose search overhead
- Using direct index-based interpolation on structured grids
- JIT compilation of inner loops
- Parallel execution across evaluation points

---

## Example Usage

```python
from gamma_engine import gamma


def make_grid(nx=101, ny=101, spacing=1.0):
    x = np.linspace(0, (nx-1)*spacing, nx)
    y = np.linspace(0, (ny-1)*spacing, ny)
    xv, yv = np.meshgrid(x, y)
    return x, y, xv, yv


def gaussian_2D(xx,yy,sigma,x0,y0):
    return np.exp(-((xx-x0)**2 + (yy-y0)**2)/2/sigma**2)


def gaussian(x, b, c):
    return np.exp(-(x-b)**2/(2*c**2))


x_ref = np.arange(-10,10,0.1, dtype=np.float64)
x_eval = x_ref + 0.1
y_ref = gaussian(x_ref, 0.7, 2)*100
y_eval = gaussian(x_ref, 0, 1.98)*100
# note the comma in (x_ref,) is necessary to ensure compatibility with the higher dimension formatting
gamma_me = gamma(x_eval,(x_ref,),y_eval,y_ref,2,2,0.001,1)

x, y, xx, yy = make_grid()
z1 = gaussian_2D(xx, yy, 9.5, 0, 0)*100
z2 = gaussian_2D(xx, yy, 10, 1, 0)*100 + 1
coords1 = np.stack((yy.ravel(), xx.ravel()), axis=1)
gamma_me = gamma(coords1, (y,x), z1.ravel(), z2, 2.0, 2.0, 0.02, 1)



