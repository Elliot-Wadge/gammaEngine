# gammaEngine

High-performance gamma analysis for radiotherapy quality assurance.

gammaEngine is a fast NumPy/Numba-based implementation of gamma index evaluation designed for clinical and research QA workflows. It focuses on computational efficiency, numerical correctness, and flexible handling of both regular and irregular dose grids for comparison between treatment planning system dose planes and measurement data.

---

## Key Features

- High-performance gamma computation using Numba-accelerated kernels
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
from gamma_engine import gamma_1d

gamma = gamma_1d(
    x_evaluated,
    x_reference,
    dose_eval,
    dose_reference,
    2.0, # dose threshold
    0.03, # distance threshold
    0.001, # interpolation resolution
    100/max(reference) # definition of 100 percent
)
