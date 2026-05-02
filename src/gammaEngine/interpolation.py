
from numba import njit
import numpy as np
import numpy.typing as npt

@njit
def f_bilinear_interp(y:float, 
                      x:float, 
                      y_axes:npt.NDArray[np.float64], 
                      x_axes:npt.NDArray[np.float64], 
                      values:npt.NDArray[np.float64], 
                      dy:float, 
                      dx:float):
    '''
    function for extra fast bilinear interpolation, by passing the dy and dx we avoid issues with njit and avoid repeated 
    division. This also avoid the use of np.searchsorted for finding the indices.

    :param y: y value at which to interpolate values
    :param x: x value at which to interpolate values
    :param y_axes: the y values corresponding to the rows in values, y_axes[i] => values[i,:]
    :param x_axes: the x values corresponding to columns in values, x_axes[j] => values[:,j] 
    :param dx: the inverse step size of the x_axes, used to calculate the column index, j = x*dx
    :param dy: the inverse step size of the y_axes, used to calculate the row index, i = y*dy
    '''
    # 1. Find the "cell" the point is in
    
    # someimtes if y = yaxes[-1] it causes i = len(y_axes) which causes an error when we try to do y_axes[i+1], tried fixing it outside of this function but couldn't get it to work so settled on this
    i = int(np.floor((y-y_axes[0])*dy))
    j = int(np.floor((x-x_axes[0])*dx))


    if i == len(y_axes)-1:
        i -= 1

    if j == len(x_axes)-1:
        j -= 1

    # 2. Get coordinates of the 4 corners
    y0, y1 = y_axes[i], y_axes[i+1]
    x0, x1 = x_axes[j], x_axes[j+1]
    
    # 3. Calculate relative weights (0 to 1)
    wa = (y - y0) * dy
    wb = (x - x0) * dx
    
    # 4. Mix the values
    v00 = values[i, j]
    v10 = values[i+1, j]
    v01 = values[i, j+1]
    v11 = values[i+1, j+1]
    return v00*(1-wa)*(1-wb) + v10*wa*(1-wb) + v01*(1-wa)*wb + v11*wa*wb


@njit
def f_linear_interp(x:float, 
                    x_axes:npt.NDArray[np.float64], 
                    values:npt.NDArray[np.float64], 
                    dx:float):
    
    '''1D fast linear interpolation of a regularly spaced axes
    
    :param x: the x value at which to return the interpolated is calculated
    :param x_axes: regularly spaced axes
    :param values: the corresponding values to the x_axes
    :param dx: the inverse spacing between entries in x_axes, dx=1/(x_axes[1] - x_axes[0])'''
    
    i = int(np.floor((x-x_axes[0])*dx))
    if i == len(x_axes)-1:
        i -= 1
    x0 = x_axes[i]
    x1 = x_axes[i+1]
    w1 = (x-x0)*dx
    w0 = 1-w1

    return w0*values[i] + w1*values[i+1]