
from numba import njit
import numpy as np
import numpy.typing as npt


@njit()
def f_trilinear_interp(pz: float, py: float, px: float,
                     z_ax: np.ndarray, y_ax: np.ndarray, x_ax: np.ndarray,
                     values: np.ndarray, dz: float, dy: float, dx: float):

    ix = int((px - x_ax[0]) * dx)
    iy = int((py - y_ax[0]) * dy)
    iz = int((pz - z_ax[0]) * dz)

    if iy == len(y_ax)-1:
        iy -= 1

    if ix == len(x_ax)-1:
        ix -= 1

    if iz == len(z_ax)-1:
        iz -= 1

    xd = (px - x_ax[ix]) * dx
    yd = (py - y_ax[iy]) * dy
    zd = (pz - z_ax[iz]) * dz

    ixd = 1 - xd
    iyd = 1 - yd

    # corners (z,y,x)
    c00 = values[iz, iy, ix] * ixd + values[iz, iy, ix+1] * xd
    c01 = values[iz+1, iy, ix] * ixd + values[iz+1, iy, ix+1] * xd
    c10 = values[iz, iy+1, ix] * ixd + values[iz, iy+1, ix+1] * xd
    c11 = values[iz+1, iy+1, ix] * ixd + values[iz+1, iy+1, ix+1] * xd

    c0 = c00 * iyd + c10 * yd
    c1 = c01 * iyd + c11 * yd

    return c0 * (1 - zd) + c1 * zd


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


@njit
def fast_sin(angle, sin_table, dangle, inv_dangle):
    i = int(angle * inv_dangle)
    x1 = angle - dangle * i
    x2 = dangle * (i+1) - angle
    value = sin_table[i] * (x2*inv_dangle) + sin_table[i+1] * (x1*inv_dangle)

    return value


@njit
def fast_cos(angle, cos_table, dangle, inv_dangle):
    i = int(angle * inv_dangle)
    x1 = angle - dangle * i
    x2 = dangle * (i+1) - angle
    value = cos_table[i] * (x2*inv_dangle) + cos_table[i+1] * (x1*inv_dangle)

    return value


@njit(fastmath=True)
def uniform_spaced_points_on_sphere(N,R):
    a = 4*np.pi*R**2/(N)
    d = np.sqrt(a)
    M1 = int(np.round(np.pi/d)+1)
    M1_inv = 1/M1
    d1 = np.pi*M1_inv
    d2 = a/d1
    d2_inv = 1/d2
    res = np.empty((N + M1,3), dtype=np.float64)
    count = 0
    twopi = 2*np.pi
    for i in range(int(M1/2)):
        theta = np.pi*(i+0.5)*M1_inv
        M2 = int(np.round(2*np.pi*np.sin(theta)*d2_inv))
        M2_inv = 1/M2
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        z = R*costheta
        rsintheta = R*sintheta
        c = twopi*M2_inv
        for j in range(M2):
            phi = j*c
            cosphi = np.cos(phi)
            sinphi = np.sin(phi)
            x = rsintheta*cosphi
            y = rsintheta*sinphi
            res[count][0] = x
            res[count][1] = y
            res[count][2] = z
            count += 1
            res[count][0] = x
            res[count][1] = y
            res[count][2] = -z
            count += 1
            
    
    return res[:count]


