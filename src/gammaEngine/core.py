
from numba import njit, prange
import numpy as np
import numpy.typing as npt
from .interpolation import f_bilinear_interp, f_linear_interp


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


@njit(parallel=True)
def _gamma_2D(coord_eval: npt.NDArray[np.float64], 
              ref_y_axes: npt.NDArray[np.float64], 
              ref_x_axes: npt.NDArray[np.float64], 
              dose_eval: npt.NDArray[np.float64], 
              dose_ref: npt.NDArray[np.float64], 
              dose_threshold: float, 
              distance_threshold: float, 
              interp_res: float, 
              norm: float):
    
    '''
    :param coord_eval: the coordinates in form [[x1,y1],[x2,y2],...] which are the position coordinates of point 1, the gamma index is evaluated for every set of coordinates in this array
    :param ref_y_axes: a uniformly spaced axes that defines the reference grid in the y direction, this data will be used for interpolation
    :param ref_x_axes: a uniformly spaced axes that defines the reference grid in the x direction, this data will be used for interpolation
    :param dose_eval: the flattened dose array [d1,d2,d3,...] corresponding to the coordinates in coord_eval
    :param dose_ref: a 2D dose array corresponding to the ref_x_axes and ref_y_axes, dose at position (x,y) = dose_ref[y_index, x_index]
    :param dose_threshold: the dose threshold in arbitrary units defined by norm
    :param distance_threshold: the distance threshold in units defined by ref_y_axes and ref_x_axes
    :param interp_res: the resolution at which the ref dose distribution is interpolated and searched at
    :param norm: the normalization constant that defines the relative dose difference, to have the dose threshold be defined as the percentage relative to dmax then norm = 100/max(dose_ref)
    '''

    res = np.empty(dose_eval.shape, dtype=np.float64)

    if distance_threshold <= 0 or dose_threshold <= 0:
        raise ValueError("Thresholds must be positive")
    if dose_threshold == 0:
        raise ValueError("Dose threshold cannot be zero")
    if distance_threshold == 0:
        raise ValueError("Distance threshold cannot be zero")
    
    # precompute useful values
    d = interp_res
    dx = 1/(ref_x_axes[1] - ref_x_axes[0])
    dy = 1/(ref_y_axes[1] - ref_y_axes[0])
    dist_scale = 1/distance_threshold
    dose_scale = 1*norm/dose_threshold
    dist_scale_sq = dist_scale**2
    dose_scale_sq = dose_scale**2
    dose_thresh_sq = dose_threshold**2
    distance_threshold_sq = distance_threshold**2

    x_min, x_max = ref_x_axes[0], ref_x_axes[-1]
    y_min, y_max = ref_y_axes[0], ref_y_axes[-1]
    two_pi = 6.283185307179586

    N = 2045
    angles = np.linspace(0,2*two_pi+4*np.pi/N,N)
    sin_table = np.sin(angles)
    cos_table = np.cos(angles)
    dangle = (2*two_pi + 2*two_pi/N)/N
    inv_dangle = 1/dangle

    for i in prange(len(coord_eval)):
        pos = coord_eval[i]
        dose = dose_eval[i]

        # start with an initial gamma at the same position
        if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
                value = f_bilinear_interp(pos[0], pos[1], ref_y_axes, ref_x_axes, dose_ref, dy, dx)
                min_g_sq =(value - dose)**2*dose_scale_sq 
        else:
            min_g_sq = np.inf
        
        max_distance = max(x_min-pos[1], x_max-pos[1])**2 + max(y_min-pos[0], y_max-pos[0])**2
        d = interp_res
        k = 1
        while d**2 < min_g_sq*distance_threshold_sq and d < max_distance:
            
            m = int(two_pi * k)
            dist_term = d**2*dist_scale_sq
            increment = two_pi/m
            bound = dose_thresh_sq * (min_g_sq - dist_term)
            
            for j in range(m):
                angle = j*increment
                xj = pos[1] + fast_sin(angle, sin_table, dangle, inv_dangle)*d#+ np.cos(angle)*d 
                yj = pos[0] + fast_cos(angle, cos_table, dangle, inv_dangle)*d#+ np.sin(angle)*d 
                if x_min <= xj <= x_max and y_min <= yj <= y_max:
                    value = f_bilinear_interp(yj,xj, ref_y_axes, ref_x_axes, dose_ref, dy, dx)
                    diff_sq = (value - dose)**2
                else:
                    diff_sq = np.inf
                if diff_sq < bound:
                    min_g_sq = diff_sq*dose_scale_sq + dist_term
                    bound = dose_thresh_sq * (min_g_sq - dist_term)
            d += interp_res
            k += 1
        res[i] = np.sqrt(min_g_sq)

    return res


@njit(parallel=True)
def _gamma_1D(x_eval:npt.NDArray[np.float64], 
              x_ref:npt.NDArray[np.float64], 
              dose_eval:npt.NDArray[np.float64], 
              dose_ref:npt.NDArray[np.float64], 
              dose_threshold:float, 
              distance_threshold:float, 
              interp_res:float, 
              norm:float):
    
    # precompute useful values
    
    dx = 1/(x_ref[1] - x_ref[0])
    dist_scale = 1/distance_threshold
    dose_scale = 1*norm/dose_threshold
    dist_scale_sq = dist_scale**2
    dose_scale_sq = dose_scale**2
    dose_thresh_sq = dose_threshold**2
    distance_threshold_sq = distance_threshold**2
    x_min, x_max = x_ref[0], x_ref[-1]
    signs = [-1,1]
    res = np.empty(x_eval.shape)


    for i in prange(len(x_eval)):
        d = interp_res
        x = x_eval[i]
        dose_e = dose_eval[i]
        if x_min <= x <= x_max:
            value = f_linear_interp(x, x_ref, dose_ref, dx)
            min_g_sq =(value - dose_e)**2*dose_scale_sq
        else:
            min_g_sq = np.inf

        max_distance = max(abs(x-x_min), abs(x-x_max))
        
        while d**2 < min_g_sq*distance_threshold_sq and d < max_distance:
            
            dist_term = d**2*dist_scale_sq
            bound = dose_thresh_sq * (min_g_sq - dist_term)
            # take a step in both directions away from the starting point
            for sign in signs:
                check_x1 = x + sign*d
                if x_min <= check_x1 <= x_max:
                    dose_r = f_linear_interp(check_x1, x_ref, dose_ref, dx)
                    diff_sq = (dose_e - dose_r)**2
                else:
                    diff_sq = np.inf

                if diff_sq < bound:
                    min_g_sq = diff_sq*dose_scale_sq + dist_term
                    bound = dose_thresh_sq * (min_g_sq - dist_term)

            d += interp_res

        res[i] = np.sqrt(min_g_sq)

    return res
