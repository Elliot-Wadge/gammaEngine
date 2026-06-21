from .core import _gamma_1D, _gamma_1D_pr, _gamma_2D, _gamma_2D_pr, _gamma_3D, _gamma_3D_pr
import numpy as np
import numpy.typing as npt
import warnings



def prescreen(eval_cords: npt.NDArray[np.float64],
          ref_axis: tuple,
          dose_eval: npt.NDArray[np.float64],
          dose_ref: npt.NDArray[np.float64],
          dose_threshold: float,
          distance_threshold: float,
          interp_res: float,
          norm: float,
          pass_rate_only:bool = False):

    

    if distance_threshold <= 0:
        raise ValueError('Distance threshold must be greater than zero')

    if dose_threshold <= 0:
        raise ValueError('Dose threshold must be greater than zero')

    if interp_res <= 0:
        raise ValueError('The interp res must be greater than zero')

    if norm <= 0:
        raise ValueError('The normalization must be greater than zero')


    coord_dim = eval_cords.shape

    if len(coord_dim) == 1:
        coord_dim = (*coord_dim,1)

    ref_axis_dim = len(ref_axis)
    ref_axis_lens = np.array([len(array) for array in ref_axis])
    dose_eval_dim = dose_eval.shape
    dose_ref_dim = dose_ref.shape

    if coord_dim[1] > 3:
        raise ValueError(f"the dimension of eval_cords ({coord_dim[1]}) must be less than three")

    if dose_eval.ndim != 1:
        raise ValueError("dose_eval must be pre-flattened 1D arrays.")

    if coord_dim[1] != ref_axis_dim:
        raise ValueError(f'The dimensions for the eval_cords ({coord_dim[1]}) and ref_axis ({ref_axis_dim}) do not match')

    if coord_dim[0] != dose_eval_dim[0]:
        raise ValueError(f'The length of the eval_cords ({coord_dim[1]}) and dose_eval ({dose_eval_dim[0]}) do not match')

    if not np.all(ref_axis_lens == dose_ref_dim):
        raise ValueError(f'The shape of the ref axis ({ref_axis_lens}) do not match the shape of the dose_ref matrix ({dose_ref_dim})')

    for i,arr in enumerate(ref_axis):
        diff = np.diff(arr)
        if not np.all(diff > 0):
            raise ValueError(f'The ref_axis[{i}] is not strictly increasing')
        
        if not np.allclose(diff, np.mean(diff), rtol=1e-5, atol=1e-8):
            raise ValueError(f'The ref_axis[{i}] is not regularly spaced.')

    min_ref_spacing = min(np.min(np.diff(arr)) for arr in ref_axis)
    if interp_res > min_ref_spacing:
    # A warning or exception depending on how strict you want to be
        warnings.warn(f"Interpolation resolution ({interp_res}) is coarser than minimum reference grid spacing ({min_ref_spacing}).")

    if interp_res > 0.1*distance_threshold:
        warnings.warn(f"the interp res ({interp_res}) is coarser than the 10% of the distance threshold ({distance_threshold}), this may result in unreliable gamma estimates")

    if interp_res > distance_threshold:
        raise ValueError(f"the interp res ({interp_res}) is coarser than the distance threshold ({distance_threshold}), this gaurantees all gamma scores will fail")



def match_gamma_function(ref_axis, pass_rate_only):
    function_key = {'2-0':_gamma_2D,
                '2-1':_gamma_2D_pr,
                '1-0':_gamma_1D,
                '1-1':_gamma_1D_pr,
                '3-0':_gamma_3D,
                '3-1':_gamma_3D_pr}

    key = str(len(ref_axis)) + '-' + str(int(pass_rate_only))
    return function_key[key]


def gamma(eval_cords: npt.NDArray[np.float64],
          ref_axis: tuple,
          dose_eval: npt.NDArray[np.float64],
          dose_ref: npt.NDArray[np.float64],
          dose_threshold: float,
          distance_threshold: float,
          interp_res: float,
          norm: float,
          pass_rate_only:bool = False):

    prescreen(eval_cords, ref_axis, dose_eval, dose_ref, dose_threshold, distance_threshold, interp_res, norm, pass_rate_only)
    gamma_f = match_gamma_function(ref_axis, pass_rate_only)
    res = gamma_f(eval_cords,
            *ref_axis,
            dose_eval,
            dose_ref,
            dose_threshold,
            distance_threshold,
            interp_res,
            norm)
    
    return res

    
# def _gamma_2D(coord_eval: npt.NDArray[np.float64], 
#               ref_y_axes: npt.NDArray[np.float64], 
#               ref_x_axes: npt.NDArray[np.float64], 
#               dose_eval: npt.NDArray[np.float64], 
#               dose_ref: npt.NDArray[np.float64], 
#               dose_threshold: float, 
#               distance_threshold: float, 
#               interp_res: float, 
#               norm: float):

# def _gamma_3D(coord_eval: npt.NDArray[np.float64],
#               ref_z_axes: npt.NDArray[np.float64], 
#               ref_y_axes: npt.NDArray[np.float64], 
#               ref_x_axes: npt.NDArray[np.float64], 
#               dose_eval: npt.NDArray[np.float64], 
#               dose_ref: npt.NDArray[np.float64], 
#               dose_threshold: float, 
#               distance_threshold: float, 
#               interp_res: float, 
#               norm: float):

# def _gamma_1D(x_eval:npt.NDArray[np.float64], 
#               x_ref:npt.NDArray[np.float64], 
#               dose_eval:npt.NDArray[np.float64], 
#               dose_ref:npt.NDArray[np.float64], 
#               dose_threshold:float, 
#               distance_threshold:float, 
#               interp_res:float, 
#               norm:float):