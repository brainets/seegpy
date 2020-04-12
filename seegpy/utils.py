"""Utility functions."""
import numpy as np


def hemi_to_load(hemi, output_for='fs'):
    """Define the hemispheres to load.

    Parameters
    ----------
    hemi : {'both', 'left', 'right'}
        Hemispheres to load
    ouput_for : {'fs', 'bv'}
        Output for either Freesurfer ('fs') or Brainvisa ('bv')
    """
    assert hemi in ['both', 'left', 'right']
    assert output_for in ['fs', 'bv']
    # define the list of hemispheres to load
    if hemi is 'both':
        load_hemi = ['left', 'right']
    else:
        load_hemi = [hemi]
    # change according to brainvisa / freesurfer
    if output_for is 'bv':
        load_hemi = [k[0].upper() for k in load_hemi]
    elif output_for is 'fs':
        hemi_conv = dict(left='lh', right='rh')
        load_hemi = [hemi_conv[k] for k in load_hemi]
    return load_hemi


def compute_roi_size(vol, roi_values, ratio=True):
    """Compute the ROI size in a volume.

    Parameters
    ----------
    vol : array_like
        3D volume array
    roi_values : array_like
        Unique ROI values in the volume the get the size
    ratio : bool | True
        If True, divide the number of points in the volume by the total size
        of the volume and multiply by 100. As a result, each size reflect the
        percentage of the total volume (for non-zero values)

    Returns
    -------
    sizes : array_like
        Array of size of shape (n_roi, 2) where the first column refers to the
        ROI number and the second column the size
    """
    n_pts = (vol != 0).sum()
    n_roi = len(roi_values)
    sizes = np.zeros((n_roi, 2), dtype=float)
    sizes[:, 0] = roi_values
    for n_r, r in enumerate(roi_values):
        sizes[n_r, 1] = (vol == r).sum()
    if ratio:
        sizes[:, 1] = 100 * sizes[:, 1] / n_pts
    return sizes
