"""Utility functions."""
# import numpy as np


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
