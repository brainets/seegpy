"""Labelling pipeline."""
import numpy as np
import pandas as pd

from seegpy.io import set_log_level
from seegpy.labelling import (labelling_contacts_surf_ma,
                              labelling_contacts_surf_fs,
                              labelling_contacts_vol_fs_mgz)


def pipeline_labelling_ss(fs_root, bv_root, suj, c_xyz, c_names, bipolar=True,
                          radius=5., bad_label='none', verbose=None):
    """Single subject contact labelling pipeline.

    Parameters
    ----------
    fs_root : string
        Path to the Brainvisa folder where subject are stored
    bv_root : string
        Path to the Brainvisa folder where subject are stored
    suj : string
        Subject name (e.g 'subject_01')
    c_xyz : array_like
        Array of contacts' coordinates in the scanner-based referential
        (T1.mgz) of shape (n_contacts, 3)
    c_names : array_like
        Array of contact's names
    bipolar : bool | True
        Consider the provided contacts as monopolar and enable to compute
        labelling on bipolar derivations
    radius : float | 5.
        Distance under which to consider vertices and voxels in order to infer
        the label
    bad_label : string | 'none'
        Label to use for contacts that have no close roi
    """
    set_log_level(verbose)
    # -------------------------------------------------------------------------
    # monopolar and bipolar
    contacts = dict()

    
    # -------------------------------------------------------------------------
    # scanner and mni coordinates
    # -------------------------------------------------------------------------
    # MarsAtlas cortical : surface labelling
    # -------------------------------------------------------------------------
    # Freesurfer cortical : surface labelling
    # -------------------------------------------------------------------------
    # Freesurfer cortical and subcortical : volumique labelling
    # -------------------------------------------------------------------------
    # post-processing
