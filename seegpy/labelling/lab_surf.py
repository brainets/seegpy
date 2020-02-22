"""Labelling contacts using a surface."""
import logging

import numpy as np
from scipy.spatial.distance import cdist

from seegpy.io.load import (load_ma_mesh, load_ma_labmap, load_ma_table,
                            load_fs_mesh, load_fs_labmap, load_fs_table)
from seegpy.io.syslog import set_log_level

logger = logging.getLogger('seegpy')


def labelling_contacts_surf(vert, labmap, xyz, lab_idx, lab_names, radius=5.,
                            bad_label='none', verbose=None):
    """Infer contacts' ROI using a labellized surface.

    Parameters
    ----------
    vert : array_like
        Array of vertices of the surface of shape (n_vertices, 3)
    labmap : array_like
        Array of indices assigned to each vertex of the surface of shape
        (n_vertices,)
    xyz : array_like
        Array of contacts' coordinates of shape (n_contacts, 3)
    lab_idx : array_like
        Labels' indices of shape (n_labels,)
    lab_names : array_like
        Labels' names associated to each indices of shape (n_labels,)
    radius : float | 5.
        Use the vertices that are contained in a sphere centered arround each
        contact
    bad_label : string | 'none'
        Label to use for contacts that have no close roi

    Returns
    -------
    labels : array_like
        Array of labels of shape (n_contacts,)
    """
    set_log_level(verbose)
    assert vert.shape[1] == xyz.shape[1] == 3
    assert vert.shape[0] == len(labmap)
    assert len(lab_idx) == len(lab_names)
    n_contacts = xyz.shape[0]
    if lab_names.ndim == 1:
        lab_names = lab_names.reshape(-1, 1)
    n_labs = lab_names.shape[-1]
    logger.info(f"-> Labelling {n_contacts} contacts (radius={radius})")
    # compute euclidian distance between vertices and contacts
    eucl = cdist(vert, xyz)
    is_close = eucl <= radius
    # infer contacts' roi
    good_bad, labels = [], []
    for n_c in range(n_contacts):
        is_vertices_inside = is_close[:, n_c]
        any_is_close = is_vertices_inside.any()
        if not any_is_close:
            labels += [np.full((1, n_labs), bad_label)]
            continue
        good_bad += [any_is_close]
        # indices closed to the selected contact
        idx_close_to_c = labmap[is_vertices_inside]
        # count the number of indices and select the most probable one
        idx_unique, idx_counts = np.unique(idx_close_to_c, return_counts=True)
        idx_best = idx_unique[idx_counts.argmax()]
        # infer label if possible
        if idx_best in lab_idx:
            labels += [lab_names[lab_idx == idx_best, :]]
        else:
            labels += [np.full((1, n_labs), bad_label)]
    labels = np.concatenate(labels, axis=0)

    return labels


###############################################################################
###############################################################################
#                           BRAINVISA / MARSATLAS
###############################################################################
###############################################################################


def labelling_contacts_surf_ma(bv_root, suj, xyz, radius=5., bad_label='none',
                               verbose=None):
    """Infer contacts' ROI using MarsAtlas surface.

    Parameters
    ----------
    bv_root : string
        Path to the Brainvisa folder where subject are stored
    suj : string
        Subject name (e.g 'subject_01')
    xyz : array_like
        Array of contacts' coordinates of shape (n_contacts, 3)
    radius : float | 5.
        Use the vertices that are contained in a sphere centered arround each
        contact
    bad_label : string | 'none'
        Label to use for contacts that have no close roi

    Returns
    -------
    labels : array_like
        Array of MarsAtlas labels of shape (n_contacts, 4)
    """
    set_log_level(verbose)
    # load MarsAtlas labels and indices
    ma_idx, ma_names = load_ma_table(verbose=verbose)
    # load Brainvisa's mesh
    vert, _ = load_ma_mesh(bv_root, suj, hemi='both', transform=True,
                           verbose=verbose)
    # load labmap
    labmap = load_ma_labmap(bv_root, suj, hemi='both', verbose=verbose)
    # infer roi using marsatlas
    labels = labelling_contacts_surf(vert, labmap, xyz, ma_idx, ma_names,
                                     radius=radius, bad_label=bad_label,
                                     verbose=verbose)

    return labels


###############################################################################
###############################################################################
#                                FREESURFER
###############################################################################
###############################################################################


def labelling_contacts_surf_fs(fs_root, suj, xyz, radius=5., bad_label='none',
                               verbose=None):
    """Infer contacts' ROI using Freesurfer cortical surface.

    Parameters
    ----------
    fs_root : string
        Path to the Brainvisa folder where subject are stored
    suj : string
        Subject name (e.g 'subject_01')
    xyz : array_like
        Array of contacts' coordinates of shape (n_contacts, 3)
    radius : float | 5.
        Use the vertices that are contained in a sphere centered arround each
        contact
    bad_label : string | 'none'
        Label to use for contacts that have no close roi

    Returns
    -------
    labels : array_like
        Array of Freesurfer cortical labels of shape (n_contacts,)
    """
    set_log_level(verbose)
    # detect contacts that are in the left / right hemispehres
    is_left = xyz[:, 0] <= 0
    xyz_lr = [xyz[is_left, :], xyz[~is_left, :]]
    # loop over left / right hemispheres
    labels = []
    for n_h, h in enumerate(['left', 'right']):
        if not xyz_lr[n_h].shape[0]:
            continue
        # load freesurfer labels and indices
        fs_idx, fs_names = load_fs_table(fs_root, suj, hemi=h, verbose=verbose)
        # load freesurfer's mesh
        vert, _ = load_fs_mesh(fs_root, suj, hemi=h, transform=True,
                               verbose=verbose)
        # load labmap
        labmap = load_fs_labmap(fs_root, suj, hemi=h, verbose=verbose)
        # infer roi using freesurfer
        _labels = labelling_contacts_surf(vert, labmap, xyz, fs_idx, fs_names,
                                          radius=radius, bad_label=bad_label,
                                          verbose=verbose)
        labels += [_labels]
    labels = np.concatenate(labels, axis=0)

    return labels


if __name__ == '__main__':
    from seegpy.io import read_3dslicer_fiducial

    fs_root = ("/home/etienne/Server/frioul/database/db_freesurfer/"
               "seeg_causal")
    bv_root = '/home/etienne/Server/frioul/database/db_brainvisa/seeg_causal'
    suj = 'subject_01'

    # -------------------------------------------------------------------------
    # read contacts
    seeg_path = ("/run/media/etienne/DATA/RAW/CausaL/LYONNEURO_2014_DESj/"
                 "TEST_DATA/recon.fcsv")
    df = read_3dslicer_fiducial(seeg_path)
    xyz = np.array(df[['x', 'y', 'z']])
    contact = df['label']

    # -------------------------------------------------------------------------
    # Brainvisa / MarsAtlas
    labs = labelling_contacts_surf_ma(bv_root, suj, xyz, radius=5.,
                                      bad_label='none')
    print(np.c_[contact, labs])

    # -------------------------------------------------------------------------
    # Freesurfer
    # labs = labelling_contacts_surf_fs(fs_root, suj, xyz, radius=5.,
    #                                   bad_label='none')
    # print(labs.shape)
    # print(np.c_[contact, labs])
