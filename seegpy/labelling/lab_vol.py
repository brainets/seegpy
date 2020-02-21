"""Labelling contacts using a volume."""
import logging
import os.path as op

import numpy as np

import nibabel
from mne.source_space import _get_lut

from seegpy.config import CONFIG
from seegpy.transform import apply_transform
from seegpy.io import set_log_level


logger = logging.getLogger('seegpy')


def get_contact_label_vol(vol, tab_idx, tab_labels, xyz, radius=5.,
                          bad_label='none'):
    """Get the label of a single contact in a volume.

    Parameters
    ----------
    vol : array_like
        The full volume that contains the indices (3D array)
    tab_idx : array_like
        Array of unique indices contained in the volume
    tab_labels : array_like
        Array of labels where each label is associated to the indices in
        `tab_idx`
    xyz : array_like
        Array of contacts' coordinates of shape (3,). The coordinates should be
        in the same voxel space as the volume
    radius : float | 5.
        Use the voxels that are contained in a sphere centered arround each
        contact
    bad_label : string | 'none'
        Label to use for contacts that have no close roi

    Returns
    -------
    label : string
        Label associate to the contact's coordinates
    """
    assert len(tab_idx) == len(tab_labels)
    # build the voxel mask (under `radius`)
    mask = np.arange(-int(radius), int(radius) + 1)
    [x_m, y_m, z_m] = np.meshgrid(mask, mask, mask)
    mask_vol = np.sqrt(x_m ** 2 + y_m ** 2 + z_m ** 2) <= radius
    # get indices for selecting voxels under `radius`
    x_m = x_m[mask_vol] + int(np.round(xyz[0]))
    y_m = y_m[mask_vol] + int(np.round(xyz[1]))
    z_m = z_m[mask_vol] + int(np.round(xyz[2]))
    subvol_idx = vol[x_m, y_m, z_m]
    # get indices and number of voxels contained in the selected subvolume
    unique, counts = np.unique(subvol_idx, return_counts=True)
    if (len(unique) == 1) and (unique[0] == 0):
        return bad_label  # skip 'Unknown' only
    elif (len(unique) > 1) and (0 in unique):
        # if there's 'Unknown' + something else, skip 'Unknown'
        unique, counts = unique[1::], counts[1::]
    # infer the label
    u_vol_idx = unique[counts.argmax()]
    is_index = tab_idx == u_vol_idx
    if is_index.any():
        return tab_labels[is_index][0]
    else:
        return bad_label


def labelling_contacts_vol_fs_mgz(fs_root, suj, xyz, radius=5., file='aseg',
                                  bad_label='none', verbose=None):
    """Labelling contacts using Freesurfer mgz volume.

    This function should be used with files like aseg.mgz, aparc+aseg.mgz
    or aparc.a2009s+aseg.mgz

    Parameters
    ----------
    fs_root : string
        Path to the Freesurfer folder where subject are stored
    suj : string
        Subject name (e.g 'subject_01')
    xyz : array_like
        Array of contacts' coordinates of shape (n_contacts, 3)
    radius : float | 5.
        Use the voxels that are contained in a sphere centered arround each
        contact
    file : string | 'aseg'
        The volume to consider. Use either :

            * 'aseg'
            * 'aparc+aseg'
            * 'aparc.a2009s+aseg'
    bad_label : string | 'none'
        Label to use for contacts that have no close roi

    Returns
    -------
    labels : array_like
        Array of labels of shape (n_contacts,)
    """
    set_log_level(verbose)
    # -------------------------------------------------------------------------
    # build path to the volume file
    mri_path = CONFIG['FS_MRI_FOLDER'].format(fs_root=fs_root, suj=suj)
    mgz_path = op.join(mri_path, f"{file}.mgz")
    if not op.isfile(mgz_path):
        raise IOError(f"File {mgz_path} doesn't exist in the /mri/ Freesurfer "
                      "subfolder.")
    n_contacts = xyz.shape[0]
    logger.info(f"-> Localize {n_contacts} using {file}.mgz")

    # -------------------------------------------------------------------------
    # load volume and transformation
    arch = nibabel.load(mgz_path)
    vol = arch.get_data()
    tr = arch.affine
    vs = nibabel.affines.voxel_sizes(tr)
    assert np.array_equal(vs, np.array([1., 1., 1.])), "Need to be updated"
    # load freesurfer LUT table using mne
    lut = _get_lut()
    fs_labels = np.array(lut['name'])
    fs_idx = np.array(lut['id'])

    # -------------------------------------------------------------------------
    # transform coordinates into the voxel space
    xyz_tr = apply_transform(tr, xyz, inverse=True)
    # localize contacts
    labels = []
    for k in range(n_contacts):
        _lab = get_contact_label_vol(vol, fs_idx, fs_labels, xyz_tr[k, :],
                                     radius=radius, bad_label=bad_label)
        labels += [_lab]

    return np.array(labels)
