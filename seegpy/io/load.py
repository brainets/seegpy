"""Loading functions."""
import os
from pkg_resources import resource_filename

import numpy as np
import pandas as pd
import nibabel

from seegpy.io.read import read_trm
from seegpy.transform import apply_transform


def get_data_path(file=None):
    """Get the path to seegpy/data/.

    Alternatively, this function can also be used to load a file inside the
    data folder.

    Parameters
    ----------
    file : str
        File name

    Returns
    -------
    path : str
        Path to the data folder if file is None otherwise the path to the
        provided file.
    """
    file = file if isinstance(file, str) else ''
    return resource_filename('seegpy', os.path.join('data', file))


def load_marsatlas():
    """Get the MarsAtlas dataframe.

    MarsAtlas parcels are described here [1]_.

    Returns
    -------
    df : DataFrame
        The MarsAtlas as a pandas DataFrame

    References
    ----------
    .. [1] Auzias, G., Coulon, O., & Brovelli, A. (2016). MarsAtlas: a cortical
       parcellation atlas for functional mapping. Human brain mapping, 37(4),
       1573-1592.
    """
    ma_path = get_data_path('MarsAtlasSurf.xls')
    df = pd.read_excel(ma_path).iloc[:-1]
    df["LR_Name"] = df["Hemisphere"].map(str) + ['_'] * len(df) + df["Name"]
    return df


def load_ma_mesh(bv_root, suj, hemi='both', transform=True):
    """Load parcellized MarsAtlas mesh.

    Parameters
    ----------
    bv_root : string
        Path to the Brainvisa folder where subject are stored
    suj : string
        Subject name (e.g 'subject_01')
    hemi : {'both', 'left', 'right'}
        Hemispehres to load
    transform : bool | True
        Tranform the mesh from Brainvisa's space to the scanner based

    Returns
    -------
    vertices : array_like
        Array of vertices
    faces : array_like
        Array of faces
    labmap : array_like
        Labmap (i.e indices associated to each vertex)
    """
    assert hemi in ['both', 'left', 'right']
    if hemi is 'both':
        load_hemi = ['left', 'right']
    else:
        load_hemi = [hemi]
    load_hemi = [k[0].upper() for k in load_hemi]

    # -------------------------------------------------------------------------
    # path to the mesh folder
    path_ref = (f'{bv_root}', f'{suj}', 't1mri', 'default_acquisition',
                'default_analysis', 'segmentation', 'mesh')
    path_ref = os.path.join(*path_ref)
    # path to the mesh
    mesh_files, lab_files = [], []
    for h in load_hemi:
        _m_f = os.path.join(path_ref, f'{suj}_{h}white.gii')
        _l_f = os.path.join(*(path_ref, 'surface_analysis',
                              f'{suj}_{h}white_parcels_marsAtlas.gii'))
        mesh_files += [_m_f]
        lab_files += [_l_f]

    # -------------------------------------------------------------------------
    # load transformation (if needed)
    if transform:
        tr_path = (f"{bv_root}/{suj}/t1mri/default_acquisition/registration/"
                   f"RawT1-{suj}_default_acquisition_TO_Scanner_Based.trm")
        tr = read_trm(tr_path)

    # -------------------------------------------------------------------------
    # load the files
    vertices, faces, labmap = [], [], []
    for k in range(len(mesh_files)):
        # load vertices / faces
        arch = nibabel.load(mesh_files[k])
        _v, _f = np.array(arch.darrays[0].data), np.array(arch.darrays[1].data)
        if len(faces):
            _f += faces[0].max() + 1
        # apply transformation (if needed)
        if transform:
            _v = apply_transform(tr, _v)
        vertices += [_v]
        faces += [_f]
        # load labmap
        labmap += [nibabel.load(lab_files[k]).darrays[0].data]
    # concatenate everything
    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)
    labmap = np.concatenate(labmap, axis=0)

    return vertices, faces, labmap
