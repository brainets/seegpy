"""Loading functions."""
import logging

import os.path as op
from pkg_resources import resource_filename

import numpy as np
import pandas as pd
import nibabel

from seegpy.io.read import read_trm
from seegpy.io.syslog import set_log_level
from seegpy.transform import apply_transform, chain_transform
from seegpy.utils import hemi_to_load
from seegpy.config import CONFIG

logger = logging.getLogger('seegpy')


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
    return resource_filename('seegpy', op.join('data', file))


###############################################################################
###############################################################################
#                           BRAINVISA / MARSATLAS
###############################################################################
###############################################################################


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


def load_ma_mesh(bv_root, suj, hemi='both', transform=True, verbose=None):
    """Load MarsAtlas mesh.

    Parameters
    ----------
    bv_root : string
        Path to the Brainvisa folder where subject are stored
    suj : string
        Subject name (e.g 'subject_01')
    hemi : {'both', 'left', 'right'}
        Hemispheres to load
    transform : bool | True
        Tranform the mesh from Brainvisa's space to the scanner based (same as
        Freesurfer)

    Returns
    -------
    vertices : array_like
        Array of vertices
    faces : array_like
        Array of faces
    """
    set_log_level(verbose)
    logger.info(f'-> Loading MarsAtlas mesh of {suj}')
    load_hemi = hemi_to_load(hemi, output_for='bv')

    # -------------------------------------------------------------------------
    # define path and files to load
    mesh_path = CONFIG['BV_MESH_FOLDER'].format(bv_root=bv_root, suj=suj)
    mesh_files = [f'{suj}_{h}white.gii' for h in load_hemi]

    # -------------------------------------------------------------------------
    # load transformation (if needed)
    if transform:
        tr_path = CONFIG['BV_TRM_FOLDER'].format(bv_root=bv_root, suj=suj)
        tr_file = f"RawT1-{suj}_default_acquisition_TO_Scanner_Based.trm"
        tr = read_trm(op.join(tr_path, tr_file))
        logger.info(f'    MarsAtlas transformation loaded (BV T1 -> SCANNER)')

    # -------------------------------------------------------------------------
    # load the files
    vertices, faces = [], []
    for k in range(len(mesh_files)):
        # load vertices / faces
        arch = nibabel.load(op.join(mesh_path, mesh_files[k]))
        _v, _f = np.array(arch.darrays[0].data), np.array(arch.darrays[1].data)
        if len(faces):
            _f += faces[0].max() + 1
        # apply transformation (if needed)
        if transform:
            _v = apply_transform(tr, _v)
        vertices += [_v]
        faces += [_f]
        logger.info(f'    MarsAtlas mesh {mesh_files[k]} loaded')
    # concatenate everything
    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)

    return vertices, faces


def load_ma_labmap(bv_root, suj, hemi='both', verbose=None):
    """Load the MarsAtlas labmap.

    Parameters
    ----------
    bv_root : string
        Path to the Brainvisa folder where subject are stored
    suj : string
        Subject name (e.g 'subject_01')
    hemi : {'both', 'left', 'right'}
        Hemispheres to load

    Returns
    -------
    labmap : array_like
        Labmap (i.e indices associated to each vertex)
    """
    set_log_level(verbose)
    logger.info(f'-> Loading MarsAtlas labmap of {suj}')
    load_hemi = hemi_to_load(hemi, output_for='bv')

    # -------------------------------------------------------------------------
    # define path and files to load
    lab_path = CONFIG['BV_LABMAP_FOLDER'].format(bv_root=bv_root, suj=suj)
    lab_files = [f'{suj}_{h}white_parcels_marsAtlas.gii' for h in load_hemi]

    # -------------------------------------------------------------------------
    # load the files
    labmap = []
    for k in range(len(lab_files)):
        arch = nibabel.load(op.join(lab_path, lab_files[k]))
        labmap += [arch.darrays[0].data]
        logger.info(f'    MarsAtlas labmap {lab_files[k]} loaded')
    # concatenate everything
    labmap = np.concatenate(labmap, axis=0)

    return labmap

def load_ma_table(verbose=None):
    """Load MarsAtlas corresponding table.

    Returns
    -------
    ma_idx : array_like
        Array of MarsAtlas indices
    ma_names : array_like
        Array of MarsAtlas names associated to each index
    """
    set_log_level(verbose)
    ma = load_marsatlas()
    ma_idx = np.array(ma['Label'])
    ma_names = np.c_[np.array(ma['LR_Name']), np.array(ma['Lobe']),
                     np.array(ma['Full name'])]
    logger.info('-> MarsAtlas table loaded')

    return ma_idx, ma_names


###############################################################################
###############################################################################
#                                FREESURFER
###############################################################################
###############################################################################


def load_fs_mesh(fs_root, suj, hemi='both', transform=True, verbose=None):
    """Load Freesurfer mesh.

    Parameters
    ----------
    fs_root : string
        Path to the Freesurfer folder where subject are stored
    suj : string
        Subject name (e.g 'subject_01')
    hemi : {'both', 'left', 'right'}
        Hemispheres to load
    transform : bool | True
        Tranform the mesh from meshes space to the scanner based

    Returns
    -------
    vertices : array_like
        Array of vertices
    faces : array_like
        Array of faces
    """
    set_log_level(verbose)
    logger.info(f'-> Loading Freesurfer mesh of {suj}')
    load_hemi = hemi_to_load(hemi, output_for='fs')

    # -------------------------------------------------------------------------
    # define path and files to load
    mesh_path = CONFIG['FS_MESH_FOLDER'].format(fs_root=fs_root, suj=suj)
    mesh_files = [f'{h}.white' for h in load_hemi]

    # -------------------------------------------------------------------------
    # load transformation (if needed)
    if transform:
        # path to transformations
        tr_path = CONFIG['FS_TRM_FOLDER'].format(fs_root=fs_root, suj=suj)
        tr_file_orig_to_mesh = f'{suj}_orig_TO_meshes.trm'
        tr_file_orig_to_scanner = f'orig_{suj}_TO_Scanner_Based.trm'
        tr_path_orig_to_mesh = op.join(tr_path, tr_file_orig_to_mesh)
        tr_path_orig_to_scanner = op.join(tr_path, tr_file_orig_to_scanner)
        # load and combine transformations
        tr_mesh_to_orig = read_trm(tr_path_orig_to_mesh, inverse=True)
        tr_orig_to_scanner = read_trm(tr_path_orig_to_scanner, inverse=True)
        tr = chain_transform([tr_mesh_to_orig, tr_orig_to_scanner])
        logger.info('    Freesurfer tranformation loaded (MESHES -> SCANNER)')

    # -------------------------------------------------------------------------
    # load the files
    vertices, faces = [], []
    for k in range(len(mesh_files)):
        # load vertices / faces
        _file = op.join(mesh_path, mesh_files[k])
        _v, _f = nibabel.freesurfer.read_geometry(_file)
        if len(faces):
            _f += faces[0].max() + 1
        # apply transformation (if needed)
        if transform:
            _v = apply_transform(tr, _v)
        vertices += [_v]
        faces += [_f]
        logger.info(f'    Freesurfer mesh {mesh_files[k]} loaded')
    # concatenate everything
    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)

    return vertices, faces


def load_fs_labmap(fs_root, suj, hemi='both', verbose=None):
    """Load the Freesurfer labmap.

    Parameters
    ----------
    fs_root : string
        Path to the Freesurfer folder where subject are stored
    suj : string
        Subject name (e.g 'subject_01')
    hemi : {'both', 'left', 'right'}
        Hemispheres to load

    Returns
    -------
    labmap : array_like
        Labmap (i.e indices associated to each vertex)
    """
    set_log_level(verbose)
    logger.info(f'-> Loading Freesurfer labmap of {suj}')
    load_hemi = hemi_to_load(hemi, output_for='fs')

    # -------------------------------------------------------------------------
    # define path and files to load
    lab_path = CONFIG['FS_LABEL_FOLDER'].format(fs_root=fs_root, suj=suj)
    lab_files = [f'{h}.aparc.a2009s.annot' for h in load_hemi]

    # -------------------------------------------------------------------------
    # load the files
    labmap = []
    for k in range(len(lab_files)):
        _file = op.join(lab_path, lab_files[k])
        _labmap, _, _ = nibabel.freesurfer.read_annot(_file, orig_ids=True)
        labmap += [_labmap]
        logger.info(f'    Freesurfer labmap {lab_files[k]} loaded')
    # concatenate everything
    labmap = np.concatenate(labmap, axis=0)

    return labmap


def load_fs_table(fs_root, suj, hemi='both', verbose=None):
    """Load Freesurfer corresponding table.

    Parameters
    ----------
    fs_root : string
        Path to the Freesurfer folder where subject are stored
    suj : string
        Subject name (e.g 'subject_01')
    hemi : {'both', 'left', 'right'}
        Hemispheres to load

    Returns
    -------
    ma_idx : array_like
        Array of Freesurfer indices
    ma_names : array_like
        Array of Freesurfer names associated to each index
    """
    set_log_level(verbose)
    logger.info(f'-> Loading Freesurfer table of {suj}')
    load_hemi = hemi_to_load(hemi, output_for='fs')

    # -------------------------------------------------------------------------
    # define path and files to load
    tab_path = CONFIG['FS_LABEL_FOLDER'].format(fs_root=fs_root, suj=suj)
    tab_files = [f'{h}.aparc.a2009s.annot' for h in load_hemi]

    # -------------------------------------------------------------------------
    # load the files
    fs_idx, fs_names = [], []
    for k in range(len(tab_files)):
        _file = op.join(tab_path, tab_files[k])
        _, _ctab, _names = nibabel.freesurfer.read_annot(_file, orig_ids=True)
        fs_names += [_names]
        fs_idx += [_ctab[:, -1]]
        logger.info(f'    Freesurfer table {tab_files[k]} loaded')
    # concatenate everything
    fs_names = np.concatenate(fs_names, axis=0)
    fs_idx = np.concatenate(fs_idx, axis=0)

    return fs_idx, fs_names


if __name__ == '__main__':
    fs_root = '/home/etienne/Server/frioul/database/db_freesurfer/seeg_causal'
    bv_root = '/home/etienne/Server/frioul/database/db_brainvisa/seeg_causal'
    suj = 'subject_01'

    # -------------------------------------------------------------------------
    # BRAINVISA TESTING
    # labmap = load_ma_labmap(bv_root, suj)
    # print(labmap.shape)
    # v, f = load_ma_mesh(bv_root, suj)
    # print(v.shape, f.shape)
    # ma_idx, ma_names = load_ma_table()
    # print(np.c_[ma_idx, ma_names])
    # -------------------------------------------------------------------------
    # FREESURFER TESTING
    # labmap = load_fs_labmap(fs_root, suj, hemi='both', verbose=None)
    # print(labmap.shape)
    # v, f = load_fs_mesh(fs_root, suj, hemi='both', transform=True)
    # print(v.shape, f.shape)
    idx, names = load_fs_table(fs_root, suj, hemi='both', verbose=None)
    print(np.c_[idx, names])

