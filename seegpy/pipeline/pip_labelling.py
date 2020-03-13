"""Labelling pipeline."""
import logging
import os.path as op

import numpy as np
import pandas as pd

from seegpy.config import CONFIG
from seegpy.io import set_log_level
from seegpy.labelling import (labelling_contacts_surf_ma,
                              labelling_contacts_surf_fs,
                              labelling_contacts_vol_fs_mgz,
                              labelling_contacts_vol_ma)
from seegpy.contacts import (successive_monopolar_contacts,
                             compute_middle_contact, contact_to_mni,
                             clean_contact)
from seegpy.testing import test_located_contacts, test_volume_ma


logger = logging.getLogger('seegpy')


def pipeline_labelling_ss(save_path, fs_root, bv_root, suj, c_xyz, c_names,
                          bipolar=True, radius=5., bad_label='none',
                          testing=True, verbose=None):
    """Single subject contact labelling pipeline.

    Parameters
    ----------
    save_path : string
        Path to the folder where the labelling file have to be saved
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
    # -------------------------------------------------------------------------
    # test the provided data
    if testing:
        # test_located_contacts(c_xyz, c_names)
        test_volume_ma(bv_root, suj)
    c_names = np.asarray(clean_contact(list(c_names)))

    # -------------------------------------------------------------------------
    # file checking
    set_log_level(verbose)
    assert op.isdir(save_path)
    kw = dict(radius=radius, bad_label=bad_label, verbose=verbose)
    fs_vol_file = 'aparc.a2009s+aseg'
    # define how the file is going to be saved
    save_as = op.join(save_path, f"{suj}_radius-{radius}.xlsx")

    # -------------------------------------------------------------------------
    # monopolar and bipolar
    contacts = dict()
    contacts['monopolar'] = (c_xyz, c_names)
    if bipolar:
        logger.info("-> Compute bipolar derivations")
        # get successive contacts
        ano_names, cat_names, ano_idx, cat_idx = successive_monopolar_contacts(
            c_names, c_xyz, radius=5., verbose=verbose)
        cat_xyz, ano_xyz = c_xyz[cat_idx, :], c_xyz[ano_idx, :]
        # get bipolar names and coordinates
        bip_xyz = compute_middle_contact(cat_xyz, ano_xyz)
        bip_names = [f"{c}-{a}" for c, a in zip(cat_names, ano_names)]
        contacts['bipolar'] = (bip_xyz, np.array(bip_names))

    # -------------------------------------------------------------------------
    # surfacique and volumique labelling using MarsAtlas and Freesurfer
    df = dict()
    for n_c, (derivation, (cur_xyz, cur_names)) in enumerate(contacts.items()):
        logger.info(f'    Processing {derivation} contacts')

        # ---------------------------------------------------------------------
        # COORDINATES
        # ---------------------------------------------------------------------
        # contact name
        df_name = pd.DataFrame(cur_names, columns=['contact'])
        # scanner coordinates
        df_coords = pd.DataFrame()
        df_coords['x_scanner'] = cur_xyz[:, 0]
        df_coords['y_scanner'] = cur_xyz[:, 1]
        df_coords['z_scanner'] = cur_xyz[:, 2]
        # mni coordinates
        cur_xyz_mni = contact_to_mni(fs_root, suj, cur_xyz)
        df_coords['x_mni'] = cur_xyz_mni[:, 0]
        df_coords['y_mni'] = cur_xyz_mni[:, 1]
        df_coords['z_mni'] = cur_xyz_mni[:, 2]

        # ---------------------------------------------------------------------
        # VOLUMIQUE LABELLING
        # ---------------------------------------------------------------------
        # MarsAtlas volumique labelling
        _ma_lab_vol = labelling_contacts_vol_ma(bv_root, suj, cur_xyz, **kw)
        df_ma_vol = pd.DataFrame()
        df_ma_vol['Lobe'] = _ma_lab_vol[:, 1]
        df_ma_vol['MarsAtlas'] = _ma_lab_vol[:, 0]
        df_ma_vol['MarsAtlas Full'] = _ma_lab_vol[:, 2]

        # Freesurfer volumique labelling
        fs_labels = labelling_contacts_vol_fs_mgz(fs_root, suj, cur_xyz,
                                                  file=fs_vol_file, **kw)
        df_fs_vol = pd.DataFrame(fs_labels.ravel(), columns=['Freesurfer'])

        # build hemisphere
        hemi = np.array(['Right'] * cur_xyz.shape[0])
        hemi[cur_xyz[:, 0] < 0] = 'Left'
        df_hemi = pd.DataFrame(hemi, columns=['Hemisphere'])

        # build the white / grey matter and subcortical in aseg
        matter = df_fs_vol["Freesurfer"].copy()
        matter.replace(CONFIG['FS_CLEANUP'], inplace=True, regex=True)
        matter_arr = np.array(matter)
        m_is_none = matter_arr == bad_label
        m_is_sub = matter_arr == 'Subcortical'
        m_is_white = matter_arr == 'White'
        nn_grey_matter = np.c_[m_is_none, m_is_sub, m_is_white].any(axis=1)
        matter_arr[~nn_grey_matter] = 'Grey'
        df_matter = pd.DataFrame(matter_arr, columns=['Matter'])

        # ---------------------------------------------------------------------
        # SURFACE LABELLING
        # ---------------------------------------------------------------------
        # Freesurfer surface labelling
        # fs_surf = labelling_contacts_surf_fs(fs_root, suj, cur_xyz, **kw)
        # df_fs_surf = pd.DataFrame(fs_surf, columns=['aparc.a2009s.surf'])

        # # MarsAtlas surface labelling
        # ma_surf = labelling_contacts_surf_ma(bv_root, suj, cur_xyz, **kw)
        # df_ma_surf = pd.DataFrame(ma_surf, columns=[
        #     'MarsAtlasSurf', 'Lobe', 'MarsAtlasSurf Full'])

        # build MarsAtlas subcortical based on Freesurfer ouputs
        # zp = zip(CONFIG['FS_SUBCORTICAL'], CONFIG['MA_SUBCORTICAL'])
        # repl = {i_fs: i_ma for i_fs, i_ma in zp}
        # ma_sub = df_fs_vol["Freesurfer"].copy()
        # ma_sub.replace(repl, inplace=True, regex=True)
        # pa = ma_sub.str.findall('(' + '|'.join(CONFIG['MA_SUBCORTICAL']) + ')')
        # is_sub = np.array(pa.astype(bool))
        # df_ma_surf['MarsAtlasSurf'].iloc[is_sub] = ma_sub.iloc[is_sub]
        # df_ma_surf['MarsAtlasSurf Full'].iloc[is_sub] = ma_sub.iloc[is_sub]
        # df_ma_surf['Lobe'].iloc[is_sub] = df_matter['Matter'].iloc[is_sub]

        # ---------------------------------------------------------------------
        # FINALIZE DATAFRAME
        # ---------------------------------------------------------------------
        # drop aseg labels (because redundant)
        # df_fs_vol.drop(columns="aseg.vol", inplace=True)
        # merge everything
        _df = pd.concat((df_name, df_matter, df_hemi, df_ma_vol, df_fs_vol,
                         df_coords), axis=1)
        df[derivation] = _df

    # -------------------------------------------------------------------------
    # Excel writting
    with pd.ExcelWriter(save_as) as writer:
        for sheet, wdf in df.items():
            wdf.to_excel(writer, sheet_name=sheet)


if __name__ == '__main__':
    from seegpy.io import read_3dslicer_fiducial

    fs_root = '/home/etienne/Server/frioul/database/db_freesurfer/seeg_causal'
    bv_root = '/home/etienne/Server/frioul/database/db_brainvisa/seeg_causal'
    suj = 'subject_01'
    save_to = '/run/media/etienne/DATA/RAW/CausaL/LYONNEURO_2014_DESj/TEST_DATA'

    # -------------------------------------------------------------------------
    # path = '/home/etienne/DATA/RAW/CausaL/LYONNEURO_2015_BARv/3dslicer/recon.fcsv'
    path = '/run/media/etienne/DATA/RAW/CausaL/LYONNEURO_2014_DESj/TEST_DATA/recon.fcsv'
    df = read_3dslicer_fiducial(path)
    c_xyz = np.array(df[['x', 'y', 'z']])
    c_names = np.array(df['label'])

    pipeline_labelling_ss(save_to, fs_root, bv_root, suj, c_xyz, c_names,
                          bipolar=True, radius=5., bad_label='none', verbose=False)