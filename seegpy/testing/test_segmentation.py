"""Tests if the segmentation is correct."""
import os.path as op

import numpy as np
import pandas as pd

import nibabel

from seegpy.config import CONFIG
from seegpy.io import get_data_path


def test_volume_ma(bv_root, suj):
    """Test if the MarsAtlas volume contains the proper transformation."""
    # define paths to the gyri and parcellation files
    bv_path = CONFIG['BV_LABMAP_FOLDER'].format(bv_root=bv_root, suj=suj)
    gyri_file = op.join(bv_path, f"{suj}_L_gyriVolume.nii.gz")
    parc_file = op.join(bv_path, f"{suj}_parcellation.nii.gz")
    ma = pd.read_excel(get_data_path('MarsAtlasSurf.xls'))
    ma_idx = np.sort(np.r_[ma['Label'], [100, 255]])

    # -------------------------------------------------------------------------
    # test the transformation
    tr_gyri = nibabel.load(gyri_file).affine
    tr_parc = nibabel.load(parc_file).affine
    err_msg = (f"The transformations inside {suj}_L_gyriVolume.nii.gz and "
               f"{suj}_parcellation.nii.gz are different. This means that the "
               "MarsAtlas parcellation is not going to be properly aligned "
               "with Freesurfer's T1")
    np.testing.assert_array_equal(tr_gyri, tr_parc, err_msg=err_msg)
    print("-> [TEST] MarsAtlas volume transformation : OK")

    # -------------------------------------------------------------------------
    # test the values
    data = nibabel.load(parc_file).get_data()
    u_data = np.unique(data)
    for k in u_data:
        if k not in ma_idx:
            print(k)
    err_msg = (f"The values in the MarsAtlas volume are not correct")
    np.testing.assert_array_equal(u_data, ma_idx, err_msg=err_msg)
    print("-> [TEST] MarsAtlas volume data : OK")



if __name__ == '__main__':
    from seegpy.io import set_log_level
    bv_root = '/home/etienne/Server/frioul/database/db_brainvisa/seeg_causal'
    suj = 'subject_02'

    test_volume_ma(bv_root, suj)

