"""Configuration file."""
import os.path as op

CONFIG = dict()

# -----------------------------------------------------------------------------
# BRAINVISA

BV_MESH_FOLDER = ('{bv_root}', '{suj}', 't1mri', 'default_acquisition',
                  'default_analysis', 'segmentation', 'mesh')
BV_LABMAP_FOLDER = ('{bv_root}', '{suj}', 't1mri', 'default_acquisition',
                    'default_analysis', 'segmentation', 'mesh',
                    'surface_analysis')
BV_TRM_FOLDER = ('{bv_root}', '{suj}', 't1mri', 'default_acquisition',
                 'registration')

CONFIG['BV_MESH_FOLDER'] = op.join(*BV_MESH_FOLDER)
CONFIG['BV_LABMAP_FOLDER'] = op.join(*BV_LABMAP_FOLDER)
CONFIG['BV_TRM_FOLDER'] = op.join(*BV_TRM_FOLDER)

# -----------------------------------------------------------------------------
# FREESURFER

FS_MRI_FOLDER = ('{fs_root}', '{suj}', 'mri')
FS_MESH_FOLDER = ('{fs_root}', '{suj}', 'surf')
FS_LABEL_FOLDER = ('{fs_root}', '{suj}', 'label')
FS_TRM_FOLDER = ('{fs_root}', '{suj}', 'mri', 'transforms')

CONFIG['FS_MRI_FOLDER'] = op.join(*FS_MRI_FOLDER)
CONFIG['FS_MESH_FOLDER'] = op.join(*FS_MESH_FOLDER)
CONFIG['FS_LABEL_FOLDER'] = op.join(*FS_LABEL_FOLDER)
CONFIG['FS_TRM_FOLDER'] = op.join(*FS_TRM_FOLDER)
