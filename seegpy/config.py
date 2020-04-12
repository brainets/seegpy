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
FS_SUBCORTICAL = [
    'Left-Accumbens-area', 'Left-Amygdala', 'Left-Caudate', 'Left-Hippocampus',
    'Left-Pallidum', 'Left-Putamen', 'Left-Thalamus-Proper',
    'Right-Accumbens-area', 'Right-Amygdala', 'Right-Caudate',
    'Right-Hippocampus', 'Right-Pallidum', 'Right-Putamen',
    'Right-Thalamus-Proper']
FS_MATTER = {
    'Right-Cerebral-Cortex': 'Grey',
    'Left-Cerebral-Cortex': 'Grey',
    'Right-Cerebral-White-Matter': 'White',
    'Left-Cerebral-White-Matter': 'White'}
FS_CLEANUP = FS_MATTER.copy()
for k in FS_SUBCORTICAL:
    FS_CLEANUP[k] = 'Subcortical'

CONFIG['FS_MRI_FOLDER'] = op.join(*FS_MRI_FOLDER)
CONFIG['FS_MESH_FOLDER'] = op.join(*FS_MESH_FOLDER)
CONFIG['FS_LABEL_FOLDER'] = op.join(*FS_LABEL_FOLDER)
CONFIG['FS_TRM_FOLDER'] = op.join(*FS_TRM_FOLDER)
CONFIG['FS_SUBCORTICAL'] = FS_SUBCORTICAL
CONFIG['FS_MATTER'] = FS_MATTER
CONFIG['FS_CLEANUP'] = FS_CLEANUP

# -----------------------------------------------------------------------------
# MARSATLAS

MA_SUBCORTICAL = [
    'L_NAc', 'L_Amyg', 'L_Cd', 'L_Hipp', 'L_GP', 'L_Put', 'L_Thal',
    'R_NAc', 'R_Amyg', 'R_Cd', 'R_Hipp', 'R_GP', 'R_Put', 'R_Thal']
CONFIG['MA_SUBCORTICAL'] = MA_SUBCORTICAL
CONFIG['LOBES'] = ['Frontal', 'Occipital', 'Parietal', 'Subcortical',
                   'Temporal']
