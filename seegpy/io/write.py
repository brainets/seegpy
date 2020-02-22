"""Writing functions."""
import numpy as np

from seegpy.io import read_trm


def write_3dslicer_fiducial(path, c_xyz, c_names=None, description=None):
    """Write fiducial files for 3D slicer.

    Parameters
    ----------
    path : string
        Path to the file to save. Should end up with .fcsv
    c_xyz : array_like
        Array of coordinates of shape (n_contacts, 3) (e.g sEEG sites)
    c_names : array_like | None
        Array of labels associated to each contact (e.g 'A1', 'A2'). Should be
        an array with a shape of (n_contacts,)
    description : array_like | None
        Array of descriptions associated to each contact (e.g the ROI of each
        site). Should be an array with a shape of (n_contacts,)
    """
    # data checking
    assert c_xyz.shape[1] == 3
    n_contacts = c_xyz.shape[0]
    if c_names is None: c_names = np.full((n_contacts,), '')  # noqa
    if description is None: description = np.full((n_contacts,), '')  # noqa
    assert len(c_names) == len(description) == n_contacts
    assert ('.fcsv' in path), "File should end up with .fcsv"
    # line prototype
    col = ('# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,'
           'associatedNodeID\n')
    line = ("vtkMRMLMarkupsFiducialNode_{n_c},{x},{y},{z},"
            "0.000,0.000,0.000,1.000,1,1,0,{label},{desc},\n")
    # open a file and write required line
    file = open(path, 'w')
    file.write('# Markups fiducial file version = 4.10\n')
    file.write('# CoordinateSystem = 0\n')
    file.write(col)
    # now loop over sites and write them
    for n_c in range(n_contacts):
        x, y, z = c_xyz[n_c, :]
        file.write(line.format(n_c=n_c, x=x, y=y, z=z, label=c_names[n_c],
                               desc=description[n_c]))

    file.close()


def write_3dslicer_transform(path, trm):
    assert ('.txt' in path), "File should end up with .txt"
    if isinstance(trm, str):
        trm = read_trm(trm)
    assert isinstance(trm, np.ndarray) and (trm.shape == (4, 4))
    # flatten the transform
    rt = trm[:-1, :].T.ravel()
    rt_str = ' '.join([str(k) for k in rt])
    # open a file and write required line
    file = open(path, 'w')
    file.write('#Insight Transform File V1.0\n')
    file.write('#Transform 0\n')
    file.write('Transform: AffineTransform_double_3_3\n')
    file.write(f'Parameters: {rt_str}\n')
    file.write('FixedParameters: 0 0 0\n')
    file.close()


if __name__ == '__main__':
    tr_path = '/home/etienne/Server/frioul/database/db_brainvisa/seeg_causal/subject_01/t1mri/default_acquisition/registration/RawT1-subject_01_default_acquisition_TO_Scanner_Based.trm'
    save_to = '/home/etienne/DATA/RAW/CausaL/LYONNEURO_2014_DESj/TEST_DATA/x3d_bv-scanner_to_fs-scanner.txt'
    write_3dslicer_transform(save_to, tr_path)