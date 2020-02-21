"""Writing functions."""
import numpy as np


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
