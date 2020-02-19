"""Functions for applying transformations."""
import numpy as np


def apply_transform(tr, xyz, inverse=False):
    """Apply the transformation to coordinates.

    Parameters
    ----------
    tr : array_like
        The (4, 4) transformation array
    xyz : array_like
        Array of coordinates (e.g vertices, sEEG contacts etc.)
    inverse : bool | False
        Inverse transformation

    Returns
    -------
    xyz_m : array_like
        Transformed coordinates
    """
    assert tr.shape == (4, 4)
    assert xyz.shape[1] == 3
    n_xyz = xyz.shape[0]
    if inverse:
        tr = np.linalg.inv(tr)
    return tr.dot(np.c_[xyz, np.ones(n_xyz)].T).T[:, 0:-1]


def chain_transform(trs, inverse=False):
    """Chain of affine transformations.

    Parameters
    ----------
    trs : list
        List of transformations.

    Returns
    -------
    tr : array_like
        Transformation array
    """
    tr_chain = None
    for tr in trs:
        if tr_chain is None:
            tr_chain = tr
        else:
            tr_chain = np.dot(tr_chain, tr)
    if inverse:
        tr_chain = np.linalg.inv(tr_chain)
    return tr_chain
