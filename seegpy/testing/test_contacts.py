"""Test contacts."""
import numpy as np
import pandas as pd

from re import findall


def test_located_contacts(xyz, names):
    """Test the contacts located by the user."""
    # -------------------------------------------------------------------------
    # test shapes
    assert (xyz.shape[0] == len(names)) and (xyz.shape[1] == 3)
    assert isinstance(xyz, np.ndarray)
    print("-> [TEST] Contact shape : OK")

    # -------------------------------------------------------------------------
    # extract letter and numbers
    letter, number = [], []
    for k in names:
        _l = findall(r'[A-Za-z]+', k)
        _n = findall(r'\d+', k)
        _s = findall(r'[!@#$%^&*(),.?":{}|<>]', k)
        assert len(_l) == 1, f"No letter detected for contact {k}"
        assert len(_n) == 1, f"No number detected for contact {k}"
        assert len(_s) == 0, f"Contact {k} contains special caractere"
        letter += _l
        number += [int(_n[0])]
    letter, number = np.array(letter), np.array(number)
    print("-> [TEST] Contact form : OK")
    df = pd.DataFrame({'let': letter, 'nb': number})
    gp = df.groupby('let').groups

    # -------------------------------------------------------------------------
    # test that tip contact are first (e.g A1)
    for l, idx in gp.items():
        _x = np.abs(xyz[np.array(list(idx)), 0])
        _nb = number[idx]
        _nb_min, _nb_max = _x.argmin(), _x.argmax()
        assert _x[0]  < _x[-1], (
            f"First contact should be the deepest (min={l})")
        assert _nb_max == len(_x) - 1, (f"Last contact should be the closest "
                                        f"to the surface (max={l}{_nb_max})")
        assert _nb[0] < _nb[-1]
        # contact numbers are sorted (increasing)
        err = (f'Contacts number of electrode {l} not sorted properly')
        np.testing.assert_array_equal(_nb.argsort(), np.arange(len(_x)),
                                      err_msg=err)
        # contact numbers should be consecutives
        err = (f'Contacts number of electrode {l} are not consecutives')
        np.testing.assert_array_equal(_nb, np.arange(1, len(_x) + 1),
                                      err_msg=err)
    print("-> [TEST] Tip contact are first : OK")
    print("-> [TEST] Contacts are sorted : OK")



if __name__ == '__main__':
    from seegpy.io import read_3dslicer_fiducial
    root = '/home/etienne/Server/frioul/database/db_anatomy/seeg_causal/'
    suj = 'subject_09'
    path = f'{root}/{suj}/implantation/recon.fcsv'
    # path = '/run/media/etienne/Samsung_T5/BACKUPS/RAW/CausaL/LYONNEURO_2015_BOUa/3dsl/implantation/recon.fcsv'
    df = read_3dslicer_fiducial(path)
    xyz = np.array(df[['x', 'y', 'z']])
    names = df['label']

    test_located_contacts(xyz, names)
