"""Reading functinons."""
import numpy as np
import pandas as pd


def read_trm(path, as_transform=True):
    """Read a transformation file.

    Parameters
    ----------
    path : string
        Path to the trm file
    as_transform : bool
        Get either the array as a usable (4, 4) array (True) or just simply
        or just the array contained in the file (False)

    Returns
    -------
    tr : array_like
        Transformation array
    """
    tr = np.genfromtxt(path)
    if not as_transform:
        return tr
    return np.vstack((np.c_[tr[1::, :], tr[0, :]], np.array([0, 0, 0, 1])))


def read_contacts_trc(path, report=True):
    """Read the channels that are contained inside a TRC file.

    This function uses the neo python package.

    Parameters
    ----------
    path : string
        Path to the trc file
    report : bool | True
        Print report about channel informations

    Returns
    -------
    seeh_chan : list
        List of contacts
    """
    import neo

    # -------------------------------------------------------------------------
    # read the channels
    micro = neo.MicromedIO(filename=path)
    seg = micro.read_segment(signal_group_mode='split-all', lazy=True)
    all_chan = [sig.name.replace(' ', '').strip().upper()
            for sig in seg.analogsignals]

    # -------------------------------------------------------------------------
    # detect seeg contacts
    is_chan = []
    for n_c, c in enumerate(all_chan):
        # first char is a letter and second a number
        if (c[0].isalpha()) and (c[1].isdigit()): is_chan += [n_c]  # noqa
    seeg_chan = np.array(all_chan)[is_chan]

    # -------------------------------------------------------------------------
    # isolate first letter :
    letter = np.sort([k[0] for k in seeg_chan])
    # count the number of contacts per channels
    contact_count = dict()
    for l in letter:
        num = 0
        for c in seeg_chan:
            if l in c: num += 1  # noqa
        contact_count[l] = num

    if report:
        print('-' * 79)
        print(f"#Channels in the TRC file : {len(all_chan)}")
        print(f"#sEEG channel : {len(seeg_chan)}")
        print(f"#Non sEEG channel : {len(all_chan) - len(seeg_chan)}")
        print(f"#Electrodes : {len(letter)}")
        print("#Contacts per electrodes :")
        # print([print(f"{c} = {n}") for c, n in contact_count.items()])
        s = ' | '.join([f'{c} : {n}' for (c, n) in contact_count.items()])
        print("\n".join(wrap(s, width=79)))
        print('-' * 79)

    return seeg_chan


def read_3dslicer_fiducial(path):
    """Read coordinates in a fiducial fcsv file.

    Parameters
    ----------
    path : string
        Path to the fcsv file

    Returns
    -------
    df : DataFrame
        DataFrame with the columns label, x, y and z
    """
    return pd.read_csv(path, skiprows=[0, 1])[['label', 'x', 'y', 'z']]
