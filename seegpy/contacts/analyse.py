"""Analyse the channels contains in TRC or mat files.

- TRC files : Lyon / Grenoble
- MAT files : Pragues
"""
import numpy as np
import pandas as pd

from re import findall

from seegpy.io import read_contacts_trc


def _analyse_channels(letter, number, seeg_chan, all_chan, print_report=True):
    """Sub function for analyzing channels."""
    # -------------------------------------------------------------------------
    # group letters
    gp_letter = pd.Series(letter).groupby(by=letter).groups
    contact_info = dict()
    for k in gp_letter.keys():
        _gp = dict()
        _gp['idx'] = np.array(gp_letter[k])
        _gp['len'] = len(_gp['idx'])
        _gp['nb'] = number[_gp['idx']]
        _gp['suc'] = np.array_equal(_gp['nb'], np.arange(1, _gp['len'] + 1))
        contact_info[k] = _gp

    # -------------------------------------------------------------------------
    # build the report

    report = f"""
    {'-' * 79}
    # Number of channels : {len(all_chan)}
    # Number of sEEG channels : {len(seeg_chan)}
    # Number of non-sEEG channel : {len(all_chan) - len(seeg_chan)}
    # Number of electrodes : {len(contact_info.keys())}
    {'-' * 79}
    List of channels :
    {', '.join(all_chan)}
    {'-' * 79}
    List of sEEG channels :
    {', '.join(seeg_chan)}
    {'-' * 79}\n"""
    for l, r in contact_info.items():
        report += f"    * Electrode : {l}\n"
        report += f"        Number of contacts : {r['len']}\n"
        report += f"        Numbers : {r['nb']}\n"
        report += f"        Max : {max(r['nb'])}\n"
        report += f"        Successive : {r['suc']}\n"


    # -------------------------------------------------------------------------
    # count the number of contacts per channels
    if isinstance(print_report, bool) and print_report:
        print(report)
    elif isinstance(print_report, str):
        file = open(print_report, 'w')
        file.write(report)
        file.close()


def analyse_channels_in_trc(path, print_report=True):
    """Read the channels that are contained inside a TRC file.

    Parameters
    ----------
    path : string
        Path to a TRC file
    print_report : bool | True
        Print the report

    Returns
    -------
    seeg_chan : array_like
        Array of channels contained in the TRC file
    """
    # -------------------------------------------------------------------------
    # read the channels
    all_chan, units = read_contacts_trc(path)

    # -------------------------------------------------------------------------
    # detect seeg contacts
    is_chan, letter, number = [], [], []
    for n_c, c in enumerate(all_chan):
        # define conditions
        s_letter = (np.sum([_c.isalpha() for _c in c]) == 1) and c[0].isalpha()
        any_digit = np.any([_c.isdigit() for _c in c])
        len_range = (2 <= len(c) <= 4)
        is_uv = 'uV' in units[n_c]
        if s_letter and any_digit and len_range and is_uv:
            letter += [''.join([i for i in c if not i.isdigit()])]
            number += [int(findall(r'\d+', c)[0])]
            is_chan += [n_c]
    letter, number = np.array(letter), np.array(number)
    seeg_chan = np.array(all_chan)[is_chan]

    # analyse and report
    _analyse_channels(letter, number, seeg_chan, all_chan,
                      print_report=print_report)

    return seeg_chan


def analyse_channels_in_mat(path, print_report=True):
    """Read the channels that are contained inside a MAT file.

    This function is really made for Pragues data.

    Parameters
    ----------
    path : string
        Path to a MAT header file
    print_report : bool | True
        Print the report

    Returns
    -------
    seeg_chan : array_like
        Array of channels contained in the MAT file
    """
    import h5py
    f = h5py.File(path, 'r')['H']
    fc = f['channels']
    # read channel names and types
    _ref_chan = list(fc['name'])
    cn = [''.join(chr(i) for i in f[k[0]][:]) for k in list(fc['name'])]
    ct = [''.join(chr(i) for i in f[k[0]][:]) for k in list(fc['signalType'])]
    # conversion
    all_chan = np.array(cn)
    is_chan = np.array(ct) == 'SEEG'
    seeg_chan = all_chan[is_chan]
    letter, number = [], []
    for c in seeg_chan:
        letter += [str(findall(r'[A-Za-z]+', c)[0])]
        number += [int(findall(r'\d+', c)[0])]
    letter, number = np.array(letter), np.array(number)

    # analyse and report
    _analyse_channels(letter, number, seeg_chan, all_chan,
                      print_report=print_report)

    return seeg_chan


if __name__ == '__main__':
    path = '/run/media/etienne/Samsung_T5/BACKUPS/RAW/CausaL/PRAGUES_2019_PR7_day1/alignedData/19_PR7_day1_header.mat'

    analyse_channels_in_mat(path, print_report=True)
