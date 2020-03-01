"""Utility functions for sEEG contacts."""
import os.path as op
import logging

import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from re import findall
from textwrap import wrap

from seegpy.io import set_log_level, read_trm
from seegpy.config import CONFIG
from seegpy.transform import apply_transform


logger = logging.getLogger('seegpy')


def clean_contact(c_names):
    """Clean contact's names.

    For example "A02 - A01" -> "A2-A1"

    Parameters
    ----------
    c_names : list
        List of contacts' names.

    Returns
    -------
    contact_c : list
        List of cleaned contacts.
    """
    assert isinstance(c_names, list)
    chan_repl = {'01': '1', '02': '2', '03': '3', '04': '4', '05': '5',
                 '06': '6', '07': '7', '08': '8', '09': '9', ' ': '', 'p': "'"}
    if not isinstance(c_names, pd.Series):
        c_names = pd.Series(data=c_names, name='c_names')
    c_names.replace(chan_repl, regex=True, inplace=True)
    c_names = c_names.str.upper()
    c_names = c_names.str.strip()

    return list(c_names)


def contact_mono_to_bipo(c_names, sep='-', verbose=None):
    """Convert a list of monopolar contacts into bipolar contacts.

    Parameters
    ----------
    c_names : list
        List of monopolar contact.
    sep : string | '-'
        String separator between bipolar contact.

    Returns
    -------
    c_names_bip : list
        List of bipolar contacts.
    """
    set_log_level(verbose)
    assert isinstance(c_names, list)
    # start by cleaning up monopolar contacts
    c_names = clean_contact(c_names)
    # loop over monopolar contacts
    c_names_bip = []
    for k in c_names:
        try:
            letter = ''.join([i for i in k if not i.isdigit()])
            number = int(findall(r'\d+', k)[0])
            previous_contact = '%s%i' % (letter, number - 1)
            if previous_contact in c_names:
                c_names_bip += ['%s%s%s' % (k, sep, previous_contact)]
        except:
            logger.warning(f'{k} is not a proper sEEG channel')

    return c_names_bip


def contact_bipo_to_mono(c_names, sep='-'):
    """Convert a list of bipolar contacts into monopolar sites.

    Parameters
    ----------
    c_names : list
        List of bipolar contacts.

    Returns
    -------
    contact_r : list
        List of (unsorted) monopolar contacts.
    """
    from textwrap import wrap
    c_names = [k.strip().replace(' ', '').replace(sep, '') for k in c_names]
    c_names = clean_contact(c_names)
    _split = []
    for k in c_names:
        _k = wrap(k, int(np.ceil(len(k) / 2)))
        assert len(_k) == 2, "Wrong channel conversion %s" % str(_k)
        _split += list(_k)
    _split = np.ravel(_split)
    c_names_mono = []
    _ = [c_names_mono.append(k) for k in _split if k not in c_names_mono]  # noqa
    return c_names_mono


def successive_monopolar_contacts(c_names, c_xyz, radius=5., verbose=None):
    """Find successive contacts based on names and spatial locations.

    Successive contacts have to pass three conditions :

        * They must have the same letter (e.g 'A', 'B')
        * Their digits should be seperated by 1 (e.g 'A1' and 'A2')
        * Their spatial distance should be below `radius`

    Parameters
    ----------
    c_names : array_like
        Array of monopolar contacts names of shape (n_contacts,)
    c_xyz : array_like
        Array of monopolar contacts coordinates of shape (n_contacts, 3)
    radius : float | 5.
        Distance to consider as a threshold when considering the inter-contact
        spatial distance

    Returns
    -------
    ano_names : array_like
        Array of anode names
    cat_names : array_like
        Array of cathodes names
    ano_idx : array_like
        Array of anode indices
    cat_idx : array_like
        Array of cathode indices
    """
    set_log_level(verbose)
    c_names = np.asarray(clean_contact(c_names.tolist()))
    letter, number = [], []
    for k in c_names:
        letter += [''.join([i for i in k if not i.isdigit()])]
        number += [int(findall(r'\d+', k)[0])]
    letter, number = np.array(letter), np.array(number).astype(int)
    # computes masks on letters and numbers
    mask_letter = letter.reshape(-1, 1) == letter.reshape(1, -1)
    mask_number = number.reshape(-1, 1) == number.reshape(1, -1) - 1
    c_dist = cdist(c_xyz, c_xyz)
    # if no radius is provided, compute it
    if radius is None:
        # distance between contacts that have the same number and letter
        mask_radius = np.stack((mask_letter, mask_number), axis=0).all(0)
        (c_1, c_2) = np.where(mask_radius)
        c_good_dist = c_dist[c_1, c_2]
        # compute inter-quartile
        q1, q3 = np.percentile(c_good_dist, [0,95])
        iqr = q3 - q1
        radius = q3 + (1.5 * iqr)
        # sanity check plot
        # import matplotlib.pyplot as plt
        # plt.plot(np.sort(c_good_dist), '*')
        # plt.axhline(radius)
        # plt.show()
        logger.info("-> Successive contacts are going to be considered only "
                    f"when the distance between them is under {radius}")
    mask_dist = c_dist < radius
    # merge masks and find cathodes and anodes
    mask = np.stack((mask_letter, mask_number, mask_dist)).all(0)
    (an_idx, cat_idx) = np.where(mask)
    an_names, cat_names = c_names[an_idx], c_names[cat_idx]
    
    return an_names, cat_names, an_idx, cat_idx


def compute_middle_contact(cat_xyz, ano_xyz):
    """Compute the coordinates of the middle point.

    Parameters
    ----------
    cat_xyz : array_like
        Array of coordinates for the cathodes of shape (n_contacts, 3)
    ano_xyz : array_like
        Array of coordinates for the anodes of shape (n_contacts, 3)

    Returns
    -------
    middle : array_like
        Middle point coordinate
    """
    return (cat_xyz + ano_xyz) / 2.


def contact_to_mni(fs_root, suj, c_xyz):
    """Transform contacts from the scanner based to the MNI space.

    Parameters
    ----------
    fs_root : string
        Path to the Freesurfer folder where subject are stored
    suj : string
        Subject name (e.g 'subject_01')
    c_xyz : array_like
        Array of monopolar contacts coordinates of shape (n_contacts, 3)

    Returns
    -------
    c_xyz_mni : array_like
        Contacts in the MNI space
    """
    # path to the transformation
    trm_path = CONFIG['FS_TRM_FOLDER'].format(fs_root=fs_root, suj=suj)
    trm_file = f'{suj}_scanner_to_mni.trm'
    trm = read_trm(op.join(trm_path, trm_file))
    # apply the transformation
    c_xyz_mni = apply_transform(trm, c_xyz)

    return c_xyz_mni


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
    import neo

    # -------------------------------------------------------------------------
    # read the channels
    micro = neo.MicromedIO(filename=path)
    seg = micro.read_segment(signal_group_mode='split-all', lazy=True)
    all_chan = [sig.name.replace(' ', '').strip().upper()
            for sig in seg.analogsignals]
    units = [str(sig.units) for sig in seg.analogsignals]

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

    return seeg_chan


if __name__ == '__main__':
    from seegpy.io import read_3dslicer_fiducial
    # -------------------------------------------------------------------------
    # clean contact names
    # c_names = ['A02 - A01', 'B2- B10']
    # print(clean_contact(c_names))

    # # -------------------------------------------------------------------------
    # # mono -> bipo
    # c_names = ['A01', 'A02', 'A03', 'x', 'B01', 'B02', 'B03', 'B04', 'C1']
    # print(contact_mono_to_bipo(c_names))

    # -------------------------------------------------------------------------
    # path = '/home/etienne/DATA/RAW/CausaL/LYONNEURO_2015_BARv/3dslicer/recon.fcsv'
    path = '/run/media/etienne/DATA/RAW/CausaL/LYONNEURO_2014_DESj/TEST_DATA/recon.fcsv'
    df = read_3dslicer_fiducial(path)
    c_xyz = np.array(df[['x', 'y', 'z']])
    c_names = np.array(df['label'])

    # n_1, n_2, c_1, c_2 = successive_monopolar_contacts(c_names, c_xyz)
    # print(np.c_[n_2, n_1])

    fs_root = '/home/etienne/Server/frioul/database/db_freesurfer/seeg_causal'
    bv_root = '/home/etienne/Server/frioul/database/db_brainvisa/seeg_causal'
    suj = 'subject_01'
    xyz_mni = contact_to_mni(fs_root, suj, c_xyz.copy())

    from visbrain.objects import BrainObj, SourceObj, SceneObj

    sc = SceneObj()
    sc.add_to_subplot(BrainObj("B1"))
    sc.add_to_subplot(SourceObj('orig', c_xyz, color='red'))
    sc.add_to_subplot(SourceObj('mni', xyz_mni, color='green'))
    sc.preview()
