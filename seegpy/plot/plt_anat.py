"""Plot anatomical results."""
import numpy as np
import pandas as pd

from seegpy.io import load_marsatlas
from seegpy.config import CONFIG


def subplot_bar(df, hemi, lobe, gs, color, xlim, total, merge_lr):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # subselect the dataframe for the hemi / lobe
    sub_df = df.loc[hemi].loc[lobe].iloc[::-1]
    count = sub_df['Count']
    coun_all = f" - {np.sum(count)} sites" if total else ''
    roi = sub_df['ROI']
    # plot it using seaborn
    ax = plt.subplot(gs)
    sns.barplot(x='Count', y='ROI', orient='h', data=sub_df, color=color,
                ax=ax)
    plt.xlabel(''), plt.ylabel('')  # noqa
    lr = '' if merge_lr else f"({hemi})"
    plt.title(f"{lobe} {lr}{coun_all}", fontweight='bold')
    plt.xlim(xlim)


def plot_anat_repartition_ma(roi, n_subjects=False, merge_lr=False,
                             color='black', title=None, total=True, ma=None):
    """Plot the ROI contact repartition using MarsAtlas.

    Parameters
    ----------
    roi : list
        List of length (n_subjects,) where each element of the list is an array
        of length (n_roi_suj,) describing ROI names (e.g MarsAtlas labels)
    n_subjects : bool | False
        Plot either the number of sites per ROI (`n_subjects=False`) or the
        number of subjects per ROI (`n_subjects=True`)
    merge_lr : bool | False
        Merge the left and right hemisphere
    color : string | 'black'
        Bar color
    title : str | None
        Title of the figure
    total : bool | True
        Display the number of sites per lobe

    Returns
    -------
    fig : plt.figure
        The matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # -------------------------------------------------------------------------
    # test inputs types
    assert isinstance(roi, list)
    assert all([isinstance(k, np.ndarray) for k in roi])
    if n_subjects:
        roi = [np.unique(k) for k in roi]
    all_roi = np.r_[tuple(roi)]
    u_roi = np.unique(all_roi)

    # -------------------------------------------------------------------------
    # group by roi
    df_roi = pd.DataFrame(dict(roi=all_roi, count=np.ones((len(all_roi),))))
    if merge_lr:
        df_roi.replace({'R\\_': 'L_'}, regex=True, inplace=True)
        n_c, n_r, figsize = 30, 11, (12, 10)
    else:
        n_c, n_r, figsize = 30, 23, (16, 10)
    gp_roi = df_roi.groupby('roi').count()
    # load the MarsAtlas reference table
    if not isinstance(ma, pd.DataFrame):
        ma = load_marsatlas()[['Lobe', 'LR_Name', 'Hemisphere']]
        ma.rename(columns={'LR_Name': 'ROI'}, inplace=True)
    ma_s = ma[['ROI']]
    # build the merged dataframe and group by roi
    df = ma_s.set_index('ROI').reindex(gp_roi.index).dropna(axis=0)
    df['Count'] = gp_roi['count'].loc[df.index]
    # reset to the true indexes
    df = df.reindex(ma_s['ROI'], fill_value=0)
    df['Lobe'] = ma.set_index('ROI')['Lobe'].loc[df.index]
    df['Hemisphere'] = ma.set_index('ROI')['Hemisphere'].loc[df.index]
    df = df.reset_index().replace({'L\\_': '', 'R\\_': ''}, regex=True)
    # build a multiindex
    df.index = pd.MultiIndex.from_frame(df[['Hemisphere', 'Lobe', 'ROI']])
    all_counts = np.array(list(df['Count']))
    xlim = (all_counts.min(), all_counts.max())

    # -------------------------------------------------------------------------
    # prepare the figure
    fig = plt.figure(figsize=figsize)
    if isinstance(title, str):
        fig.suptitle(title, fontweight='bold', fontsize=15, y=.99)
    gs = GridSpec(n_c, n_r, left=0.05, bottom=0.03, right=0.99, top=0.90,
                  wspace=.01)
    # define subplot variable
    if merge_lr:
        subplots = {
            'Temporal': {'L' : gs[21:26, 0:5]},
            'Frontal': {'L' : gs[0:19, 0:5]},
            'Parietal': {'L' : gs[14:23, 6:11]},
            'Occipital': {'L' : gs[25:29, 6:11]},
            'Subcortical': {'L' : gs[6:12, 6:11]}}
        hemi = ['L']
    else:
        subplots = {
            'Temporal': {'L' : gs[21:26, 0:5], 'R' : gs[21:26, 18:23]},
            'Frontal': {'L' : gs[0:19, 0:5], 'R' : gs[0:19, 18:23]},
            'Parietal': {'L' : gs[14:23, 6:11], 'R' : gs[14:23, 12:17]},
            'Occipital': {'L' : gs[25:29, 6:11], 'R' : gs[25:29, 12:17]},
            'Subcortical': {'L' : gs[6:12, 6:11], 'R' : gs[6:12, 12:17]}}
        hemi = ['L', 'R']

    # -------------------------------------------------------------------------
    # plot each hemisphere / lobe

    for l in CONFIG['LOBES']:
        for h in hemi:
            subplot_bar(df, h, l, subplots[l][h], color, xlim, total, merge_lr)

    return fig



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from brainpipe.system import Study
    st = Study('CausaL')

    roi, lobe = [], []
    files = st.search('3.5', folder='anatomy', full_path=False, verbose=False)
    for f in files:
        df = st.search(f, folder='anatomy', load=True)
        roi += [np.array(list(df['MarsAtlas']))]

    title = f'Number of bipolar derivations per roi (n_suj={len(roi)})'

    plot_anat_repartition_ma(roi, title=title, merge_lr=True)
    plt.show()
