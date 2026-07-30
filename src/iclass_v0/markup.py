import logging
import numpy as np
import pandas as pd
from astropy.coordinates import angular_separation


def mkmarkup(input_fname: str, key: str, ebinsdec: float, cuts: str = '') -> pd.DataFrame:
    """
    Marks up the PSF classes within the MC file.

    Calculates the reconstructed event angular offset wrt to the
    true coordinates and splits events into "PSF classes" defined by
    the 25, 50, 75 and 100% offset population percentiles within the
    energy bins of the pre-defined width.

    The input MC file should be of DL2 level - i.e. include the
    reconstructed event directions ("reco_src_x" and "reco_src_y"
    columns describing it in the telescope camera frame).

    Parameters
    ----------
    input_fname: str
        input Monte Carlo file name
    key: str
        input HDF5 file key to read from
    ebinsdec: float
        number of true energy bins per dec to assume
    cuts: str
        event cuts to apply

    Returns
    -------
    df: pd.DataFrame
        MC event list with the "psf_class" column
    """

    log = logging.getLogger(__name__)
    data = pd.read_hdf(input_fname, key=key)

    if cuts:
        data = data.query(cuts)

    data.loc[:, 'reco_offset'] = 180 / np.pi * angular_separation(
        data['mc_az'].values,
        data['mc_alt'].values,
        data["reco_az"].values,
        data["reco_alt"].values,
    )
    data['psf_class'] = -1

    energy_edges = 10**np.arange(
        np.log10(data['mc_energy'].min()),
        np.log10(data['mc_energy'].max()),
        step=1 / ebinsdec
    )

    energy_ids = np.digitize(data['mc_energy'], energy_edges)

    for energy_id in np.unique(energy_ids):
        selection = energy_ids == energy_id
        mid_edges = np.percentile(
            data['reco_offset'][selection],
            [25, 50, 75]
        )
        offset_edges = np.concatenate(
            ([0], mid_edges, [np.inf])
        )
        psf_class = np.digitize(
            data['reco_offset'][selection],
            offset_edges
        )
        data.loc[selection, 'psf_class'] = psf_class

    if any(data['psf_class'].values == -1):
        log.warning(
            "not marked events found and will be dropped; "
            "this may indicate reconstructed offsets were "
            "outside the [0;inf] range"
        )
        data = data.query('psf_class != -1')

    return data