import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

from iclass.markup import mkmarkup


def get_ref_df(log_emin: float, log_emax: float, ebinsdec: int, nclasses: int, nsamples: int) -> pd.DataFrame:
    """
    Creates a mock event data frame.

    Parameters
    ----------
    log_emin: float
        log10 of the minimal energy to take
    log_emax: float
        log10 of the maximal energy to take
    ebinsdec: int
        number of bins per decade to assume
    nclasses: int
        number of PSF classes to assume
    nsamples: int
        number of event samples to take
        (each with the shape (nclasses, nenergies))

    Returns
    -------
    df: pd.DataFrame
        examplary event list
    """

    _offset = np.arange(nclasses)
    _loge = np.linspace(log_emin, log_emax, num=ebinsdec+1)[:-1]
    _sample_id = np.arange(nsamples)

    offset, loge, _ = np.meshgrid(_offset, _loge, _sample_id)
    phi = np.random.uniform(0, 2*np.pi, size=offset.shape)
    
    offset = offset.flatten()
    phi = phi.flatten()
    loge = loge.flatten()

    data = dict(
        mc_az = np.zeros(len(offset)),
        mc_alt = np.zeros(len(offset)),
        reco_az = np.pi/180 * offset * np.cos(phi),
        reco_alt = np.pi/180 * offset * np.sin(phi),
        mc_energy = 10**loge,
        psf_class_true = (offset + 1).astype(int)
    )

    return pd.DataFrame(data)


class MarkupTest(unittest.TestCase):
    @patch('pandas.read_hdf')
    def test_mkmarkup(self, mock_read_hdf):
        ebinsdec = 4

        ref = get_ref_df(
            log_emin = 0,
            log_emax = 2,
            ebinsdec = ebinsdec,
            nclasses = 4,
            nsamples = 100
        )

        mock_read_hdf.configure_mock(
            return_value = ref
        )
        result = mkmarkup(
            input_fname = 'dummy_input',
            key = 'dummy_key',
            ebinsdec = ebinsdec,
            cuts = ''
        )

        self.assertTrue(
            np.allclose(
                result['psf_class'].values,
                result['psf_class_true'].values
            )
        )
