import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch


from iclass.split import evtsplit, cfgsplit


def get_event_df(n_showers: int = 100, n_obs: int = 5) -> pd.DataFrame:
    event_id, obs_id = np.meshgrid(
        np.arange(n_showers),
        np.arange(n_obs)
    )
    obs_id = obs_id.flatten()
    event_id = event_id.flatten()

    data = dict(
        obs_id = obs_id,
        event_id = event_id,
    )

    return pd.DataFrame(data)


def get_config_df(n_showers: int = 100, n_obs: int = 5) -> pd.DataFrame:
    n_showers = np.repeat(n_showers, n_obs)
    obs_id = np.arange(n_obs)

    data = dict(
        obs_id = obs_id,
        n_showers = n_showers,
    )

    return pd.DataFrame(data)


class SplitTest(unittest.TestCase):
    @patch('pandas.read_hdf')
    def test_evtsplit(self, mock_read_hdf):
        n_showers = 100
        n_obs = 5

        events = get_event_df(n_showers, n_obs)

        mock_read_hdf.configure_mock(
            return_value = events
        )

        fractions_sample = (
            (0.7, 0.3),
            (0.2, 0.3, 0.5),
            (0.3, 0.4)
        )

        for fractions in fractions_sample:
            parts = evtsplit(
                input_fname = 'dummy_input',
                key = 'dummy_key',
                fractions = fractions,
            )

            nevents = len(events)
            for part, frac in zip(parts, fractions):
                self.assertAlmostEqual(
                    len(part), frac * nevents, places=1
                )

        with self.assertRaises(ValueError):
            parts = evtsplit(
                input_fname = 'dummy_input',
                key = 'dummy_key',
                fractions = (0.5, 0.6),
            )

    @patch('iclass.io.open_file')
    @patch('pandas.read_hdf')
    def test_cfgsplit(self, mock_read_hdf, mock_open_file):
        n_showers = 100
        n_obs = 5

        cfg = get_config_df(n_showers, n_obs)

        mock_read_hdf.configure_mock(
            return_value = cfg
        )
        mock_open_file.configure_mock(
            return_value = Mock()
        )
        mock_open_file.return_value.__enter__ = Mock()
        mock_open_file.return_value.__exit__ = Mock()

        fractions_sample = (
            (0.7, 0.3),
            (0.2, 0.3, 0.5),
            (0.3, 0.4)
        )

        for fractions in fractions_sample:
            parts = cfgsplit(
                input_fname = 'dummy_input',
                key = 'dummy_key',
                fractions = fractions,
            )

            for part, frac in zip(parts, fractions):
                self.assertTrue(
                    np.allclose(part.n_showers, frac * cfg.n_showers, atol=1)
                )

        with self.assertRaises(ValueError):
            parts = cfgsplit(
                input_fname = 'dummy_input',
                key = 'dummy_key',
                fractions = (0.5, 0.6),
            )

