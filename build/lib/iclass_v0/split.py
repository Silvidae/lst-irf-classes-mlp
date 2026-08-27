import logging
import random
import numpy as np
import pandas as pd

from iclass.io import read_simulation_config


def evtsplit(input_fname: str, key: str, fractions: tuple) -> tuple:
    """
    Splits the input MC events into parts with the counts
    proportional to the indicated fractions.

    Parameters
    ----------
    input_fname: str
        input Monte Carlo file name
    key: str
        input HDF5 file key to read from
    fractions: tuple
        Relative fractions to split into; must total to <1.

    Returns
    -------
    samples: list
        List of pd.DataFrame instances with event lists 
        corresponding to the specifed fractions
    """
    log = logging.getLogger(__name__)

    if sum(fractions) > 1:
        raise ValueError(
            f"total of the fractions should be <=1"
            f" but is {sum(fractions)}"
            f" ({fractions})"
        )
    
    if sum(fractions) < 1:
        log.warning(
            'total of the fractions is %f < 1,'
            'some events will be lost',
            sum(fractions)
        )

    events = pd.read_hdf(input_fname, key=key)

    cfractions = np.cumsum(fractions)
    if cfractions[0] > 0:
        cfractions = np.concatenate(([0], cfractions))

    samples = []
    seed = random.randint(0, 2**32 - 1)

    for obs_id in events.obs_id.unique():
        _events = events.query(f'obs_id == {obs_id}')
        _events = _events.sample(frac=1, random_state=seed).reset_index(drop=True)
        nevents = len(_events)

        _samples = [
            _events.iloc[int(fstart * nevents) : int(fstop * nevents)]
            for fstart, fstop in zip(cfractions[:-1], cfractions[1:])
        ]

        for sample in _samples:
            sample.reset_index(drop=True, inplace=True) 

        samples.append(_samples)

    parts = [
        pd.concat([sample[i] for sample in samples])
        for i in range(len(fractions))
    ]

    for part in parts:
        part.reset_index(drop=True, inplace=True)

    return parts


def cfgsplit(input_fname: str, key: str, fractions: tuple) -> tuple:
    """
    Splits the input MC simulation configuration into parts with 
    the event counts proportional to the indicated fractions.

    Parameters
    ----------
    input_fname: str
        input Monte Carlo file name
    key: str
        input HDF5 file key to read from
    fractions: tuple
        Relative fractions to split into; must total to <1.

    Returns
    -------
    samples: list
        List of pd.DataFrame instances with event lists 
        corresponding to the specifed fractions
    """
    log = logging.getLogger(__name__)

    if sum(fractions) > 1:
        raise ValueError(
            f"total of the fractions should be <=1"
            f" but is {sum(fractions)}"
            f" ({fractions})"
        )
    
    if sum(fractions) < 1:
        log.warning(
            'total of the fractions is %f < 1,'
            'some events will be lost',
            sum(fractions)
        )

    config = read_simulation_config(input_fname, key=key)

    parts = [
        config.copy()
        for _ in fractions
    ]
    if hasattr(config, 'attrs'):
        for cfg in parts:
            cfg.attrs = config.attrs.copy()

    for part, frac in zip(parts, fractions):
        part.n_showers = (part.n_showers * frac).astype(int)

    return parts
