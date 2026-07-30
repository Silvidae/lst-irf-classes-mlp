import argparse
import logging
import os
import joblib
import pandas as pd

from iclass.rf import apply_rf
from iclass.io import read_simulation_config, write_simulation_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=r"""
        Event class computation tool for CTA-compatible event files.

        Applies the pre-trained random forest to calculate
        the PSF quality class.

        The input data file should be of DL2 level
        and include all the columns used during the RF training.
        """
    )

    parser.add_argument(
        '-i',
        "--input",
        default='',
        help='input Monte Carlo file name'
    )
    parser.add_argument(
        '-r',
        "--rf",
        default='',
        help='pre-trained random forest path'
    )
    parser.add_argument(
        '-p',
        "--prefix",
        default='./out_',
        help='output file name prefix. '
        "It will be appended with the original file name."
        "If '--split' option is specified, mulptiple files "
        "with endings like 'class0.h5', 'class1.h5' etc, "
        "containing events of with individual PSF classes will be created."
    )
    parser.add_argument(
        '-e',
        "--event-key",
        default='/dl2/event/telescope/parameters/LST_LSTCam',
        help='input HDF5 file key to read the events from'
    )
    parser.add_argument(
        '-c',
        "--cfg-key",
        default='',
        help='input HDF5 file key to read the config from. '
            'For LST MCs the path is "/simulation/run_config".'
    )
    parser.add_argument(
        '-s',
        "--split",
        action='store_true',
        help='split output MC file into the parts with individual PSF classes'
    )
    parser.add_argument(
        '-z',
        "--complevel",
        type=int,
        default=7,
        help='HDF5 data compression level'
    )
    args = parser.parse_args()

    rf = joblib.load(args.rf)
    sample = pd.read_hdf(args.input, key=args.event_key)
    sample = apply_rf(sample, rf)

    if args.cfg_key:
        cfg = read_simulation_config(args.input, key=args.cfg_key)

    if args.split:
        _, file_name = os.path.split(args.input)
        fname, _ = os.path.splitext(file_name)

        for psf_class in sample['reco_psf_class'].unique():
            output = f'{args.prefix}{fname}_class{psf_class}.h5'
            subsample = sample.query(f'reco_psf_class == {psf_class}')
            subsample.to_hdf(output, key=args.event_key, complevel=args.complevel)
            if args.cfg_key:
                # MC configuration table has to be written with `tables`
                # as DataFrame.to_hdf(..., format='table') stores the resulting
                # table under the additional '.../table' key.
                write_simulation_config(cfg, output, args.cfg_key)
    else:
        _, file_name = os.path.split(args.input)
        output = f'{args.prefix}{file_name}'
        sample.to_hdf(output, key=args.event_key, complevel=args.complevel)
        if args.cfg_key:
            # MC configuration table has to be written with `tables`
            # as DataFrame.to_hdf(..., format='table') stores the resulting
            # table under the additional '.../table' key.
            write_simulation_config(cfg, output, args.cfg_key)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s %(name)-30s : %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    main()
