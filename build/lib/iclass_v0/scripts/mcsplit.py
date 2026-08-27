import argparse
import logging

from iclass.split import evtsplit, cfgsplit
from iclass.io import write_simulation_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=r"""
        CTA-compatible Monte Carlo files splitting tool.

        Input MC file is split following the specified fractions
        and each part is saved as a separate file with the corresponding name.
        """
    )

    parser.add_argument(
        '-i',
        "--input",
        default='',
        help='input Monte Carlo file name'
    )
    parser.add_argument(
        '-p',
        "--prefix",
        default='./out_',
        help='output file name prefix. '
        "It will be appended with 'part0.5h', 'part1.h5' etc "
        "when generating the output files"
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
        default='/simulation/run_config',
        help='input HDF5 file key to read the config from'
    )
    parser.add_argument(
        '-f',
        "--fractions",
        default=[0.5, 0.5],
        nargs="+",
        type=float,
        help='factions into which to split the input MC file'
    )
    parser.add_argument(
        '-z',
        "--complevel",
        type=int,
        default=7,
        help='HDF5 data compression level'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s %(name)-30s : %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    evt_samples = evtsplit(args.input, args.event_key, args.fractions)
    cfg_samples = cfgsplit(args.input, args.cfg_key, args.fractions)

    for i, (evt, cfg) in enumerate(zip(evt_samples, cfg_samples)):
        output = f'{args.prefix}part{i}.h5'
        evt.to_hdf(output, key=args.event_key, complevel=args.complevel)
        # MC configuration table has to be written with `tables`
        # as DataFrame.to_hdf(..., format='table') stores the resulting
        # table under the additional '.../table' key.
        write_simulation_config(cfg, output, args.cfg_key)


if __name__ == "__main__":
    main()
