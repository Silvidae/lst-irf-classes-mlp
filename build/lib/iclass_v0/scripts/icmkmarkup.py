import argparse
import logging

from shutil import copyfile

from iclass.markup import mkmarkup


def main() -> None:
    parser = argparse.ArgumentParser(
        description=r"""
        Event class markup tool for CTA-compatible Monte Carlo files.

        Calculates the reconstructed event angular offset wrt to the
        true coordinates and splits events into "PSF classes" defined by
        the 25, 50, 75 and 100% offset population percentiles within the
        energy bins of the pre-defined width.

        The input MC file should be of DL2 level - i.e. include the
        reconstructed event directions ("reco_src_x" and "reco_src_y"
        columns describing it in the telescope camera frame).
        """
    )

    parser.add_argument(
        '-i',
        "--input",
        default='',
        help='input Monte Carlo file name'
    )
    parser.add_argument(
        '-o',
        "--output",
        default='out.h5',
        help='output Monte Carlo file name with event classes marked'
    )
    parser.add_argument(
        '-k',
        "--key",
        default='dl2/event/telescope/parameters/LST_LSTCam',
        help='input HDF5 file key to read from'
    )
    parser.add_argument(
        '-e',
        "--ebinsdec",
        default=10,
        help='number of true energy bins per dec to assume'
    )
    parser.add_argument(
        '-c',
        "--cuts",
        default='gammaness > 0.7 & intensity > 50 & r < 1 & wl > 0.01 & wl < 1 & leakage_intensity_width_2 < 1',
        help='event cuts to apply'
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

    copyfile(args.input, args.output)
    data = mkmarkup(args.input, args.key, args.ebinsdec, args.cuts)
    data.to_hdf(args.output, key=args.key, complevel=args.complevel)


if __name__ == "__main__":
    main()
