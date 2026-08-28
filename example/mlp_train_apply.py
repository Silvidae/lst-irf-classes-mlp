#!/usr/bin/env python
# coding: utf-8

import argparse
import subprocess
import os
import shutil

import pandas as pd 
import numpy as np
from shutil import copy2
import matplotlib.pyplot as pyplot

import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion

from gammapy.data import Observation, FixedPointingInfo, observatory_locations
from gammapy.datasets import Datasets, SpectrumDataset
from gammapy.estimators import FluxPointsEstimator
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel, Models

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main(args):
    fraction_ints = [int(f*100) for f in args.fraction]
    name = f"f-{'_'.join(map(str, fraction_ints))}_p-{'_'.join(map(str, args.partition))}"
    path = args.path
    dir_name = f'{path}/processed_{name}'

    shutil.rmtree(dir_name, ignore_errors=True)
    os.makedirs(dir_name, exist_ok=True)

    # Create processed directory
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(f'{dir_name}/results_{name}', exist_ok=True)

    # Copy MC files
    try:
        copy2(args.proton_file, "./protons/")
        copy2(args.gamma_file, "./GammaDiffuse/")
    except Exception as e:
        print(f"Copy failed: {e}")

    # Split gamma file
    fraction_str = ' '.join(map(str, args.fraction))
    run(f"icmcsplit -i GammaDiffuse/{os.path.basename(args.gamma_file)} -o {dir_name}/gamma_ -f {fraction_str}")

    partition_str = ' '.join(map(str, args.partition))
    # Markup
    run("icmkmarkup -i " + dir_name + "/gamma_part0.h5 -o " + dir_name + "/gamma_part0_marked.h5 "
        f"-c 'gammaness > 0.7 & intensity > 50 & r < 1 & wl > 0.01 & wl < 1 & leakage_intensity_width_2 < 1' -p {partition_str}")
    
    run(f"icmkmarkup -i {dir_name}/gamma_part1.h5 -o {dir_name}/gamma_part1_marked.h5 -c '' -p {partition_str}")

    # Train MLP
    run(f"ictrainmlp -i {dir_name}/gamma_part0_marked.h5 -o {dir_name}/psf_ -c ic_mlp_std_config.json")
    
    # Apply MLP
    run(f"icapplymlp -i {dir_name}/gamma_part1_marked.h5 "
        f"-r {dir_name}/psf_ic_mlp.pkl "
        f"-o {dir_name}/ic_ "
        f"--partition {partition_str}")
    run(f"icapplymlp -i {dir_name}/gamma_part1_marked.h5 "
        f"-r {dir_name}/psf_ic_mlp.pkl "
        f"-c /simulation/run_config "
        f"-o {dir_name}/ic_ "
        f"--partition {partition_str} "
        "--split")

    processed_files = os.listdir(dir_name)
    print("Processed files:")
    for f in processed_files:
        print(f)

    #Plot confusion matrix
    sample = pd.read_hdf(
        f'{dir_name}/ic_gamma_part1_marked.h5', 
        key='/dl2/event/telescope/parameters/LST_LSTCam'
    )
    sample = sample.query('gammaness > 0.7 & intensity > 50 & r < 1 & wl > 0.01 & wl < 1 & leakage_intensity_width_2 < 1')
    n_classes = len(np.unique(sample['psf_class']))
    edges = 0.5 + np.arange(n_classes + 1)
    counts, _, _ = np.histogram2d(
        sample['psf_class'].values,
        sample['pred_psf_class'].values,
        bins=(edges, edges)
    )

    density = counts / counts.sum(axis=1)[..., None]
    pyplot.figure(figsize=(12, 5))

    pyplot.subplot(121, aspect='equal')
    pyplot.xlabel('PSF class (True)')
    pyplot.ylabel('PSF class (predicted)')
    pyplot.hist2d(
        sample['psf_class'].values,
        sample['pred_psf_class'].values,
        bins=(edges, edges)
    );
    pyplot.colorbar(label='counts')

    pyplot.subplot(122, aspect='equal')
    pyplot.xlabel('PSF class (True)')
    pyplot.ylabel('PSF class (predicted)')
    pyplot.pcolormesh(
        edges,
        edges,
        density.transpose()
    );
    pyplot.colorbar(label='density')
    pyplot.savefig(f'{dir_name}/results_{name}/MLP_matrix_{name}.png')

    
    # Apply MLP to protons
    run(f"icapplymlp -i protons/dl2_simtel_corsika_theta_19.807_az_66.706_merged.h5 "
        f"-r {dir_name}/psf_ic_mlp.pkl "
        f"-c /simulation/run_config "
        f"-o {dir_name}/ic_ "
        f"--partition {partition_str} "
        "--split")

    # Create IRF files for all events
    run("lstchain_create_irf_files --overwrite "
        f"-g 'GammaDiffuse/{os.path.basename(args.gamma_file)}' "
        f"-p 'protons/{os.path.basename(args.proton_file)}' "
        f"-e 'GammaDiffuse/{os.path.basename(args.gamma_file)}' "
        f"-o '{dir_name}/irf.fits' "
        f"-c 'irf_dl3_tool_config.json'")

    # Create IRF files for each PSF class
    proton_base = os.path.basename(args.proton_file).replace('.h5', '')
    for idx in range(1, n_classes + 1):
        run(f"lstchain_create_irf_files --overwrite "
            f"-g {dir_name}/ic_gamma_part1_marked_class{idx}.h5 "
            f"-p {dir_name}/ic_{proton_base}_class{idx}.h5 "
            f"-e {dir_name}/ic_gamma_part1_marked_class{idx}.h5 "
            f"-o {dir_name}/irf_MLP_class{idx}.fits "
            f"-c irf_dl3_tool_config.json")


    # Calculate sensitivity and save results
    energy_axis = MapAxis.from_energy_bounds(0.03 * u.TeV, 30 * u.TeV, nbin=20)
    energy_axis_true = MapAxis.from_energy_bounds(
        0.01 * u.TeV, 100 * u.TeV, nbin=100, name="energy_true"
    )

    pointing = SkyCoord(ra=0 * u.deg, dec=0 * u.deg)
    pointing_info = FixedPointingInfo(fixed_icrs=pointing)
    offset = 0.5 * u.deg

    source_position = pointing.directional_offset_by(0 * u.deg, offset)
    on_region_radius = 0.1 * u.deg
    on_region = CircleSkyRegion(source_position, radius=on_region_radius)

    geom = RegionGeom.create(on_region, axes=[energy_axis])

    location = observatory_locations["cta_north"]
    livetime = 50.0 * u.h

    maker = SpectrumDatasetMaker(selection=["exposure", "background", "edisp"])



    # Define the models (Crab Nebula)
    spectral_model = PowerLawSpectralModel(index=2.62)
    crab_model = SkyModel(
        spectral_model=spectral_model,
        name="Crab Nebula",
    )
    models = Models([crab_model])


    #Sensitivity estimation with UL method 
    def get_sensitivity_ul(observations):
        datasets = Datasets()
        for obs in observations:
            empty_dataset = SpectrumDataset.create(geom, energy_axis_true=energy_axis_true)
            dataset = maker.run(empty_dataset, obs)
            containment = 0.68
            # correct exposure
            dataset.exposure *= containment
            # correct background estimation
            on_radii = obs.psf.containment_radius(
                energy_true=energy_axis.center, offset=offset, fraction=containment
            )
            factor = (1 - np.cos(on_radii)) / (1 - np.cos(on_region_radius))
            dataset.background *= factor.value.reshape((-1, 1, 1))
            # Set the counts to only background
            dataset.counts = dataset.background
            dataset.counts.data = np.floor(dataset.counts.data)
            dataset.models = models
            datasets.append(dataset)
        # Estimate flux points
        fpe = FluxPointsEstimator(
            energy_edges=energy_axis.edges,
            source="Crab Nebula",
            n_sigma_ul=5,
            selection_optional=["all"],         # upper limits is one of the options included in "all"
            norm = {"max": 1e4, "min": 0.0},    # increasing norm_max to prevent the fit from failing 
        ).run(datasets)
        result = fpe.e2dnde_ul.data[:, 0, 0] * fpe.e2dnde_ul.unit
        fix_10_counts = True
        if fix_10_counts:
            back_zero = fpe.counts.data[:,0,0,0] == 0
            result[back_zero] *= 10/12.5
            
        return result



    irf_ref = load_irf_dict_from_file(f'{dir_name}/irf.fits')
    irfs = {
        evcls: load_irf_dict_from_file(f'{dir_name}/irf_MLP_class{evcls}.fits')
        for evcls in range(1, n_classes + 1)
    }    

    observations_ref = [Observation.create(
        pointing=pointing_info, irfs=irf_ref, livetime=livetime, location=location
    )]
    observations_evcls = [Observation.create(
        pointing=pointing_info, irfs=irfs[evcls], livetime=livetime, location=location
    )for evcls in range(1, n_classes + 1)]
    # compute bin centers from edges
    energy_centers = 0.5 * (energy_axis.edges[1:] + energy_axis.edges[:-1])

    sens_ref = get_sensitivity_ul(observations_ref)
    sens_evcls_combined = get_sensitivity_ul(observations_evcls)
    sens_evcls = [
        get_sensitivity_ul([obs]) for obs in observations_evcls
    ]

    np.savez(
        f"{dir_name}/results_{name}/MLP_sens_{name}.npz",
        energy=energy_centers.value,
        energy_unit=str(energy_centers.unit),
        sens_ref=sens_ref.value,
        sens_evcls_combined=sens_evcls_combined.value,
        sens_evcls=np.array([s.value for s in sens_evcls]),
        flux_unit=str(sens_ref.unit)
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP Train and Apply Pipeline")
    parser.add_argument(
        '--proton_file', 
        type=str, 
        required=True, 
        help='Path to proton H5 file'
        )
    parser.add_argument(
        '--gamma_file', 
        type=str, 
        required=True, 
        help='Path to gamma H5 file'
        )
    parser.add_argument(
        '-f',
        '--fraction',
        nargs='+',
        type=float,
        default=[0.5, 0.5],
        help='Train/test statistics (e.g. --fraction 0.5 0.5)'
    )
    parser.add_argument(
        '-p',
        '--partition',
        nargs='+',
        type=int,
        default=[25, 50, 75],
        help='Event class cumulative partition (e.g. -p 25 50 75)'
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Base project path"
    )
    args = parser.parse_args()
    main(args)
