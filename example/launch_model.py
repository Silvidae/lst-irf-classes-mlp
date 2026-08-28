# Script to run the MLP training and application for different partitions
import subprocess
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=["rf", "mlp"],
    required=True,
    help="Model to train and apply"
)
parser.add_argument(
    "--path",
    required=True,
    help="Base path of the project directory"
)

args = parser.parse_args()


partitions = [
    "20",
    "35",
    "50",
    "15_35",
    "20_45",
    "33_66",
    "10_20_35",
    "15_30_60",
    "25_50_75",
    "20_40_60_80",
]

path = args.path
proton_file = path+"MC/Protons/dl2_simtel_corsika_theta_19.807_az_66.706_merged.h5"
gamma_file  = path+"MC/Gamma/dl2_simtel_corsika_theta_19.807_az_66.706_merged.h5"
fraction    = [0.5, 0.5]   # train / test 
log_dir     = Path(path + "/logs")
log_dir.mkdir(exist_ok=True)

script = {
    "rf": "rf_train_apply.py",
    "mlp": "mlp_train_apply.py",
}[args.model]

processes = []

for part_str in partitions:
    part_args = part_str.split("_")

    logfile = log_dir / f"{args.model}_{part_str}.log"

    cmd = [
        "nohup", "python", script,
        "--proton_file", proton_file,
        "--gamma_file", gamma_file,
        "--path", path,
        "-f", *map(str, fraction),
        "-p", *part_args
    ]

    print("Running:", " ".join(cmd))
    with open(logfile, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        processes.append(p)
