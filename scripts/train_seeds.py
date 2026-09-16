"""Sequential multi-seed training without platform-specific shell loops."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--seeds", default="11,22,33")
    p.add_argument("--output", default="runs/full")
    p.add_argument("--steps", type=int)
    p.add_argument("--ablation-charging", action="store_true")
    args = p.parse_args()
    for seed in [int(s) for s in args.seeds.split(",")]:
        cmd = [
            sys.executable,
            "-m",
            "fleetrl",
            "train",
            "--config",
            args.config,
            "--seed",
            str(seed),
            "--output",
            str(Path(args.output) / f"seed_{seed}"),
        ]
        if args.steps:
            cmd += ["--steps", str(args.steps)]
        if args.ablation_charging:
            cmd += ["--ablation-charging"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
