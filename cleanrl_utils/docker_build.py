import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="cleanrl:latest", help="the name of this experiment")
args = parser.parse_args()

subprocess.run(
    f"docker build -t {args.tag} .",
    shell=True,
    check=True,
)
