import subprocess


def test_submit_exp_no_build():
    subprocess.run(
        "poetry run python -m cleanrl_utils.submit_exp --docker-tag vwxyzjn/cleanrl:latest --wandb-key xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        shell=True,
        check=True,
    )
